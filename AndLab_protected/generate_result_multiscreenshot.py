import argparse
import concurrent.futures
import datetime
import json
import os
import threading
from typing import Dict, List, Optional, Tuple

import jsonlines

from evaluation.configs import AppConfig
from evaluation.definition import detect_answer_test
from evaluation.multiscreenshot_judge import judge_complete_with_multiscreenshot
from evaluation.task import Evaluation_Task, deanonymize_xml_tree, dump_xml
from generate_result import find_all_task_files, find_all_traces_files
from recalculate_metrics import calculate_average_metrics
from utils_mobile.privacy_protection import get_privacy_layer


def parse_args():
    parser = argparse.ArgumentParser(
        description="使用多张 before/end 截图 + 任务描述 + 规则结果来重新评估 complete 字段，支持 OpenAI 兼容接口的第三方 LLM"
    )
    parser.add_argument("--input_folder", type=str, default="logs/evaluation")
    parser.add_argument("--output_folder", type=str, default="outputs")
    parser.add_argument("--judge_model", type=str, default="gpt-4o")
    parser.add_argument("--api_base", type=str, default="")
    parser.add_argument("--api_key", type=str, default="")
    parser.add_argument(
        "--target_dirs",
        type=str,
        nargs="+",
        default=None,
        help="只评估指定的运行目录名",
    )
    parser.add_argument(
        "--vision_mode",
        type=str,
        choices=["auto", "multiscreenshot", "per_image"],
        default="auto",
        help="auto: 先尝试多图/分批，再回退逐图；multiscreenshot: 只用多图链路；per_image: 直接逐图判断",
    )
    parser.add_argument(
        "--max_images_per_request",
        type=int,
        default=20,
        help="单次发给模型的最大图片数；默认 20，避免把 25 张图写死到接口限制里",
    )
    return parser.parse_args()


class MultiScreenshotEvaluationTask(Evaluation_Task):
    def __init__(self, config, traces, args, detail=False):
        super().__init__(config, traces, args, detail=detail)
        self.write_lock = threading.Lock()

    @staticmethod
    def _screen_sort_key(filename: str) -> Tuple[int, int, str]:
        if filename.startswith("screenshot-end"):
            timestamp = filename.replace("screenshot-end-", "").replace(".png", "")
            try:
                numeric = int(float(timestamp))
            except ValueError:
                numeric = 10**18
            return (1, numeric, filename)

        parts = filename.replace(".png", "").split("-")
        if len(parts) >= 4 and parts[0] == "screenshot" and parts[-1] == "before":
            try:
                step = int(parts[1])
            except ValueError:
                step = 10**9
            try:
                timestamp = int(float(parts[2]))
            except ValueError:
                timestamp = step
            return (0, step * 10**9 + timestamp, filename)

        return (2, 10**18, filename)

    @classmethod
    def _collect_eval_images(cls, screen_dir: str) -> Tuple[List[str], List[str]]:
        before_images: List[str] = []
        end_images: List[str] = []

        if not os.path.exists(screen_dir):
            return before_images, end_images

        filenames = sorted(os.listdir(screen_dir), key=cls._screen_sort_key)
        for filename in filenames:
            path = os.path.join(screen_dir, filename)
            if not filename.endswith(".png"):
                continue
            if filename.endswith("-before.png"):
                before_images.append(path)
            elif filename.startswith("screenshot-end"):
                end_images.append(path)

        return before_images, end_images

    def _build_manual_review_entry(self, task, result):
        return {
            "task_id": task.get("task_id"),
            "task": self.config.task_name.get(task.get("task_id")),
            "metric_type": self.config.metrics_type.get(task.get("task_id")),
            "complete": result.get("complete"),
            "rule_complete_before_vision": result.get("rule_complete_before_vision"),
            "vision_status": result.get("vision_status"),
            "vision_reason": result.get("vision_reason"),
            "used_images": result.get("used_images", []),
            "conflict": result.get("conflict", False),
            "needs_manual_review": result.get("needs_manual_review", False),
        }

    def save_single(self, task, result):
        save_dir = self.config.output_dir
        output_dict = {
            "task_id": task.get("task_id"),
            "task": self.config.task_name[task.get("task_id")],
            "metric_type": self.config.metrics_type[task.get("task_id")],
            "result": result,
        }
        if self.show_detail_metrics:
            for metric, metric_value in self.additional_metrics.items():
                output_dict[metric] = metric_value[task.get("task_id")]

        with self.write_lock:
            with jsonlines.open(os.path.join(save_dir, "results.jsonl"), mode="a") as writer:
                writer.write(output_dict)
            self.all_result.append(output_dict)

            if result.get("needs_manual_review"):
                review_entry = self._build_manual_review_entry(task, result)
                with jsonlines.open(os.path.join(save_dir, "manual_review.jsonl"), mode="a") as writer:
                    writer.write(review_entry)

    def _evaluate_single_task(self, task) -> None:
        try:
            assert task.get("task_id") in self.metrics, f"No valid function mapped for {task.get('task_id')}"
        except AssertionError:
            print(f"No valid function mapped for {task.get('task_id')}")
            return

        task_id = task.get("task_id")
        metric = self.metrics[task_id](self.args)
        metric.token_mapping = None
        final_result = {"complete": False}

        if task_id not in self.traces:
            print(f"Trace for task '{task_id}' not found.")
            return

        if not os.path.exists(self.traces[task_id]["trace_file"]):
            return

        privacy_layer = get_privacy_layer()
        task_trace_root = self.traces[task_id]["trace_root"]
        token_mapping_loaded = privacy_layer.load_token_mapping(task_trace_root)
        local_token_to_real = privacy_layer.token_to_real.copy() if token_mapping_loaded else {}
        if token_mapping_loaded:
            metric.token_mapping = local_token_to_real

        all_operation_trace = []
        original_task_prompt = None
        last_action = None
        last_action_context = None
        num_repeat = 0

        with jsonlines.open(self.traces[task_id]["trace_file"]) as reader:
            trace_root = self.traces[task_id]["trace_root"]
            for line in reader:
                if original_task_prompt is None and "original_instruction" in line:
                    original_task_prompt = line["original_instruction"]
                elif original_task_prompt is None and "target" in line:
                    original_task_prompt = line["target"]

                current_action = json.dumps(line.get("parsed_action", {}), ensure_ascii=False, sort_keys=True)
                if current_action == last_action:
                    num_repeat += 1
                    if num_repeat > 5:
                        break
                else:
                    num_repeat = 0
                    last_action = current_action

                if line.get("ac_xml") is None:
                    xml_path = line["xml"]
                else:
                    xml_path = line["ac_xml"]
                xml_path = os.path.join(self.traces[task_id]["xml_path"], xml_path.split("/")[-1])

                if not os.path.exists(xml_path):
                    print(f"XML file not found: {xml_path}")
                    continue

                xml_compressed = dump_xml(xml_path)
                if xml_compressed is not None and local_token_to_real:
                    xml_compressed = deanonymize_xml_tree(xml_compressed, local_token_to_real)

                try:
                    result = metric.judge(xml_compressed, line)
                    all_operation_trace.append(line)

                    if line.get("parsed_action"):
                        last_action_context = {
                            "parsed_action": line.get("parsed_action"),
                            "current_response": line.get("current_response"),
                        }

                    if "judge_page" in result.keys() and not result.get("judge_page"):
                        continue
                    final_result = result
                except Exception:
                    pass

        if original_task_prompt is None:
            original_task_prompt = task.get("task") or self.config.task_name.get(task_id)

        screen_dir = os.path.join(self.traces[task_id]["trace_root"], "Screen")
        before_images, end_images = self._collect_eval_images(screen_dir)

        ordered_images = []
        seen_images = set()
        for image_path in [*before_images, *end_images]:
            if image_path in seen_images:
                continue
            seen_images.add(image_path)
            ordered_images.append(image_path)

        rule_result = dict(final_result)
        rule_complete = rule_result.get("complete", False)
        vision_result = judge_complete_with_multiscreenshot(
            image_paths=ordered_images,
            task_prompt=original_task_prompt,
            rule_result=rule_result,
            final_action=last_action_context,
            args=self.args,
        )

        final_result["rule_complete_before_vision"] = rule_complete
        final_result["complete"] = vision_result.get("complete")
        for key, value in vision_result.items():
            if key == "complete":
                continue
            final_result[key] = value

        if self.show_detail_metrics:
            self.add_metrics(task, all_operation_trace, before_images, final_result)

        self.save_single(task, final_result)

        if token_mapping_loaded:
            privacy_layer.token_to_real.clear()
            privacy_layer.real_to_token.clear()


def evaluate_all_tasks(tasks: List[Evaluation_Task]):
    for task in tasks:
        try:
            task.evaluate()
            del task
        except Exception as exc:
            import traceback
            print(traceback.format_exc())
            print(f"Generated an exception while evaluating task config: {exc}")


def evaluate_input_dir(input_dir, task_yamls, create_time, args):
    test_name = input_dir.split("/")[-1]
    output_root_dir = os.path.join(args.output_folder, test_name + "_multiscreenshot_" + create_time)
    if not os.path.exists(output_root_dir):
        os.makedirs(output_root_dir)

    task_files = find_all_task_files(task_yamls)
    traces = find_all_traces_files(input_dir)

    tasks = []
    print("> Loading task configs")
    for app_task_config_path in task_files:
        app_config = AppConfig(app_task_config_path, output_dir=output_root_dir)
        app_task = MultiScreenshotEvaluationTask(app_config, traces, args, detail=True)
        print(f"    MultiScreenshotEvaluationTask '{app_task.name}' loaded from config {app_task_config_path}")
        tasks.append(app_task)
    print(f"> Successfully load {len(tasks)} task{'s' if len(tasks) > 1 else ''}")

    evaluate_all_tasks(tasks)

    total_jsonl_path = os.path.join(output_root_dir, "total.jsonl")
    if os.path.exists(total_jsonl_path):
        average_metrics_path = os.path.join(output_root_dir, "average_metrics.json")
        calculate_average_metrics(total_jsonl_path, average_metrics_path)

    manual_review_path = os.path.join(output_root_dir, "manual_review.jsonl")
    if os.path.exists(manual_review_path):
        count = 0
        with jsonlines.open(manual_review_path) as reader:
            for _ in reader:
                count += 1
        with open(os.path.join(output_root_dir, "manual_review_summary.json"), "w", encoding="utf-8") as fp:
            json.dump(
                {
                    "manual_review_cases": count,
                    "manual_review_file": manual_review_path,
                },
                fp,
                ensure_ascii=False,
                indent=2,
            )


def main():
    args = parse_args()
    detect_answer_test(args)

    task_yamls = os.listdir("evaluation/config")
    task_yamls = ["evaluation/config/" + name for name in task_yamls if name.endswith(".yaml")]
    create_time = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

    all_input_dirs = [os.path.join(args.input_folder, input_dir) for input_dir in os.listdir(args.input_folder)]
    if args.target_dirs:
        input_dirs = [path for path in all_input_dirs if os.path.basename(path) in args.target_dirs]
        if not input_dirs:
            print(f"Warning: No matching directories found in {args.input_folder} for {args.target_dirs}")
            return
        print(f"> Will evaluate only: {[os.path.basename(path) for path in input_dirs]}")
    else:
        input_dirs = all_input_dirs

    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder)

    filtered_input_dirs = []
    for input_dir in input_dirs:
        if "emulator_output.txt" in input_dir:
            continue
        filtered_input_dirs.append(input_dir)

    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        futures = [
            executor.submit(evaluate_input_dir, input_dir, task_yamls, create_time, args)
            for input_dir in filtered_input_dirs
        ]
        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()
            except Exception as exc:
                import traceback
                traceback.print_exc()
                print(f"Generated an exception: {exc}")


if __name__ == "__main__":
    main()
