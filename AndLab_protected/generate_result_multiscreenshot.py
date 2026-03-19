"""
Multi-screenshot evaluator for AndLab task results.

This script extends the original result generation flow with two behaviors:
1. `operation` tasks can send the last N chronological before/end screenshots
   to a vision model in one request and overwrite `complete`.
2. `query_detect` tasks keep the original text-answer judging logic, but also
   use the LLM judge result to overwrite `complete`, while also recording
   the judge prompt / response / reason into the output result.

Common usage:
    python generate_result_multiscreenshot.py \
      --input_folder logs/evaluation \
      --target_dirs test_gliner \
      --operation_judge_model Qwen2.5-VL-72B-Instruct \
      --query_judge_model Qwen3-VL-235B-A22B-Thinking \
      --api_base https://your-openai-compatible-endpoint/v1 \
      --api_key <your-api-key> \
      --tail_image_count 8 \
      --max_workers 8 \
      --evaluate_metric_type both

Important arguments:
    --input_folder
        Evaluation log root produced by `eval.py`.
    --output_folder
        Directory where new `results.jsonl` / `total.jsonl` will be written.
    --target_dirs
        One or more run directory names under `input_folder`.
    --operation_judge_model / --query_judge_model
        Type-specific judge model names for `operation` and `query_detect`.
    --api_base / --api_key
        Shared base URL and key for all judge model requests.
    --tail_image_count
        Only send the last N ordered before/end screenshots for `operation`.
    --max_workers
        Task evaluation parallelism. Default is 1, which is serial.
    --evaluate_metric_type
        Which task type to evaluate: `both`, `operation`, or `query_detect`.
"""

import argparse
import datetime
import json
import os
import threading
from typing import Any, Dict, List, Optional, Tuple

import jsonlines

from evaluation.configs import AppConfig
from evaluation.definition import detect_answer_test, detect_answer_with_details
from evaluation.multiscreenshot_judge import judge_complete_with_multiscreenshot
from evaluation.task import (
    Evaluation_Task,
    deanonymize_text_content,
    deanonymize_xml_tree,
    dump_xml,
    extract_unresolved_tokens,
    load_task_token_mapping,
)
from generate_result import find_all_task_files, find_all_traces_files
from recalculate_metrics import calculate_average_metrics
from utils_mobile.privacy_protection import get_privacy_layer


def parse_args():
    parser = argparse.ArgumentParser(
        description="使用多张 before/end 截图 + 任务描述 + 规则结果来重新评估 complete 字段，支持 OpenAI 兼容接口的第三方 LLM",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog=(
            "示例:\n"
            "  python generate_result_multiscreenshot.py \\\n"
            "    --input_folder logs/evaluation \\\n"
            "    --target_dirs test_gliner \\\n"
            "    --operation_judge_model Qwen2.5-VL-72B-Instruct \\\n"
            "    --query_judge_model Qwen3-VL-235B-A22B-Thinking \\\n"
            "    --api_base https://your-openai-compatible-endpoint/v1 \\\n"
            "    --api_key <your-api-key> \\\n"
            "    --tail_image_count 8 \\\n"
            "    --max_workers 8 \\\n"
            "    --evaluate_metric_type both"
        ),
    )
    parser.add_argument(
        "--input_folder",
        type=str,
        default="logs/evaluation",
        help="`eval.py` 生成的运行日志根目录",
    )
    parser.add_argument(
        "--output_folder",
        type=str,
        default="outputs",
        help="评估结果输出目录；会在其中新建 `<run_name>_multiscreenshot_<time>` 子目录",
    )
    parser.add_argument(
        "--operation_judge_model",
        type=str,
        default="",
        help="operation 多图判定使用的 judge model",
    )
    parser.add_argument(
        "--query_judge_model",
        type=str,
        default="",
        help="query_detect 判题使用的 judge model",
    )
    parser.add_argument(
        "--api_base",
        type=str,
        default="",
        help="OpenAI 兼容接口的 base URL；第三方服务通常需要写到 `/v1`",
    )
    parser.add_argument(
        "--api_key",
        type=str,
        default="",
        help="Judge model 的 API key",
    )
    parser.add_argument(
        "--target_dirs",
        type=str,
        nargs="+",
        default=None,
        help="只评估指定的运行目录名；不传则扫描 input_folder 下全部目录",
    )
    parser.add_argument(
        "--tail_image_count",
        type=int,
        default=8,
        help="operation 任务只发送按时间顺序排列后的最后 N 张 before/end 截图；不足 N 张则全部发送",
    )
    parser.add_argument(
        "--max_workers",
        type=int,
        default=1,
        help="单个运行目录内的任务评估并行度；默认 1 表示串行",
    )
    parser.add_argument(
        "--evaluate_metric_type",
        type=str,
        choices=["both", "operation", "query_detect"],
        default="both",
        help="选择要评估的 metric_type；默认 both 表示 operation 和 query_detect 都评估",
    )
    return parser.parse_args()


def _build_args_with_judge_model(args, judge_model: str):
    return argparse.Namespace(**{**vars(args), "judge_model": judge_model})


def _get_judge_model_for_metric_type(args, metric_type: str) -> str:
    if metric_type == "operation":
        return getattr(args, "operation_judge_model", "")
    if metric_type == "query_detect":
        return getattr(args, "query_judge_model", "")
    return ""


def _validate_args(args):
    metric_type = getattr(args, "evaluate_metric_type", "both")
    missing = []
    if metric_type in {"both", "operation"} and not getattr(args, "operation_judge_model", ""):
        missing.append("--operation_judge_model")
    if metric_type in {"both", "query_detect"} and not getattr(args, "query_judge_model", ""):
        missing.append("--query_judge_model")

    if missing:
        raise ValueError(
            "Missing required judge model arguments for the selected metric type: "
            + ", ".join(missing)
        )


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
            "operation_extra_prompt": result.get("operation_extra_prompt"),
            "used_images": result.get("used_images", []),
            "needs_manual_review": result.get("needs_manual_review", False),
        }

    def _should_evaluate_metric_type(self, metric_type: str) -> bool:
        selected = getattr(self.args, "evaluate_metric_type", "both")
        return selected == "both" or selected == metric_type

    @staticmethod
    def _is_finish_line(line: Optional[Dict[str, Any]]) -> bool:
        if not line:
            return False
        parsed_action = line.get("parsed_action", {})
        return parsed_action.get("action") == "finish" or parsed_action.get("type") == "finish"

    @staticmethod
    def _build_query_detect_judge_details(
        metric,
        line: Optional[Dict[str, Any]],
        args,
    ) -> Optional[Dict[str, Any]]:
        if line is None:
            return None
        parsed_action = line.get("parsed_action", {})
        if parsed_action.get("action") != "finish" and parsed_action.get("type") != "finish":
            return None
        if getattr(metric, "final_ground_truth", None) is None:
            return None

        try:
            question = line["target"]
            if "kwargs" in parsed_action:
                model_answer = parsed_action["kwargs"]["message"]
            else:
                model_answer = parsed_action["input"]

            question = deanonymize_text_content(question, getattr(metric, "token_mapping", None))
            model_answer = deanonymize_text_content(model_answer, getattr(metric, "token_mapping", None))
            ground_truth = deanonymize_text_content(metric.final_ground_truth, getattr(metric, "token_mapping", None))
            judge_details = detect_answer_with_details(question, model_answer, ground_truth, args)
            unresolved_tokens = {
                "question": extract_unresolved_tokens(question),
                "model_answer": extract_unresolved_tokens(model_answer),
                "standard_answer": extract_unresolved_tokens(ground_truth),
            }
            return {
                "query_detect_judge_prompt": judge_details.get("judge_prompt"),
                "query_detect_judge_response": judge_details.get("judge_response"),
                "query_detect_judge_reason": judge_details.get("judge_reason"),
                "query_detect_judge_question": question,
                "query_detect_judge_model_answer": model_answer,
                "query_detect_judge_standard_answer": ground_truth,
                "query_detect_judge_complete": judge_details.get("complete"),
                "query_detect_unresolved_tokens": unresolved_tokens,
            }
        except Exception as exc:
            return {
                "query_detect_judge_prompt": None,
                "query_detect_judge_response": None,
                "query_detect_judge_question": None,
                "query_detect_judge_model_answer": None,
                "query_detect_judge_standard_answer": getattr(metric, "final_ground_truth", None),
                "query_detect_judge_complete": None,
                "query_detect_unresolved_tokens": None,
                "query_detect_judge_error": str(exc),
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
        metric_type = self.config.metrics_type[task_id]
        if not self._should_evaluate_metric_type(metric_type):
            return
        judge_args = _build_args_with_judge_model(
            self.args, _get_judge_model_for_metric_type(self.args, metric_type)
        )
        metric = self.metrics[task_id](self.args)
        metric.token_mapping = None
        final_result = {"complete": False}

        if task_id not in self.traces:
            print(f"Trace for task '{task_id}' not found.")
            return

        if not os.path.exists(self.traces[task_id]["trace_file"]):
            return

        task_trace_root = self.traces[task_id]["trace_root"]
        local_token_to_real = load_task_token_mapping(task_trace_root)
        if local_token_to_real:
            metric.token_mapping = local_token_to_real

        all_operation_trace = []
        original_task_prompt = None
        last_action = None
        last_action_context = None
        final_result_line = None
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

                if line.get("parsed_action"):
                    last_action_context = {
                        "parsed_action": line.get("parsed_action"),
                        "current_response": line.get("current_response"),
                    }

                if not os.path.exists(xml_path):
                    print(f"XML file not found: {xml_path}")
                    if metric_type == "query_detect" and self._is_finish_line(line):
                        try:
                            result = metric.judge(None, line)
                            all_operation_trace.append(line)
                            if "judge_page" in result.keys() and not result.get("judge_page"):
                                continue
                            final_result = result
                            final_result_line = line
                        except Exception:
                            pass
                    continue

                xml_compressed = dump_xml(xml_path)
                if xml_compressed is not None and local_token_to_real:
                    xml_compressed = deanonymize_xml_tree(xml_compressed, local_token_to_real)

                try:
                    result = metric.judge(xml_compressed, line)
                    all_operation_trace.append(line)

                    if "judge_page" in result.keys() and not result.get("judge_page"):
                        continue
                    final_result = result
                    final_result_line = line
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

        if metric_type == "operation":
            rule_result = dict(final_result)
            rule_complete = rule_result.get("complete", False)
            vision_result = judge_complete_with_multiscreenshot(
                task_id=task_id,
                image_paths=ordered_images,
                task_prompt=original_task_prompt,
                rule_result=rule_result,
                final_action=last_action_context,
                args=judge_args,
            )

            final_result["rule_complete_before_vision"] = rule_complete
            final_result["complete"] = vision_result.get("complete")
            for key, value in vision_result.items():
                if key == "complete":
                    continue
                final_result[key] = value
        elif metric_type == "query_detect":
            rule_complete = final_result.get("complete", False)
            query_detect_judge_details = self._build_query_detect_judge_details(metric, final_result_line, judge_args)
            if query_detect_judge_details is not None:
                final_result.update(query_detect_judge_details)
                final_result["rule_complete_before_llm"] = rule_complete
                judge_complete = query_detect_judge_details.get("query_detect_judge_complete")
                if isinstance(judge_complete, bool):
                    final_result["complete"] = judge_complete

        if self.show_detail_metrics:
            self.add_metrics(task, all_operation_trace, before_images, final_result)

        self.save_single(task, final_result)

        if local_token_to_real:
            privacy_layer = get_privacy_layer()
            privacy_layer.token_to_real.clear()
            privacy_layer.real_to_token.clear()


def evaluate_all_tasks(tasks: List[Evaluation_Task], max_workers: int):
    for task in tasks:
        try:
            task.evaluate(max_workers=max_workers)
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

    evaluate_all_tasks(tasks, max_workers=max(1, args.max_workers))

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
    _validate_args(args)
    query_test_args = _build_args_with_judge_model(
        args, _get_judge_model_for_metric_type(args, "query_detect")
    )
    detect_answer_test(query_test_args)

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

    for input_dir in filtered_input_dirs:
        evaluate_input_dir(input_dir, task_yamls, create_time, args)


if __name__ == "__main__":
    main()
