from collections import defaultdict
from typing import Generic, TypeVar, Optional, List
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Any
from tqdm import tqdm
import json
import os
import re
import jsonlines
import numpy as np
from PIL import Image

from evaluation.definition import *
from evaluation.utils import *
from utils_mobile.utils import get_compressed_xml
from utils_mobile.privacy_protection import get_privacy_layer


T_INPUT = TypeVar('T_INPUT')
T_OUTPUT = TypeVar('T_OUTPUT')
T_TARGET = TypeVar('T_TARGET')

TOKEN_PATTERN = re.compile(r"\[?([A-Z_]+#[A-Za-z0-9]+)\]?")





def dump_xml(xml_path):
    xml_compressed = get_compressed_xml(xml_path)
    if xml_compressed is None:
        return None
    return json.loads(xml_compressed)


def deanonymize_xml_tree(xml_tree, token_mapping):
    """
    递归地将 XML 树中的匿名 token 还原为真实值。
    
    Args:
        xml_tree: XML 压缩后的树结构（字典）
        token_mapping: token 到真实值的映射字典
    
    Returns:
        还原后的 XML 树
    """
    if not token_mapping or not xml_tree:
        return xml_tree
    
    if isinstance(xml_tree, dict):
        result = {}
        for key, value in xml_tree.items():
            # 还原 key 中的 token
            deanonymized_key = deanonymize_xml_tree(key, token_mapping)
            # 递归还原 value
            deanonymized_value = deanonymize_xml_tree(value, token_mapping)
            result[deanonymized_key] = deanonymized_value
        return result
    elif isinstance(xml_tree, list):
        return [deanonymize_xml_tree(item, token_mapping) for item in xml_tree]
    elif isinstance(xml_tree, str):
        # 对字符串进行 token 还原
        result = xml_tree
        # 按 token 长度从长到短排序，避免部分匹配问题
        sorted_tokens = sorted(token_mapping.items(), key=lambda x: len(x[0]), reverse=True)
        for token, real_value in sorted_tokens:
            if isinstance(token, str) and isinstance(real_value, str):
                result = result.replace(token, real_value)
        return result
    else:
        return xml_tree


def load_task_token_mapping(task_trace_root: str) -> Dict[str, str]:
    """
    Load task-scoped token mapping even if the privacy layer is not currently enabled.
    """
    privacy_layer = get_privacy_layer()
    token_mapping_loaded = privacy_layer.load_token_mapping(task_trace_root)
    if token_mapping_loaded and privacy_layer.token_to_real:
        return privacy_layer.token_to_real.copy()

    mapping_file = os.path.join(task_trace_root, "privacy_token_mapping.json")
    if not os.path.exists(mapping_file):
        return {}

    try:
        with open(mapping_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        token_to_real = data.get("token_to_real", {})
        if isinstance(token_to_real, dict):
            return {
                token: real_value
                for token, real_value in token_to_real.items()
                if isinstance(token, str) and isinstance(real_value, str)
            }
    except Exception:
        return {}

    return {}


def deanonymize_text_content(text: Any, token_mapping: Optional[Dict[str, str]] = None) -> Any:
    """
    Replace anonymized tokens in free-form text with real values.
    Handles both `TOKEN#abc12` and `[TOKEN#abc12]` forms.
    """
    if not isinstance(text, str):
        return text

    result = text
    mapping = token_mapping or {}

    sorted_items = sorted(mapping.items(), key=lambda x: len(x[0]), reverse=True)
    for token, real_value in sorted_items:
        if not isinstance(token, str) or not isinstance(real_value, str):
            continue
        result = result.replace(f"[{token}]", real_value)
        result = result.replace(token, real_value)

    privacy_layer = get_privacy_layer()
    if getattr(privacy_layer, "token_to_real", None):
        result = privacy_layer.convert_token_to_real(result)

    def replace_unresolved(match):
        token = match.group(1)
        real_value = mapping.get(token)
        if real_value is None and getattr(privacy_layer, "token_to_real", None):
            real_value = privacy_layer.token_to_real.get(token)
        return real_value if isinstance(real_value, str) else match.group(0)

    return TOKEN_PATTERN.sub(replace_unresolved, result)


def extract_unresolved_tokens(text: Any) -> List[str]:
    if not isinstance(text, str):
        return []
    return sorted(set(match.group(1) for match in TOKEN_PATTERN.finditer(text)))


def calculate_partial_acc(dict):
    tt = 0
    acc = 0
    for key, values in dict.items():
        if key != "complete" and key != "judge_page":
            tt += 1
            if values:
                acc += 1
    if tt == 0:
        return 0
    return acc / tt


def compute_image_similarity(image_paths):
    if len(image_paths) <= 2:
        return [], 0
    image_paths = image_paths[:-1]
    image_list = []
    for path in image_paths:
        try:
            image_list.append(np.array(Image.open(path)))
        except Exception as e:
            image_list.append(np.zeros((1, 1, 3)))

    simi = []
    sum_simi = 0

    for i in range(len(image_list) - 1):
        try:
            either_not_255 = np.logical_or(np.not_equal(image_list[i], 255), np.not_equal(image_list[i + 1], 255))
            values_match = np.equal(image_list[i], image_list[i + 1])
            match_in_either_not_255 = np.logical_and(values_match, either_not_255)

            similarity = np.sum(match_in_either_not_255.astype(np.float32)) / np.sum(either_not_255.astype(np.float32))
            simi.append(float(similarity))

            if similarity > 0.999:
                sum_simi += 1
        except Exception as e:
            simi.append(0)

    return simi, sum_simi


class Evaluation_Task(Generic[T_INPUT, T_OUTPUT, T_TARGET]):
    def __init__(self, config, traces, args, detail=False):
        self.config = config
        self.args = args
        assert self.config is not None, "Task config is required."
        self.name = self.config.APP
        self.task_list = self.config.get_tasks()
        self.metrics = self.config.get_metrics()
        self.traces = traces
        self.all_result = []
        self.show_detail_metrics = detail
        self.total_tasks_num = 138  # TODO: change this number if the number of all tasks changes
        if self.show_detail_metrics:
            self.additional_metrics = defaultdict(dict)
            with open("evaluation/tasks/human_ground_turth/ground_truth_length.json") as f:
                self.length_gt = json.load(f)

    def evaluate(self, max_workers: int = 4) -> Dict[str, Any]:
        # 使用 ThreadPoolExecutor 来控制并发任务数
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_task = {
                executor.submit(self._evaluate_single_task, task): task for task in self.task_list
            }

            for future in as_completed(future_to_task):
                task = future_to_task[future]
                try:
                    future.result()  # 获取每个并发任务的结果，如果有异常会在此处抛出
                except Exception as e:
                    print(f"Error evaluating task {task.get('task_id')}: {e}")

        self.print_metric()

    def _evaluate_single_task(self, task) -> None:
        try:
            assert task.get('task_id') in self.metrics, f"No valid function mapped for {task.get('task_id')}"
        except AssertionError:
            print(f"No valid function mapped for {task.get('task_id')}")
            return

        task_id = task.get('task_id')
        metric = self.metrics[task_id](self.args)
        # Initialize token_mapping attribute for thread-safe concurrent evaluation
        metric.token_mapping = None
        final_result = {"complete": False}

        if task_id not in self.traces:
            print(f"Trace for task '{task_id}' not found.")
            return

        if not os.path.exists(self.traces[task_id]['trace_file']):
            return

        # Load token mapping for this specific task if privacy protection was enabled
        # Each task has its own token mapping because the same real value may be
        # anonymized to different tokens in different tasks
        # IMPORTANT: Copy mapping to local variable to avoid race conditions in concurrent evaluation
        task_trace_root = self.traces[task_id]['trace_root']
        local_token_to_real = load_task_token_mapping(task_trace_root)
        
        # Store mapping in metric instance so check_answer can use it
        # This avoids race conditions when multiple tasks evaluate concurrently
        if local_token_to_real:
            metric.token_mapping = local_token_to_real
        else:
            metric.token_mapping = None

        all_operation_trace = []
        all_images = []
        agent_name = self.traces[task_id]["trace_root"].split("/")[-2]
        original_task_prompt = None  # 保存原始任务prompt
        end_screenshot_path = None  # 保存最后一张截图路径

        num_repeat = 0
        last_action = None

        with jsonlines.open(self.traces[task_id]['trace_file']) as reader:
            trace_root = self.traces[task_id]['trace_root']
            for line in reader:
                # 保存第一行的原始任务prompt
                if original_task_prompt is None and "original_instruction" in line:
                    original_task_prompt = line["original_instruction"]
                elif original_task_prompt is None and "target" in line:
                    original_task_prompt = line["target"]
                current_action = json.dumps(line["parsed_action"])
                if current_action == last_action:
                    num_repeat += 1
                    if num_repeat > 5:
                        break
                else:
                    num_repeat = 0
                    last_action = current_action

                if line["ac_xml"] is None:
                    xml_path = line["xml"]
                else:
                    xml_path = line["ac_xml"]
                xml_path = os.path.join(self.traces[task_id]['xml_path'], xml_path.split("/")[-1])
                metric_type = self.config.metrics_type[task.get('task_id')]

                if not os.path.exists(xml_path):
                    print(f"XML file not found: {xml_path}")
                    continue

                xml_compressed = dump_xml(xml_path)
                # 如果启用了隐私保护，将 XML 树中的匿名 token 还原为真实值
                if xml_compressed is not None and local_token_to_real:
                    xml_compressed = deanonymize_xml_tree(xml_compressed, local_token_to_real)
                try:
                    result = metric.judge(xml_compressed, line)
                    all_operation_trace.append(line)
                    image_path = line["image"]
                    image_filename = image_path.split("/")[-1]
                    image_path = os.path.join(trace_root, "Screen", image_filename)
                    if image_path.split('/')[-4] != agent_name:
                        image_path = image_path.replace(image_path.split('/')[-4], agent_name)
                    all_images.append(image_path)

                    if "judge_page" in result.keys() and not result.get("judge_page"):
                        continue
                    else:
                        final_result = result
                except Exception as e:
                    pass

        # 查找最后一张截图（screenshot-end）
        screen_dir = os.path.join(trace_root, "Screen")
        if os.path.exists(screen_dir):
            end_screenshots = [f for f in os.listdir(screen_dir) if f.startswith("screenshot-end")]
            if end_screenshots:
                end_screenshot_path = os.path.join(screen_dir, end_screenshots[0])
        
        # 如果启用了基于截图的判断，且是operation类型任务，使用LLM判断最后一张截图
        use_screenshot_judge = getattr(self.args, 'use_screenshot_judge', False)
        metric_type = self.config.metrics_type[task.get('task_id')]
        
        if use_screenshot_judge and metric_type == "operation" and end_screenshot_path and original_task_prompt:
            try:
                # 使用LLM判断最后一张截图
                screenshot_complete = judge_task_by_screenshot(
                    end_screenshot_path, 
                    original_task_prompt, 
                    self.args
                )
                # 覆盖complete字段，但保留其他判断结果（如judge_page等）
                if "judge_page" not in final_result:
                    final_result["judge_page"] = True  # 假设在正确页面
                final_result["complete"] = screenshot_complete
                print(f"[Screenshot Judge] Task {task_id}: complete={screenshot_complete}")
            except Exception as e:
                print(f"[Screenshot Judge] Error for task {task_id}: {e}")
                # 如果判断失败，保持原有结果

        if self.show_detail_metrics:
            self.add_metrics(task, all_operation_trace, all_images, final_result)

        self.save_single(task, final_result)
        
        # Clear this thread's in-memory mapping after evaluation (get_privacy_layer is per-thread).
        if local_token_to_real:
            privacy_layer = get_privacy_layer()
            privacy_layer.token_to_real.clear()
            privacy_layer.real_to_token.clear()

    def evaluate_old(self) -> Dict[str, Any]:
        for task in self.task_list:
            try:
                assert task.get('task_id') in self.metrics, f"No valid function mapped for {task.get('task_id')}"
            except:
                print(f"No valid function mapped for {task.get('task_id')}")
                continue
            task_id = task.get('task_id')
            metric = self.metrics[task_id](self.args)
            final_result = {"complete": False}
            if task_id not in self.traces:
                print(f"Trace for task '{task_id}' not found.")
                continue
            if not os.path.exists(self.traces[task_id]['trace_file']):
                print(f"Trace file not found: {self.traces[task_id]['trace_file']}")
                continue
            all_operation_trace = []
            all_images = []
            agent_name = self.traces[task_id]["trace_root"].split("/")[-2]

            num_repeat = 0
            last_action = None

            with jsonlines.open(self.traces[task_id]['trace_file']) as reader:
                trace_root = self.traces[task_id]['trace_root']
                for line in reader:
                    current_action = json.dumps(line["parsed_action"])
                    if current_action == last_action:
                        num_repeat += 1
                        if num_repeat > 5:
                            break
                    else:
                        num_repeat = 0
                        last_action = current_action

                    if line["ac_xml"] is None:
                        xml_path = line["xml"]
                    else:
                        xml_path = line["ac_xml"]
                    xml_path = os.path.join(self.traces[task_id]['xml_path'], xml_path.split("/")[-1])
                    metric_type = self.config.metrics_type[task.get('task_id')]
                    if not os.path.exists(xml_path):
                        print(f"XML file not found: {xml_path}")
                        continue
                    xml_compressed = dump_xml(xml_path)
                    try:
                        result = metric.judge(xml_compressed, line)
                        all_operation_trace.append(line)
                        image_path = line["image"]
                        image_filename = image_path.split("/")[-1]
                        image_path = os.path.join(trace_root, "Screen", image_filename)
                        if image_path.split('/')[-4] != agent_name:
                            image_path = image_path.replace(image_path.split('/')[-4], agent_name)
                        all_images.append(image_path)
                        if "judge_page" in result.keys() and not result.get("judge_page"):
                            continue
                        else:
                            final_result = result
                    except:
                        result = {"complete": False}
                        #import traceback
                        #traceback.print_exc()
                        #print(f"Error in judging {task_id} at line {line}")

            if self.show_detail_metrics:
                self.add_metrics(task, all_operation_trace, all_images, final_result)

            self.save_single(task, final_result)
        self.print_metric()

    def add_metrics(self, task, traces, all_images, final_result):
        # Reversed Redundancy Ratio
        length = len(traces)
        if not final_result.get("complete") or length == 0:
            RRR = None
        else:
            RRR = self.length_gt[task["task_id"]] / length if task["task_id"] in self.length_gt else None
        self.additional_metrics["RRR"][task["task_id"]] = RRR

        # Final Task Ratio
        # if traces[-1]["parsed_action"]["operation"] == "finish":
        # self.additional_metrics["final_task_ratio"][task["task_id"]] = 1
        # else:
        # self.additional_metrics["final_task_ratio"][task["task_id"]] = 0

        # Reasonable Operation Ratio
        simi, sum_simi = compute_image_similarity(all_images)
        if length - 1 == 0:
            self.additional_metrics["reasonable_operation_ratio"][task["task_id"]] = 1
        else:
            self.additional_metrics["reasonable_operation_ratio"][task["task_id"]] = 1 - (sum_simi / (length - 1))

    def save_single(self, task, result):
        save_dir = self.config.output_dir
        with jsonlines.open(os.path.join(save_dir, "results.jsonl"), mode='a') as writer:
            output_dict = {}
            output_dict["task_id"] = task.get('task_id')
            output_dict["task"] = self.config.task_name[task.get('task_id')]
            output_dict["metric_type"] = self.config.metrics_type[task.get('task_id')]
            output_dict["result"] = result
            if self.show_detail_metrics:
                for metric, metric_value in self.additional_metrics.items():
                    output_dict[metric] = metric_value[task.get('task_id')]
            # print(f"Task '{task.get('task_id')}' evaluated.")
            # print(f"Result: {result}")
            writer.write(output_dict)
            self.all_result.append(output_dict)

    def print_metric(self):
        complete_metric = defaultdict(list)
        partial_metric = defaultdict(list)

        for result in self.all_result:
            app = result["task_id"].split("_")[0]
            if result["result"].get("complete") == True:
                complete_metric[app].append(1)
                partial_metric[app].append(1)
            else:
                complete_metric[app].append(0)
                partial_metric[app].append(calculate_partial_acc(result["result"]))
        for key, values in complete_metric.items():
            with jsonlines.open(os.path.join(self.config.output_dir, "total.jsonl"), mode='a') as writer:
                output_dir = {"App": key, "Acc": sum(values) / len(values), "Total": len(values),
                              "Complete_Correct": sum(values), "Sum_Partial_Acc": sum(partial_metric[key]),
                              "Partial_Acc": sum(partial_metric[key]) / len(values)}
                if self.show_detail_metrics:
                    for metric, metric_value in self.additional_metrics.items():
                        values_set = [i for i in metric_value.values() if i is not None]
                        try:
                            output_dir[metric] = sum(values_set) / len(values_set)
                            output_dir["Sum_" + metric] = sum(values_set)
                        except:
                            output_dir[metric] = 0
                            output_dir["Sum_" + metric] = 0
                writer.write(output_dir)


class SingleTask():
    def __init__(self, args):
        self.metric_type = ""
        self.final_ground_truth = None
        self.args = args

    def check_answer(self, line):
        if line["parsed_action"].get("action") != "finish" and line["parsed_action"].get("type") != "finish":
            return False
        if self.final_ground_truth is None:
            return False

        try:
            question = line["target"]
            if "kwargs" in line["parsed_action"]:
                model_answer = line["parsed_action"]["kwargs"]["message"]
            else:
                model_answer = line["parsed_action"]["input"]

            question = deanonymize_text_content(question, getattr(self, 'token_mapping', None))
            model_answer = deanonymize_text_content(model_answer, getattr(self, 'token_mapping', None))
            ground_truth = deanonymize_text_content(self.final_ground_truth, getattr(self, 'token_mapping', None))

            if detect_answer(question, model_answer, ground_truth, self.args):
                return True
            else:
                return False
        except:
            return False

    def judge_page(self, xml_compressed_tree):
        return True

    def judge(self, xml_compressed_tree, line):
        raise NotImplementedError

    def save_answer(self, answer):
        self.final_ground_truth = answer
