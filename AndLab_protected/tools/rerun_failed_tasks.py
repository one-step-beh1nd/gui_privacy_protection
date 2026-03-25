#!/usr/bin/env python3
import argparse
import datetime
from glob import glob
import json
import os
from os.path import isdir, isfile, join, relpath
import subprocess
import sys
from typing import Dict, List, Tuple

import jsonlines
import yaml

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def parse_args():
    task_config_dir = os.path.join(PROJECT_ROOT, "evaluation", "config")
    default_task_configs = []
    if os.path.isdir(task_config_dir):
        default_task_configs = [
            os.path.join("evaluation", "config", name)
            for name in os.listdir(task_config_dir)
            if name.endswith(".yaml")
        ]

    parser = argparse.ArgumentParser(
        description="Find failed tasks in an eval run and rerun them with the same task filtering logic as multiscreenshot evaluation."
    )
    parser.add_argument("--run_name", required=True, help="Directory name under logs/evaluation to inspect")
    parser.add_argument("-c", "--config", required=True, help="eval.py config yaml used for rerun")
    parser.add_argument("--task_config", nargs="+", default=default_task_configs, help="Task config yaml(s) to load")
    parser.add_argument("--task_id", nargs="+", default=None, help="Optional task ids to restrict the candidate set")
    parser.add_argument("--app", nargs="+", default=None, help="Optional app names to restrict the candidate set")
    parser.add_argument("-p", "--parallel", type=int, default=1, help="Parallel workers for each rerun invocation")
    parser.add_argument("--evaluation_root", default="logs/evaluation", help="Root directory containing eval runs")
    parser.add_argument("--max_retries", type=int, default=3, help="Maximum rerun attempts")
    parser.add_argument("--rerun_name", default=None, help="Optional output run name; defaults to <run_name>_rerun_<timestamp>")
    return parser.parse_args()


def normalize_task_ids(raw_task_ids):
    if raw_task_ids is None:
        return None
    task_ids = []
    for item in raw_task_ids:
        if "," in item:
            task_ids.extend(part.strip() for part in item.split(",") if part.strip())
        elif item.strip():
            task_ids.append(item.strip())
    return task_ids


def load_autotask_class(config_path: str) -> str:
    with open(config_path, "r", encoding="utf-8") as fp:
        yaml_data = yaml.safe_load(fp)
    task_config = yaml_data.get("task", {})
    return task_config.get("class", "ScreenshotMobileTask_AutoTest")


def find_all_task_files(task_config_paths: List[str]) -> List[str]:
    task_files = []
    for task in task_config_paths:
        abs_task = task if os.path.isabs(task) else os.path.join(PROJECT_ROOT, task)
        if isdir(abs_task):
            task_files.extend(relpath(path, PROJECT_ROOT) for path in glob(join(abs_task, "**/*.yaml"), recursive=True))
        elif isfile(abs_task):
            task_files.append(relpath(abs_task, PROJECT_ROOT))
        else:
            print(f"Ignored invalid task config path: {task}")
    return task_files


def load_task_catalog(task_config_path: str):
    abs_path = task_config_path if os.path.isabs(task_config_path) else os.path.join(PROJECT_ROOT, task_config_path)
    with open(abs_path, "r", encoding="utf-8") as fp:
        data = yaml.safe_load(fp) or {}
    tasks = data.get("tasks", []) or []
    return {
        "app": data.get("APP"),
        "task_ids": [task.get("task_id") for task in tasks if task.get("task_id")],
    }


def find_all_traces_files(traces_path_fold: str) -> Dict[str, Dict[str, str]]:
    traces = {}
    if not os.path.isdir(traces_path_fold):
        return traces
    for trace in os.listdir(traces_path_fold):
        trace_root = os.path.join(traces_path_fold, trace)
        if not os.path.isdir(trace_root):
            continue
        parts = trace.split("_")
        if len(parts) < 2:
            continue
        task_id = f"{parts[0]}_{parts[1]}"
        traces[task_id] = {
            "task_id": task_id,
            "trace_file": os.path.join(trace_root, "traces", "trace.jsonl"),
            "xml_path": os.path.join(trace_root, "xml"),
            "trace_root": trace_root,
        }
    return traces


def collect_expected_tasks(task_config_paths: List[str], task_ids_filter=None, app_filter=None) -> List[str]:
    expected = []
    normalized_task_ids = normalize_task_ids(task_ids_filter)
    for app_task_config_path in find_all_task_files(task_config_paths):
        task_catalog = load_task_catalog(app_task_config_path)
        current_task_ids = normalized_task_ids or task_catalog["task_ids"]
        if app_filter is not None and task_catalog["app"] not in app_filter:
            continue
        for task_id in current_task_ids:
            if task_id in task_catalog["task_ids"] and task_id not in expected:
                expected.append(task_id)
    return expected


def trace_finishes(trace_file: str) -> bool:
    try:
        with jsonlines.open(trace_file) as reader:
            lines = list(reader)
        if not lines:
            return False
        parsed_action = lines[-1].get("parsed_action", {})
        return parsed_action.get("action") == "finish" or parsed_action.get("type") == "finish"
    except Exception:
        return False


def detect_failed_tasks(run_dir: str, expected_task_ids: List[str], autotask_class: str) -> Tuple[List[str], Dict[str, str]]:
    traces = find_all_traces_files(run_dir)
    reasons: Dict[str, str] = {}

    for task_id in expected_task_ids:
        trace = traces.get(task_id)
        if trace is None:
            reasons[task_id] = "missing_trace_dir"
            continue
        if not os.path.exists(trace["trace_file"]):
            reasons[task_id] = "missing_trace_file"
            continue
        if not trace_finishes(trace["trace_file"]):
            reasons[task_id] = "trace_not_finished"
            continue

        if autotask_class == "ScreenshotMobileTask_AutoTest":
            screen_dir = os.path.join(trace["trace_root"], "Screen")
            has_end_screenshot = os.path.isdir(screen_dir) and any(
                name.startswith("screenshot-end") for name in os.listdir(screen_dir)
            )
            if not has_end_screenshot:
                reasons[task_id] = "missing_end_screenshot"

    return list(reasons.keys()), reasons


def build_rerun_name(run_name: str, user_value: str = None) -> str:
    if user_value:
        return user_value
    ts = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    return f"{run_name}_rerun_{ts}"


def run_eval_for_tasks(args, output_run_name: str, task_ids: List[str]) -> int:
    command = [
        sys.executable,
        os.path.join(PROJECT_ROOT, "eval.py"),
        "-n",
        output_run_name,
        "-c",
        args.config,
        "--parallel",
        str(max(1, args.parallel)),
    ]
    if args.task_config:
        command.extend(["--task_config", *args.task_config])
    if args.app:
        command.extend(["--app", *args.app])
    command.extend(["--task_id", *task_ids])

    completed = subprocess.run(command, cwd=PROJECT_ROOT)
    return completed.returncode


def write_report(report_path: str, payload: dict):
    with open(report_path, "w", encoding="utf-8") as fp:
        json.dump(payload, fp, ensure_ascii=False, indent=2)


def main():
    args = parse_args()
    run_dir = os.path.join(PROJECT_ROOT, args.evaluation_root, args.run_name)
    if not os.path.isdir(run_dir):
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    autotask_class = load_autotask_class(args.config)
    expected_task_ids = collect_expected_tasks(args.task_config, args.task_id, args.app)
    failed_task_ids, initial_reasons = detect_failed_tasks(run_dir, expected_task_ids, autotask_class)

    output_run_name = build_rerun_name(args.run_name, args.rerun_name)
    report = {
        "source_run_name": args.run_name,
        "rerun_name": output_run_name,
        "autotask_class": autotask_class,
        "expected_task_ids": expected_task_ids,
        "initial_failed_tasks": initial_reasons,
        "attempts": [],
        "final_remaining_failed_tasks": {},
    }

    remaining = failed_task_ids[:]
    last_reasons = initial_reasons
    for attempt in range(1, args.max_retries + 1):
        if not remaining:
            break
        exit_code = run_eval_for_tasks(args, output_run_name, remaining)
        rerun_dir = os.path.join(PROJECT_ROOT, args.evaluation_root, output_run_name)
        current_failed, current_reasons = detect_failed_tasks(rerun_dir, remaining, autotask_class)
        last_reasons = current_reasons
        report["attempts"].append(
            {
                "attempt": attempt,
                "requested_task_ids": remaining[:],
                "exit_code": exit_code,
                "remaining_failed_tasks": current_reasons,
            }
        )
        remaining = current_failed

    report["final_remaining_failed_tasks"] = {task_id: last_reasons.get(task_id, "rerun_failed") for task_id in remaining}

    report_path = os.path.join(PROJECT_ROOT, args.evaluation_root, output_run_name, "rerun_failed_tasks_report.json")
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    write_report(report_path, report)

    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"Report saved to: {report_path}")


if __name__ == "__main__":
    main()
