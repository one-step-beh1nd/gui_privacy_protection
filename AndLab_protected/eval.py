import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import argparse
import glob
import json
import shutil
import yaml

from evaluation.auto_test import *
from generate_result import find_all_task_files
from evaluation.configs import AppConfig, PrivacyConfig, TaskConfig
from evaluation.parallel import parallel_worker


def calculate_overall_anonymization_stats(task_dir: str):
    """
    Calculate overall anonymization statistics from all task folders.
    
    Args:
        task_dir: Base directory containing all task folders (e.g., "logs/evaluation/20251214_1")
    """
    if not os.path.exists(task_dir):
        print(f"[PrivacyProtection] Task directory {task_dir} does not exist, skipping statistics calculation.")
        return
    
    # Find all privacy_anonymization_stats.json files in subdirectories
    stats_files = glob.glob(os.path.join(task_dir, "**", "privacy_anonymization_stats.json"), recursive=True)
    
    if not stats_files:
        print(f"[PrivacyProtection] No anonymization statistics files found in {task_dir}")
        return
    
    # Aggregate statistics from all files
    total_original_length = 0
    total_anonymized_length = 0
    total_records = 0
    by_type: dict = {}
    task_stats = []
    
    for stats_file in stats_files:
        try:
            with open(stats_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                task_dir_name = os.path.basename(os.path.dirname(stats_file))
                
                # Aggregate from this task
                task_original = 0
                task_anonymized = 0
                task_records = data.get("total_records", 0)
                
                for record in data.get("records", []):
                    original_len = record.get("original_length", 0)
                    # Use anonymized_chars_count if available, otherwise fall back to anonymized_length for backward compatibility
                    anonymized_chars = record.get("anonymized_chars_count", record.get("anonymized_length", 0))
                    record_type = record.get("type", "unknown")
                    
                    task_original += original_len
                    task_anonymized += anonymized_chars
                    
                    # Group by type
                    if record_type not in by_type:
                        by_type[record_type] = {
                            "count": 0,
                            "original_length": 0,
                            "anonymized_chars_count": 0
                        }
                    by_type[record_type]["count"] += 1
                    by_type[record_type]["original_length"] += original_len
                    by_type[record_type]["anonymized_chars_count"] += anonymized_chars
                
                total_original_length += task_original
                total_anonymized_length += task_anonymized
                total_records += task_records
                
                # Calculate ratio for this task
                task_ratio = (task_anonymized / task_original * 100) if task_original > 0 else 0.0
                task_stats.append({
                    "task_dir": task_dir_name,
                    "original_length": task_original,
                    "anonymized_chars_count": task_anonymized,
                    "anonymization_ratio": task_ratio,
                    "records": task_records
                })
        except Exception as e:
            print(f"[PrivacyProtection] Failed to read statistics from {stats_file}: {e}")
            continue
    
    # Calculate overall ratio
    overall_ratio = (total_anonymized_length / total_original_length * 100) if total_original_length > 0 else 0.0
    
    # Calculate ratios for each type
    for stat_type in by_type:
        if by_type[stat_type]["original_length"] > 0:
            by_type[stat_type]["anonymization_ratio"] = (
                by_type[stat_type]["anonymized_chars_count"] / by_type[stat_type]["original_length"] * 100
            )
        else:
            by_type[stat_type]["anonymization_ratio"] = 0.0
    
    # Create summary
    summary = {
        "task_dir": task_dir,
        "overall_statistics": {
            "total_original_length": total_original_length,
            "total_anonymized_chars_count": total_anonymized_length,
            "anonymization_ratio_percent": round(overall_ratio, 2),
            "total_records": total_records
        },
        "by_type": by_type,
        "per_task_statistics": task_stats
    }
    
    # Save summary to file
    summary_file = os.path.join(task_dir, "privacy_anonymization_summary.json")
    try:
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        
        print("\n" + "="*60)
        print("[PrivacyProtection] Anonymization Statistics Summary")
        print("="*60)
        print(f"Total Original Length: {total_original_length:,} characters")
        print(f"Total Anonymized Chars Count: {total_anonymized_length:,} characters")
        print(f"Anonymization Ratio: {overall_ratio:.2f}%")
        print(f"Total Records: {total_records}")
        print("\nBy Type:")
        for stat_type, stats in by_type.items():
            print(f"  {stat_type}:")
            print(f"    Records: {stats['count']}")
            print(f"    Original: {stats['original_length']:,} chars")
            print(f"    Anonymized Chars: {stats['anonymized_chars_count']:,} chars")
            print(f"    Ratio: {stats.get('anonymization_ratio', 0):.2f}%")
        print(f"\nSummary saved to: {summary_file}")
        print("="*60 + "\n")
    except Exception as e:
        print(f"[PrivacyProtection] Failed to save summary: {e}")


def load_incomplete_tasks(path):
    """
    Load incomplete_tasks.json. Returns (items, ok) where ok is True only if
    the file existed and had a valid {"incomplete": list} shape.
    """
    if not os.path.isfile(path):
        return [], False
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        items = data.get("incomplete") if isinstance(data, dict) else None
        if not isinstance(items, list):
            print("[eval] incomplete_tasks.json: invalid format (missing incomplete list), ignoring")
            return [], False
        return items, True
    except Exception as exc:
        print(f"[eval] incomplete_tasks.json: failed to load ({exc}), ignoring")
        return [], False


def safe_rmtree_under(parent_dir, name):
    """Remove join(parent_dir, name) only if it is a directory under abspath(parent_dir)."""
    parent_dir = os.path.abspath(parent_dir)
    target = os.path.abspath(os.path.join(parent_dir, name))
    if not os.path.isdir(target):
        return
    try:
        if os.path.commonpath([parent_dir, target]) != parent_dir:
            print(f"[eval] Refusing to remove outside run_dir: {target}")
            return
    except ValueError:
        print(f"[eval] Refusing to remove path not under run_dir: {target}")
        return
    shutil.rmtree(target)


def remove_abort_task_folders(run_dir, task_id, recorded_folder):
    """Delete recorded task folder and any other run_dir/<task_id>_*/ directories."""
    if recorded_folder:
        safe_rmtree_under(run_dir, recorded_folder)
    if not task_id or not os.path.isdir(run_dir):
        return
    prefix = task_id + "_"
    for name in list(os.listdir(run_dir)):
        path = os.path.join(run_dir, name)
        if os.path.isdir(path) and name.startswith(prefix):
            safe_rmtree_under(run_dir, name)


def task_ids_from_run_dir(run_dir):
    """Task ids inferred from immediate subdirectories (name like app_num_...)."""
    if not os.path.isdir(run_dir):
        return set()
    found = set()
    for name in os.listdir(run_dir):
        path = os.path.join(run_dir, name)
        if not os.path.isdir(path):
            continue
        parts = name.split("_")
        if len(parts) < 2:
            continue
        found.add(parts[0] + "_" + parts[1])
    return found


def merge_incomplete_records(prev_incomplete, outcomes):
    """
    Keep prior entries for tasks not in this run's outcomes; append max_step/abort
    from outcomes for tasks that ran.
    """
    ran_ids = {o["task_id"] for o in outcomes}
    merged = [dict(x) for x in prev_incomplete if x.get("task_id") not in ran_ids]
    for o in outcomes:
        st = o.get("status")
        if st == "max_step":
            merged.append({
                "task_id": o["task_id"],
                "task_folder": o["task_folder"],
                "status": "max_step",
            })
        elif st == "abort":
            entry = {
                "task_id": o["task_id"],
                "task_folder": o["task_folder"],
                "status": "abort",
            }
            if o.get("last_agent_raw_output") is not None:
                entry["last_agent_raw_output"] = o["last_agent_raw_output"]
            merged.append(entry)
    return merged


if __name__ == '__main__':
    task_yamls = os.listdir('evaluation/config')
    task_yamls = ["evaluation/config/" + i for i in task_yamls if i.endswith(".yaml")]

    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("-n", "--name", default="test", type=str)
    arg_parser.add_argument("-c", "--config", default="config-mllm-0409.yaml", type=str)
    arg_parser.add_argument("--task_config", nargs="+", default=task_yamls, help="All task config(s) to load")
    arg_parser.add_argument("--task_id", nargs="+", default=None)
    arg_parser.add_argument("--debug", action="store_true", default=False)
    arg_parser.add_argument("--app", nargs="+", default=None)
    arg_parser.add_argument("-p", "--parallel", default=1, type=int)

    args = arg_parser.parse_args()
    with open(args.config, "r") as file:
        yaml_data = yaml.safe_load(file)

    agent_config = yaml_data["agent"]
    task_config = yaml_data["task"]
    eval_config = yaml_data["eval"]
    privacy_config = PrivacyConfig.from_raw(yaml_data.get("privacy"))

    autotask_class = task_config["class"] if "class" in task_config else "ScreenshotMobileTask_AutoTest"

    single_config = TaskConfig(**task_config["args"])
    single_config = single_config.add_config(eval_config)
    single_config.privacy = privacy_config
    if "True" == agent_config.get("relative_bbox"):
        single_config.is_relative_bbox = True

    task_files = find_all_task_files(args.task_config)
    run_dir = os.path.join(single_config.save_dir, args.name)
    incomplete_path = os.path.join(run_dir, "incomplete_tasks.json")
    prev_incomplete, incomplete_ok = load_incomplete_tasks(incomplete_path)
    abort_task_ids = {e["task_id"] for e in prev_incomplete if e.get("status") == "abort"}

    if incomplete_ok:
        for entry in prev_incomplete:
            if entry.get("status") != "abort":
                continue
            tid = entry.get("task_id")
            remove_abort_task_folders(run_dir, tid, entry.get("task_folder"))

    already_run = task_ids_from_run_dir(run_dir)

    all_task_start_info = []
    for app_task_config_path in task_files:
        app_config = AppConfig(app_task_config_path)
        if args.task_id is None:
            task_ids = list(app_config.task_name.keys())
        else:
            # Handle both space-separated and comma-separated task_ids
            task_ids = []
            for task_id_arg in args.task_id:
                # Split by comma if comma-separated, otherwise use as-is
                if ',' in task_id_arg:
                    task_ids.extend([tid.strip() for tid in task_id_arg.split(',')])
                else:
                    task_ids.append(task_id_arg)
        for task_id in task_ids:
            if task_id in already_run and task_id not in abort_task_ids:
                print(f"Task {task_id} already run, skipping")
                continue
            if task_id not in app_config.task_name:
                continue
            task_instruction = app_config.task_name[task_id].strip()
            app = app_config.APP
            if args.app is not None:
                print(app, args.app)
                if app not in args.app:
                    continue
            package = app_config.package
            command_per_step = app_config.command_per_step.get(task_id, None)

            task_instruction = f"You should use {app} to complete the following task: {task_instruction}"
            all_task_start_info.append({
                "agent_name": agent_config["name"],
                "agent_args": dict(agent_config["args"]),
                "task_id": task_id,
                "task_instruction": task_instruction,
                "package": package,
                "command_per_step": command_per_step,
                "app": app
            })

    class_ = globals().get(autotask_class)
    if class_ is None:
        raise AttributeError(f"Class {autotask_class} not found. Please check the class name in the config file.")

    Auto_Test = class_(single_config.subdir_config(args.name))
    if args.parallel < 1:
        raise ValueError("--parallel must be at least 1")

    if args.parallel == 1:
        outcomes = Auto_Test.run_serial(all_task_start_info)
    else:
        outcomes = parallel_worker(class_, single_config.subdir_config(args.name), args.parallel,
                                   all_task_start_info.copy())

    # Calculate and save overall anonymization statistics
    task_dir = os.path.abspath(single_config.subdir_config(args.name).save_dir)
    # Ensure the run-level directory exists even if no tasks were executed (e.g., filtered/skipped),
    # so the stats aggregator does not report a confusing "directory does not exist".
    try:
        os.makedirs(task_dir, exist_ok=True)
    except Exception:
        pass

    incomplete = merge_incomplete_records(prev_incomplete, outcomes)
    try:
        with open(os.path.join(task_dir, "incomplete_tasks.json"), "w", encoding="utf-8") as f:
            json.dump({"incomplete": incomplete}, f, ensure_ascii=False, indent=2)
    except Exception as exc:
        print(f"Failed to write incomplete_tasks.json: {exc}")

    if single_config.privacy.enabled and single_config.privacy.method != "none":
        calculate_overall_anonymization_stats(task_dir)


