#!/usr/bin/env python3
"""
检查logs/evaluation下指定文件夹中每个task的trace文件情况
- 检查是否有traces/trace.jsonl文件
- 检查trace是否以finish结尾
"""

import argparse
import os
import jsonlines
from pathlib import Path
from typing import List, Tuple


def check_trace_finish(trace_file: str) -> bool:
    """
    检查trace文件是否以finish结尾
    返回True表示以finish结尾，False表示不是
    """
    try:
        with jsonlines.open(trace_file) as reader:
            lines = list(reader)
            if not lines:
                return False
            
            # 检查最后一行
            last_line = lines[-1]
            parsed_action = last_line.get("parsed_action", {})
            action = parsed_action.get("action", "")
            
            return action == "finish"
    except Exception as e:
        print(f"  错误：读取trace文件失败: {e}")
        return False


def check_traces(evaluation_folder: str, target_dir: str) -> Tuple[List[str], List[str]]:
    """
    检查指定文件夹下的所有task
    返回: (没有trace.jsonl的task列表, 不以finish结尾的task列表)
    """
    base_path = Path(evaluation_folder) / target_dir
    
    if not base_path.exists():
        print(f"错误：文件夹 {base_path} 不存在")
        return [], []
    
    if not base_path.is_dir():
        print(f"错误：{base_path} 不是一个目录")
        return [], []
    
    tasks_without_trace = []
    tasks_not_finished = []
    
    # 遍历所有子目录（每个task一个目录）
    print(f"正在检查文件夹: {base_path}")
    print("=" * 60)
    
    task_dirs = sorted([d for d in base_path.iterdir() if d.is_dir()])
    total_tasks = len(task_dirs)
    
    for task_dir in task_dirs:
        task_name = task_dir.name
        
        # 提取task_id（格式：app_id_timestamp）
        parts = task_name.split('_')
        if len(parts) >= 2:
            task_id = f"{parts[0]}_{parts[1]}"
        else:
            task_id = task_name
        
        trace_file = task_dir / "traces" / "trace.jsonl"
        
        # 检查trace.jsonl是否存在
        if not trace_file.exists():
            tasks_without_trace.append(task_id)
            print(f"❌ {task_id}: trace.jsonl 不存在")
            continue
        
        # 检查是否以finish结尾
        is_finished = check_trace_finish(str(trace_file))
        if not is_finished:
            tasks_not_finished.append(task_id)
            print(f"⚠️  {task_id}: trace 不以finish结尾")
        else:
            print(f"✓   {task_id}: 正常")
    
    return tasks_without_trace, tasks_not_finished


def main():
    parser = argparse.ArgumentParser(
        description="检查logs/evaluation下指定文件夹中每个task的trace文件情况"
    )
    parser.add_argument(
        "--evaluation_folder",
        type=str,
        default="logs/evaluation",
        help="evaluation文件夹路径 (默认: logs/evaluation)"
    )
    parser.add_argument(
        "--target_dir",
        type=str,
        required=True,
        help="要检查的文件夹名称 (例如: test_gliner)"
    )
    
    args = parser.parse_args()
    
    # 检查
    tasks_without_trace, tasks_not_finished = check_traces(
        args.evaluation_folder,
        args.target_dir
    )
    
    # 打印统计结果
    print("\n" + "=" * 60)
    print("统计结果:")
    print("=" * 60)
    
    print(f"\n📊 没有trace.jsonl的任务数: {len(tasks_without_trace)}")
    if tasks_without_trace:
        print("   任务列表:")
        for task_id in tasks_without_trace:
            print(f"     - {task_id}")
    
    print(f"\n📊 不以finish结尾的任务数: {len(tasks_not_finished)}")
    if tasks_not_finished:
        print("   任务列表:")
        for task_id in tasks_not_finished:
            print(f"     - {task_id}")
    
    # 统计总数
    base_path = Path(args.evaluation_folder) / args.target_dir
    if base_path.exists():
        total_tasks = len([d for d in base_path.iterdir() if d.is_dir()])
        print(f"\n📊 总任务数: {total_tasks}")
        print(f"📊 正常任务数: {total_tasks - len(tasks_without_trace) - len(tasks_not_finished)}")


if __name__ == "__main__":
    main()

