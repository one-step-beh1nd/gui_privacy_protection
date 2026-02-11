#!/usr/bin/env python3
"""
检测 logs/evaluation 下指定文件夹的所有 task 是否都有 trace 文件
如果没有 trace 文件，打印出 task id
"""

import os
import sys
from pathlib import Path


def check_trace_files(evaluation_folder):
    """
    检查指定文件夹下所有 task 是否都有 trace 文件
    
    Args:
        evaluation_folder: logs/evaluation 下的文件夹名称（如 'gemini_xml'）
    """
    base_dir = Path(__file__).parent
    logs_dir = base_dir / "logs" / "evaluation" / evaluation_folder
    
    if not logs_dir.exists():
        print(f"错误: 文件夹 {logs_dir} 不存在")
        return
    
    if not logs_dir.is_dir():
        print(f"错误: {logs_dir} 不是一个目录")
        return
    
    print(f"正在检查文件夹: {logs_dir}")
    print("-" * 60)
    
    missing_trace_tasks = []
    total_tasks = 0
    
    # 遍历所有 task 文件夹
    for task_dir in sorted(logs_dir.iterdir()):
        if not task_dir.is_dir():
            continue
        
        total_tasks += 1
        task_id = task_dir.name
        
        # 检查 traces/trace.jsonl 文件是否存在
        trace_file = task_dir / "traces" / "trace.jsonl"
        
        if not trace_file.exists():
            missing_trace_tasks.append(task_id)
            print(f"缺少 trace 文件: {task_id}")
    
    print("-" * 60)
    print(f"总共检查了 {total_tasks} 个 task")
    print(f"缺少 trace 文件的 task 数量: {len(missing_trace_tasks)}")
    
    if missing_trace_tasks:
        print("\n缺少 trace 文件的 task id 列表:")
        for task_id in missing_trace_tasks:
            print(f"  - {task_id}")
    else:
        print("\n所有 task 都有 trace 文件！")


def main():
    if len(sys.argv) < 2:
        print("用法: python check_trace_files.py <evaluation_folder>")
        print("示例: python check_trace_files.py gemini_xml")
        print("\n可用的文件夹:")
        base_dir = Path(__file__).parent
        logs_dir = base_dir / "logs" / "evaluation"
        if logs_dir.exists():
            for folder in sorted(logs_dir.iterdir()):
                if folder.is_dir():
                    print(f"  - {folder.name}")
        sys.exit(1)
    
    evaluation_folder = sys.argv[1]
    check_trace_files(evaluation_folder)


if __name__ == "__main__":
    main()

