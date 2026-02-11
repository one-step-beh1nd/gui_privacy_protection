#!/usr/bin/env python3
"""
根据修改后的 results.jsonl 重新计算指标（RRR 和 total.jsonl）
需要从原始 trace 文件获取 traces 长度来计算 RRR
"""

import json
import jsonlines
import os
import argparse
from collections import defaultdict
from typing import Dict, List, Optional


def load_ground_truth_lengths(gt_file: str) -> Dict[str, int]:
    """加载 ground truth 长度文件"""
    with open(gt_file, 'r') as f:
        return json.load(f)


def get_trace_length(trace_file: str) -> Optional[int]:
    """从 trace 文件获取操作序列长度"""
    if not os.path.exists(trace_file):
        return None
    
    try:
        with jsonlines.open(trace_file) as reader:
            count = 0
            for line in reader:
                count += 1
            return count
    except Exception as e:
        print(f"Error reading trace file {trace_file}: {e}")
        return None


def find_trace_file(task_id: str, traces_base_dir: str) -> Optional[str]:
    """查找任务的 trace 文件路径"""
    # 尝试不同的路径模式
    # 模式1: traces_base_dir/task_id/traces/trace.jsonl
    trace_file = os.path.join(traces_base_dir, task_id, "traces", "trace.jsonl")
    if os.path.exists(trace_file):
        return trace_file
    
    # 模式2: traces_base_dir/*task_id*/traces/trace.jsonl
    if os.path.exists(traces_base_dir):
        for item in os.listdir(traces_base_dir):
            if task_id in item:
                trace_file = os.path.join(traces_base_dir, item, "traces", "trace.jsonl")
                if os.path.exists(trace_file):
                    return trace_file
    
    return None


def recalculate_rrr(results: List[Dict], length_gt: Dict[str, int], 
                    traces_base_dir: Optional[str] = None) -> List[Dict]:
    """重新计算 RRR 值"""
    updated_results = []
    
    for result in results:
        task_id = result["task_id"]
        complete = result["result"].get("complete", False)
        current_rrr = result.get("RRR")
        
        # 如果 complete 为 True 且 RRR 为 None，尝试重新计算
        if complete and current_rrr is None:
            if task_id in length_gt:
                # 尝试获取 trace 长度
                trace_length = None
                if traces_base_dir:
                    trace_file = find_trace_file(task_id, traces_base_dir)
                    if trace_file:
                        trace_length = get_trace_length(trace_file)
                
                if trace_length and trace_length > 0:
                    rrr = length_gt[task_id] / trace_length
                    result["RRR"] = rrr
                    print(f"✓ 重新计算 {task_id}: RRR = {rrr:.4f} (ground_truth={length_gt[task_id]}, length={trace_length})")
                else:
                    print(f"⚠ {task_id}: 无法获取 trace 长度，RRR 保持为 None")
            else:
                print(f"⚠ {task_id}: 不在 ground_truth_length.json 中，RRR 保持为 None")
        
        updated_results.append(result)
    
    return updated_results


def calculate_partial_acc(result_dict: Dict) -> float:
    """计算部分准确率"""
    tt = 0
    acc = 0
    for key, values in result_dict.items():
        if key != "complete" and key != "judge_page":
            tt += 1
            if values:
                acc += 1
    if tt == 0:
        return 0
    return acc / tt


def regenerate_total_jsonl(results: List[Dict], output_file: str):
    """重新生成 total.jsonl 文件"""
    complete_metric = defaultdict(list)
    partial_metric = defaultdict(list)
    rrr_metric = defaultdict(dict)
    ror_metric = defaultdict(dict)
    
    # 按应用分组统计
    for result in results:
        app = result["task_id"].split("_")[0]
        complete = result["result"].get("complete", False)
        
        if complete:
            complete_metric[app].append(1)
            partial_metric[app].append(1)
        else:
            complete_metric[app].append(0)
            partial_metric[app].append(calculate_partial_acc(result["result"]))
        
        # 保存 RRR 和 reasonable_operation_ratio
        if "RRR" in result:
            rrr_metric[app][result["task_id"]] = result["RRR"]
        if "reasonable_operation_ratio" in result:
            ror_metric[app][result["task_id"]] = result["reasonable_operation_ratio"]
    
    # 生成 total.jsonl
    output_data = []
    for app, values in complete_metric.items():
        output_dict = {
            "App": app,
            "Acc": sum(values) / len(values),
            "Total": len(values),
            "Complete_Correct": sum(values),
            "Sum_Partial_Acc": sum(partial_metric[app]),
            "Partial_Acc": sum(partial_metric[app]) / len(values)
        }
        
        # 计算 RRR 平均值和总和
        if app in rrr_metric:
            rrr_values = [v for v in rrr_metric[app].values() if v is not None]
            if rrr_values:
                output_dict["RRR"] = sum(rrr_values) / len(rrr_values)
                output_dict["Sum_RRR"] = sum(rrr_values)
            else:
                output_dict["RRR"] = 0
                output_dict["Sum_RRR"] = 0
        
        # 计算 reasonable_operation_ratio 平均值和总和
        if app in ror_metric:
            ror_values = [v for v in ror_metric[app].values() if v is not None]
            if ror_values:
                output_dict["reasonable_operation_ratio"] = sum(ror_values) / len(ror_values)
                output_dict["Sum_reasonable_operation_ratio"] = sum(ror_values)
            else:
                output_dict["reasonable_operation_ratio"] = 0
                output_dict["Sum_reasonable_operation_ratio"] = 0
        
        output_data.append(output_dict)
    
    # 写入文件
    with jsonlines.open(output_file, mode='w') as writer:
        for item in output_data:
            writer.write(item)
    
    print(f"\n✓ 已重新生成 {output_file}")
    print(f"  共 {len(output_data)} 个应用的统计结果")
    
    return output_data


def calculate_average_metrics(total_jsonl_file: str, output_file: str):
    """计算所有应用指标的平均值并保存到 JSON 文件
    按照论文中的定义计算全局指标（而非应用级别的平均值）：
    - SR (Acc): 全局的 Complete_Correct / Total
    - Sub-SR (Partial_Acc): 全局的 Sum_Partial_Acc / Total
    - RRR: 全局的 Sum_RRR / Complete_Correct（按成功任务数加权）
    - ROR (reasonable_operation_ratio): 全局的 Sum_reasonable_operation_ratio / Total
    """
    # 读取 total.jsonl
    app_data = []
    with jsonlines.open(total_jsonl_file) as reader:
        for line in reader:
            app_data.append(line)
    
    if not app_data:
        print(f"⚠ 警告: {total_jsonl_file} 为空，无法计算平均值")
        return
    
    # 计算总和类型的指标（全局统计）
    num_apps = len(app_data)
    total = sum(item["Total"] for item in app_data)
    complete_correct = sum(item["Complete_Correct"] for item in app_data)
    sum_partial_acc = sum(item["Sum_Partial_Acc"] for item in app_data)
    sum_rrr = sum(item.get("Sum_RRR", 0) for item in app_data)
    sum_reasonable_operation_ratio = sum(item.get("Sum_reasonable_operation_ratio", 0) for item in app_data)
    
    # 按照论文定义计算全局指标（与 output_to_excel 逻辑一致）
    # SR (Success Rate): 全局的 Complete_Correct / Total
    acc = complete_correct / total if total > 0 else 0
    
    # Sub-SR (Sub-Success Rate): 全局的 Sum_Partial_Acc / Total
    partial_acc = sum_partial_acc / total if total > 0 else 0
    
    # RRR (Reversed Redundancy Ratio): 全局的 Sum_RRR / Complete_Correct（按成功任务数加权）
    rrr = sum_rrr / complete_correct if complete_correct > 0 else 0
    
    # ROR (Reasonable Operation Ratio): 全局的 Sum_reasonable_operation_ratio / Total
    reasonable_operation_ratio = sum_reasonable_operation_ratio / total if total > 0 else 0
    
    # 构建结果
    result = {
        "num_apps": num_apps,
        "Acc": acc,
        "Partial_Acc": partial_acc,
        "RRR": rrr,
        "reasonable_operation_ratio": reasonable_operation_ratio,
        "Total": total,
        "Complete_Correct": complete_correct,
        "Sum_Partial_Acc": sum_partial_acc,
        "Sum_RRR": sum_rrr,
        "Sum_reasonable_operation_ratio": sum_reasonable_operation_ratio,
    }
    
    # 写入 JSON 文件
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    
    print(f"\n✓ 已计算全局指标并保存到 {output_file}")
    print(f"  应用数量: {num_apps}")
    print(f"  全局指标: Acc={result['Acc']:.4f} ({result['Acc']*100:.2f}%), "
          f"Partial_Acc={result['Partial_Acc']:.4f} ({result['Partial_Acc']*100:.2f}%), "
          f"RRR={result['RRR']:.4f} ({result['RRR']*100:.2f}%), "
          f"reasonable_operation_ratio={result['reasonable_operation_ratio']:.4f} ({result['reasonable_operation_ratio']*100:.2f}%)")


def main():
    parser = argparse.ArgumentParser(description="根据修改后的 results.jsonl 重新计算指标")
    parser.add_argument("--results_file", type=str, required=True,
                       help="修改后的 results.jsonl 文件路径")
    parser.add_argument("--output_dir", type=str, default=None,
                       help="输出目录（默认与 results_file 同目录）")
    parser.add_argument("--traces_base_dir", type=str, default=None,
                       help="原始 trace 文件的基础目录（用于计算 RRR）")
    parser.add_argument("--gt_length_file", type=str,
                       default="evaluation/tasks/human_ground_turth/ground_truth_length.json",
                       help="ground truth 长度文件路径")
    
    args = parser.parse_args()
    
    # 读取 results.jsonl
    print(f"读取 {args.results_file}...")
    results = []
    with jsonlines.open(args.results_file) as reader:
        for line in reader:
            results.append(line)
    print(f"✓ 读取了 {len(results)} 条结果")
    
    # 加载 ground truth 长度
    if os.path.exists(args.gt_length_file):
        length_gt = load_ground_truth_lengths(args.gt_length_file)
        print(f"✓ 加载了 {len(length_gt)} 个任务的 ground truth 长度")
    else:
        print(f"⚠ 警告: 未找到 {args.gt_length_file}，将无法计算 RRR")
        length_gt = {}
    
    # 重新计算 RRR
    if args.traces_base_dir and length_gt:
        print(f"\n尝试从 {args.traces_base_dir} 获取 trace 文件...")
        results = recalculate_rrr(results, length_gt, args.traces_base_dir)
    elif length_gt:
        print(f"\n⚠ 未提供 traces_base_dir，将只更新已有 RRR 的任务")
        results = recalculate_rrr(results, length_gt, None)
    
    # 保存更新后的 results.jsonl
    output_dir = args.output_dir or os.path.dirname(args.results_file)
    updated_results_file = os.path.join(output_dir, "results.jsonl")
    with jsonlines.open(updated_results_file, mode='w') as writer:
        for result in results:
            writer.write(result)
    print(f"\n✓ 已保存更新后的结果到 {updated_results_file}")
    
    # 重新生成 total.jsonl
    total_jsonl_file = os.path.join(output_dir, "total.jsonl")
    regenerate_total_jsonl(results, total_jsonl_file)
    
    # 计算所有应用的平均值
    average_json_file = os.path.join(output_dir, "average_metrics.json")
    calculate_average_metrics(total_jsonl_file, average_json_file)
    
    print("\n完成！")


if __name__ == "__main__":
    main()

