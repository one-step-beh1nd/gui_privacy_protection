# 如何使用并行执行

## 快速开始

现在 `eval.py` 已经支持并行执行了！只需要添加 `-p` 或 `--parallel` 参数即可。

### 基本用法

```bash
# 串行执行（默认，1个任务接1个任务）
python eval.py -n test -c config-mllm-0409.yaml

# 并行执行（4个任务同时运行）
python eval.py -n test -c config-mllm-0409.yaml -p 4

# 或者使用长参数名
python eval.py -n test -c config-mllm-0409.yaml --parallel 4
```

## 参数说明

- `-p, --parallel`: 并行任务数量（默认值为1，即串行执行）
  - `-p 1`: 串行执行（等同于不指定该参数）
  - `-p 2`: 2个任务并行执行
  - `-p 4`: 4个任务并行执行
  - 等等...

## 完整示例

```bash
# 并行执行所有任务，使用4个并行worker
python eval.py \
    -n my_evaluation \
    -c config-mllm-0409.yaml \
    --parallel 4

# 只执行特定任务，使用2个并行worker
python eval.py \
    -n my_evaluation \
    -c config-mllm-0409.yaml \
    --task_id task1 task2 task3 \
    --parallel 2

# 只执行特定app的任务，使用8个并行worker
python eval.py \
    -n my_evaluation \
    -c config-mllm-0409.yaml \
    --app com.example.app1 com.example.app2 \
    --parallel 8
```

## 工作原理

### 线程本地存储

每个并行任务都会自动获得独立的 `PrivacyProtectionLayer` 实例，通过线程本地存储（Thread-Local Storage）实现：

- **串行执行**：所有任务共享一个全局实例（向后兼容）
- **并行执行**：每个线程（任务）有自己独立的实例

### 任务隔离

每个并行任务都有：
- 独立的 token 映射（`token_to_real`, `real_to_token`）
- 独立的白名单（`whitelist`）
- 独立的统计信息（`_anonymization_stats`）
- 独立的任务目录设置（`_task_dir`）

这确保了不同任务之间不会互相干扰。

## 注意事项

### 1. 资源限制

并行执行会消耗更多资源：
- **内存**：每个并行任务需要独立的内存空间
- **CPU**：多个任务同时运行会增加CPU负载
- **Android模拟器**：每个并行任务需要一个独立的AVD实例

**建议**：
- 根据你的机器配置选择合适的并行数
- 如果使用Docker，确保有足够的容器资源
- 如果使用本地AVD，确保有足够的系统资源

### 2. 实例管理

`parallel.py` 使用实例池（Instance Pool）管理Android模拟器：
- 预先创建指定数量的实例（AVD或Docker容器）
- 任务完成后，实例会被回收并分配给新任务
- 这避免了频繁创建/销毁实例的开销

### 3. 错误处理

如果某个并行任务失败：
- 错误会被捕获并打印
- 其他任务会继续执行
- 失败的任务不会影响其他任务的结果

### 4. 统计信息

每个任务的统计信息会保存到各自的任务目录：
- `{task_dir}/privacy_anonymization_stats.json`
- `{task_dir}/privacy_token_mapping.json`

所有任务完成后，`eval.py` 会聚合所有任务的统计信息到：
- `{save_dir}/{name}/privacy_anonymization_summary.json`

## 性能建议

### 选择合适的并行数

```bash
# 小规模测试：2-4个并行任务
python eval.py -n test -c config.yaml -p 2

# 中等规模：4-8个并行任务
python eval.py -n test -c config.yaml -p 4

# 大规模：8-16个并行任务（需要足够的资源）
python eval.py -n test -c config.yaml -p 8
```

### 监控资源使用

并行执行时，建议监控：
- CPU使用率
- 内存使用情况
- 磁盘I/O（如果使用Docker）
- Android模拟器性能

如果发现资源不足，可以减少并行数。

## 故障排除

### 问题1：任务执行失败

**症状**：某些任务报错或失败

**解决**：
1. 检查错误日志
2. 确认AVD/Docker资源是否足够
3. 尝试减少并行数（`-p 2` 或 `-p 1`）

### 问题2：内存不足

**症状**：系统变慢或任务被杀死

**解决**：
1. 减少并行数
2. 关闭其他占用内存的程序
3. 如果使用Docker，检查容器内存限制

### 问题3：统计信息混乱

**症状**：统计信息不正确或混合

**解决**：
- 这个问题已经通过线程本地存储修复
- 如果仍然出现问题，检查是否有其他代码直接访问全局实例

## 验证并行执行

运行并行执行后，检查：

1. **每个任务目录都有独立的统计文件**：
   ```bash
   ls logs/evaluation/test_name/*/privacy_anonymization_stats.json
   ```

2. **每个任务都有独立的token映射**：
   ```bash
   ls logs/evaluation/test_name/*/privacy_token_mapping.json
   ```

3. **汇总统计文件包含所有任务**：
   ```bash
   cat logs/evaluation/test_name/privacy_anonymization_summary.json
   ```

## 与串行执行的对比

| 特性 | 串行执行 | 并行执行 |
|------|---------|---------|
| 执行方式 | 一个接一个 | 同时执行多个 |
| 速度 | 较慢 | 较快（取决于并行数） |
| 资源使用 | 较低 | 较高 |
| Privacy Layer | 全局单例 | 线程本地实例 |
| 适用场景 | 资源有限、调试 | 大规模评估、快速执行 |

## 总结

现在你可以安全地使用并行执行了！只需要：

```bash
python eval.py -n test -c config.yaml -p 4
```

每个并行任务都会自动获得独立的 privacy layer 实例，不会互相干扰。
