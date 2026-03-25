# 并行执行支持修复说明

## 问题概述

之前的实现中，`PrivacyProtectionLayer` 使用全局单例模式，所有任务共享同一个实例。这导致在并行执行时会出现以下严重问题：

1. **数据竞争（Race Condition）**：多个任务同时读写共享的 `token_to_real` 和 `real_to_token` 字典
2. **Token映射污染**：不同任务的token映射会互相干扰
3. **白名单污染**：不同任务的prompt中的非实体词会混合
4. **统计信息混乱**：所有任务的统计信息会混合在一起
5. **任务目录冲突**：`set_task_dir()` 会被不同任务覆盖

## 修复方案

### 1. 线程本地存储（Thread-Local Storage）

修改了 `utils_mobile/privacy/layer.py` 中的全局实例管理：

- 使用 `threading.local()` 为每个线程（任务）创建独立的 `PrivacyProtectionLayer` 实例
- `get_privacy_layer()` 现在优先返回线程本地实例，如果没有则返回全局实例（向后兼容）
- `set_privacy_layer()` 新增 `thread_local` 参数，支持设置线程本地或全局实例

### 2. 任务级别的实例创建

修改了 `evaluation/auto_test.py` 中的 `run_task()` 方法：

- 在每个任务开始时，创建独立的 `PrivacyProtectionLayer` 实例
- 使用 `set_privacy_layer(..., thread_local=True)` 设置为线程本地实例
- 任务结束时，调用 `clear_mappings()` 确保状态清理

## 使用方法

### 串行执行（原有方式）

串行执行不需要任何修改，继续使用 `run_serial()` 方法即可。代码会自动使用全局实例。

### 并行执行

现在可以安全地使用并行执行。每个并行任务会自动获得独立的 privacy layer 实例。

示例（使用 `evaluation/parallel.py`）：

```python
from evaluation.parallel import parallel_worker

# 并行执行任务
parallel_worker(
    class_=ScreenshotMobileTask_AutoTest,
    config=config,
    parallel=4,  # 4个并行任务
    tasks=all_task_start_info
)
```

每个任务会：
1. 自动创建独立的 `PrivacyProtectionLayer` 实例
2. 维护独立的 token 映射、白名单和统计信息
3. 在任务结束时保存统计信息到各自的任务目录

## 技术细节

### 线程本地存储的工作原理

```python
_thread_local = threading.local()

def get_privacy_layer():
    # 优先检查线程本地实例（并行执行）
    if hasattr(_thread_local, 'privacy_layer'):
        return _thread_local.privacy_layer
    
    # 否则使用全局实例（串行执行）
    return _privacy_layer
```

### 任务生命周期

1. **任务开始**：`run_task()` 创建新的 `PrivacyProtectionLayer` 实例并设置为线程本地
2. **任务执行**：所有 privacy layer 操作都使用线程本地实例
3. **任务结束**：调用 `save_stats()` 保存统计信息，然后 `clear_mappings()` 清理状态

## 向后兼容性

- **串行执行**：完全兼容，无需修改任何代码
- **现有代码**：所有使用 `get_privacy_layer()` 的代码都能正常工作
- **环境变量**：`PRIVACY_REPLACEMENT_STYLE` 仍然有效

## 注意事项

1. **线程安全**：现在每个线程有独立的实例，不再有数据竞争问题
2. **内存使用**：并行执行时，每个任务会创建独立的实例，内存使用会增加
3. **统计信息**：每个任务的统计信息会保存到各自的任务目录，不会混合

## 测试建议

在启用并行执行前，建议：

1. 先用少量任务测试并行执行（如 `parallel=2`）
2. 检查每个任务的 `privacy_token_mapping.json` 和 `privacy_anonymization_stats.json` 是否正确保存
3. 验证不同任务的 token 映射是否独立（相同真实值在不同任务中可能生成不同token，这是正常的）
4. 确认统计信息没有混合

## 相关文件

- `utils_mobile/privacy/layer.py` - PrivacyProtectionLayer 实现和全局实例管理
- `evaluation/auto_test.py` - AutoTest 类和 run_task() 方法
- `evaluation/parallel.py` - 并行执行实现
