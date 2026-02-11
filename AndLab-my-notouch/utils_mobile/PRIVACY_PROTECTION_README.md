# 隐私保护层使用说明

## 概述

隐私保护层（Privacy Protection Layer）用于在 Android-Lab 中保护敏感信息，通过以下方式工作：

1. **输入侧（对Agent隐藏）**：在获取UI/XML/截图后，识别并模糊化敏感信息（如电话号码、邮箱等），替换为token（如 `token_mom_phone_number`）
2. **输出侧（执行命令前）**：当Agent使用token进行操作时，在执行ADB命令前将token转换回真实值

## 架构设计

### 核心组件

1. **`utils_mobile/privacy_protection.py`** - 隐私保护层核心实现
   - `PrivacyProtectionLayer` 类：管理token映射和转换
   - 提供识别、模糊化、转换的接口

2. **集成点**：
   - **输入侧**：`recorder/json_recoder.py` - 在获取XML和截图后进行模糊化
   - **输出侧**：`utils_mobile/and_controller.py` - 在执行ADB命令前进行token转换

### 数据流

```
1. 获取UI数据（XML/截图）
   ↓
2. PrivacyProtectionLayer.identify_and_mask_xml/screenshot()
   ↓ 识别敏感信息并替换为token
3. Agent看到的是token化的数据
   ↓
4. Agent生成操作（可能包含token，如 "call token_mom_phone_number"）
   ↓
5. PageExecutor执行操作
   ↓
6. AndroidController执行ADB命令前
   ↓
7. PrivacyProtectionLayer.convert_token_to_real()
   ↓ 将token转换回真实值
8. 执行真实的ADB命令
```

## 使用示例

### 基本使用

```python
from utils_mobile.privacy_protection import get_privacy_layer, set_privacy_layer, PrivacyProtectionLayer

# 启用隐私保护层
privacy_layer = PrivacyProtectionLayer(enabled=True)
set_privacy_layer(privacy_layer)

# 或者使用默认实例
privacy_layer = get_privacy_layer()
```

### 手动添加Token映射

```python
# 手动添加电话号码映射
privacy_layer.add_token_mapping("token_mom_phone_number", "1234567890")
privacy_layer.add_token_mapping("token_dad_phone_number", "0987654321")
```

### 在代码中获取Token

```python
# 如果Agent需要知道某个值的token
phone_number = "1234567890"
token = privacy_layer.get_token_for_value(phone_number, category="phone", identifier="mom")
# token = "token_mom_phone_1"
```

### 转换Token回真实值

```python
# Agent生成的命令可能包含token
agent_command = "call token_mom_phone_number"
real_command = privacy_layer.convert_token_to_real(agent_command)
# real_command = "call 1234567890"
```

## 需要实现的函数

以下函数需要根据实际需求实现隐私信息识别逻辑：

### 1. `identify_and_mask_xml(xml_content: str)`

在XML内容中识别敏感信息并替换为token。

**需要实现的功能**：
- `detect_phone_numbers_in_xml(xml_content)` - 检测XML中的电话号码
- `detect_emails_in_xml(xml_content)` - 检测XML中的邮箱地址
- `detect_names_in_xml(xml_content)` - 检测XML中的姓名
- `detect_addresses_in_xml(xml_content)` - 检测XML中的地址
- 其他敏感信息检测...

**示例实现思路**：
```python
def identify_and_mask_xml(self, xml_content: str):
    # 1. 解析XML
    root = ET.fromstring(xml_content)
    
    # 2. 遍历所有元素，检测敏感信息
    phone_numbers = detect_phone_numbers_in_xml(root)
    emails = detect_emails_in_xml(root)
    
    # 3. 为每个敏感信息生成token
    for phone in phone_numbers:
        token = self._generate_token("phone", "mom")  # 可以从上下文推断identifier
        self.real_to_token[phone] = token
        self.token_to_real[token] = phone
        # 替换XML中的电话号码
        xml_content = xml_content.replace(phone, token)
    
    return xml_content, new_tokens
```

### 2. `identify_and_mask_screenshot(image_path: str)`

在截图中识别敏感信息（通过OCR）并模糊化。

**需要实现的功能**：
- `perform_ocr(image_path)` - 对截图进行OCR识别
- `detect_sensitive_info_in_ocr(ocr_results)` - 在OCR结果中检测敏感信息
- `blur_region_in_image(image_path, bbox)` - 模糊图片中的特定区域
- `mask_text_in_image(image_path, text, bbox)` - 在图片中遮挡文本

**示例实现思路**：
```python
def identify_and_mask_screenshot(self, image_path: str):
    # 1. 执行OCR
    ocr_results = perform_ocr(image_path)  # 返回 [(text, bbox), ...]
    
    # 2. 检测敏感信息
    sensitive_regions = []
    for text, bbox in ocr_results:
        if is_phone_number(text):
            sensitive_regions.append((text, bbox, "phone"))
        elif is_email(text):
            sensitive_regions.append((text, bbox, "email"))
    
    # 3. 模糊化敏感区域
    image = Image.open(image_path)
    masked_image_path = image_path.replace('.png', '_masked.png')
    for text, bbox, category in sensitive_regions:
        token = self.get_token_for_value(text, category=category)
        # 模糊化bbox区域
        blur_region(image, bbox)
        # 可选：在图片上绘制token文本
    
    image.save(masked_image_path)
    return masked_image_path, new_tokens
```

### 3. `identify_and_mask_text(text: str)`

在纯文本中识别敏感信息。

**需要实现的功能**：
- `detect_phone_numbers(text)` - 使用正则表达式检测电话号码
- `detect_emails(text)` - 使用正则表达式检测邮箱
- `detect_credit_cards(text)` - 检测信用卡号
- 其他敏感信息检测...

**示例实现思路**：
```python
def identify_and_mask_text(self, text: str):
    # 使用正则表达式检测电话号码
    phone_pattern = r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b'
    phone_numbers = re.findall(phone_pattern, text)
    
    for phone in phone_numbers:
        token = self.get_token_for_value(phone, category="phone")
        text = text.replace(phone, token)
    
    return text, new_tokens
```

## 扩展：支持特定操作

如果Agent需要执行特定操作（如打电话），可以在 `page_executor` 中添加支持：

```python
# 在 page_executor/text_executor.py 或 simple_vision_executor.py 中

def call(self, phone_token):
    """
    Call a phone number using token.
    
    Args:
        phone_token: Token like "token_mom_phone_number"
    """
    privacy_layer = get_privacy_layer()
    real_phone = privacy_layer.convert_token_to_real(phone_token)
    
    # 执行打电话的ADB命令
    # 例如：adb shell am start -a android.intent.action.CALL -d tel:1234567890
    command = f"adb -s {self.device} shell am start -a android.intent.action.CALL -d tel:{real_phone}"
    self.controller.run_command(command)
    self.current_return = {"operation": "do", "action": "Call", "kwargs": {"phone_token": phone_token}}
```

## 配置

可以在配置文件中添加隐私保护层的配置：

```yaml
# config.yaml
privacy_protection:
  enabled: true
  categories:
    - phone
    - email
    - name
    - address
  detection_method: "regex"  # or "ml" for machine learning based
```

然后在初始化时加载配置：

```python
# 在 eval.py 或 auto_test.py 中
from utils_mobile.privacy_protection import PrivacyProtectionLayer

if yaml_data.get("privacy_protection", {}).get("enabled", False):
    privacy_layer = PrivacyProtectionLayer(enabled=True)
    set_privacy_layer(privacy_layer)
```

## 注意事项

1. **Token持久化**：如果需要跨会话保持token映射，可以实现序列化/反序列化功能
2. **上下文识别**：在识别敏感信息时，尽量从上下文推断identifier（如"mom"、"dad"），使token更有意义
3. **性能考虑**：OCR和图像处理可能较慢，考虑异步处理或缓存
4. **安全性**：token映射存储在内存中，生产环境需要考虑加密存储

## 测试

```python
# 测试示例
privacy_layer = PrivacyProtectionLayer(enabled=True)
privacy_layer.add_token_mapping("token_mom_phone", "1234567890")

# 测试转换
assert privacy_layer.convert_token_to_real("call token_mom_phone") == "call 1234567890"

# 测试XML模糊化
xml = "<text>Call mom at 1234567890</text>"
masked_xml, tokens = privacy_layer.identify_and_mask_xml(xml)
assert "1234567890" not in masked_xml
assert "token" in masked_xml
```


