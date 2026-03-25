import sys
import re
import os
import json
import base64
try:
    import backoff
except ImportError:
    class _BackoffShim:
        @staticmethod
        def expo(*args, **kwargs):
            return None

        @staticmethod
        def on_exception(*args, **kwargs):
            def decorator(func):
                return func
            return decorator

    backoff = _BackoffShim()
from openai import OpenAI
try:
    from zhipuai import ZhipuAI
except ImportError:
    ZhipuAI = None
from utils_mobile.and_controller import AndroidController, list_all_devices
from utils_mobile.utils import print_with_color


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def get_code_snippet(content):
    code_block = re.search(r"```(?:python)?\s*([\s\S]+?)```", content)
    if code_block is not None:
        content = code_block.group(1).strip()

    action_match = re.search(r"Action:\s*([\s\S]+)", content)
    if action_match is not None:
        content = action_match.group(1).strip()

    lines = [line.strip() for line in content.splitlines() if line.strip()]
    if not lines:
        return content.strip()

    for line in lines:
        if line.startswith(("do(", "finish(", "tap(", "type(", "swipe(", "long_press(", "press_back(", "press_home(", "press_enter(", "wait(", "launch(")):
            return line

    for line in reversed(lines):
        if "(" in line and ")" in line:
            return line

    return lines[-1]


def handle_backoff(details):
    print(f"Retry {details['tries']} for Exception: {details['exception']}")


def handle_giveup(details):
    # Handle both cases: details could be a dict or an exception object
    if isinstance(details, dict):
        try:
            print(
                "Backing off {wait:0.1f} seconds after {tries} tries calling function {target} with args {args} and kwargs {kwargs}"
                .format(**details))
        except (TypeError, KeyError) as e:
            print(f"Give up after retries. Details: {details}, Format error: {e}")
    else:
        # If details is an exception object, just print it
        print(f"Give up after retries. Exception: {details}")


def detect_answer(question: str, model_answer: str, standard_answer: str, args):
    result = detect_answer_with_details(question, model_answer, standard_answer, args)
    return result["complete"]


def build_detect_answer_prompt(question: str, model_answer: str, standard_answer: str) -> str:
    return (
        "You are grading whether the model answer is semantically correct for the user's question. "
        "The standard answer is a reference meaning, not a required exact string match. "
        "Accept paraphrases, equivalent explanations, longer answers, and yes/no answers that clearly imply the same meaning as the reference answer. "
        "Mark the answer incorrect only if it contradicts the reference meaning, answers a different question, or misses the key conclusion. "
        "For yes/no reference answers, decide based on the final meaning of the model answer, not whether it literally says Yes or No. "
        "If the reference answer is 'No', then explanations like 'not every day', 'turned off', 'not enabled', or other negative conclusions should usually be treated as correct when they answer the same question. "
        "Focus on the final conclusion that answers the question, not on intermediate details that may still be mentioned in the explanation.\n\n"
        "Examples:\n"
        "- Question: Is there an alarm set on 4PM everyday?\n"
        "- Reference Answer: No\n"
        "- Model Answer: There is an alarm at 4 PM on some weekdays, but not every day.\n"
        "- This should be graded as correct because the final conclusion is 'not every day', which matches 'No'.\n\n"
        "- Question: Have my alarm at 9AM been turned on?\n"
        "- Reference Answer: No\n"
        "- Model Answer: The 9 AM alarm is turned off.\n"
        "- This should be graded as correct because 'turned off' means the answer to the question is No.\n\n"
        f"Question:\n{question}\n\n"
        f"Model Answer:\n{model_answer}\n\n"
        f"Reference Answer:\n{standard_answer}\n\n"
        "Return strict JSON only with this schema:\n"
        "{\n"
        '  "correct": true,\n'
        '  "reason": "short explanation"\n'
        "}"
    )


def detect_answer_with_details(question: str, model_answer: str, standard_answer: str, args):
    detect_prompt = build_detect_answer_prompt(question, model_answer, standard_answer)
    call_time = 0
    last_response = None
    while call_time <= 5:
        call_time += 1
        if args.judge_model == "glm4":
            return_message = get_completion_glm(prompt=detect_prompt, glm4_key=args.api_key)
        else:
            return_message = get_completion_gpt(
                prompt=detect_prompt,
                model_name=args.judge_model,
                api_key=getattr(args, 'api_key', None),
                api_base=getattr(args, 'api_base', None),
            )
        last_response = return_message
        parsed_correct = None
        parsed_reason = None
        try:
            response_text = return_message.strip()
            if response_text.startswith("```"):
                match = re.search(r"```(?:json)?\s*(\{.*\})\s*```", response_text, flags=re.DOTALL)
                if match:
                    response_text = match.group(1).strip()
            try:
                payload = json.loads(response_text)
            except json.JSONDecodeError:
                match = re.search(r"\{.*\}", response_text, flags=re.DOTALL)
                payload = json.loads(match.group(0)) if match else None
            if isinstance(payload, dict):
                parsed_correct = payload.get("correct")
                parsed_reason = payload.get("reason")
        except Exception:
            parsed_correct = None

        if isinstance(parsed_correct, bool):
            return {
                "complete": parsed_correct,
                "judge_prompt": detect_prompt,
                "judge_response": return_message,
                "judge_reason": parsed_reason,
            }
        if "correct" in return_message.lower() and "incorrect" not in return_message.lower():
            return {
                "complete": True,
                "judge_prompt": detect_prompt,
                "judge_response": return_message,
                "judge_reason": parsed_reason,
            }
        if "incorrect" in return_message.lower():
            return {
                "complete": False,
                "judge_prompt": detect_prompt,
                "judge_response": return_message,
                "judge_reason": parsed_reason,
            }
    return {
        "complete": False,
        "judge_prompt": detect_prompt,
        "judge_response": last_response,
        "judge_reason": None,
    }

def detect_answer_test(args):
    # print(f"Question: {question}\nModel Answer: {model_answer}\nStandard Answer: {standard_answer}")
    detect_prompt = "hello! who are you"
    call_time = 0
    while call_time <= 5:
        call_time += 1
        return_message = None
        if args.judge_model == "glm4":
            return_message = get_completion_glm(prompt=detect_prompt, glm4_key=args.api_key)
        else:
            return_message = get_completion_gpt(
                prompt=detect_prompt,
                model_name=args.judge_model,
                api_key=getattr(args, 'api_key', None),
                api_base=getattr(args, 'api_base', None),
            )
        print("Here is the judge_model test: ")
        print("Question: ", detect_prompt)
        print("Model Answer: ", return_message)
        if not isinstance(return_message, str):
            print("ERROR: Judge model error!")
            sys.exit()
        else:
            return


@backoff.on_exception(backoff.expo,
                      Exception,  # 捕获所有异常
                      max_tries=5,
                      on_backoff=handle_backoff,  # 指定重试时的回调函数
                      giveup=handle_giveup)  # 指定放弃重试时的回调函数
def get_completion_glm(prompt, glm4_key):
    if ZhipuAI is None:
        raise ImportError("zhipuai is required when judge_model is glm4")
    client = ZhipuAI(api_key=glm4_key)
    response = client.chat.completions.create(
        model="glm-4",  # 填写需要调用的模型名称
        messages=[
            {"role": "user", "content": prompt},
        ],
    )
    return response.choices[0].message.content

@backoff.on_exception(backoff.expo,
                      Exception,  # 捕获所有异常
                      max_tries=5,
                      on_backoff=handle_backoff,  # 指定重试时的回调函数
                      giveup=handle_giveup)  # 指定放弃重试时的回调函数
def get_completion_gpt(prompt, model_name, api_key=None, api_base=None):
    # Handle empty string as None
    if api_key == "":
        api_key = None
    if api_base == "":
        api_base = None
    
    if api_key and api_base:
        client = OpenAI(api_key=api_key, base_url=api_base)
    elif api_key:
        client = OpenAI(api_key=api_key)
    else:
        client = OpenAI()
    messages = [{
            "role": "user",
            "content": prompt
        }]
    r = client.chat.completions.create(
        model=model_name,
        messages=messages,
        max_tokens=512,
        temperature=0.001
    )
    return r.choices[0].message.content


@backoff.on_exception(backoff.expo,
                      Exception,
                      max_tries=5,
                      on_backoff=handle_backoff,
                      giveup=handle_giveup)
def judge_task_by_screenshot(image_path, task_prompt, args):
    """
    使用LLM vision模型根据最后一张截图判断任务是否成功完成。
    
    Args:
        image_path: 最后一张截图的路径（screenshot-end-*.png）
        task_prompt: 原始任务prompt
        args: 包含judge_model, api_key, api_base等参数
    
    Returns:
        True if task is completed successfully, False otherwise
    """
    # 检查图片是否存在
    if not os.path.exists(image_path):
        print(f"Warning: Screenshot not found: {image_path}")
        return False
    
    # 编码图片
    base64_image = encode_image(image_path)
    
    # 构建判断prompt
    judge_prompt = f"""You are evaluating whether an Android mobile task has been completed successfully.

Task instruction: {task_prompt}

Please examine the screenshot carefully and determine if the task has been completed successfully. 
Consider:
1. Whether the task requirements have been fully met
2. Whether the UI shows the expected final state
3. Whether all required actions have been performed

Answer with ONLY [True] or [False]. Do not provide any explanation."""

    call_time = 0
    while call_time <= 5:
        call_time += 1
        try:
            if args.judge_model == "glm4":
                # GLM4 可能不支持vision，这里先尝试文本判断
                return_message = get_completion_glm(prompt=judge_prompt, glm4_key=args.api_key)
            else:
                # 使用 OpenAI 兼容的多模态 chat completions 接口
                api_key = getattr(args, 'api_key', None)
                api_base = getattr(args, 'api_base', None)
                if api_key == "":
                    api_key = None
                if api_base == "":
                    api_base = None
                
                if api_key and api_base:
                    client = OpenAI(api_key=api_key, base_url=api_base)
                elif api_key:
                    client = OpenAI(api_key=api_key)
                else:
                    client = OpenAI()
                
                messages = [{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": judge_prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{base64_image}"
                            }
                        }
                    ]
                }]
                
                r = client.chat.completions.create(
                    model=args.judge_model,
                    messages=messages,
                    max_tokens=512,
                    temperature=0.001
                )
                return_message = r.choices[0].message.content
            
            if "True" in return_message:
                return True
            elif "False" in return_message:
                return False
        except Exception as e:
            if call_time >= 5:
                print(f"Error judging task by screenshot: {e}")
                return False
            continue
    
    return False


def get_mobile_device():
    device_list = list_all_devices()
    if not device_list:
        print_with_color("ERROR: No device found!", "red")
        sys.exit()
    print_with_color(f"List of devices attached:\n{str(device_list)}", "yellow")
    if len(device_list) == 1:
        device = device_list[0]
        print_with_color(f"Device selected: {device}", "yellow")
    else:
        print_with_color("Please choose the Android device to start demo by entering its ID:", "blue")
        device = input()

    controller = AndroidController(device)
    width, height = controller.get_device_size()
    if not width and not height:
        print_with_color("ERROR: Invalid device size!", "red")
        sys.exit()
    print_with_color(f"Screen resolution of {device}: {width}x{height}", "yellow")

    return controller


def get_mobile_device_and_name():
    device_list = list_all_devices()
    if not device_list:
        print_with_color("ERROR: No device found!", "red")
        sys.exit()
    print_with_color(f"List of devices attached:\n{str(device_list)}", "yellow")
    if len(device_list) == 1:
        device = device_list[0]
        print_with_color(f"Device selected: {device}", "yellow")
    else:
        print_with_color("Please choose the Android device to start demo by entering its ID:", "blue")
        device = input()

    controller = AndroidController(device)
    width, height = controller.get_device_size()
    if not width and not height:
        print_with_color("ERROR: Invalid device size!", "red")
        sys.exit()
    print_with_color(f"Screen resolution of {device}: {width}x{height}", "yellow")

    return controller, device
