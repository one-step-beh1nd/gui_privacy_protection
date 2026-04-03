import ast
import inspect
import json
import re
import time
from functools import partial

from templates.packages import find_package
from .utils import call_dino, plot_bbox
from utils_mobile.privacy_protection import get_privacy_layer


def remove_leading_zeros_in_string(s):
    # 使用正则表达式匹配列表中的每个数值并去除前导零
    return re.sub(r'\b0+(\d)', r'\1', s)


class TextOnlyExecutor:
    def __init__(self, controller, config):
        self.config = config
        self.controller = controller
        self.device = controller.device
        self.screenshot_dir = config.screenshot_dir
        self.task_id = int(time.time())

        self.new_page_captured = False
        self.current_screenshot = None
        self.current_return = None

        self.last_turn_element = None
        self.last_turn_element_tagname = None
        self.is_finish = False
        self.device_pixel_ratio = None
        self.latest_xml = None
        # self.glm4_key = config.glm4_key

        # self.device_pixel_ratio = self.page.evaluate("window.devicePixelRatio")

    # ------------------------------------------------------------------ #
    # Helpers for parsing cloud_agent_compute_with_tokens(...) calls
    # ------------------------------------------------------------------ #
    def _parse_cloud_agent_compute_instruction(self, expr: str):
        """
        Parse an expression like:
        cloud_agent_compute_with_tokens(
            anon_tokens=["phone_number#0abc1"],
            compute_instruction="...",
            usage_reason="..."
        )

        Only keyword arguments are supported. Any extra parameters (e.g.
        temperature, do_sample, model_dir) will be ignored so that the
        cloud agent cannot control local LLM hyper-parameters.
        """
        try:
            tree = ast.parse(expr.strip(), mode="eval")
        except SyntaxError as exc:
            raise ValueError(f"Invalid cloud_agent_compute_with_tokens expression: {exc}") from exc

        if not isinstance(tree, ast.Expression) or not isinstance(tree.body, ast.Call):
            raise ValueError("Instruction must be a single function call expression.")

        call = tree.body
        func_name = ""
        if isinstance(call.func, ast.Name):
            func_name = call.func.id
        elif isinstance(call.func, ast.Attribute):
            func_name = call.func.attr

        if func_name != "cloud_agent_compute_with_tokens":
            raise ValueError("Only cloud_agent_compute_with_tokens(...) is supported in Call_API.")

        kwargs = {}
        for kw in call.keywords:
            if kw.arg is None:
                continue
            # 只允许云端 agent 指定这三个参数，忽略其他参数以防止控制本地 LLM 超参数
            if kw.arg not in {"anon_tokens", "compute_instruction", "usage_reason"}:
                continue
            kwargs[kw.arg] = ast.literal_eval(kw.value)

        anon_tokens = kwargs.get("anon_tokens") or []
        compute_instruction = kwargs.get("compute_instruction") or ""
        usage_reason = kwargs.get("usage_reason") or ""

        if not isinstance(anon_tokens, list) or not all(isinstance(t, str) for t in anon_tokens):
            raise ValueError("anon_tokens must be a list of strings.")
        if not isinstance(compute_instruction, str) or not isinstance(usage_reason, str):
            raise ValueError("compute_instruction and usage_reason must be strings.")

        return anon_tokens, compute_instruction, usage_reason

    def __get_current_status__(self):
        page_position = None
        scroll_height = None
        status = {
            "Current URL": self.controller.get_current_activity(),
        }
        return json.dumps(status, ensure_ascii=False)

    def modify_relative_bbox(self, relative_bbox):
        viewport_width, viewport_height = self.controller.viewport_size
        modify_x1 = relative_bbox[0] * viewport_width / 1000
        modify_y1 = relative_bbox[1] * viewport_height / 1000
        modify_x2 = relative_bbox[2] * viewport_width / 1000
        modify_y2 = relative_bbox[3] * viewport_height / 1000
        return [modify_x1, modify_y1, modify_x2, modify_y2]

    def __call__(self, code_snippet):
        '''
        self.new_page_captured = False
        self.controller.on("page", self.__capture_new_page__)
        self.current_return = None'''

        local_context = self.__get_class_methods__()
        local_context.update(**{'self': self})
        print(code_snippet.strip())
        if len(code_snippet.split("\n")) > 1:
            for code in code_snippet.split("\n"):
                if "Action: " in code:
                    code_snippet = code
                    break

        code = remove_leading_zeros_in_string(code_snippet.strip())
        exec(code, {}, local_context)
        return self.current_return

    def __get_class_methods__(self, include_dunder=False, exclude_inherited=True):
        """
        Returns a dictionary of {method_name: method_object} for all methods in the given class.

        Parameters:
        - cls: The class object to inspect.
        - include_dunder (bool): Whether to include dunder (double underscore) methods.
        - exclude_inherited (bool): Whether to exclude methods inherited from parent classes.
        """
        methods_dict = {}
        cls = self.__class__
        for name, method in inspect.getmembers(cls, predicate=inspect.isfunction):
            if exclude_inherited and method.__qualname__.split('.')[0] != cls.__name__:
                continue
            if not include_dunder and name.startswith('__'):
                continue
            methods_dict[name] = partial(method, self)
        return methods_dict

    def update_screenshot(self, prefix=None, suffix=None):
        # time.sleep(2)
        if prefix is None and suffix is None:
            self.current_screenshot = f"{self.screenshot_dir}/screenshot-{time.time()}.png"
        elif prefix is not None and suffix is None:
            self.current_screenshot = f"{self.screenshot_dir}/screenshot-{prefix}-{time.time()}.png"
        elif prefix is None and suffix is not None:
            self.current_screenshot = f"{self.screenshot_dir}/screenshot-{time.time()}-{suffix}.png"
        else:
            self.current_screenshot = f"{self.screenshot_dir}/screenshot-{prefix}-{time.time()}-{suffix}.png"
        self.controller.save_screenshot(self.current_screenshot)

    def do(self, action=None, element=None, **kwargs):
        assert action in ["Tap", "Type", "Swipe", "Enter", "Home", "Back", "Long Press", "Wait", "Launch",
                          "Call_API"], "Unsupported Action"
        if self.config.is_relative_bbox:
            if element is not None:
                element = self.modify_relative_bbox(element)
        if action == "Tap":
            self.tap(element)
        elif action == "Type":
            self.type(**kwargs)
        elif action == "Swipe":
            self.swipe(element, **kwargs)
        elif action == "Enter":
            self.press_enter()
        elif action == "Home":
            self.press_home()
        elif action == "Back":
            self.press_back()
        elif action == "Long Press":
            self.long_press(element)
        elif action == "Wait":
            self.wait()
        elif action == "Launch":
            self.launch(**kwargs)
        elif action == "Call_API":
            self.call_api(**kwargs)
        else:
            raise NotImplementedError()
        # self.__update_screenshot__() # update screenshot 全部移到recoder内

    def get_relative_bbox_center(self, instruction, screenshot):
        # 获取相对 bbox
        relative_bbox = call_dino(instruction, screenshot)

        viewport_width, viewport_height = self.controller.get_device_size()

        center_x = (relative_bbox[0] + relative_bbox[2]) / 2 * viewport_width / 1000
        center_y = (relative_bbox[1] + relative_bbox[3]) / 2 * viewport_height / 1000
        width_x = (relative_bbox[2] - relative_bbox[0]) * viewport_width / 1000
        height_y = (relative_bbox[3] - relative_bbox[1]) * viewport_height / 1000

        # 点击计算出的中心点坐标
        # print(center_x, center_y)
        plot_bbox([int(center_x - width_x / 2), int(center_y - height_y / 2), int(width_x), int(height_y)], screenshot,
                  instruction)

        return (int(center_x), int(center_y)), relative_bbox

    def tap(self, element):
        if isinstance(element, list) and len(element) == 4:
            center_x = (element[0] + element[2]) / 2
            center_y = (element[1] + element[3]) / 2
        elif isinstance(element, list) and len(element) == 2:
            center_x, center_y = element
        else:
            raise ValueError("Invalid element format")
        self.controller.tap(center_x, center_y)
        self.current_return = {"operation": "do", "action": 'Tap', "kwargs": {"element": element}}

    def long_press(self, element):
        if isinstance(element, list) and len(element) == 4:
            center_x = (element[0] + element[2]) / 2
            center_y = (element[1] + element[3]) / 2
        elif isinstance(element, list) and len(element) == 2:
            center_x, center_y = element
        else:
            raise ValueError("Invalid element format")
        self.controller.long_press(center_x, center_y)
        self.current_return = {"operation": "do", "action": 'Long Press', "kwargs": {"element": element}}

    def swipe(self, element=None, **kwargs):
        if element is None:
            center_x, center_y = self.controller.width // 2, self.controller.height // 2
        elif element is not None:
            if isinstance(element, list) and len(element) == 4:
                center_x = (element[0] + element[2]) / 2
                center_y = (element[1] + element[3]) / 2
            elif isinstance(element, list) and len(element) == 2:
                center_x, center_y = element
            else:
                raise ValueError("Invalid element format")
        assert "direction" in kwargs, "direction is required for swipe"
        direction = kwargs.get("direction")
        dist = kwargs.get("dist", "medium")
        self.controller.swipe(center_x, center_y, direction, dist)
        self.current_return = {"operation": "do", "action": 'Swipe',
                               "kwargs": {"element": element, "direction": direction, "dist": dist}}
        time.sleep(1)

    def type(self, **kwargs):
        assert "text" in kwargs, "text is required for type"
        instruction = kwargs.get("text")
        self.controller.text(instruction)
        self.controller.enter()
        self.current_return = {"operation": "do", "action": 'Type',
                               "kwargs": {"text": instruction}}

    def press_enter(self):
        self.controller.enter()
        self.current_return = {"operation": "do", "action": 'Press Enter'}

    def press_back(self):
        self.controller.back()
        self.current_return = {"operation": "do", "action": 'Press Back'}

    def press_home(self):
        self.controller.home()
        self.current_return = {"operation": "do", "action": 'Press Home'}

    def finish(self, message=None):
        self.is_finish = True
        self.current_return = {"operation": "finish", "action": 'finish', "kwargs": {"message": message}}

    def wait(self):
        time.sleep(5)
        self.current_return = {"operation": "do", "action": 'Wait'}

    def launch(self, **kwargs):
        assert "app" in kwargs, "app is required for launch"
        app = kwargs.get("app")
        try:
            package = find_package(app)
        except:
            import traceback
            traceback.print_exc()
        self.controller.launch_app(package)
        self.current_return = {"operation": "do", "action": 'Launch',
                               "kwargs": {"package": package}}

    def call_api(self, **kwargs):
        """
        Handle `do(action="Call_API", instruction=...)` from the cloud agent.

        We overload this to route privacy-sensitive requests to the local
        privacy LLM interface cloud_agent_compute_with_tokens() implemented
        in the privacy layer.

        Expected pattern from the cloud agent:

            do(
                action="Call_API",
                instruction="cloud_agent_compute_with_tokens(anon_tokens=[...], compute_instruction='...', usage_reason='...')",
                with_screen_info=False
            )

        The agent is **not** allowed to pass any local LLM hyper-parameters
        (e.g. temperature, top_p, do_sample); such fields are ignored.
        """
        assert "instruction" in kwargs, "instruction is required for Call_API"
        instruction = kwargs.get("instruction")
        privacy_layer = get_privacy_layer()
        if not privacy_layer.supports_cloud_api():
            response = {
                "approved": False,
                "decision_reason": "Current privacy strategy does not enable local privacy APIs.",
                "result": None,
                "missing_tokens": [],
                "raw_llm_output": "",
            }
            self.current_return = {
                "operation": "do",
                "action": "Call_API",
                "kwargs": {
                    "instruction": instruction,
                    "response": response,
                    "with_screen_info": False,
                },
            }
            return

        # 解析 cloud_agent_compute_with_tokens(...) 形式的表达式
        anon_tokens, compute_instruction, usage_reason = self._parse_cloud_agent_compute_instruction(instruction)

        # 获取原始用户任务，用于本地隐私审批（只在本地使用，不发给云端）
        original_task = getattr(self, "original_instruction", None)
        if original_task is None:
            # 兼容旧版本：如果执行器上没有 original_instruction，则尝试从 JSONRecorder 里读取
            try:
                original_task = getattr(self, "record", None).instruction  # type: ignore[attr-defined]
            except Exception:
                original_task = ""

        # 本地大模型路径：优先从 config 读取，其次从环境变量，最后使用一个默认值
        import os

        privacy_args = getattr(getattr(self.config, "privacy", None), "args", {}) or {}
        model_dir = (
            privacy_args.get("local_llm_model_dir")
            or getattr(self.config, "local_llm_model_dir", None)
            or os.environ.get(
            "LOCAL_PRIVACY_LLM_DIR", "./Qwen3-8B"
            )
        )

        response = privacy_layer.cloud_agent_compute_with_tokens(
            anon_tokens=anon_tokens,
            compute_instruction=compute_instruction,
            usage_reason=usage_reason,
            original_task=original_task or "",
            model_dir=model_dir,
        )

        # 结果记录到 current_return，后续在 JSONRecorder.update_after 里拼接到对话中
        self.current_return = {
            "operation": "do",
            "action": "Call_API",
            "kwargs": {
                "instruction": instruction,
                "response": response,
                "with_screen_info": False,
            },
        }
