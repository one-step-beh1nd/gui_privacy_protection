import inspect
import json
import re
import time
from functools import partial

from templates.packages import find_package
from .utils import call_dino, plot_bbox


def remove_leading_zeros_in_string(s):
    # 使用正则表达式匹配列表中的每个数值并去除前导零
    return re.sub(r'\b0+(\d)', r'\1', s)


def _extract_action_call(text: str) -> str:
    candidate = (text or "").strip()
    if not candidate:
        return candidate

    action_idx = candidate.find("Action:")
    if action_idx != -1:
        candidate = candidate[action_idx + len("Action:"):].strip()

    lines = [line.strip() for line in candidate.splitlines() if line.strip()]
    if not lines:
        return candidate

    joined = " ".join(lines)
    start = None
    for func_name in ("do(", "finish(", "tap(", "type(", "swipe(", "long_press(", "press_back(", "press_home(", "press_enter(", "wait(", "launch("):
        idx = joined.find(func_name)
        if idx != -1 and (start is None or idx < start):
            start = idx
    if start is None:
        return lines[-1]

    joined = joined[start:]
    depth = 0
    in_string = False
    quote_char = ""
    escaped = False

    for idx, ch in enumerate(joined):
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == quote_char:
                in_string = False
            continue

        if ch in ("'", '"'):
            in_string = True
            quote_char = ch
            continue
        if ch == "(":
            depth += 1
            continue
        if ch == ")":
            depth -= 1
            if depth == 0:
                return joined[: idx + 1]

    return joined


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
        code = _extract_action_call(code_snippet)
        code = remove_leading_zeros_in_string(code.strip())
        if code.startswith("Action:"):
            code = code[len("Action:"):].strip()
        compile(code, "<action>", "exec")
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
        assert action in ["Tap", "Type", "Swipe", "Enter", "Home", "Back", "Long Press", "Wait", "Launch"], "Unsupported Action"
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

