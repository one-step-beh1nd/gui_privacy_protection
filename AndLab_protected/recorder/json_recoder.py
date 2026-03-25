import json
import os

import jsonlines

from utils_mobile.utils import draw_bbox_multi
from utils_mobile.xml_tool import UIXMLTree
from utils_mobile.privacy_protection import get_privacy_layer



def get_compressed_xml(xml_path, type="plain_text", version="v1"):
    xml_parser = UIXMLTree()
    with open(xml_path, 'r', encoding='utf-8') as f:
        xml_str = f.read()
    try:
        compressed_xml = xml_parser.process(xml_str, level=1, str_type=type)
        if isinstance(compressed_xml, tuple):
            compressed_xml = compressed_xml[0]

        if type == "plain_text":
            compressed_xml = compressed_xml.strip()
    except Exception as e:
        compressed_xml = None
        print(f"XML compressed failure: {e}")
    return compressed_xml


class JSONRecorder:
    def __init__(self, id, instruction, anonymized_instruction, page_executor, config):
        self.id = id
        # 原始用户任务指令（未匿名）
        self.instruction = instruction
        # 匿名后的任务指令（发给云端 agent 的版本）
        self.anonymized_instruction = anonymized_instruction
        self.page_executor = page_executor

        self.turn_number = 0
        trace_dir = os.path.join(config.task_dir, 'traces')
        xml_dir = os.path.join(config.task_dir, 'xml')
        log_dir = config.task_dir
        if not os.path.exists(trace_dir):
            os.makedirs(trace_dir)
        if not os.path.exists(xml_dir):
            os.makedirs(xml_dir)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        self.trace_file_path = os.path.join(trace_dir, 'trace.jsonl')
        self.xml_file_path = os.path.join(xml_dir)
        self.log_dir = log_dir
        # Create prompts directory for saving prompts sent to cloud agent
        prompts_dir = os.path.join(log_dir, 'prompts')
        if not os.path.exists(prompts_dir):
            os.makedirs(prompts_dir)
        self.prompts_dir = prompts_dir
        self.contents = []
        self.xml_history = []
        self.history = []
        self.command_per_step = []
        
        # Set task directory for privacy protection layer statistics
        privacy_layer = get_privacy_layer()
        privacy_layer.set_task_dir(log_dir)
        if config.version is None or config.version == "v1":
            self.xml_compressed_version = "v1"
        elif config.version == "v2":
            self.xml_compressed_version = "v2"

    def update_response_deprecated(self, controller, response=None, prompt="** screenshot **", need_screenshot=False,
                                   ac_status=False):
        if need_screenshot:
            self.page_executor.update_screenshot(prefix=str(self.turn_number), suffix="before")
        xml_path = None
        ac_xml_path = None

        if not ac_status:
            xml_status = controller.get_xml(prefix=str(self.turn_number), save_dir=self.xml_file_path)
            if "ERROR" in xml_status:
                xml_path = "ERROR"
            else:
                xml_path = os.path.join(self.xml_file_path, str(self.turn_number) + '.xml')
        else:
            xml_status = controller.get_ac_xml(prefix=str(self.turn_number), save_dir=self.xml_file_path)
            if "ERROR" in xml_status:
                ac_xml_path = "ERROR"
            else:
                ac_xml_path = os.path.join(self.xml_file_path, 'ac_' + str(self.turn_number) + '.xml')
        step = {
            "trace_id": self.id,
            "index": self.turn_number,
            "prompt": prompt if self.turn_number > 0 else f"{self.anonymized_instruction}",
            "image": self.page_executor.current_screenshot,
            "xml": xml_path,
            "ac_xml": ac_xml_path,
            "response": response,
            # "url": map_url_to_real(page.url),
            "window": controller.viewport_size,
            "target": self.instruction,
            "original_instruction": self.instruction,
            "anonymized_instruction": self.anonymized_instruction,
            "current_activity": controller.get_current_activity()
        }
        step = self.test_per_step(step, controller)
        self.contents.append(step)

        return xml_status

    def test_per_step(self, step, controller):
        if len(self.command_per_step) == 0 or self.command_per_step[0] is None:
            return step
        step["command"] = {}
        for command in self.command_per_step:
            if "adb" not in command:
                continue
            result = controller.run_command(command)
            step["command"][command] = result
        return step

    def update_before(self, controller, prompt="** XML **", need_screenshot=False, ac_status=False, need_labeled=False):
        privacy_layer = get_privacy_layer()
        
        if need_screenshot:
            self.page_executor.update_screenshot(prefix=str(self.turn_number), suffix="before")
            # Apply privacy protection to screenshot
            if privacy_layer.enabled and self.page_executor.current_screenshot:
                try:
                    masked_image_path, new_tokens = privacy_layer.identify_and_mask_screenshot(
                        self.page_executor.current_screenshot
                    )
                    # Update screenshot path to masked version
                    if masked_image_path != self.page_executor.current_screenshot:
                        self.page_executor.current_screenshot = masked_image_path
                except Exception as e:
                    print(f"Warning: Failed to apply privacy protection to screenshot: {e}")
        
        xml_path = None
        ac_xml_path = None

        if not ac_status:
            xml_status = controller.get_xml(prefix=str(self.turn_number), save_dir=self.xml_file_path)
            if "ERROR" in xml_status:
                xml_path = "ERROR"
                # Try fallback to ac_xml if regular xml fails
                ac_xml_status = controller.get_ac_xml(prefix=str(self.turn_number), save_dir=self.xml_file_path)
                if "ERROR" not in ac_xml_status:
                    # get_ac_xml returns the full path, use it directly
                    ac_xml_path = ac_xml_status
                else:
                    ac_xml_path = "ERROR"
            else:
                xml_path = os.path.join(self.xml_file_path, str(self.turn_number) + '.xml')
                # Note: Privacy protection is applied to compressed XML in get_latest_xml(),
                # not to the original XML file which is kept for record-keeping purposes.
        else:
            xml_status = controller.get_ac_xml(prefix=str(self.turn_number), save_dir=self.xml_file_path)
            if "ERROR" in xml_status:
                ac_xml_path = "ERROR"
                # Try fallback to regular xml if ac_xml fails
                regular_xml_status = controller.get_xml(prefix=str(self.turn_number), save_dir=self.xml_file_path)
                if "ERROR" not in regular_xml_status:
                    xml_path = os.path.join(self.xml_file_path, str(self.turn_number) + '.xml')
                else:
                    xml_path = "ERROR"
            else:
                # get_ac_xml returns the full path, use it directly
                ac_xml_path = xml_status
                # Note: Privacy protection is applied to compressed XML in get_latest_xml(),
                # not to the original XML file which is kept for record-keeping purposes.

        step = {
            "trace_id": self.id,
            "index": self.turn_number,
            "prompt": prompt if self.turn_number > 0 else f"{self.anonymized_instruction}",
            "image": self.page_executor.current_screenshot,
            "xml": xml_path,
            "ac_xml": ac_xml_path,
            "current_activity": controller.get_current_activity(),
            "window": controller.viewport_size,
            "target": self.instruction,
            "original_instruction": self.instruction,
            "anonymized_instruction": self.anonymized_instruction,
        }
        step = self.test_per_step(step, controller)
        if need_labeled:
            try:
                # 确定要使用的XML路径
                selected_xml_path = None
                if xml_path != "ERROR" and xml_path is not None:
                    selected_xml_path = xml_path
                elif ac_xml_path != "ERROR" and ac_xml_path is not None:
                    selected_xml_path = ac_xml_path
                
                # 只有当找到有效的XML路径时才调用set_elem_list
                if selected_xml_path is not None:
                    self.page_executor.set_elem_list(selected_xml_path)
                else:
                    print("Warning: No valid XML path found for labeling. xml_path:", xml_path, "ac_xml_path:", ac_xml_path)
                    # 初始化一个空的elem_list以避免后续AttributeError
                    if not hasattr(self.page_executor, 'elem_list'):
                        self.page_executor.elem_list = []
            except Exception as e:
                print("xml_path:", xml_path)
                print("ac_xml_path:", ac_xml_path)
                import traceback
                print(traceback.print_exc())
                # 如果set_elem_list失败，初始化一个空的elem_list
                if not hasattr(self.page_executor, 'elem_list'):
                    self.page_executor.elem_list = []
            
            # 只有当elem_list存在且不为空时才绘制边界框
            if hasattr(self.page_executor, 'elem_list') and len(self.page_executor.elem_list) > 0:
                draw_bbox_multi(self.page_executor.current_screenshot,
                                self.page_executor.current_screenshot.replace(".png", "_labeled.png"),
                                self.page_executor.elem_list)
                self.labeled_current_screenshot_path = self.page_executor.current_screenshot.replace(".png", "_labeled.png")
                step["labeled_image"] = self.labeled_current_screenshot_path
            else:
                # 如果没有可用的元素列表，使用原始截图
                self.labeled_current_screenshot_path = self.page_executor.current_screenshot
                step["labeled_image"] = self.labeled_current_screenshot_path
                print("Warning: No elements found for labeling, using original screenshot")

        self.contents.append(step)

    def dectect_auto_stop(self):
        if len(self.contents) <= 5:
            return
        should_stop = True
        parsed_action = self.contents[-1]['parsed_action']
        for i in range(1, 6):
            if self.contents[-i]['parsed_action'] != parsed_action:
                should_stop = False
                break
        if should_stop:
            self.page_executor.is_finish = True

    def get_latest_xml(self):
        if len(self.contents) == 0:
            return None
        # print(self.contents[-1])
        if self.contents[-1]['xml'] == "ERROR" or self.contents[-1]['xml'] is None:
            xml_path = self.contents[-1]['ac_xml']
        else:
            xml_path = self.contents[-1]['xml']
        
        # If xml_path is None or "ERROR", return a clear error message instead of None
        if xml_path is None or xml_path == "ERROR":
            return "[XML fetch failed: Unable to retrieve UI hierarchy. Element-based operations (tap, long_press, swipe) are not available. Please use coordinate-based operations or wait and retry.]"
        
        xml_compressed = get_compressed_xml(xml_path, version=self.xml_compressed_version)
        
        # Apply privacy protection to compressed XML only.
        privacy_layer = get_privacy_layer()
        if privacy_layer.enabled and xml_compressed:
            try:
                masked_xml_compressed, _ = privacy_layer.identify_and_mask_xml(xml_compressed)
                xml_compressed = masked_xml_compressed
            except Exception as e:
                print(f"Warning: Failed to apply privacy protection to compressed XML: {e}")
        elif not privacy_layer.enabled:
            print("[PrivacyProtection] Privacy protection is disabled, XML will not be anonymized")
        elif not xml_compressed:
            print("[PrivacyProtection] Warning: compressed XML is empty, skipping anonymization")
        
        with open(os.path.join(self.xml_file_path, f"{self.turn_number}_compressed_xml.txt"), 'w',
                  encoding='utf-8') as f:
            f.write(xml_compressed)
        self.page_executor.latest_xml = xml_compressed
        return xml_compressed

    def get_latest_xml_tree(self):
        if len(self.contents) == 0:
            return None
        print(self.contents[-1])
        if self.contents[-1]['xml'] == "ERROR" or self.contents[-1]['xml'] is None:
            xml_path = self.contents[-1]['ac_xml']
        else:
            xml_path = self.contents[-1]['xml']
        
        # If xml_path is None or "ERROR", return None instead of trying to process it
        if xml_path is None or xml_path == "ERROR":
            return None
        
        xml_compressed = get_compressed_xml(xml_path, type="json")
        return json.loads(xml_compressed)

    def update_execution(self, exe_res):
        if len(self.contents) == 0:
            return
        self.contents[-1]['parsed_action'] = exe_res
        with jsonlines.open(self.trace_file_path, 'a') as f:
            f.write(self.contents[-1])

    def update_after(self, exe_res, rsp):
        if len(self.contents) == 0:
            return
        self.contents[-1]['parsed_action'] = exe_res
        self.history.append({"role": "user", "content": "** XML **"})
        self.history.append({"role": "assistant", "content": rsp})
        self.contents[-1]["current_response"] = rsp
        with jsonlines.open(self.trace_file_path, 'a') as f:
            f.write(self.contents[-1])
        self.dectect_auto_stop()

    def save_prompt_to_cloud_agent(self, messages, turn_number=None, stage=None):
        """
        Save the prompt (messages) sent to cloud agent to a JSON file.
        
        Args:
            messages: List of message dictionaries sent to the agent
            turn_number: The turn number (if None, uses self.turn_number)
            stage: Optional stage identifier (e.g., "query", "referring" for SeeAct tasks)
        """
        if turn_number is None:
            turn_number = self.turn_number
        
        # Create filename with turn number and optional stage
        if stage:
            filename = f"prompt_turn_{turn_number}_{stage}.json"
        else:
            filename = f"prompt_turn_{turn_number}.json"
        
        filepath = os.path.join(self.prompts_dir, filename)
        
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(messages, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"Warning: Failed to save prompt to {filepath}: {e}")
