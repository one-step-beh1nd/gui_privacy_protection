from typing import List, Dict, Any, Optional

import backoff
import json
import requests
from openai import OpenAI

from agent.utils import *
from templates.android_screenshot_template import *


def capture_llm_raw_response(agent: Any, response_obj: Any) -> None:
    """
    Store the full LLM/API response on agent.last_llm_raw_response (string: JSON or raw body).
    Call only after a successful API return, before parsing return value for the pipeline.
    """
    agent.last_llm_raw_response = None
    if response_obj is None:
        return
    try:
        if hasattr(response_obj, "model_dump_json"):
            agent.last_llm_raw_response = response_obj.model_dump_json()
            return
        if hasattr(response_obj, "model_dump"):
            agent.last_llm_raw_response = json.dumps(
                response_obj.model_dump(), ensure_ascii=False, default=str
            )
            return
    except Exception:
        pass
    try:
        if isinstance(response_obj, requests.Response):
            agent.last_llm_raw_response = response_obj.text
            return
    except Exception:
        pass
    try:
        agent.last_llm_raw_response = json.dumps(
            response_obj, ensure_ascii=False, default=str
        )
    except TypeError:
        agent.last_llm_raw_response = str(response_obj)


def handle_giveup(details):
    print(
        "Backing off {wait:0.1f} seconds afters {tries} tries calling fzunction {target} with args {args} and kwargs {kwargs}"
        .format(**details))


def handle_backoff(details):
    args = str(details['args'])[:1000]
    print(f"Backing off {details['wait']:0.1f} seconds after {details['tries']} tries "
          f"calling function {details['target'].__name__} with args {args} and kwargs ")

    import traceback
    print(traceback.format_exc())


class Agent:
    name: str

    @backoff.on_exception(
        backoff.expo, Exception,
        on_backoff=handle_backoff,
        on_giveup=handle_giveup,
    )
    def act(self, messages: List[Dict[str, Any]]) -> str:
        raise NotImplementedError

    def prompt_to_message(self, prompt, images):
        raise NotImplementedError

    def system_prompt(self, instruction) -> str:
        raise NotImplementedError


class OpenAIAgent(Agent):
    def __init__(
            self,
            api_key: str = '',
            api_base: str = '',
            model_name: str = '',
            max_new_tokens: int = 16384,
            temperature: float = 0,
            top_p: float = 0.7,
            **kwargs
    ) -> None:
        self.client = OpenAI(api_key=api_key, base_url=api_base)
        # openai.api_base = api_base
        # openai.api_key = api_key
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.kwargs = kwargs
        self.name = "OpenAIAgent"
        self.last_llm_raw_response: Optional[str] = None

    @backoff.on_exception(
        backoff.expo, Exception,
        on_backoff=handle_backoff,
        on_giveup=handle_giveup,
        max_tries=10
    )
    def act(self, messages: List[Dict[str, Any]]) -> str:
        r = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            max_tokens=self.max_new_tokens,
            temperature=self.temperature,
            top_p=self.top_p
        )
        capture_llm_raw_response(self, r)
        return r.choices[0].message.content

    def prompt_to_message(self, prompt, images):
        content = [
            {
                "type": "text",
                "text": prompt
            }
        ]
        for img in images:
            base64_img = image_to_base64(img)
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_img}"
                }
            })
        message = {
            "role": "user",
            "content": content
        }
        return message

    def system_prompt(self, instruction) -> str:
        return SYSTEM_PROMPT_ANDROID_MLLM_DIRECT + f"\n\nTask Instruction: {instruction}"


class HTTPAgent(Agent):
    def __init__(
            self,
            url: str,
            headers: Dict[str, Any] = {},
            body: Dict[str, Any] = {},
            return_format: str = "{response[choices][0][message][content]}"
    ) -> None:
        self.url = url
        self.headers = headers
        self.body = body
        self.return_format = return_format
        self.last_llm_raw_response: Optional[str] = None

    def update_messages(self, body: Dict[str, Any], messages: List[Dict[str, Any]]):
        body.update({"messages": messages})
        return body

    @backoff.on_exception(
        backoff.expo, Exception,
        on_backoff=handle_backoff,
        on_giveup=handle_giveup,
    )
    def act(self, messages: List[Dict[str, Any]]):
        body = self.update_messages({**self.body}, messages)
        response = requests.post(
            self.url, headers=self.headers, json=body
        )
        capture_llm_raw_response(self, response)
        try:
            resp_json = response.json()
        except Exception:
            resp_json = {"_raw_text": response.text}
        return self.return_format.format(response=resp_json)
