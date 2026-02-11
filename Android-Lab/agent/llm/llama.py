from agent.model import *

class LLaMAModelAgent(OpenAIAgent):
    @backoff.on_exception(
        backoff.expo, Exception,
        on_backoff=handle_backoff,
        on_giveup=handle_giveup,
        max_tries=10
    )

    def act(self, messages: List[Dict[str, Any]] = None, prefix=None, prompt=None) -> str:
        if messages is not None:
            prompt = self.format_prompt(messages, prefix=prefix)
        headers = {
            'Content-Type': 'application/json',
        }

        data = {
            'model': 'llama3-8b',
            'messages': [{"role": "user", "content": prompt}],
            'seed': 34,
            "do_sample": False if self.temperature < 0.001 else True,
            'temperature': self.temperature,
            'max_tokens': self.max_tokens,
            'stream': False
        }

        response = requests.post(self.api_base, headers=headers, json=data)
        return response.json()["choices"][0]["message"]["content"]

    

    def format_prompt(self, messages: List[Dict[str, Any]], prefix=None) -> str:
        history = ""
        turn = 0
        for message in messages:
            if message == messages[-1]:
                break

            if message["role"] == "assistant":
                history += f"Round {turn}\n\n<|user|>\n** XML **\n\n<|assistant|>\n{message['content']}\n\n"
                turn += 1

        if messages[-1].get("current_app"):
            current_app_name = messages[-1]['current_app']
            current_turn = f"Round {turn}\n\n<|user|>\n{json.dumps({'current_app': current_app_name}, ensure_ascii=False)}\n{messages[-1]['content']}\n\n<|assistant|>\n"
        else:
            current_turn = f"Round {turn}\n\n<|user|>\n{messages[-1]['content']}\n\n<|assistant|>\n"

        prompt = history + current_turn
        if prefix is not None:
            prompt = prefix + "\n\n" + prompt
        elif prefix is None and messages[0].get("role") == "system":
            prompt = messages[0].get("content") + "\n\n" + prompt

        return prompt