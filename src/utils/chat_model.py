from abc import ABC, abstractmethod
from openai import OpenAI
import json


class ChatModel(ABC):
    @abstractmethod
    def generate(
        self,
        prompt: str,
        system="You are a helpful assistant.",
        return_type="json",
        **kwargs,
    ): ...


class OpenAIChatModel(ChatModel):
    def __init__(self, model_name="gpt-4o", base_url=None, api_key=None):
        self.model = OpenAI(base_url=base_url, api_key=api_key)
        self.model_name = model_name
    def generate(
        self,
        prompt: str,
        system="You are a helpful assistant.",
        return_type="json",
        **kwargs,
    ):
        completion = self.model.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            **kwargs,
        )
        response = completion.choices[0].message.content

        if return_type == "json":
            response = extract_json(response)

        return response


def extract_json(response: str):
    if response.startswith("```json"):
        # 去掉 ```
        response = "\n".join(response.splitlines()[1:-1])
    return json.loads(response)
