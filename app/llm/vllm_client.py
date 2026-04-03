import requests
from app.llm.base import LLMClient


class VLLMClient(LLMClient):
    def __init__(self, base_url: str, model: str):
        self.base_url = base_url.rstrip("/")
        self.model = model

    def generate_text(self, messages: list[str]) -> str:
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": "\n\n".join(messages)}],
            "temperature": 0
        }

        r = requests.post(
            f"{self.base_url}/v1/chat/completions",
            json=payload,
            timeout=60
        )
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]