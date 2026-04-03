import requests
from app.llm.base import LLMClient


class GroqClient(LLMClient):
    def __init__(self, api_key: str, model: str):
        self.api_key = api_key
        self.model = model
        self.url = "https://api.groq.com/openai/v1/chat/completions"

    def generate_text(self, messages: list[str]) -> str:
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": "\n\n".join(messages)}],
            "temperature": 0
        }

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        r = requests.post(self.url, json=payload, headers=headers, timeout=60)
        r.raise_for_status()

        data = r.json()
        return data["choices"][0]["message"]["content"]