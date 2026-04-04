import requests
from app.llm.base import LLMClient


class GroqClient(LLMClient):
    """Groq LLM client using their OpenAI-compatible API."""

    def __init__(self, api_key: str, model: str):
        """Initialize Groq client.
        
        Args:
            api_key: Groq API key.
            model: Model identifier (e.g. 'meta-llama/llama-4-scout-17b-16e-instruct').
        """
        self.api_key = api_key
        self.model = model
        self.url = "https://api.groq.com/openai/v1/chat/completions"

    def generate_text(self, messages: list[dict]) -> str:
        """Generate text using Groq API.
        
        Args:
            messages: List of dicts with 'role' and 'content' keys.
        
        Returns:
            Generated text from the LLM.
        """
        payload = {
            "model": self.model,
            "messages": messages,
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