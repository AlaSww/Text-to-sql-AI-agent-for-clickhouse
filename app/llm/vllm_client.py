import requests
from app.llm.base import LLMClient


class VLLMClient(LLMClient):
    """Self-hosted vLLM client using OpenAI-compatible API."""

    def __init__(self, base_url: str, model: str):
        """Initialize vLLM client.
        
        Args:
            base_url: Base URL of vLLM server (e.g. 'http://localhost:8000').
            model: Model name to request from vLLM.
        """
        self.base_url = base_url.rstrip("/")
        self.model = model

    def generate_text(self, messages: list[dict]) -> str:
        """Generate text using vLLM server.
        
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

        r = requests.post(
            f"{self.base_url}/v1/chat/completions",
            json=payload,
            timeout=60
        )
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]