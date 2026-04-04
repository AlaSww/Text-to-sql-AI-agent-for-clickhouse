from abc import ABC, abstractmethod
from typing import List


class LLMClient(ABC):
    @abstractmethod
    def generate_text(self, messages: List[dict]) -> str:
        """Generate text from a list of messages with role and content.
        
        Args:
            messages: List of dicts with 'role' ('system' or 'user') and 'content' keys.
        
        Returns:
            Generated text from the LLM.
        """
        pass