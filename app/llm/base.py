from abc import ABC, abstractmethod
from typing import List


class LLMClient(ABC):
    @abstractmethod
    def generate_text(self, messages: List[str]) -> str:
        pass