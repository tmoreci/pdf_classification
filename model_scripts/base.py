from pydantic import BaseModel
from abc import ABC, abstractmethod


class Extraction(BaseModel):
    title: str
    abstract: str


class Summary(BaseModel):
    title: str
    summary: str


class LLM(ABC):
    """Abstract base class for LLM interactions"""

    def __init__(self, api_key, database, config_path="llm_config.yaml"):
        self.api_key = api_key
        self.db = database
        self.config_path = config_path

    @abstractmethod
    def generate_answer(self, question, document, temperature=None):
        """Generate answer using retrieved documents"""
        pass

    @abstractmethod
    def _handle_tool_call(self, response):
        """Handle tool call responses"""
        pass
