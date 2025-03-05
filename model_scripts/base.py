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

    def __init__(self, api_key, database, model, temperature):
        self.api_key = api_key
        self.db = database
        self.model = model
        self.temperature = temperature

    @abstractmethod
    def generate_answer(self, question, document, temperature=None):
        """Generate answer using retrieved documents"""
        pass

    @abstractmethod
    def _handle_tool_call(self, response):
        """Handle tool call responses"""
        pass
