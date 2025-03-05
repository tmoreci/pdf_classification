import cohere
from prompts import (
    preamble,
    tool_description,
    user_message,
    gemini_prompt,
    gemini_retrieved_prompt,
)
from db import DocumentDatabase
from jinja2 import Template
import json
import re
from google import genai
from google.genai import types
from pathlib import Path
from base import LLM
from typing import Dict, List, Any, Optional, Tuple


class CohereLLM(LLM):
    """Handles LLM interactions for Q&A over retrieved documents using Cohere's API"""

    def __init__(
        self,
        api_key: str,
        database: DocumentDatabase,
        model: str,
        temperature: float,
    ):
        """
        Initialize the CohereLLM with the necessary configurations.

        Args:
            api_key: Cohere API key
            database: DocumentDatabase instance for retrieval
            model: Name of the Cohere model to use
            temperature: Temperature parameter for response generation
        """
        super().__init__(api_key, database, model, temperature)
        # Initialize LLM client
        self.llm = cohere.ClientV2(api_key)
        self.tools = [
            cohere.ToolV2(
                type="function",
                function={
                    "name": "abstract_search",
                    "description": tool_description,
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "description": (
                                    "A natural language query of the database. Include any keywords necessary"
                                ),
                                "type": "string",
                            }
                        },
                    },
                },
            ),
        ]
        self.functions_map = {"abstract_search": self.db.vector_search}

    def _handle_tool_call(
        self, messages: List[Dict[str, Any]], response: Any
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Handle the tool calls from the Cohere API response.

        Args:
            messages: List of message objects in the conversation
            response: Response object from Cohere API

        Returns:
            Tuple containing updated messages and tool result
        """
        messages.append(
            {
                "role": "assistant",
                "tool_plan": response.message.tool_plan,
                "tool_calls": response.message.tool_calls,
            }
        )
        for tc in response.message.tool_calls:
            tool_result = self.functions_map[tc.function.name](
                **json.loads(tc.function.arguments)
            )
            # for data in tool_result:
            tool_content = []
            for data in tool_result["documents"][0]:
                # Optional: the "document" object can take an "id" field for use in citations, otherwise auto-generated
                tool_content.append(
                    {
                        "type": "document",
                        "document": {"data": data},
                    }
                )
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": tool_content,
                }
            )
        return messages, tool_result

    def generate_answer(
        self, question: str, document: str, temperature: Optional[float] = None
    ) -> str:
        """
        Generate an answer to a question using retrieved documents.

        Args:
            question: User's question to answer
            document: Document content to use as context
            temperature: Optional temperature override

        Returns:
            Generated answer text
        """
        if temperature is None:
            temperature = self.temperature
        prompt_template = Template(user_message)
        user_input = prompt_template.render(
            document=document, user_query=question
        )
        sources = []
        # Generate response
        messages = [
            {
                "role": "system",
                "content": preamble,
            },
            {"role": "user", "content": user_input},
        ]
        response = self.llm.chat(
            model=self.model,
            messages=messages,
            tools=self.tools,
            temperature=temperature,
            seed=42,
        )
        print("initial response complete")
        if response.message.tool_calls:
            messages, sources = self._handle_tool_call(messages, response)
            response = self.llm.chat(
                model=self.model, messages=messages, tools=self.tools
            )
            # output = self._add_citations(response)
        # print(response)
        return response.message.content[0].text


class GeminiLLM(LLM):
    """Handles LLM interactions for Q&A over retrieved documents using Google's Gemini API"""

    def __init__(
        self,
        api_key: str,
        database: DocumentDatabase,
        model: str,
        temperature: float,
    ):
        """
        Initialize the GeminiLLM with the necessary configurations.

        Args:
            api_key: Gemini API key
            database: DocumentDatabase instance for retrieval
            model: Name of the Gemini model to use
            temperature: Temperature parameter for response generation
        """
        super().__init__(api_key, database, model, temperature)
        # Initialize LLM client
        self.llm = genai.Client(api_key=api_key)
        self.tool = types.Tool(
            function_declarations=[
                types.FunctionDeclaration(
                    name="query",
                    description="Returns a list of relevant documents from a vector database",
                    parameters=types.Schema(
                        properties={
                            "query": types.Schema(type="STRING"),
                        },
                        type="OBJECT",
                    ),
                )
            ]
        )
        self.model_config = {
            "tools": [self.tool],
            "automatic_function_calling": {"disable": True},
        }

    def _extract_citations(self, llm_response: str) -> List[int]:
        """
        Extracts document citations from an LLM response that uses [i] citation format.

        Args:
            llm_response: The text response from the LLM with citations in [i] format

        Returns:
            List of unique document indices that were cited in the response
        """
        # Find all citations in the format [i]
        citation_pattern = r"\[(\d+)\]"
        citations = re.findall(citation_pattern, llm_response)

        # Convert to integers and get unique citations
        cited_doc_indices = [int(idx) for idx in citations]
        unique_cited_docs = sorted(set(cited_doc_indices))

        return unique_cited_docs

    def _handle_tool_call(self, response: List[Any]) -> Dict[str, Any]:
        """
        Handle tool calls from the Gemini API response.

        Args:
            response: Function call response from Gemini API

        Returns:
            Results from the database query
        """
        print(response)
        query = response[0].args["query"]
        tool_result = self.db.hybrid_search(query)
        return tool_result

    def generate_answer(
        self, question: str, document: str, temperature: Optional[float] = None
    ) -> Tuple[Any, List[int], Dict[str, Any]]:
        """
        Generate an answer to a question using a document and retrieved context.

        Args:
            question: User's question to answer
            document: Path to the PDF document
            temperature: Optional temperature override

        Returns:
            Tuple containing the response object, cited document indices, and retrieved documents
        """
        if temperature is None:
            temperature = self.temperature
        prompt_template = Template(gemini_prompt)
        user_input = prompt_template.render(user_query=question)
        file_path = Path(document)
        cited_docs = []
        retrieved_docs = []
        # Generate response
        response = self.llm.models.generate_content(
            model=self.model,
            contents=[
                types.Part.from_bytes(
                    data=file_path.read_bytes(),
                    mime_type="application/pdf",
                ),
                user_input,
            ],
            config=self.model_config,
        )
        if response.function_calls:
            retrieved_docs = self._handle_tool_call(response.function_calls)
            retrieval_template = Template(gemini_retrieved_prompt)
            # ! To Do: Add titles to prompt
            user_input = retrieval_template.render(
                user_query=question,
                documents=retrieved_docs["documents"][0],
                enumerate=enumerate,
            )
            response = self.llm.models.generate_content(
                model=self.model,
                contents=[
                    types.Part.from_bytes(
                        data=file_path.read_bytes(),
                        mime_type="application/pdf",
                    ),
                    user_input,
                ],
                config=self.model_config,
            )
            cited_docs = self._extract_citations(response.text)

        return response, cited_docs, retrieved_docs


# Example usage
if __name__ == "__main__":
    load_dotenv()

    cohere_api_key = os.getenv("COHERE_API")
    gemini_api_key = os.getenv("GOOGLE_API")
    database = DocumentDatabase(gemini_api=gemini_api_key)
    doc_path = "../data/2408.02545v1.pdf"
    model = "gemini-2.0-flash"
    temperature = 0.1
    document = full_text_parse(doc_path)
    query = "How could this research be combined with other research to enhance multi-agent systems?"
    # Initialize QA system
    # model = CohereLLM(cohere_api_key, database)
    model = GeminiLLM(gemini_api_key, database, model, temperature)
    model_response, cited_docs, retrieved_docs = model.generate_answer(
        query, doc_path
    )
    print(model_response.text)
