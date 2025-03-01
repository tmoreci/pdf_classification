import cohere
import yaml
from prompts import preamble, tool_description, user_message
from db import DocumentDatabase
from jinja2 import Template
import json
from model_scripts.pdf_utils import full_text_parse
from dotenv import load_dotenv
import os


class LLM:
    """Handles LLM interactions for Q&A over retrieved documents"""

    def __init__(self, api_key, database, config_path="llm_config.yaml"):
        # Initialize LLM client
        self.llm = cohere.ClientV2(api_key)
        self.db = database
        # Load configuration
        with open(config_path, "r") as config_file:
            self.config = yaml.safe_load(config_file)

        self.model = self.config.get("model", "command-r-08-2024")
        self.temperature = self.config.get("temperature", 0.1)
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
                                "description": "A natural language query of the database. Include any keywords necessary",
                                "type": "string",
                            }
                        },
                    },
                },
            ),
        ]
        self.functions_map = {"abstract_search": self.db.vector_search}

    def _handle_tool_call(self, messages, response):
        """
        Based on Cohere API docs
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

    def generate_answer(self, question, document, temperature=None):
        """Generate answer using retrieved documents"""
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


# Example usage
if __name__ == "__main__":
    load_dotenv()

    cohere_api_key = os.getenv("COHERE_API")
    gemini_api_key = os.getenv("GOOGLE_API")
    database = DocumentDatabase(gemini_api=gemini_api_key)
    document = full_text_parse("../data/2408.02545v1.pdf")
    query = "How could this research be combined with other research to enhance multi-agent systems?"
    # Initialize QA system
    model = LLM(cohere_api_key, database)
    model_response = model.generate_answer(query, document)
    print(model_response)
