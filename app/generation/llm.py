import os

from dotenv import load_dotenv
from google import genai
from google.genai import types


load_dotenv()


class LLM:
    def __init__(self, model="gemini-3-flash-preview"):
        api_key = os.getenv("GEMINI_API_KEY")

        if not api_key:
            raise ValueError("GEMINI_API_KEY is not set")

        self.client = genai.Client(api_key=api_key)
        self.model = model

    def generate(self, prompt):
        response = self.client.models.generate_content(
            model=self.model,
            contents=prompt,
            config=types.GenerateContentConfig(
                automatic_function_calling=types.AutomaticFunctionCallingConfig(
                    disable=True
                )
            )
        )

        return response.text