from generation.llm import LLM


llm = LLM()

response = llm.generate(
    "Explain Retrieval-Augmented Generation in one paragraph."
)

print("\n=== GEMINI RESPONSE ===")
print(response)