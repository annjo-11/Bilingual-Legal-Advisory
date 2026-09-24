from huggingface_hub import InferenceClient


# Connect to Hugging Face
client = InferenceClient()


# Send a simple question to the LLM
response = client.chat.completions.create(
    model="openai/gpt-oss-120b",
    messages=[
        {
            "role": "user",
            "content": "What is identity theft? Explain it in simple words."
        }
    ],
    max_tokens=200
)


# Print the answer
print("\nLLM Response:")
print(response.choices[0].message.content)