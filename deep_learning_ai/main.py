import openai
import os

from dotenv import load_dotenv, find_dotenv
__ = load_dotenv(find_dotenv())  # read local .env file

client = openai.OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY") # key is stored in .env file, and would be outdated in 7 days, so you need to update it every 7 days
)

# There is no currently free model
def get_completion(prompt, model="gpt-5.5", max_tokens=100):
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "user", "content": prompt}
        ],
        max_tokens=max_tokens
    )
    return response.choices[0].message.content.strip()

print(get_completion("What is the capital of France?"))