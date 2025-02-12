import os
from openai import OpenAI
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv(), override=True)
# print(os.getenv("API_KEY"))
# os.environ["HTTP_PROXY"] = "http://127.0.0.1:7834"
# os.environ["HTTPS_PROXY"] = "http://127.0.0.1:7834"
client = OpenAI(
    api_key=os.getenv("API_KEY"),  # This is the default and can be omitted
    base_url = os.getenv("BASE_URL")
)
print(client.api_key)
print(client.base_url)
chat_completion = client.chat.completions.create(
    messages=[
        {
            "role": "user",
            "content": "Say this is a test",
        }
    ],
    model="gpt-4o",
)
print(chat_completion.choices[0].message.content)