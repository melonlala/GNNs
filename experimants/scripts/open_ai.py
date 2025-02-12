import os
from openai import OpenAI
os.environ["HTTP_PROXY"] = "http://127.0.0.1:7866"
os.environ["HTTPS_PROXY"] = "http://127.0.0.1:7866"

client = OpenAI(
    api_key="sk-kqFOf0Z2CxakzAfsFduRRaNdId3J9VkJYAMpduciXJ7hNsvk",  # This is the default and can be omitted
    base_url="https://cld.gpt5api.cc/v1",
)

chat_completion = client.chat.completions.create(
    messages=[
        {
            "role": "user",
            "content": "Say this is a test",
        }
    ],
    model="gpt-4o",
)
print(chat_completion.choices[0].message)