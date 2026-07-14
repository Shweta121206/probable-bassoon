from google import genai
from dotenv import load_dotenv
import os

print("Step 1")

load_dotenv()

print("Step 2")

api_key = os.getenv("GEMINI_API_KEY")
print("API Key Found:", api_key is not None)

client = genai.Client(api_key=api_key)

print("Step 3")

try:
    response = client.models.generate_content(
        model="gemini-2.0-flash-lite",
        contents="Say only Hello Shweta"
    )

    print("Step 4")
    print(response)
    print("----------------")
    print("TEXT:")
    print(repr(response.text))

except Exception as e:
    print("ERROR:")
    print(type(e))
    print(e)