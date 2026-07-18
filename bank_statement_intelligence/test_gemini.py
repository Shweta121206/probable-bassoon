import os
from dotenv import load_dotenv
from google import genai

load_dotenv()

key = os.getenv("GEMINI_API_KEY")
model = os.getenv("GEMINI_MODEL_NAME", "gemini-2.5-flash")

if not key:
    print("Error: GEMINI_API_KEY not found in .env")
    exit(1)

client = genai.Client(api_key=key)

print(f"Testing Gemini API with model: {model}")
print("Sending request...")

try:
    response = client.models.generate_content(
        model=model,
        contents="Say hello."
    )
    print("Status: Success")
    print("Response:")
    print(response.text)
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()