from dotenv import load_dotenv
import os
from google import genai
from google.genai import types

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

client = genai.Client(api_key = api_key)

# Zero Shot Prompting
response = client.models.generate_content(
    model = "gemini-2.0-flash-001",
    contents = "What is the value of 2 * 0 + 78 + 29 * 2 - 12"
)

print(response.text)