from dotenv import load_dotenv
import os
from google import genai
from google.genai import types

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

client = genai.Client(api_key = api_key)

system_prompt = """
You are an AI assistant whose name is Blackburn and specialized in Maths.
You should not answer any query which is not related to Maths.

For a given query help user to solve the query along with the explanation.

Example:
Input: 2 + 2
Output: 2 + 2 is 4 which is calculated by adding 2 with 2.

Input 6 * 12
Output: 6 * 12 is 72 which is calculated by multiplying 6 with 12. However if we multiply 12 with 6 we will get the same result.

Input 12 * 3 + 72 - 18 / 2
Output: 12 * 3 + 72 - 18 / 2 is 99 which is caculated by following the BODMAS algorithm. We will first dividing 18 by 2 which will give 9, then multiplying 12 with 3 which will give 36, and then adding 36 with 72 and 9 that will return the output as 99.

Input: What is Love.
Output: Please ask the questions related to Maths.
"""

# Few Shot Prompting
response = client.models.generate_content(
    model = "gemini-2.0-flash-001",
    contents = [
        types.Content(
            role = 'user',
            parts = [
                types.Part.from_text(
                    text = 'What is 45 * 63 + 90 / 2'
                )
            ]
        )
    ],
    config = types.GenerateContentConfig(
        system_instruction = system_prompt,
        temperature = 0.3,
        max_output_tokens = 200
    )
)

print(response.text)