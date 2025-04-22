from dotenv import load_dotenv
import os
import json
from google import genai
from google.genai import types

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

client = genai.Client(api_key = api_key)

system_prompt = """
You are an AI assistant whose name is Blackburn and specialized in breaking down complex problems and then resolve the user input query.
For the given user input, analyse the input and break down the problem step by step.
Atleast think 5 to 6 steps on how to solve the problem before solving it down.

The steps are you get a user input, you analyse, you think, and think again for several times and then return an output with explanation, and then finally you validate the output as well before returning the final result.

Follow the provided steps in a sequence, that is "analyse", "think", "output", "validate", and finally "result".

Rules:
1. Follow the strict JSON output as per the output schema.
2. Always perform one step at a time and wait for next input.
3. Carefully analyse the user query.

Output Format:
{{ step: "string", content: "string" }}

Example:
Input: What is 23 - 3
Output: {{ step: "analyse", content: "Alright! The user is interested in maths query and he's asking a basic arithmetic operation." }}
Output: {{ step: "think", content: "To perform the addition I must go from left to right and add all the operands." }}
Output: {{ step: "output", content: "20" }}
Output: {{ step: "validate", content: "seems like 20 is correct answer for 23 - 3" }}
Output: {{ step: "result", content: "23 - 3 is 20 which we can get by subtracting 3 from 23." }}
"""

# Chain of Thoughts
response = client.models.generate_content(
    model = "gemini-2.0-flash-001",
    contents = [
        types.Content(
            role = 'user',
            parts = [
                types.Part.from_text(
                    text = "What is 3 * 97 + 57 - 23 * 2"
                )
            ]
        ),
        types.Content(
            role = 'assistant',
            parts = [
                types.Part.from_text(
                    text = json.dumps({
                        "content": "Okay! The user wants me to evaluate an arithmetic expression involving multiplication, addition, and subtraction.",
                        "step": "analyse"
                    })
                )
            ]
        ),
        types.Content(
            role = 'assistant',
            parts = [
                types.Part.from_text(
                    text = json.dumps({
                        "content": "I need to break this problem into smaller steps by following the order of operations (PEMDAS/BODMAS). First, I'll perform the multiplications, then the addition, and finally the subtraction.",
                        "step": "think"
                    })
                )
            ]
        ),
        types.Content(
            role = 'assistant',
            parts = [
                types.Part.from_text(
                    text = json.dumps({
                        "content": "First, I'll calculate 3 * 97 and 23 * 2. Then, I'll add 57 to the result of 3 * 97. Finally, I'll subtract the result of 23 * 2 from the sum.",
                        "step": "think"
                    })
                )
            ]
        ),
        types.Content(
            role = 'assistant',
            parts = [
                types.Part.from_text(
                    text = json.dumps({
                        "content": "3 * 97 = 291, 23 * 2 = 46. Now the expression is 291 + 57 - 46",
                        "step": "output"
                    })
                )
            ]
        ),
        types.Content(
            role = 'assistant',
            parts = [
                types.Part.from_text(
                    text = json.dumps({
                        "content": "Now I need to perform the addition and subtraction from left to right: 291 + 57 - 46.  First, 291 + 57 = 348. Then, 348 - 46 = 302.",
                        "step": "output"
                    })
                )
            ]
        )
    ],
    config = types.GenerateContentConfig(
        system_instruction = system_prompt,
        temperature = 0.3,
        response_mime_type = 'application/json',
        response_schema={
            'required': [
                'step',
                'content'
            ],
            'properties': {
                'step': {'type': 'STRING'},
                'content': {'type': 'STRING'}
            },
            'type': 'OBJECT'
        }
    )
)

print(response.text)

# Final Output Result
# {
#   "content": "The expression 3 * 97 + 57 - 23 * 2 equals 302. I performed the calculations step by step following the order of operations.",
#   "step": "result"
# }