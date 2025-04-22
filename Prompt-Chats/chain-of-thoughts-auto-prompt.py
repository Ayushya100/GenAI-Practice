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
Atleast think 5-6 steps on how to solve the problem before solving it down.

These steps are - you get a user input, you analyse, you think, and think again for several times and then return an output with explanation, and then finally you validate the output as well before returning the final result.

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

messages = []

query = input("Ask: ")
messages.append(types.Content(
    role = 'user',
    parts = [
        types.Part.from_text(
            text = query
        )
    ]
))

# Chain of Thoughts
while True:
    response = client.models.generate_content(
        model = "gemini-2.0-flash-001",
        contents = messages,
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

    parsed_response = json.loads(response.text)
    messages.append(types.Content(
        role = 'assistant',
        parts = [
            types.Part.from_text(
                text = json.dumps(parsed_response)
            )
        ]
    ))

    if parsed_response.get('step') != 'result':
        print(f"system generated output: {parsed_response.get("step")} - {parsed_response.get('content')}")
        continue

    print(f"Final response: {parsed_response.get("step")} - {parsed_response.get('content')}")
    break