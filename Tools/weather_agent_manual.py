from dotenv import load_dotenv
import os
import json
from google import genai
from google.genai import types

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

client = genai.Client(api_key = api_key)

# Tool
def get_weather(city: str):
    return "36 degree celcius"

system_prompt = """
You are an helpful AI assistant who is specialized in resolving user query.
You work on start, plan, action, observe mode.
For the given user query and available tools, plan the step-by-step execution. Based on the planning select the relevant tool from the available tools. And based on the
tool selection you perform an action to call the tool. Wait for the observation, and based on the observation from the tool call resolve the user query.

Rules:
1. Follow the output JSON format.
2. Always perform one step at a time and wait for next input.
3. Carefully analyse the user query.

Output JSON Format:
{{
    "step": "string",
    "content": "string",
    "function": "The name of function if the step is action",
    "input": "The input parameter for the function"
}}

Available Tools:

Example:
User query: What is the weather of New Delhi?
Output: {{ "step": "plan", "content": "The user is interested in weather data of new delhi" }}
Output: {{ "step": "plan", "content": "From the available tool I should call get_weather" }}
Output: {{ "step": "action", "function": "get_weather", "input": "new delhi" }}
Output: {{ "step": "observe", "output": "28 Degree Celcius" }}
Output: {{ "step": "output", "output": "The weather for New Delhi seems to be 28 degrees." }}
"""

response = client.models.generate_content(
    model = "gemini-2.0-flash-001",
    contents = [
        types.Content(
            role = 'user',
            parts = [
                types.Part.from_text(
                    text = "What is the current weather in Bangalore"
                )
            ]
        ),
        types.Content(
            role = 'assistant',
            parts = [
                types.Part.from_text(
                    text = json.dumps({
                        "content": "The user wants to know the current weather in Bangalore.",
                        "step": "plan"
                    })
                )
            ]
        ),
        types.Content(
            role = 'assistant',
            parts = [
                types.Part.from_text(
                    text = json.dumps({
                        "content": "I should use the available tools to get the weather information for Bangalore.",
                        "step": "plan"
                    })
                )
            ]
        ),
        types.Content(
            role = 'assistant',
            parts = [
                types.Part.from_text(
                    text = json.dumps({
                        "content": "I should call the get_weather function to get the weather information.",
                        "step": "plan"
                    })
                )
            ]
        ),
        types.Content(
            role = 'assistant',
            parts = [
                types.Part.from_text(
                    text = json.dumps({
                        "content": "Calling the get_weather function with the location Bangalore.",
                        "step": "action",
                        "function": "get_weather",
                        "input": "Bangalore"
                    })
                )
            ]
        ),
        types.Content(
            role = 'assistant',
            parts = [
                types.Part.from_text(
                    text = json.dumps({
                        "content": "Waiting for the weather information for Bangalore.",
                        "step": "observe"
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
                'content': {'type': 'STRING'},
                'function': {'type': 'STRING'},
                'input': {'type': 'STRING'}
            },
            'type': 'OBJECT'
        }
    )
)

print(response.text)

# Output
# {
#     "content": "The weather in Bangalore is 24 degrees Celsius with partly cloudy skies.",
#     "step": "output"
# }