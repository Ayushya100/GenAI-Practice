from dotenv import load_dotenv
import os
import json
import requests
from google import genai
from google.genai import types

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

client = genai.Client(api_key = api_key)

# Tool
def get_weather(city: str):
    url = f"https://wttr.in/{city}?format=%C+%t"
    response = requests.get(url)

    if response.status_code == 200:
        return f"The weather in {city} is {response.text}"
    return "Something went wrong"

def add(x, y):
    print("Tool called to add two numbers")
    return x + y

available_tools = {
    "get_weather": {
        "fn": get_weather,
        "description": "Takes a city name as an input and returns the current weather as an output"
    },
    "add": {
        "fn": add,
        "description": "Takes two numbers x and y and returns sum of the given input, that is x + y"
    }
}

system_prompt = f"""
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
- get_weather: Takes a city name as an input and returns the current weather as an output
- add: Takes two numbers x and y and returns sum of the given input, that is x + y

Example:
User query: What is the weather of New Delhi?
Output: {{ "step": "plan", "content": "The user is interested in weather data of new delhi" }}
Output: {{ "step": "plan", "content": "From the available tool I should call get_weather" }}
Output: {{ "step": "action", "function": "get_weather", "input": "new delhi" }}
Output: {{ "step": "observe", "content": "28 Degree Celcius" }}
Output: {{ "step": "output", "content": "The weather for New Delhi seems to be 28 degrees." }}
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

while True:
    response = client.models.generate_content(
        model = "gemini-2.0-flash-001",
        contents = messages,
        config = types.GenerateContentConfig(
            system_instruction = system_prompt,
            temperature = 0.3,
            response_mime_type = 'application/json',
            response_schema = {
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

    parsed_response = json.loads(response.text)
    messages.append(types.Content(
        role = 'assistant',
        parts = [
            types.Part.from_text(
                text = json.dumps(parsed_response)
            )
        ]
    ))

    if parsed_response.get("step") == "plan":
        print(f"system generated output: {parsed_response.get("step")} - {parsed_response.get('content')}")
        continue

    if parsed_response.get("step") == "action":
        tool_name = parsed_response.get("function")
        tool_input = parsed_response.get("input")

        if available_tools.get(tool_name, False) != False:
            output = available_tools[tool_name].get("fn")(tool_input)
            messages.append(types.Content(
                role = 'assistant',
                parts = [
                    types.Part.from_text(
                        text = json.dumps({
                            'step': 'observe',
                            'output': output
                        })
                    )
                ]
            ))
        continue

    if parsed_response.get("step") == "output":
        print(f"Final response: {parsed_response.get("step")} - {parsed_response.get('content')}")
        break