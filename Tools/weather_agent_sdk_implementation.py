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

available_tools = types.FunctionDeclaration(
    name = 'get_weather',
    description = 'Takes a city name as an input and returns the current weather as an output',
    parameters = types.Schema(
        type = 'OBJECT',
        properties = {
            'city': types.Schema(
                type = 'STRING',
                description = 'The city name, e.g. New Delhi'
            )
        },
        required = ['city']
    )
)

system_prompt = """
You are an helpful AI assistant who is specialized in resolving user query.
You work on start, plan, action, observe mode.
For the given user query and available tools, plan the step-by-step execution. Based on the planning select the relevant tool from the available tools. And based on the
tool selection you perform an action to call the tool. Wait for the observation, and based on the observation from the tool call resolve the user query.

Rules:
1. Always perform one step at a time.
2. Carefully analyse the user query.

Example:
User query: What is the weather of New Delhi?
Output: The weather for New Delhi seems to be 28 degrees."
"""

query = input("Ask: ")
user_prompt = types.Content(
    role = 'user',
    parts = [
        types.Part.from_text(
            text = query
        )
    ]
)

# Defining tool and ampping
tool = types.Tool(function_declarations = [available_tools])

# Initial Model response
response = client.models.generate_content(
    model = "gemini-2.0-flash-001",
    contents = user_prompt,
    config = types.GenerateContentConfig(
        system_instruction = system_prompt,
        temperature = 0.3,
        tools = [tool]
    )
)
function_call_part = response.function_calls[0]
function_call_content = response.candidates[0].content

try:
    function_result = get_weather(
        function_call_part.args.get('city')
    )
    function_response = { 'result': function_result }
except (
    Exception
) as e:
    function_response = { 'error': str(e) }

function_response_part = types.Part.from_function_response(
    name = function_call_part.name,
    response = function_response
)
function_response_content = types.Content(
    role = 'tool',
    parts = [ function_response_part ]
)

response = client.models.generate_content(
    model = "gemini-2.0-flash-001",
    contents = [
        user_prompt,
        function_call_content,
        function_response_content
    ],
    config = types.GenerateContentConfig(
        system_instruction = system_prompt,
        temperature = 0.3,
        tools = [tool]
    )
)

print(response.text)