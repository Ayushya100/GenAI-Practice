from dotenv import load_dotenv
import os
import json
from google import genai
from google.genai import types
from langchain_community.document_loaders import PyPDFLoader
from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_qdrant import QdrantVectorStore

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

current_dir = Path(__file__).parent
pdf_path = current_dir.parent / "assets" / "nodejs.pdf"

# Retrieval
## Step 1 - Data Source
loader = PyPDFLoader(file_path = pdf_path)
docs = loader.load()

## Step 2 - Chunking
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1000,
    chunk_overlap = 200     # To keep the context of last chunk
)

split_docs = text_splitter.split_documents(documents = docs)

## Step 3 - Vector Embedding
embeddings = GoogleGenerativeAIEmbeddings(model = "models/text-embedding-004")

# Above part is same as what we've in Injection
## Step 4 - Retrieval of relevant chunks
retriever = QdrantVectorStore.from_existing_collection(
    url = 'http://localhost:6333',
    collection_name = 'learning_langchain',
    embedding = embeddings
)

relevant_chunks = retriever.similarity_search(
    query = "What is FS Module?"
)

# Model Building
client = genai.Client(api_key = api_key)

system_prompt = f"""
You are an helpfull AI assistant who responds based on the available context.
For the given user input, analyse the input and then generate the response.

Rules:
1. Follow the strict string format as per the output format.
2. Carefully analyse the user query.

Output Format: "string"

Context: {relevant_chunks}

Example:
Input: How to import fs module?
Output: fs module is a built-in module that provides a way to interact with the file system on the computer.
"""

message = []

query = input("Ask: ")
message.append(types.Content(
    role = 'user',
    parts = [
        types.Part.from_text(
            text = query
        )
    ]
))

response = client.models.generate_content(
    model = "gemini-2.0-flash-001",
    contents = message,
    config = types.GenerateContentConfig(
        system_instruction = system_prompt,
        temperature = 0.3
    )
)

print(response.text)