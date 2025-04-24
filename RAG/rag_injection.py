from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_qdrant import QdrantVectorStore

load_dotenv()

current_dir = Path(__file__).parent
pdf_path = current_dir.parent / "assets" / "nodejs.pdf"

# Injection
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

# ## Step 4 - Vector Store
vector_store = QdrantVectorStore.from_documents(
    documents = [],
    url = 'http://localhost:6333',
    collection_name = 'learning_langchain',
    embedding = embeddings
)

vector_store.add_documents(documents = split_docs)

print(len(docs))
print(len(split_docs))