import sys
import os
from langchain_core.documents import Document

# Allow importing from src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.vector_store import add_to_vector_db

# Create a fake document
print("--- TESTING DATABASE ---")
test_docs = [
    Document(
        page_content="Gemini 1.5 Flash is a multimodal AI model developed by Google.",
        metadata={"source": "test_data"}
    )
]

# Run the function
db = add_to_vector_db(test_docs)

# Check if the folder was created
if db and os.path.exists("faiss_index"):
    print("✅ Success! Database index created at 'faiss_index/'")
else:
    print("❌ Failed to create database.")