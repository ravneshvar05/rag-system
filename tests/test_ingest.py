import sys
import os

# Allow importing from src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.ingest import load_file

# Run the test
print("--- TESTING INGESTION ---")
docs = load_file("test.txt")

if docs:
    print(f"✅ Success! Content: {docs[0].page_content}")
else:
    print("❌ Failed.")