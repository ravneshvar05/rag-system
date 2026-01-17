import sys
import os

# Allow importing from src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.rag import ask_question

# Ensure you have run test_vector_db.py first so the DB exists!
print("--- TESTING RAG (CHAT) ---")
answer = ask_question("Who developed Gemini 1.5 Flash?")
print("\n🤖 AI Answer:\n", answer)