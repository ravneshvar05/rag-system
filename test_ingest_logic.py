import os
import shutil
from src.ingest import process_document
from src.logger import logger
from dotenv import load_dotenv

# Load env to start fresh
load_dotenv()

# 📝 CONFIG: Put the name of a REAL file here (PDF or Image)
TEST_FILE = "test_document.pdf" 

def create_dummy_file():
    """Creates a dummy PDF if one doesn't exist (renamed text file for testing)"""
    if not os.path.exists(TEST_FILE):
        with open(TEST_FILE, "w") as f:
            f.write("This is a dummy PDF content for testing fallback logic.")
        print(f"⚠️ Created dummy file '{TEST_FILE}' for testing.")

def test_smart_mode():
    print("\n" + "="*50)
    print("🧪 TEST 1: Smart Mode (LlamaParse)")
    print("="*50)
    
    # Ensure Key is present
    if not os.getenv("LLAMA_CLOUD_API_KEY"):
        print("❌ SKIPPING: No API Key found in .env")
        return

    try:
        docs = process_document(TEST_FILE)
        print(f"\n✅ SUCCESS: Extracted {len(docs)} pages/chunks.")
        print("👀 Check logs above: Should say 'Attempting LlamaParse' or 'Successfully extracted'")
    except Exception as e:
        print(f"❌ FAILED: {e}")

def test_fallback_mode():
    print("\n" + "="*50)
    print("🧪 TEST 2: Fallback Mode (Local Tesseract)")
    print("="*50)
    
    # 1. Temporarily Hide the Key
    original_key = os.environ.get("LLAMA_CLOUD_API_KEY")
    if original_key:
        del os.environ["LLAMA_CLOUD_API_KEY"]
        print("🕵️  Simulating Internet/API Failure (Key removed from memory)...")
    
    try:
        # 2. Run Processing
        docs = process_document(TEST_FILE)
        print(f"\n✅ SUCCESS: Extracted {len(docs)} pages/chunks.")
        print("👀 Check logs above: Should say 'LlamaParse failed... Falling back to Local'")
        
    except Exception as e:
        print(f"❌ FAILED: {e}")
        
    finally:
        # 3. Restore Key (So we don't break anything for real)
        if original_key:
            os.environ["LLAMA_CLOUD_API_KEY"] = original_key
            print("🔧 API Key restored.")

if __name__ == "__main__":
    # Ensure we have a file to test with
    if not os.path.exists(TEST_FILE):
        print(f"❌ Please place a file named '{TEST_FILE}' in this folder to test!")
        # create_dummy_file() # Uncomment to auto-create a fake file
    else:
        test_smart_mode()
        test_fallback_mode()