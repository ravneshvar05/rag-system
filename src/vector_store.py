import os
import time
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.embeddings import Embeddings
from huggingface_hub import InferenceClient
from src.logger import logger
from dotenv import load_dotenv

load_dotenv()

# Custom Embedding Class (Simplified)
class HuggingFaceAPIEmbeddings(Embeddings):
    def __init__(self, api_key: str, model_name: str):
        self.client = InferenceClient(token=api_key)
        self.model_name = model_name
    
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of documents with simple retry logic"""
        embeddings = []
        for text in texts:
            # Retry up to 3 times if the API is busy
            for attempt in range(3):
                try:
                    # Direct call to Hugging Face API
                    result = self.client.feature_extraction(
                        text=text,
                        model=self.model_name
                    )
                    
                    # Handle different return types (list or numpy array)
                    if hasattr(result, 'tolist'):
                        embedding = result.tolist()
                    else:
                        embedding = result
                        
                    embeddings.append(embedding)
                    break # Success! Stop retrying.
                    
                except Exception as e:
                    if attempt < 2:
                        time.sleep(2) # Wait 2 seconds before retry
                    else:
                        logger.error(f"❌ Failed to embed after 3 attempts: {e}")
                        return []
        return embeddings
    
    def embed_query(self, text: str) -> list[float]:
        """Embed a single query"""
        result = self.embed_documents([text])
        return result[0] if result else []

# Initialize Embeddings
try:
    embeddings = HuggingFaceAPIEmbeddings(
        api_key=os.getenv("HF_TOKEN"),
        model_name="BAAI/bge-m3"
    )
except Exception as e:
    logger.error(f"❌ Failed to initialize embeddings: {e}")
    raise

DB_PATH = "faiss_index"

def get_vector_db():
    if os.path.exists(DB_PATH):
        return FAISS.load_local(
            DB_PATH, 
            embeddings, 
            allow_dangerous_deserialization=True
        )
    return None

def add_to_vector_db(docs):
    try:
        if not docs:
            logger.warning("⚠️ No documents to add!")
            return None

        # Split Text
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100
        )
        split_docs = text_splitter.split_documents(docs)
        logger.info(f"✂️ Split into {len(split_docs)} chunks.")

        # Create & Save FAISS Index
        logger.info("📊 Creating FAISS index...")
        db = FAISS.from_documents(split_docs, embeddings)
        db.save_local(DB_PATH)
        logger.info(f"💾 Saved FAISS index to {DB_PATH}")
        return db

    except Exception as e:
        logger.error(f"❌ Error adding to Vector DB: {e}")
        return None