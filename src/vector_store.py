import os
import time
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.embeddings import Embeddings
from huggingface_hub import InferenceClient
from src.logger import logger
from dotenv import load_dotenv

load_dotenv()

# Custom Embedding Class with BATCHING & RETRY Logic
class HuggingFaceAPIEmbeddings(Embeddings):
    def __init__(self, api_key: str, model_name: str):
        self.client = InferenceClient(token=api_key)
        self.model_name = model_name
    
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """
        Embeds documents in BATCHES to avoid hitting API rate limits.
        """
        all_embeddings = []
        batch_size = 32  # Send 32 chunks at a time (Standard practice)
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            
            # Retry logic for the whole batch
            for attempt in range(3):
                try:
                    # Send list of texts to HF API
                    # logger.info(f"    Processing batch {i} to {i+len(batch)}...") 
                    result = self.client.feature_extraction(
                        text=batch,
                        model=self.model_name
                    )
                    
                    # Hugging Face sometimes returns different shapes
                    if hasattr(result, 'tolist'):
                        result = result.tolist()
                    
                    all_embeddings.extend(result)
                    
                    # ✅ CRITICAL: Wait 0.5s to be polite to the free API
                    time.sleep(0.5) 
                    break 

                except Exception as e:
                    if attempt < 2:
                        wait_time = 2 * (attempt + 1)
                        logger.warning(f"⚠️ Batch failed. Retrying in {wait_time}s... Error: {e}")
                        time.sleep(wait_time)
                    else:
                        logger.error(f"❌ Batch failed permanently: {e}")
                        return []
                        
        return all_embeddings

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
            return 0  # Return 0 chunks if empty

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
        
        # ✅ FIX: Return the COUNT of chunks so the API knows the truth.
        return len(split_docs)

    except Exception as e:
        logger.error(f"❌ Error adding to Vector DB: {e}")
        return 0