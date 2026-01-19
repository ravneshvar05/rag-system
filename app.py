import shutil
import os
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.responses import StreamingResponse
# ✅ FIX: Import 'load_file' (Matches your ingest.py)
from src.ingest import load_file 
from src.vector_store import add_to_vector_db
from src.rag import ask_question
from src.logger import logger

app = FastAPI(title="Multimodal RAG API", version="2.0")

@app.get("/")
def home():
    return {"message": "Multimodal RAG System is Online 🟢"}

# --- BACKGROUND TASK ---
def background_ingest(file_path: str, file_name: str):
    try:
        logger.info(f"⏳ Starting background ingestion for {file_name}...")
        
        # Use load_file here
        docs = load_file(file_path)
        
        if docs:
            count = add_to_vector_db(docs)
            logger.info(f"✅ Background ingestion finished. Added {count} chunks.")
        else:
            logger.warning(f"⚠️ No text extracted from {file_name}.")
    except Exception as e:
        logger.error(f"❌ Background Task Failed: {e}")
    finally:
        if os.path.exists(file_path):
            os.remove(file_path)

@app.post("/ingest/")
async def ingest_document(
    background_tasks: BackgroundTasks, 
    file: UploadFile = File(...)
):
    try:
        temp_dir = "temp_data"
        os.makedirs(temp_dir, exist_ok=True)
        file_path = os.path.join(temp_dir, file.filename)
        
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # 🚀 Send to Background
        background_tasks.add_task(background_ingest, file_path, file.filename)
        
        return {
            "filename": file.filename, 
            "status": "Accepted", 
            "message": "File received! Processing in background."
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat/")
def chat_with_docs(query: str = Form(...)):
    """
    Streaming Chat Endpoint.
    """
    # Get the generator function from rag.py
    response_generator = ask_question(query)
    
    # Return it as a stream
    return StreamingResponse(response_generator, media_type="text/plain")

@app.delete("/clear-db/")
def clear_database():
    db_path = "faiss_index"
    if os.path.exists(db_path):
        shutil.rmtree(db_path)
        return {"status": "✅ Database cleared successfully!"}
    return {"status": "⚠️ Database was already empty."}
# @app.post("/chat/")
# def chat_with_docs(request: QuestionRequest):
#     """
#     Ask a question to the AI about your ingested documents.
#     """
#     answer = ask_question(request.query)
#     return {"question": request.query, "answer": answer}