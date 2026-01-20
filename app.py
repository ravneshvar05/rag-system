import shutil
import os
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import StreamingResponse
# ✅ Import your modules
from src.ingest import load_file 
from src.vector_store import add_to_vector_db
from src.rag import ask_question
from src.logger import logger

app = FastAPI(title="Multimodal RAG API", version="2.5")

@app.get("/")
def home():
    return {"message": "Multimodal RAG System is Online 🟢"}

@app.post("/ingest/")
async def ingest_document(file: UploadFile = File(...)):
    """
    Ingests a file synchronously so the UI waits for it to finish.
    """
    try:
        temp_dir = "temp_data"
        os.makedirs(temp_dir, exist_ok=True)
        file_path = os.path.join(temp_dir, file.filename)
        
        # 1. Save the file locally
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        logger.info(f"⏳ Starting ingestion for {file.filename}...")

        # 2. Process the file (Synchronously)
        # We run this NOW so the UI waits for the result
        docs = load_file(file_path)
        
        chunk_count = 0
        if docs:
            chunk_count = add_to_vector_db(docs)
            logger.info(f"✅ Ingestion finished. Added {chunk_count} chunks.")
            message = f"Successfully ingested {file.filename} ({chunk_count} chunks)."
        else:
            logger.warning(f"⚠️ No text extracted from {file.filename}.")
            message = f"Warning: No text found in {file.filename}."

        # 3. Clean up
        if os.path.exists(file_path):
            os.remove(file_path)

        # 4. Return success message
        return {
            "filename": file.filename, 
            "status": "success", 
            "message": message
        }

    except Exception as e:
        logger.error(f"❌ Ingestion Failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat/")
def chat_with_docs(query: str = Form(...)):
    # Streaming Chat Endpoint
    response_generator = ask_question(query)
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