import shutil
import os
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from src.ingest import load_file
from src.vector_store import add_to_vector_db
from src.rag import ask_question

app = FastAPI(title="Multimodal RAG API", version="2.0")

@app.get("/")
def home():
    return {"message": "Multimodal RAG System is Online 🟢"}

@app.post("/ingest/")
async def ingest_document(file: UploadFile = File(...)):
    try:
        temp_dir = "temp_data"
        os.makedirs(temp_dir, exist_ok=True)
        file_path = os.path.join(temp_dir, file.filename)
        
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        docs = load_file(file_path)
        
        if not docs:
            raise HTTPException(status_code=400, detail="Could not extract text.")
            
        # ✅ FIX: Capture the real number of chunks returned by the function
        real_chunk_count = add_to_vector_db(docs)
        
        return {
            "filename": file.filename, 
            "status": "Successfully Ingested", 
            "chunks_created": real_chunk_count 
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat/")
def chat_with_docs(query: str = Form(...)):
    """
    Ask a question. 
    Uses Form data: Easy to type in Swagger, but supports long text safely.
    """
    answer = ask_question(query)
    return {"question": query, "answer": answer}

# ✅ NEW: The Safety Reset Switch
@app.delete("/clear-db/")
def clear_database():
    """
    ⚠️ Safe Reset: Deletes the vector database folder.
    This allows you to start fresh. The app handles the missing folder gracefully.
    """
    db_path = "faiss_index"
    try:
        if os.path.exists(db_path):
            shutil.rmtree(db_path)
            return {"status": "✅ Database cleared successfully!"}
        else:
            return {"status": "⚠️ Database was already empty."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
# @app.post("/chat/")
# def chat_with_docs(request: QuestionRequest):
#     """
#     Ask a question to the AI about your ingested documents.
#     """
#     answer = ask_question(request.query)
#     return {"question": request.query, "answer": answer}