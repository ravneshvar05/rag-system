import shutil
import os
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from src.ingest import load_file
from src.vector_store import add_to_vector_db
from src.rag import ask_question

app = FastAPI(title="Multimodal RAG API", version="2.0")

# Input Schema for Chat
class QuestionRequest(BaseModel):
    query: str

@app.get("/")
def home():
    return {"message": "Multimodal RAG System is Online 🟢"}

@app.post("/ingest/")
async def ingest_document(file: UploadFile = File(...)):
    """
    Upload a file (PDF, TXT, or IMAGE). 
    It will be processed (OCR if needed) and saved to the Vector DB.
    """
    try:
        # 1. Save uploaded file to a temp folder
        temp_dir = "temp_data"
        os.makedirs(temp_dir, exist_ok=True)
        file_path = os.path.join(temp_dir, file.filename)
        
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        # 2. Process File (OCR + Text Split)
        # This calls your smart load_file function
        docs = load_file(file_path)
        
        if not docs:
            raise HTTPException(status_code=400, detail="Could not extract text. (File might be empty or unreadable)")
            
        # 3. Save to Vector DB
        add_to_vector_db(docs)
        
        return {
            "filename": file.filename, 
            "status": "Successfully Ingested", 
            "chunks_created": len(docs)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat/")
def chat_with_docs(request: QuestionRequest):
    """
    Ask a question to the AI about your ingested documents.
    """
    answer = ask_question(request.query)
    return {"question": request.query, "answer": answer}