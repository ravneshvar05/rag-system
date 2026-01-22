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

import zipfile

# ... (other imports are assumed to be present at top of file, we just add zipfile if needed, but here replacing the block so will include relevant logic)

@app.post("/ingest/")
async def ingest_documents(files: list[UploadFile] = File(...)):
    """
    Ingests multiple files, including ZIP archives (expanding them efficiently).
    """
    try:
        temp_dir = "temp_data"
        os.makedirs(temp_dir, exist_ok=True)
        
        all_docs = []
        file_summary = []
        
        for file in files:
            file_path = os.path.join(temp_dir, file.filename)
            
            # 1. Save uploaded file locally
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            
            # 2. Check if ZIP
            if file.filename.lower().endswith(".zip"):
                logger.info(f"📦 Detected ZIP archive: {file.filename}")
                extract_path = os.path.join(temp_dir, "extracted_" + file.filename)
                os.makedirs(extract_path, exist_ok=True)
                
                try:
                    with zipfile.ZipFile(file_path, 'r') as zip_ref:
                        zip_ref.extractall(extract_path)
                    
                    # Walk through extracted files
                    for root, _, extracted_files in os.walk(extract_path):
                        for extracted_file in extracted_files:
                            # Skip hidden files
                            if extracted_file.startswith("."):
                                continue
                                
                            full_path = os.path.join(root, extracted_file)
                            logger.info(f"⏳ Processing extracted file: {extracted_file}")
                            
                            zip_docs = load_file(full_path)
                            if zip_docs:
                                all_docs.extend(zip_docs)
                                file_summary.append(f"{extracted_file} (from zip)")
                            
                except zipfile.BadZipFile:
                    logger.error(f"❌ Invalid ZIP file: {file.filename}")
                finally:
                    # Clean up extracted folder
                    if os.path.exists(extract_path):
                        shutil.rmtree(extract_path)

            else:
                # 3. Process Regular File
                logger.info(f"⏳ Starting ingestion for {file.filename}...")
                docs = load_file(file_path)
                if docs:
                    all_docs.extend(docs)
                    file_summary.append(file.filename)
            
            # 4. Clean up original uploaded file
            if os.path.exists(file_path):
                os.remove(file_path)
        
        if not all_docs:
            return {
                "status": "warning", 
                "message": "⚠️ No text extracted from any of the uploaded files."
            }

        # 5. Add ALL docs to Vector DB at once
        chunk_count = add_to_vector_db(all_docs)
        logger.info(f"✅ Ingestion finished. Added {chunk_count} chunks from {len(file_summary)} files.")
        
        message = f"Successfully ingested {len(file_summary)} files ({chunk_count} chunks)."

        return {
            "filenames": file_summary, 
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