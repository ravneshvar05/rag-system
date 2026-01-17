import os
import fitz  # PyMuPDF
import PIL.Image
import io
from langchain_core.documents import Document
from src.logger import logger
import google.generativeai as genai
from dotenv import load_dotenv

# 1. Load Secrets
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# --- HELPER 1: Check if PDF is a Scan ---
def is_pdf_scanned(pdf_path, page_limit=3):
    """
    Returns True if the PDF has very little selectable text (likely a scan).
    """
    try:
        doc = fitz.open(pdf_path)
        text_length = 0
        # Check first few pages only (to save time)
        pages_to_check = min(len(doc), page_limit)
        
        for i in range(pages_to_check):
            text_length += len(doc[i].get_text())
            
        doc.close()
        
        # If average characters per page is < 50, it's probably an image
        if (text_length / pages_to_check) < 50:
            return True
        return False
    except Exception as e:
        logger.error(f"Error checking scan status: {e}")
        return False

# --- HELPER 2: The "Eyes" (Gemini Vision) ---
def gemini_ocr_page(image_bytes):
    """
    Sends an image to Gemini to read the text inside it.
    """
    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        img = PIL.Image.open(io.BytesIO(image_bytes))
        
        prompt = "Transcribe this document exactly. If you see tables, format them as Markdown."
        response = model.generate_content([prompt, img])
        return response.text
    except Exception as e:
        logger.error(f"Gemini OCR Failed: {e}")
        return ""

# --- MAIN FUNCTION: Load the File ---
def load_file(file_path):
    """
    The Master Function. Give it a file path, get back a list of Documents.
    """
    file_ext = os.path.splitext(file_path)[1].lower()
    docs = []
    
    logger.info(f"📂 Starting Ingestion for: {file_path}")

    try:
        # CASE A: Standard Text Files
        if file_ext in ['.txt', '.md']:
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
            docs.append(Document(page_content=text, metadata={"source": file_path}))

        # CASE B: PDFs (The Smart Part)
        elif file_ext == '.pdf':
            # Check if it needs OCR
            if is_pdf_scanned(file_path):
                logger.warning(f"⚠️ Scan detected! Converting {file_path} to images for AI reading...")
                
                pdf_doc = fitz.open(file_path)
                full_text = ""
                
                for page_num in range(len(pdf_doc)):
                    # Convert PDF page to Image
                    page = pdf_doc.load_page(page_num)
                    pix = page.get_pixmap()
                    img_bytes = pix.tobytes("png")
                    
                    # AI Read
                    page_text = gemini_ocr_page(img_bytes)
                    full_text += f"\n\n--- Page {page_num+1} ---\n{page_text}"
                
                docs.append(Document(page_content=full_text, metadata={"source": file_path, "type": "scanned_pdf"}))
                
            else:
                # Standard PDF Read
                logger.info("✅ Native text detected. Reading directly.")
                pdf_doc = fitz.open(file_path)
                for page in pdf_doc:
                    text = page.get_text()
                    docs.append(Document(page_content=text, metadata={"source": file_path, "page": page.number}))

        # CASE C: Images (Direct OCR)
        elif file_ext in ['.png', '.jpg', '.jpeg']:
            logger.info("🖼️ Image detected. Sending to Gemini Vision...")
            with open(file_path, "rb") as f:
                img_bytes = f.read()
            text = gemini_ocr_page(img_bytes)
            docs.append(Document(page_content=text, metadata={"source": file_path, "type": "image"}))

        else:
            logger.error(f"❌ Unsupported file type: {file_ext}")
            return []

        logger.info(f"🎉 Processed {file_path} successfully!")
        return docs

    except Exception as e:
        logger.error(f"❌ Critical Error in load_file: {e}")
        return []