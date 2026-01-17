import os
import fitz  # This is PyMuPDF
import pytesseract
from PIL import Image
import io
from langchain_community.document_loaders import TextLoader
from langchain_core.documents import Document
from src.logger import logger

# ✅ CONFIG: Tesseract Path
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

def load_file(file_path):
    """
    Robust Loader: Handles Text, Native PDFs, Scanned PDFs (OCR), and Images.
    """
    try:
        ext = os.path.splitext(file_path)[1].lower()
        logger.info(f"📂 Loading file type: {ext}")

        # 1. Handle Text Files
        if ext == ".txt":
            loader = TextLoader(file_path, encoding="utf-8")
            return loader.load()

        # 2. Handle PDF Files (Hybrid Mode: Text + OCR)
        elif ext == ".pdf":
            documents = []
            
            # Open PDF with PyMuPDF
            with fitz.open(file_path) as doc:
                for page_num, page in enumerate(doc):
                    
                    # A. Try extracting text directly (Fastest)
                    text = page.get_text()
                    
                    # B. If little/no text, it's likely a SCANNED page -> Use OCR
                    if len(text.strip()) < 10:  # Threshold: if less than 10 chars, assume scan
                        logger.info(f"🔍 Page {page_num+1} seems scanned. Running OCR...")
                        
                        # Render page as an image (300 DPI for good OCR)
                        pix = page.get_pixmap(dpi=300)
                        
                        # Convert to PIL Image
                        img_data = pix.tobytes("png")
                        img = Image.open(io.BytesIO(img_data))
                        
                        # Run Tesseract
                        text = pytesseract.image_to_string(img)
                        logger.info(f"✅ OCR finished for Page {page_num+1}")

                    if text.strip():
                        documents.append(Document(page_content=text, metadata={"source": file_path, "page": page_num+1}))
            
            return documents

        # 3. Handle Images
        elif ext in [".png", ".jpg", ".jpeg"]:
            try:
                image = Image.open(file_path)
                logger.info("🔍 Performing OCR on image...")
                text = pytesseract.image_to_string(image)
                
                if not text.strip():
                    return []
                
                return [Document(page_content=text, metadata={"source": file_path})]
            except Exception as e:
                logger.error(f"❌ OCR Error: {e}")
                return []

        else:
            logger.warning(f"⚠️ Unsupported file type: {ext}")
            return []

    except Exception as e:
        logger.error(f"❌ Error loading file: {e}")
        return []