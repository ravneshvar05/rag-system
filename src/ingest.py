import os
import fitz  # PyMuPDF
import pdfplumber  # Better table extraction
import pytesseract
from PIL import Image
import io
from langchain_community.document_loaders import TextLoader
from langchain_core.documents import Document
from src.logger import logger
from dotenv import load_dotenv
import platform

load_dotenv()

# ✅ CONFIG: Tesseract Path
if platform.system() == "Windows":
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
else:
    # On Hugging Face (Linux), Tesseract is already in the PATH
    pytesseract.pytesseract.tesseract_cmd = "/usr/bin/tesseract"

def format_table_to_markdown(table):
    """Helper to convert a raw list of lists into a Markdown string."""
    if not table:
        return ""
    
    table_text = ""
    for row_idx, row in enumerate(table):
        clean_row = [str(cell).strip().replace("\n", " ") if cell else "" for cell in row]
        
        if row_idx == 0:
            table_text += "| " + " | ".join(clean_row) + " |\n"
            table_text += "|" + "|".join(["---" for _ in clean_row]) + "|\n"
        else:
            table_text += "| " + " | ".join(clean_row) + " |\n"
            
    return table_text

def extract_tables_as_text(page):
    """FALLBACK: PyMuPDF table extraction."""
    try:
        tables = page.find_tables()
        if not tables:
            return ""
        
        table_text = "\n\n=== TABLES DETECTED (PyMuPDF) ===\n"
        for i, table in enumerate(tables):
            table_text += f"\n--- Table {i+1} ---\n"
            table_data = table.extract()
            for row in table_data:
                clean_row = [str(cell) if cell is not None else "" for cell in row]
                row_text = " | ".join(clean_row)
                table_text += row_text + "\n"
            table_text += "\n"
        return table_text
    except Exception as e:
        logger.warning(f"⚠️ PyMuPDF table extraction failed: {e}")
        return ""

def load_file(file_path):
    """
    SPEED OPTIMIZED LOADER:
    1. Opens PDF files ONCE (Massive speedup).
    2. Uses PyMuPDF as a 'Gatekeeper' to only run slow pdfplumber on pages with actual tables.
    """
    try:
        ext = os.path.splitext(file_path)[1].lower()
        logger.info(f"📂 Loading file type: {ext}")

        if ext == ".txt":
            loader = TextLoader(file_path, encoding="utf-8")
            return loader.load()

        elif ext == ".pdf":
            documents = []
            
            # ✅ OPTIMIZATION 1: Open BOTH libraries ONCE at the start
            # This prevents re-parsing the file 100 times for 100 pages.
            with fitz.open(file_path) as doc, pdfplumber.open(file_path) as plumber_pdf:
                
                total_pages = len(doc)
                logger.info(f"📄 Processing {total_pages} pages...")
                
                for page_num, page in enumerate(doc):
                    
                    # A. Extract Regular Text (Instant)
                    text = page.get_text()
                    text_length = len(text.strip())
                    
                    # B. Intelligent Table Extraction
                    table_text = ""
                    
                    # ✅ OPTIMIZATION 2: The Gatekeeper
                    # PyMuPDF is C++ fast. We ask it: "Are there tables here?"
                    # If NO, we skip the slow pdfplumber entirely.
                    possible_tables = page.find_tables()
                    
                    if possible_tables:
                        # Found a potential table! Now use the slow but accurate tool.
                        try:
                            if page_num < len(plumber_pdf.pages):
                                plumber_page = plumber_pdf.pages[page_num]
                                tables = plumber_page.extract_tables()
                                
                                if tables:
                                    table_text += "\n\n=== TABLES DETECTED ===\n"
                                    for i, table in enumerate(tables):
                                        table_text += f"\n--- Table {i+1} ---\n"
                                        table_text += format_table_to_markdown(table)
                                        table_text += "\n"
                                    logger.info(f"📊 Extracted {len(tables)} table(s) on Page {page_num+1}")
                        except Exception as e:
                            logger.warning(f"⚠️ pdfplumber failed on page {page_num}: {e}")
                            # Fallback to the PyMuPDF tables we already found
                            table_text = extract_tables_as_text(page)
                    
                    # C. OCR Check (Only if text is missing)
                    if text_length < 50:
                        logger.info(f"🔍 Page {page_num+1} appears scanned. Running OCR...")
                        
                        # ✅ DPI 200 is 2x faster than 300 and good enough
                        pix = page.get_pixmap(dpi=200)
                        img_data = pix.tobytes("png")
                        img = Image.open(io.BytesIO(img_data))
                        
                        ocr_text = pytesseract.image_to_string(img, config='--psm 6')
                        
                        if len(ocr_text.strip()) > text_length:
                            text = ocr_text
                            logger.info(f"✅ OCR completed for page {page_num+1}")
                    
                    # D. Combine
                    combined_text = text + table_text
                    
                    if combined_text.strip():
                        documents.append(Document(
                            page_content=combined_text, 
                            metadata={"source": file_path, "page": page_num+1}
                        ))
            
            return documents

        elif ext in [".png", ".jpg", ".jpeg"]:
            try:
                image = Image.open(file_path)
                logger.info("🔍 Performing OCR on image...")
                text = pytesseract.image_to_string(image, config='--psm 6')
                if not text.strip(): return []
                return [Document(page_content=text, metadata={"source": file_path, "page": 1})]
            except Exception as e:
                logger.error(f"❌ OCR Error: {e}")
                return []
        else:
            logger.warning(f"⚠️ Unsupported file type: {ext}")
            return []

    except Exception as e:
        logger.error(f"❌ Error loading file: {e}")
        return []