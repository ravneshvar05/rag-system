import os
import sys
from fpdf import FPDF
from PIL import Image, ImageDraw
import io

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.ingest import load_file
from src.logger import logger

def create_normal_pdf(filename):
    """Creates a standard PDF with selectable text."""
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt="This is a normal selectable text PDF.", ln=1, align="C")
    pdf.output(filename)
    logger.info(f"📄 Created Normal PDF: {filename}")

def create_scanned_pdf(filename):
    """Creates a PDF that CONTAINS an image of text (simulating a scan)."""
    # 1. Create an Image with text
    img = Image.new('RGB', (500, 200), color='white')
    d = ImageDraw.Draw(img)
    d.text((10, 80), "This is a SCANNED image inside a PDF.", fill='black')
    
    # 2. Save image to memory
    img_path = "temp_scan_image.png"
    img.save(img_path)

    # 3. Create PDF and insert the image
    pdf = FPDF()
    pdf.add_page()
    pdf.image(img_path, x=10, y=10, w=180)
    pdf.output(filename)
    
    # Cleanup temp image
    os.remove(img_path)
    logger.info(f"📷 Created Scanned PDF: {filename}")

def run_tests():
    normal_pdf = "test_normal.pdf"
    scanned_pdf = "test_scanned.pdf"

    try:
        # --- TEST 1: Normal PDF ---
        create_normal_pdf(normal_pdf)
        print("\n🔍 Testing Normal PDF...")
        docs = load_file(normal_pdf)
        if docs and "normal selectable text" in docs[0].page_content:
            print("✅ Normal PDF Passed!")
        else:
            print(f"❌ Normal PDF Failed! Content: {docs}")

        # --- TEST 2: Scanned PDF (The Big One) ---
        create_scanned_pdf(scanned_pdf)
        print("\n🔍 Testing Scanned PDF (OCR via PyMuPDF)...")
        docs = load_file(scanned_pdf)
        
        if docs and "SCANNED image" in docs[0].page_content:
            print("✅ Scanned PDF Passed! (OCR worked)")
        else:
            print("❌ Scanned PDF Failed.")
            if docs:
                print(f"   Got text: '{docs[0].page_content}'")
            else:
                print("   Got NO text.")

    except Exception as e:
        print(f"❌ Error: {e}")
    
    finally:
        # Cleanup
        if os.path.exists(normal_pdf): os.remove(normal_pdf)
        if os.path.exists(scanned_pdf): os.remove(scanned_pdf)

if __name__ == "__main__":
    run_tests()