import pytesseract
from PIL import Image

# Explicit path fix (Windows best practice)
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

img = Image.open("download.png")
text = pytesseract.image_to_string(img)

print(text)
