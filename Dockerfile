# Use Python 3.10
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# 1. Install System Dependencies (Tesseract & Poppler)
# We update apt, install tools, and clean up to keep it small.
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    libtesseract-dev \
    poppler-utils \
    git \
    && rm -rf /var/lib/apt/lists/*

# 2. Copy Requirements & Install Python Libraries
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 3. Copy Your Application Code
COPY . .

# 4. Create & Permission Temp Folders
# This prevents "Permission Denied" errors on Hugging Face
RUN mkdir -p temp_data faiss_index && chmod -R 777 temp_data faiss_index

# 5. Expose the Secret Port
# Hugging Face ONLY listens to port 7860. We will put Streamlit here.
EXPOSE 7860

# 6. The "Dual-Process" Start Script
# ✅ FIXED: Added --server.enableCORS=false and --server.enableXsrfProtection=false
# This prevents the "AxiosError: 403" on file uploads.
RUN echo '#!/bin/bash\n\
uvicorn app:app --host 0.0.0.0 --port 8000 & \n\
streamlit run ui.py --server.port 7860 --server.address 0.0.0.0 --server.enableCORS=false --server.enableXsrfProtection=false\n\
' > start.sh && chmod +x start.sh

# 7. Start the App
CMD ["./start.sh"]