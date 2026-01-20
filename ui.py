import streamlit as st
import requests
import os

# 🔧 CONFIG: URL of your FastAPI Backend
API_URL = "http://127.0.0.1:8000"

st.set_page_config(page_title="Multimodal RAG Agent", page_icon="🤖", layout="centered")

# ✅ CSS: Hides the "Limit 200MB" text
st.markdown("""
<style>
    [data-testid="stFileUploader"] small {display: none;}
</style>
""", unsafe_allow_html=True)

# --- HEADER ---
st.title("🤖 Multimodal RAG Agent")
st.caption("Powered by Llama-3.3-70B, Groq, and Tesseract")

# --- SIDEBAR: Settings & Upload ---
with st.sidebar:
    st.header("📂 Document Manager")
    
    # 1. File Uploader
    uploaded_file = st.file_uploader("Upload PDF, TXT, or Image", type=["pdf", "txt", "png", "jpg", "jpeg"])
    
    if uploaded_file is not None:
        if st.button("🚀 Ingest Document"):
            with st.spinner("Processing... (Wait for it!)"):
                files = {"file": (uploaded_file.name, uploaded_file, uploaded_file.type)}
                try:
                    # Request waits here until backend finishes processing
                    response = requests.post(f"{API_URL}/ingest/", files=files)
                    
                    if response.status_code == 200:
                        data = response.json()
                        # ✅ Notification Pop-up
                        st.toast(data['message'], icon="✅")
                        st.success(data['message'])
                    else:
                        st.error(f"❌ Error: {response.text}")
                except Exception as e:
                    st.error(f"❌ Connection Error: {e}")

    st.markdown("---")

    # 2. Database Controls
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🗑️ Reset DB"):
            try:
                requests.delete(f"{API_URL}/clear-db/")
                st.toast("Database Cleared!", icon="🗑️")
            except Exception as e:
                st.error(f"Error: {e}")
    
    with col2:
        if st.button("🧹 New Chat"):
            st.session_state.messages = [{"role": "assistant", "content": "Chat cleared! Upload a new doc or ask away."}]
            st.rerun()

# --- MAIN CHAT INTERFACE ---

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hello! Upload a document and ask me anything."}]

# Display History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User Input
if prompt := st.chat_input("Ask a question about your documents..."):
    # 1. Add User Message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. Get AI Response (Streaming)
    with st.chat_message("assistant"):
        try:
            response = requests.post(
                f"{API_URL}/chat/", 
                data={"query": prompt}, 
                stream=True 
            )
            
            if response.status_code == 200:
                full_response = st.write_stream(response.iter_content(chunk_size=10, decode_unicode=True))
                st.session_state.messages.append({"role": "assistant", "content": full_response})
            else:
                st.error(f"❌ API Error: {response.text}")

        except requests.exceptions.ConnectionError:
            st.error("❌ Error: Could not connect to API. Is it running?")