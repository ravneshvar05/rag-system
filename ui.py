import streamlit as st
import requests

# 🔧 CONFIG: URL of your FastAPI Backend
API_URL = "http://127.0.0.1:8000"

st.set_page_config(page_title="Multimodal RAG Agent", page_icon="🤖", layout="centered")

# --- HEADER ---
st.title("🤖 Multimodal RAG Agent")
st.caption("Powered by Llama-3.3-70B, Groq, and Hugging Face")

# --- SIDEBAR: Settings & Upload ---
with st.sidebar:
    st.header("📂 Document Manager")
    
    # 1. File Uploader
    uploaded_file = st.file_uploader("Upload PDF, TXT, or Image", type=["pdf", "txt", "png", "jpg", "jpeg"])
    
    if uploaded_file is not None:
        if st.button("🚀 Ingest Document"):
            with st.spinner("Processing... (OCR + Embedding)"):
                files = {"file": (uploaded_file.name, uploaded_file, uploaded_file.type)}
                try:
                    response = requests.post(f"{API_URL}/ingest/", files=files)
                    if response.status_code == 200:
                        data = response.json()
                        st.success(f"✅ Ingested! Created {data['chunks_created']} chunks.")
                    else:
                        st.error(f"❌ Error: {response.text}")
                except Exception as e:
                    st.error(f"❌ Connection Error: {e}")

    st.markdown("---")
    
    # 2. Clear Database Button
    st.header("⚙️ Settings")
    if st.button("🗑️ Clear Database"):
        try:
            response = requests.delete(f"{API_URL}/clear-db/")
            if response.status_code == 200:
                st.toast("✅ Database Cleared!", icon="🗑️")
            else:
                st.error("❌ Failed to clear DB")
        except Exception as e:
            st.error(f"❌ Error: {e}")

# --- MAIN CHAT INTERFACE ---

# Initialize Chat History
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! Upload a document and ask me anything about it."}
    ]

# Display Chat History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User Input
if prompt := st.chat_input("Ask a question about your documents..."):
    # 1. Add User Message to History
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. Get AI Response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                # Send to API (using data= for Form input)
                response = requests.post(f"{API_URL}/chat/", data={"query": prompt})
                
                if response.status_code == 200:
                    answer = response.json().get("answer", "No answer received.")
                else:
                    answer = f"❌ API Error: {response.text}"
                    
            except requests.exceptions.ConnectionError:
                answer = "❌ Error: Could not connect to API. Is it running?"
        
        st.markdown(answer)
    
    # 3. Add AI Message to History
    st.session_state.messages.append({"role": "assistant", "content": answer})