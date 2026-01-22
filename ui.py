import streamlit as st
import requests
import os

# 🔧 CONFIG: URL of your FastAPI Backend
API_URL = "http://127.0.0.1:8000"

st.set_page_config(page_title="Multimodal RAG Agent", page_icon="🤖", layout="wide")

# ==========================================
# 🎨 CUSTOM CSS & THEME
# ==========================================
st.markdown("""
<style>
    /* Global Font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* ✨ Dark Mode Tweaks */
    .stApp {
        background-color: #0E1117;
    }

    /* 🟣 Gradient Title */
    .gradient-text {
        background: linear-gradient(45deg, #FF4B4B, #FF914D);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3em;
        font-weight: 800;
        margin-bottom: 0px;
    }
    
    .subtitle {
        color: #9CA3AF;
        font-size: 1.1em;
        margin-bottom: 20px;
    }

    /* 🗨️ Chat Bubbles */
    .stChatMessage {
        background-color: #1F2937;
        border-radius: 15px;
        padding: 10px;
        margin-bottom: 10px;
        border: 1px solid #374151;
    }
    
    /* User Bubble Accent */
    .stChatMessage[data-testid="stChatMessage"]:nth-child(odd) {
        background-color: #161b22; 
        border: 1px solid #30363d;
    }

    /* 🌫️ Sidebar Glassmorphism */
    section[data-testid="stSidebar"] {
        background-color: #161b22;
        border-right: 1px solid #30363d;
    }
    
    /* 🔘 Buttons */
    .stButton>button {
        border-radius: 8px;
        font-weight: 600;
        border: none;
        transition: all 0.2s;
    }
    .stButton>button:hover {
        opacity: 0.9;
        transform: scale(1.02);
    }
    
    /* 📄 File Uploader */
    [data-testid="stFileUploader"] {
        background-color: #1f2937;
        border: 1px dashed #4b5563;
        border-radius: 10px;
        padding: 20px;
    }
    [data-testid="stFileUploader"] small {display: none;} /* Hide limit text */

</style>
""", unsafe_allow_html=True)

# ==========================================
# 🌟 APP HEADER (HERO SECTION)
# ==========================================
col1, col2 = st.columns([3, 1])

with col1:
    st.markdown('<div class="gradient-text">Multimodal RAG Agent</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Powered by <b>Llama-3.3-70B</b> & <b>Vector Search</b>. Upload documents and ask complex questions.</div>', unsafe_allow_html=True)

with col2:
    # Status Indicator
    try:
        if requests.get(API_URL).status_code == 200:
            st.success("🟢 System Online")
        else:
            st.error("🔴 Backend Offline")
    except:
        st.error("🔴 Backend Offline")

st.markdown("---")

# ==========================================
# 📂 SIDEBAR: CONTROLS
# ==========================================
with st.sidebar:
    st.image("https://img.icons8.com/3d-fluency/94/robot-3.png", width=80)
    st.title("Control Panel")
    
    st.markdown("### 📤 Upload Documents")
    uploaded_files = st.file_uploader(
        "Supported: PDF, DOCX, TXT, CSV, MD, Images, ZIP", 
        type=["pdf", "txt", "png", "jpg", "jpeg", "md", "docx", "csv", "zip"],
        accept_multiple_files=True
    )
    
    if uploaded_files:
        if st.button("🚀 Ingest Files", type="primary", use_container_width=True):
            with st.spinner(f"Processing {len(uploaded_files)} files..."):
                files_payload = [
                    ("files", (file.name, file, file.type)) for file in uploaded_files
                ]
                try:
                    response = requests.post(f"{API_URL}/ingest/", files=files_payload)
                    if response.status_code == 200:
                        data = response.json()
                        st.toast(data['message'], icon="✅")
                        # st.balloons()
                        st.success("Files ingested!")
                    else:
                        st.error(f"❌ Error: {response.text}")
                except Exception as e:
                    st.error(f"❌ Connection Error: {e}")
    
    st.markdown("---")
    st.markdown("### ⚙️ Settings")
    
    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("🗑️ Reset DB", use_container_width=True):
            try:
                requests.delete(f"{API_URL}/clear-db/")
                st.toast("Memory Cleared!", icon="🗑️")
            except:
                st.error("Failed to clear DB")
    
    with col_b:
        if st.button("🧹 New Chat", use_container_width=True):
            st.session_state.messages = [{"role": "assistant", "content": "👋 **Hello!** I'm ready. Upload a document or ask me a question."}]
            st.rerun()

    st.markdown("---")
    st.caption("v2.5 | Multi-Question Enabled")

# ==========================================
# 💬 MAIN CHAT INTERFACE
# ==========================================

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "👋 **Hello!** Upload a document and ask me anything."}]

# Render Chat History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat Input
if prompt := st.chat_input("Ask a question (e.g., 'Summarize X AND tell me Y')..."):
    
    # 1. Append User Message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. Generate AI Response
    with st.chat_message("assistant"):
        try:
            stream_response = requests.post(
                f"{API_URL}/chat/", 
                data={"query": prompt}, 
                stream=True 
            )
            
            if stream_response.status_code == 200:
                # Stream content
                response_text = st.write_stream(stream_response.iter_content(chunk_size=10, decode_unicode=True))
                
                # Append to history
                st.session_state.messages.append({"role": "assistant", "content": response_text})
            else:
                st.error(f"❌ API Error: {stream_response.text}")
        
        except requests.exceptions.ConnectionError:
            st.error("❌ Link Error: Is the backend running?")