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
                # Send to API
                response = requests.post(f"{API_URL}/chat/", data={"query": prompt})
                
                if response.status_code == 200:
                    data = response.json()
                    
                    # ✨ NEW: Robust handling for Text + Sources
                    # We check if 'answer' is a dict (new format) or string (old format/error)
                    if isinstance(data.get("answer"), dict):
                        answer_text = data["answer"]["answer"]
                        sources = data["answer"]["sources"]
                    else:
                        # Fallback if the API structure is simple
                        answer_text = data.get("answer", "No answer received.")
                        sources = data.get("sources", [])

                    # ✨ Format the final display string with Markdown
                    final_display = answer_text
                    if sources:
                        final_display += "\n\n---\n**📚 Sources:**\n"
                        for src in sources:
                            final_display += f"\n* `{src}`"

                else:
                    final_display = f"❌ API Error: {response.text}"
                    
            except requests.exceptions.ConnectionError:
                final_display = "❌ Error: Could not connect to API. Is it running?"
        
        # 3. Render and Save to History
        st.markdown(final_display)
        st.session_state.messages.append({"role": "assistant", "content": final_display})