# import os
# from langchain_groq import ChatGroq
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_core.output_parsers import StrOutputParser
# from langchain_core.runnables import RunnablePassthrough
# from src.vector_store import get_vector_db
# from src.logger import logger
# from dotenv import load_dotenv

# load_dotenv()

# # Setup Groq with the LATEST model
# try:
#     llm = ChatGroq(
#         model="llama-3.3-70b-versatile",  # ✅ Updated to current model
#         temperature=0.3,
#         api_key=os.getenv("GROQ_API_KEY")
#     )
#     logger.info("✅ Groq LLM initialized")
# except Exception as e:
#     logger.error(f"❌ Groq Error: {e}")
#     raise

# # Prompt
# prompt_template = ChatPromptTemplate.from_template("""
# You are a helpful assistant. Use the context below to answer the question.
# If you don't know, say "I don't have enough information".

# Context:
# {context}

# Question:
# {question}

# Answer:
# """)

# def format_docs(docs):
#     return "\n\n".join(doc.page_content for doc in docs)

# def ask_question(query):
#     try:
#         logger.info(f"❓ Question: {query}")
        
#         # Load existing DB
#         db = get_vector_db()
#         if not db:
#             return "⚠️ Database is empty!"

#         # Search
#         retriever = db.as_retriever(search_kwargs={"k": 3})

#         # Chain
#         rag_chain = (
#             {"context": retriever | format_docs, "question": RunnablePassthrough()}
#             | prompt_template
#             | llm
#             | StrOutputParser()
#         )

#         answer = rag_chain.invoke(query)
#         logger.info("✅ Answer generated successfully")
#         return answer

#     except Exception as e:
#         logger.error(f"❌ Error: {e}")
#         import traceback
#         logger.error(traceback.format_exc())
#         return f"Error: {str(e)}"

# ----------------------------------------------------------------------------------------


# import os
# import time
# from langchain_groq import ChatGroq
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_core.output_parsers import StrOutputParser
# from langchain_core.runnables import RunnablePassthrough
# from src.vector_store import get_vector_db
# from src.logger import logger
# from dotenv import load_dotenv

# load_dotenv()

# # ==========================================
# # 1. SETUP MODELS (Smart vs. Fast)
# # ==========================================

# # Model A: The Genius (Llama 3.3 70B)
# # Use this for best quality. It has stricter rate limits.
# llm_70b = ChatGroq(
#     model="llama-3.3-70b-versatile",
#     temperature=0.3,
#     max_tokens=1024,
#     api_key=os.getenv("GROQ_API_KEY")
# )

# # Model B: The Backup (Llama 3.1 8B)
# # Use this if 70B is busy. It is faster and has higher limits.
# llm_8b = ChatGroq(
#     model="llama-3.1-8b-instant",
#     temperature=0.3,
#     max_tokens=856,
#     api_key=os.getenv("GROQ_API_KEY")
# )

# # ==========================================
# # 2. SETUP PROFESSIONAL PROMPT
# # ==========================================
# template_text = """
# You are a highly intelligent and professional AI assistant.
# Your task is to answer the user's question based strictly on the provided context.

# Instructions:
# 1. **Be Detailed:** If the context has lists, steps, or explanations, include them. Do not summarize too much.
# 2. **Format Cleanly:** Use **Bold** for key terms and Bullet Points for lists.
# 3. **No Hallucinations:** If the answer is NOT in the context, say exactly: "I cannot answer this based on the provided documents."
# 4. **Context Only:** Do not use outside knowledge.

# Context:
# {context}

# Question:
# {question}

# Answer:
# """

# prompt_template = ChatPromptTemplate.from_template(template_text)

# def format_docs(docs):
#     return "\n\n".join(doc.page_content for doc in docs)

# # ==========================================
# # 3. ASK QUESTION FUNCTION
# # ==========================================
# def ask_question(query: str):
#     """
#     Tries the 70B model first. If it fails (Rate Limit), falls back to 8B.
#     """
#     try:
#         logger.info(f"❓ Question: {query}")
        
#         # 1. Load DB
#         db = get_vector_db()
#         if not db:
#             return "⚠️ Database is empty! Please upload a document first."

#         # 2. Search (Fetch top 4 chunks for better context)
#         retriever = db.as_retriever(search_kwargs={"k": 4})

#         # 3. Define the Chain Runner
#         def run_chain(model_to_use):
#             chain = (
#                 {"context": retriever | format_docs, "question": RunnablePassthrough()}
#                 | prompt_template
#                 | model_to_use
#                 | StrOutputParser()
#             )
#             return chain.invoke(query)

#         # 4. Attempt 1: Try the Smart Model (70B)
#         try:
#             logger.info("🤖 Attempting with Llama-3.3-70B...")
#             return run_chain(llm_70b)
        
#         except Exception as e:
#             # Check for Rate Limit Error (429)
#             error_msg = str(e).lower()
#             if "429" in error_msg or "rate_limit" in error_msg:
#                 logger.warning("⚠️ 70B Rate Limit Hit! Falling back to 8B model...")
#                 # 5. Attempt 2: Fallback to Fast Model (8B)
#                 return run_chain(llm_8b)
#             else:
#                 # If it's a different error (e.g., Auth), raise it
#                 raise e

#     except Exception as e:
#         logger.error(f"❌ RAG Error: {e}")
#         return f"Sorry, an error occurred: {str(e)}"

# # ==========================================
# # 4. DIRECT TESTING BLOCK
# # ==========================================
# if __name__ == "__main__":
#     # This block only runs if you type `python src/rag.py`
#     print("\n--- 🧪 RAG SYSTEM TEST ---")
#     user_query = input("Type a question for your document: ")
#     print("\nThinking...")
    
#     result = ask_question(user_query)
    
#     print("\n" + "="*40)
#     print("🤖 AI ANSWER:")
#     print("="*40)
#     print(result)
#     print("="*40 + "\n")

# ----------------------------------------------------------------------------------------------

# sending source

import os
from collections import Counter
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from src.vector_store import get_vector_db
from src.logger import logger
from dotenv import load_dotenv

load_dotenv()

# ==========================================
# 1. SETUP MODELS
# ==========================================
llm_70b = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.3,
    max_tokens=1024,
    api_key=os.getenv("GROQ_API_KEY")
)

llm_8b = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0.3,
    max_tokens=856,
    api_key=os.getenv("GROQ_API_KEY")
)

# ==========================================
# 2. PROMPT
# ==========================================
template_text = """
You are a highly intelligent and professional AI assistant.
Your task is to answer the user's question based strictly on the provided context.

Instructions:
1. **Be Detailed:** If the context has lists, steps, or explanations, include them. Do not summarize too much.
2. **Format Cleanly:** Use **Bold** for key terms and Bullet Points for lists.
3. **No Hallucinations:** If the answer is NOT in the context, say exactly: "I cannot answer this based on the provided documents."
4. **Context Only:** Do not use outside knowledge.

Context:
{context}

Question:
{question}

Answer:
"""
prompt_template = ChatPromptTemplate.from_template(template_text)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# ==========================================
# 3. ASK QUESTION (STREAMING GENERATOR)
# ==========================================
def ask_question(query: str):
    """
    GENERATOR: Yields the answer token-by-token.
    """
    try:
        logger.info(f"❓ Streaming Question: {query}")
        
        db = get_vector_db()
        if not db:
            yield "⚠️ Database is empty! Please upload a document first."
            return

        # 1. Search (Fetch top 6 to find the Winner)
        retriever = db.as_retriever(search_kwargs={"k": 6})
        all_docs = retriever.invoke(query)
        
        if not all_docs:
            yield "I cannot find any relevant information in the documents."
            return

        # --- 🛡️ FILTER: WINNER TAKES ALL ---
        source_counts = Counter(doc.metadata.get("source", "Unknown") for doc in all_docs)
        best_source = source_counts.most_common(1)[0][0]
        
        # Keep only chunks from the winner (Top 4)
        final_docs = [doc for doc in all_docs if doc.metadata.get("source") == best_source][:4]
        
        logger.info(f"🎯 Context locked to: {os.path.basename(best_source)}")

        # 2. Extract Sources (Prepare string for later)
        source_list = []
        for doc in final_docs:
            source_path = doc.metadata.get("source", "Unknown")
            source_name = os.path.basename(source_path)
            page = doc.metadata.get("page", "N/A")
            s_str = f"{source_name} (Page {page})"
            if s_str not in source_list:
                source_list.append(s_str)

        context_text = format_docs(final_docs)

        # 3. Define Chain Helper
        def get_chain(model):
            return prompt_template | model | StrOutputParser()

        # 4. STREAM THE ANSWER
        # We use chain.stream() instead of chain.invoke()
        try:
            # Try 70B Model First
            chain = get_chain(llm_70b)
            for chunk in chain.stream({"context": context_text, "question": query}):
                yield chunk  # <--- Send this token immediately
        except Exception as e:
            # Fallback to 8B if Rate Limit hits
            logger.warning(f"⚠️ 70B Error ({e}). Falling back to 8B.")
            try:
                chain = get_chain(llm_8b)
                for chunk in chain.stream({"context": context_text, "question": query}):
                    yield chunk
            except Exception as inner_e:
                yield f"Error generating answer: {inner_e}"

        # 5. STREAM THE SOURCES (At the end)
        if source_list:
            yield "\n\n---\n**📚 Source (Winner):**\n"
            for source in source_list:
                yield f"* `{source}`\n"

    except Exception as e:
        yield f"❌ Error: {str(e)}"

# ==========================================
# 4. TEST BLOCK (For Terminal)
# ==========================================
if __name__ == "__main__":
    print("\n--- 🧪 STREAMING TEST ---")
    # Simulate a question
    user_query = input("Type a question: ")
    print("\n🤖 AI: ", end="", flush=True)
    
    # Iterate over the generator to simulate streaming
    for token in ask_question(user_query):
        print(token, end="", flush=True)
    
    print("\n\n" + "="*40)