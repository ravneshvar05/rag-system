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
    temperature=0.3,  # Slightly higher for more natural responses
    max_tokens=2500,
    api_key=os.getenv("GROQ_API_KEY")
)

llm_8b = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0.3,
    max_tokens=856,
    api_key=os.getenv("GROQ_API_KEY")
)

# ==========================================
# 2. IMPROVED PROMPT
# ==========================================
template_text = """You are a helpful and knowledgeable AI assistant. Answer the user's question based on the context provided below.

**Instructions:**

1. **Strict Context Adherence:**
   - Answer ONLY using information explicitly present in the context below
   - Treat the context as your entire knowledge base - nothing exists outside it
   - If information is not in the context, you genuinely don't know it
   - Never use external knowledge, training data, or general information

2. **Topic Mismatch Handling:**
   - If the question is about Topic X but context is entirely about Topic Y (completely different subjects):
     * State clearly: "The provided documents are about [Y], not [X]. I cannot answer questions about [X] based on these documents."
     * Then STOP immediately - do not add any other information
   - Examples of true mismatches:
     * Question: A.C. → Context: Refrigerator manual
     * Question: Person A → Context: Only about Person B
     * Question: Python → Context: Java documentation

3. **When to Answer:**
   - If the context contains relevant information, provide a comprehensive answer
   - Include all relevant details, steps, lists, and explanations from the context
   - Use direct information and make reasonable inferences only from what's provided
   - If you have partial information, share it: "Based on the context, [available info], but there's no information about [missing info]"

4. **Smart Inference:**
   - Make logical connections between information in the context
   - Connect related pieces across different sections
   - Draw reasonable conclusions from the provided data
   - But never infer beyond what the context supports

5. **Formatting Excellence:**
   - Use **bold** for key terms, names, and important concepts
   - Use bullet points for lists and multiple items
   - Use numbered lists for steps or sequential information
   - Create tables when comparing multiple items
   - Keep formatting clean and scannable

6. **Precision Rules:**
   - Be specific with numbers, dates, and facts
   - Cite specific details rather than vague statements
   - If context gives examples, include them
   - Preserve technical terms and proper nouns exactly as they appear


**Context:**
{context}

**Question:** {question}

**Answer:**"""

prompt_template = ChatPromptTemplate.from_template(template_text)

def format_docs(docs):
    """Format documents with clear separation"""
    formatted = []
    for i, doc in enumerate(docs, 1):
        content = doc.page_content.strip()
        source = os.path.basename(doc.metadata.get("source", "Unknown"))
        page = doc.metadata.get("page", "N/A")
        formatted.append(f"[Chunk {i} from {source}, Page {page}]\n{content}")
    return "\n\n---\n\n".join(formatted)

# ==========================================
# 3. IMPROVED RETRIEVAL STRATEGY
# ==========================================
def get_relevant_docs(query: str, db):
    """
    Enhanced retrieval with multiple strategies
    """
    # Strategy 1: Get more candidates initially
    retriever = db.as_retriever(search_kwargs={"k": 12})
    all_docs = retriever.invoke(query)
    
    if not all_docs:
        return None, []
    
    # Strategy 2: Analyze source distribution
    source_counts = Counter(doc.metadata.get("source", "Unknown") for doc in all_docs)
    
    # Get top 2 sources (in case winner doesn't have full answer)
    top_sources = [source for source, _ in source_counts.most_common(2)]
    
    # Strategy 3: Keep best chunks from top sources (up to 5 chunks)
    final_docs = []
    for source in top_sources:
        source_docs = [doc for doc in all_docs if doc.metadata.get("source") == source]
        final_docs.extend(source_docs[:3])  # Top 3 from each source
    
    final_docs = final_docs[:6]  # Max 5 chunks total
    
    # Return primary source name and docs
    primary_source = os.path.basename(top_sources[0])
    return primary_source, final_docs

# ==========================================
# 4. ASK QUESTION (STREAMING GENERATOR)
# ==========================================
def ask_question(query: str):
    """
    GENERATOR: Yields the answer token-by-token with improved reliability
    """
    try:
        logger.info(f"❓ Streaming Question: {query}")
        
        db = get_vector_db()
        if not db:
            yield "⚠️ Database is empty! Please upload a document first."
            return

        # 1. Get relevant documents
        primary_source, final_docs = get_relevant_docs(query, db)
        
        if not final_docs:
            yield "I cannot find any relevant information in the documents."
            return

        logger.info(f"🎯 Primary source: {primary_source} ({len(final_docs)} chunks)")

        # 2. Prepare context with better formatting
        context_text = format_docs(final_docs)

        # 3. Extract unique sources for citation
        source_list = []
        for doc in final_docs:
            source_path = doc.metadata.get("source", "Unknown")
            source_name = os.path.basename(source_path)
            page = doc.metadata.get("page", "N/A")
            s_str = f"{source_name} (Page {page})"
            if s_str not in source_list:
                source_list.append(s_str)

        # 4. Define Chain Helper
        def get_chain(model):
            return prompt_template | model | StrOutputParser()

        # 5. STREAM THE ANSWER with fallback
        answer_generated = False
        try:
            chain = get_chain(llm_70b)
            for chunk in chain.stream({"context": context_text, "question": query}):
                yield chunk
                answer_generated = True
        except Exception as e:
            logger.warning(f"⚠️ 70B Error ({e}). Falling back to 8B.")
            try:
                chain = get_chain(llm_8b)
                for chunk in chain.stream({"context": context_text, "question": query}):
                    yield chunk
                    answer_generated = True
            except Exception as inner_e:
                yield f"Error generating answer: {inner_e}"
                return

        # 6. STREAM THE SOURCES (if answer was generated)
        if answer_generated and source_list:
            yield "\n\n---\n**📚 Sources:**\n"
            for source in source_list:
                yield f"* `{source}`\n"

    except Exception as e:
        logger.error(f"❌ Error in ask_question: {str(e)}")
        yield f"❌ Error: {str(e)}"

# ==========================================
# 5. ALTERNATIVE: NON-STREAMING VERSION
# ==========================================
def ask_question_sync(query: str) -> dict:
    """
    Non-streaming version that returns complete response
    Useful for debugging or API endpoints
    """
    try:
        db = get_vector_db()
        if not db:
            return {"answer": "⚠️ Database is empty!", "sources": []}

        primary_source, final_docs = get_relevant_docs(query, db)
        
        if not final_docs:
            return {"answer": "No relevant information found.", "sources": []}

        context_text = format_docs(final_docs)
        
        # Extract sources
        source_list = []
        for doc in final_docs:
            source_path = doc.metadata.get("source", "Unknown")
            source_name = os.path.basename(source_path)
            page = doc.metadata.get("page", "N/A")
            s_str = f"{source_name} (Page {page})"
            if s_str not in source_list:
                source_list.append(s_str)

        # Generate answer
        chain = prompt_template | llm_70b | StrOutputParser()
        answer = chain.invoke({"context": context_text, "question": query})
        
        return {
            "answer": answer,
            "sources": source_list,
            "primary_source": primary_source
        }
        
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        return {"answer": f"Error: {str(e)}", "sources": []}

# ==========================================
# 6. TEST BLOCK (For Terminal)
# ==========================================
if __name__ == "__main__":
    print("\n--- 🧪 STREAMING TEST ---")
    user_query = input("Type a question: ")
    print("\n🤖 AI: ", end="", flush=True)
    
    for token in ask_question(user_query):
        print(token, end="", flush=True)
    
    print("\n\n" + "="*40)