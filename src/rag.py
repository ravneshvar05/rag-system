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
import time
from collections import Counter
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from src.vector_store import get_vector_db
from src.logger import logger
from dotenv import load_dotenv

load_dotenv()

from sentence_transformers import CrossEncoder

# ==========================================
# 2. SETUP MODELS
# ==========================================
try:
    # Load Cross-Encoder for Re-Ranking
    # "ms-marco-MiniLM-L-6-v2" is fast and effective for passing to LLM
    logger.info("⏳ Loading Cross-Encoder for Re-ranking...")
    cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    logger.info("✅ Cross-Encoder loaded.")
except Exception as e:
    logger.warning(f"⚠️ Could not load Cross-Encoder: {e}. Re-ranking will be skipped.")
    cross_encoder = None

llm_70b = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.3,
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
# 3. IMPROVED RETRIEVAL STRATEGY (RE-RANKING)
# ==========================================
def get_relevant_docs(query: str, db):
    """
    Enhanced retrieval with Re-Ranking (Cross-Encoder)
    """
    if not db:
        return None, []

    # Step 1: Retrieve Broad Set (Top 12)
    # Vector search is fast but approximate. We cast a wide net.
    retriever = db.as_retriever(search_kwargs={"k": 12})
    initial_docs = retriever.invoke(query)
    
    if not initial_docs:
        return None, []
    
    # Step 2: Re-Rank with Cross-Encoder (The "Judge")
    if cross_encoder:
        try:
            # Prepare pairs [Query, Doc Text]
            passages = [doc.page_content for doc in initial_docs]
            ranks = cross_encoder.rank(query, passages)
            
            # Sort by score (descending)
            # ranks is a list of {'corpus_id': int, 'score': float}
            sorted_ranks = sorted(ranks, key=lambda x: x['score'], reverse=True)
            
            # ✅ Threshold Filter: Reject chunks with very low relevance scores
            # Cross-Encoder scores range typically from -10 to +10.
            # > 0 is usually relevant. > -1 is lenient. > -10 is everything.
            MIN_SCORE = -2.0  
            filtered_ranks = [item for item in sorted_ranks if item['score'] > MIN_SCORE]

            if not filtered_ranks:
                logger.warning("⚠️ All documents filtered out by threshold! Using top 1 fallback.")
                filtered_ranks = [sorted_ranks[0]] # Fallback to at least one

            # Keep Top 5
            top_indices = [item['corpus_id'] for item in filtered_ranks[:5]]
            final_docs = [initial_docs[i] for i in top_indices]
            
            logger.info(f"✨ Re-ranked {len(initial_docs)}->{len(final_docs)} docs. Top score: {filtered_ranks[0]['score']:.4f}")
            
        except Exception as e:
            logger.warning(f"⚠️ Re-ranking failed ({e}). Falling back to vector order.")
            final_docs = initial_docs[:5]
    else:
        # Fallback if no cross-encoder
        final_docs = initial_docs[:5]
    
    # Get primary source name
    if final_docs:
        primary_source = os.path.basename(final_docs[0].metadata.get("source", "Unknown"))
    else:
        primary_source = "Unknown"

    return primary_source, final_docs

# ==========================================
# 4. ASK QUESTION (STREAMING GENERATOR)
# ==========================================
# ==========================================
# 4. QUERY DECOMPOSITION (NEW)
# ==========================================
# ==========================================
# 4. QUERY DECOMPOSITION (NEW)
# ==========================================
import re

def decompose_query(query: str) -> list[str]:
    """
    Uses a Hybrid Strategy (LLM + Heuristics + Regex) to split queries.
    """
    try:
        # Strategy 1: LLM Decomposition (The "Smart" Way)
        # Refined prompt to prevent over-splitting dependent clauses
        system = (
            "You are a query logic engine. "
            "Split the user's input into a list of distinct sub-questions ONLY if they are independent queries. "
            "DO NOT split if the user is asking for a list of items, examples, explanations, or dependent questions. "
            "Rules:"
            "1. 'What is the revenue and who is CEO?' -> ['What is the revenue?', 'Who is CEO?'] (Split)"
            "2. 'Which vector DB is best and why?' -> ['Which vector DB is best and why?'] (Keep Together - Dependent Clause)"
            "3. 'Show me scores for math, science, and physics' -> ['Show me scores for math, science, and physics'] (Keep Single)"
            "4. 'Explain X and also how it relates to Y' -> ['Explain X and also how it relates to Y'] (Keep Together - Relation)"
            "Return ONLY a JSON list of strings."
        )
        
        start_time = time.time()
        
        # Use simple non-streaming call
        response = llm_8b.invoke([
            ("system", system),
            ("user", query)
        ])
        
        content = response.content.strip()
        sub_queries = []
        
        # Clean up potential markdown code blocks
        content = content.replace("```json", "").replace("```", "").strip()
        
        import ast
        try:
            # Try to parse python/json list syntax
            parsed = ast.literal_eval(content)
            if isinstance(parsed, list):
                # Ensure flat list of strings
                for item in parsed:
                    if isinstance(item, list):
                        sub_queries.extend([str(sub) for sub in item])
                    else:
                        sub_queries.append(str(item))
            
        except:
            pass # Fallback to empty -> triggers heuristic/regex check

        # Strategy 2: Heuristic Validation (The "Safety Net")
        # Check if split result is valid. If it produced tiny fragments (e.g., "Why?"), reject it.
        is_valid_split = True
        if sub_queries and len(sub_queries) > 1:
            for q in sub_queries:
                # If a sub-question is < 4 words, it's suspiciously short (likely a broken split)
                if len(q.split()) < 4:
                    is_valid_split = False
                    logger.warning(f"⚠️ Rejecting split due to short fragment: '{q}'")
                    break
        
        if not is_valid_split or not sub_queries:
            # If LLM failed or produced bad splits, try Regex or keep original
            logger.info("⚠️ LLM split rejected or empty. Falling back to simple logic.")
            sub_queries = []

        # Strategy 3: Regex Fallback (The "Old Reliable")
        # If we still don't have sub-queries (or rejected LLM's), use Regex if '?' is present multiple times
        if not sub_queries:
             # Split by '?' followed by space/newline, but keep the '?'
            potential_splits = re.split(r'(?<=\?)\s+', query)
            potential_splits = [s.strip() for s in potential_splits if s.strip()]
            
            if len(potential_splits) > 1:
                sub_queries = potential_splits
                logger.info(f"🧩 Regex Decomposition used: {sub_queries}")
            else:
                sub_queries = [query]

        logger.info(f"🧩 Final Decomposition ({time.time() - start_time:.2f}s): {sub_queries}")
        return sub_queries

    except Exception as e:
        logger.warning(f"⚠️ Decomposition failed: {e}. Using original query.")
        return [query]

# ==========================================
# 5. ASK QUESTION (STREAMING GENERATOR)
# ==========================================
def ask_question(query: str):
    """
    GENERATOR: Handles multi-step reasoning by decomposing the query first.
    """
    try:
        logger.info(f"❓ Processing Query: {query}")
        
        db = get_vector_db()
        if not db:
            yield "⚠️ Database is empty! Please upload a document first."
            return

        # 1. Decompose Query
        sub_questions = decompose_query(query)
        
        if len(sub_questions) > 1:
            yield f"🔍 **Decomposed into {len(sub_questions)} questions:**\n"
            for q in sub_questions:
                yield f"* {q}\n"
            yield "\n---\n"
        
        # 2. Iterate through each sub-question
        for i, sub_q in enumerate(sub_questions):
            
            if len(sub_questions) > 1:
                yield f"### ❓ {sub_q}\n"

            # A. Get relevant documents for THIS sub-question
            primary_source, final_docs = get_relevant_docs(sub_q, db)
            
            if not final_docs:
                yield f"I cannot find any relevant information for: *{sub_q}*\n\n"
                continue

            # B. Prepare context
            context_text = format_docs(final_docs)

            # C. Extract sources
            source_list = []
            for doc in final_docs:
                source_path = doc.metadata.get("source", "Unknown")
                source_name = os.path.basename(source_path)
                page = doc.metadata.get("page", "N/A")
                s_str = f"{source_name} (Page {page})"
                if s_str not in source_list:
                    source_list.append(s_str)

            # D. Define Chain Helper
            def get_chain(model):
                return prompt_template | model | StrOutputParser()

            # E. STREAM THE ANSWER
            answer_generated = False
            try:
                # Use 8B for speed in multi-hop, or 70B for quality? 
                # User has free tier, let's stick to 70B for quality answer, 8B was for routing.
                chain = get_chain(llm_70b)
                for chunk in chain.stream({"context": context_text, "question": sub_q}):
                    yield chunk
                    answer_generated = True
            except Exception as e:
                logger.warning(f"⚠️ 70B Error ({e}). Falling back to 8B.")
                try:
                    chain = get_chain(llm_8b)
                    for chunk in chain.stream({"context": context_text, "question": sub_q}):
                        yield chunk
                        answer_generated = True
                except Exception as inner_e:
                    yield f"Error generating answer: {inner_e}"

            # F. STREAM SOURCES
            if answer_generated and source_list:
                yield "\n\n**Sources:** " + ", ".join([f"`{s}`" for s in source_list]) + "\n"
            
            # G. Separator for next question
            if i < len(sub_questions) - 1:
                yield "\n\n---\n\n"

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