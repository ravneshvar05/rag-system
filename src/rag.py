import os
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from src.vector_store import get_vector_db
from src.logger import logger
from dotenv import load_dotenv

load_dotenv()

# Setup Groq with the LATEST model
try:
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",  # ✅ Updated to current model
        temperature=0.3,
        api_key=os.getenv("GROQ_API_KEY")
    )
    logger.info("✅ Groq LLM initialized")
except Exception as e:
    logger.error(f"❌ Groq Error: {e}")
    raise

# Prompt
prompt_template = ChatPromptTemplate.from_template("""
You are a helpful assistant. Use the context below to answer the question.
If you don't know, say "I don't have enough information".

Context:
{context}

Question:
{question}

Answer:
""")

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def ask_question(query):
    try:
        logger.info(f"❓ Question: {query}")
        
        # Load existing DB
        db = get_vector_db()
        if not db:
            return "⚠️ Database is empty!"

        # Search
        retriever = db.as_retriever(search_kwargs={"k": 3})

        # Chain
        rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt_template
            | llm
            | StrOutputParser()
        )

        answer = rag_chain.invoke(query)
        logger.info("✅ Answer generated successfully")
        return answer

    except Exception as e:
        logger.error(f"❌ Error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return f"Error: {str(e)}"