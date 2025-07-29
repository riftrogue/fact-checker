import os
import requests
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from utils.embedding_store import embeddings
from utils.trusted_store import search_trusted_store

load_dotenv()

SYSTEM_PROMPT = """
You are a fact-checking AI. Your job is to verify a user’s claim using:

1. Uploaded document (temporary evidence)
2. Trusted facts (stored verified data)
3. If the above are insufficient, fallback to web results

Label the claim as:
- TRUE: if strong evidence supports it.
- FALSE: if evidence contradicts it.
- UNVERIFIABLE: if there's no relevant info in any sources.

Always explain your reasoning based on source(s) used.
"""

def get_rag_chain():
    index_path = os.path.join("vectorstore", "index.faiss")
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise EnvironmentError("Missing GROQ_API_KEY")

    llm = ChatGroq(
        model="llama3-70b-8192",
        temperature=0.2,
        api_key=api_key
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("user", "Claim: {question}\n\nContext:\n{context}")
    ])

    chain = prompt | llm | StrOutputParser()

    if os.path.exists(index_path):
        vectordb = FAISS.load_local(
            folder_path="vectorstore",
            embeddings=embeddings,
            index_name="index",
            allow_dangerous_deserialization=True
        )
        retriever = vectordb.as_retriever()
    else:
        retriever = None

    return chain, retriever


def retrieve_context(user_query: str, fallback_retriever=None):
    """
    Always retrieve support docs (vectorstore), then search trusted_vectorstore 
    using both the user query + doc context. Fallback to Tavily if trusted fails.
    """
    # Step 1: Get support context from uploaded documents
    support_context = ""
    if fallback_retriever:
        doc_results = fallback_retriever.invoke(user_query)
        support_context = "\n\n".join([doc.page_content for doc in doc_results]) if doc_results else ""

    # Step 2: Search trusted store using user query + support context
    combined_query = user_query + "\n\n" + support_context
    trusted_results = search_trusted_store(combined_query, k=3)

    if trusted_results:
        print("✅ Found trusted context — skipping Tavily.")
        trusted_context = "\n\n".join([doc.page_content for doc in trusted_results])
        final_context = trusted_context + "\n\n" + support_context
        return final_context, True  # Trusted hit

    # Step 3: Fallback to Tavily
    print("⚠️ No trusted match — using Tavily.")
    tavily_api_key = os.getenv("TAVILY_API_KEY")
    if tavily_api_key:
        try:
            tavily_response = requests.post(
                "https://api.tavily.com/search",
                headers={"Authorization": f"Bearer {tavily_api_key}"},
                json={"query": user_query, "search_depth": "advanced", "max_results": 5}
            )
            data = tavily_response.json()
            results = data.get("results", [])
            good_results = []

            for r in results:
                content = r.get("content", "").strip()
                title = r.get("title", "").strip()
                if len(content) > 100:  # filter out very short ones
                    formatted = f"[{title}]\n{content}"
                    good_results.append(formatted)

            if good_results:
                web_context = "\n\n".join(good_results[:3])  # avoid flooding context
                final_context = web_context + "\n\n" + support_context
                print("🌐 Using Tavily web results.")
                return final_context, False
            else:
                print("❌ Tavily returned but nothing useful.")
        except Exception as e:
            print("🚨 Tavily failed:", e)


    # Step 4: Final fallback — only support context
    print("🟡 Using support context only.")
    fallback = support_context if support_context else "No relevant context found."
    return fallback, False
