import os
import streamlit as st
from utils.embedding_store import process_docs_and_store
from utils.llm_interface import get_rag_chain, retrieve_context
from utils.history_manager import load_history, save_history
from langchain_core.messages import HumanMessage, AIMessage
import shutil


st.set_page_config(page_title="give me fact", layout="centered")
st.title("Fact Checker")

# Initialize chat history
history_file = "chat_history.json"
st.session_state["messages"] = load_history(history_file)

# Sidebar reset
if st.sidebar.button("🧹 Reset Session"):
    shutil.rmtree("vectorstore", ignore_errors=True)
    shutil.rmtree("raw_docs", ignore_errors=True)
    st.session_state["messages"] = []
    save_history(history_file, st.session_state["messages"])
    st.success("Session cleared! Upload new docs to start fresh.")

# Upload Section
st.subheader("📂 Upload Documents")
uploaded_files = st.file_uploader(
    "Upload your files (PDF, DOCX, TXT, PPTX)", 
    type=["pdf", "docx", "txt", "pptx"], 
    accept_multiple_files=True
)

if uploaded_files:
    os.makedirs("raw_docs", exist_ok=True)
    for file in uploaded_files:
        with open(os.path.join("raw_docs", file.name), "wb") as f:
            f.write(file.getbuffer())

    os.makedirs("vectorstore", exist_ok=True)
    process_docs_and_store("raw_docs", "vectorstore")

    st.success("✅ Files uploaded and processed successfully!")

# Chat Section
user_input = st.chat_input("Ask something about your documents...")

if user_input:
    st.session_state["messages"].append({"role": "user", "text": user_input})
    chain, retriever = get_rag_chain()

    history = [
        HumanMessage(content=m["text"]) if m["role"] == "user" else AIMessage(content=m["text"])
        for m in st.session_state["messages"]
    ]

    # 🔍 Get support context (from uploaded docs) + try trusted vector store first
    support_context = ""
    if retriever:
        doc_results = retriever.invoke(user_input)
        support_context = "\n\n".join([doc.page_content for doc in doc_results]) if doc_results else ""

    # 👀 First try with trusted vector store
    from utils.trusted_store import search_trusted_store, save_to_trusted_store
    combined_query = user_input + "\n\n" + support_context
    trusted_results = search_trusted_store(combined_query, k=3)

    if trusted_results:
        st.info("✅ Found trusted fact — skipping web search.")
        trusted_context = "\n\n".join([doc.page_content for doc in trusted_results])
        final_context = trusted_context + "\n\n" + support_context
        used_trusted = True

    else:
        st.warning("⚠️ No trusted fact found — using web search or doc evidence.")
        from utils.llm_interface import retrieve_context
        final_context, used_trusted = retrieve_context(user_input, fallback_retriever=retriever)

    # 🧠 Call RAG chain with final context
    response = chain.invoke({
        "context": final_context,
        "question": user_input
    })

    # 🧠 Save final output if it's a new verified fact (not already in trusted store)
    if not used_trusted and response.strip() and "please upload" not in response.lower():
        save_to_trusted_store(response)


    st.session_state["messages"].append({"role": "assistant", "text": response})
    save_history(history_file, st.session_state["messages"])
    st.rerun()

# Display chat history
for msg in st.session_state["messages"]:
    if msg["role"] == "user":
        st.chat_message("user").markdown(msg["text"])
    else:
        st.chat_message("assistant").markdown(msg["text"])
