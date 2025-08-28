import os
import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA

# ------------------------
# 1. Setup LLM with Groq
# ------------------------
llm = ChatGroq(
    groq_api_key=os.environ.get("GROQ_API_KEY"),
    model="llama3-8b-8192"   # free Groq model, good for QA
)

# ------------------------
# 2. Load and Prepare Data
# ------------------------
def load_documents():
    loaders = [
        TextLoader("salary.txt"),
        TextLoader("insurance.txt")
    ]
    docs = []
    for loader in loaders:
        docs.extend(loader.load())
    return docs

docs = load_documents()

# Split into chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
split_docs = splitter.split_documents(docs)

# Create embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Create vectorstore (in-memory Chroma)
vectorstore = Chroma.from_documents(split_docs, embedding=embeddings)

# ------------------------
# 3. Define Specialized Agents
# ------------------------
salary_retriever = vectorstore.as_retriever(search_kwargs={"filter": {"source": "salary.txt"}})
insurance_retriever = vectorstore.as_retriever(search_kwargs={"filter": {"source": "insurance.txt"}})

salary_agent = RetrievalQA.from_chain_type(llm=llm, retriever=salary_retriever)
insurance_agent = RetrievalQA.from_chain_type(llm=llm, retriever=insurance_retriever)

# ------------------------
# 4. Coordinator Logic
# ------------------------
def coordinator(query):
    # simple keyword-based routing (can be improved)
    query_lower = query.lower()
    if "salary" in query_lower or "payslip" in query_lower or "deduction" in query_lower:
        return "Salary Agent", salary_agent.run(query)
    elif "insurance" in query_lower or "premium" in query_lower or "claim" in query_lower:
        return "Insurance Agent", insurance_agent.run(query)
    else:
        return "Coordinator", "Sorry, I can only answer salary or insurance related questions."

# ------------------------
# 5. Streamlit UI
# ------------------------
st.set_page_config(page_title="Multi-Agent HR Assistant", page_icon="🤖")

st.title("🤖 Multi-Agent HR Assistant")
st.write("Ask me about **salary** or **insurance** policies.")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

 # Add a Clear Chat button
if st.button("🗑️ Clear Chat"):
    st.session_state.chat_history = []
    st.rerun()

# Chat input
user_query = st.chat_input("Type your question...")

if user_query:
    agent, response = coordinator(user_query)
    st.session_state.chat_history.append(("You", user_query))
    st.session_state.chat_history.append((agent, response))

# Display chat
for sender, msg in st.session_state.chat_history:
    if sender == "You":
        st.chat_message("user").write(msg)
    else:
        st.chat_message("assistant").write(f"**{sender}**: {msg}")
