import os
import streamlit as st

from langchain_groq import ChatGroq
from langchain.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA

# Try Streamlit secrets (for Cloud), otherwise fallback to local env variable
groq_api_key = None
try:
    groq_api_key = st.secrets["GROQ_API_KEY"]
except Exception:
    groq_api_key = os.environ.get("GROQ_API_KEY")

if not groq_api_key:
    st.error("🚨 No GROQ_API_KEY found! Please set it in Streamlit secrets or as an environment variable.")
    st.stop()

llm = ChatGroq(
    groq_api_key=groq_api_key,
    model="llama3-8b-8192"
)

# ------------------------
#  Load and Prepare Data
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

# Use a small embedding model
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-MiniLM-L3-v2")

# ------------------------
#  Create vectorstore 
# ------------------------
vectorstore = FAISS.from_documents(split_docs, embeddings)

# ------------------------
#  Define Specialized Agents
# ------------------------
salary_retriever = vectorstore.as_retriever(search_kwargs={"filter": {"source": "salary.txt"}})
insurance_retriever = vectorstore.as_retriever(search_kwargs={"filter": {"source": "insurance.txt"}})

salary_agent = RetrievalQA.from_chain_type(llm=llm, retriever=salary_retriever)
insurance_agent = RetrievalQA.from_chain_type(llm=llm, retriever=insurance_retriever)

# ------------------------
#  Coordinator Logic
# ------------------------
def coordinator(query):
    query_lower = query.lower()
    if "salary" in query_lower or "payslip" in query_lower or "deduction" in query_lower:
        return "Salary Agent", salary_agent.run(query)
    elif "insurance" in query_lower or "premium" in query_lower or "claim" in query_lower:
        return "Insurance Agent", insurance_agent.run(query)
    else:
        return "Coordinator", "Sorry, I can only answer salary or insurance related questions."

# ------------------------
#  Streamlit UI
# ------------------------
st.set_page_config(page_title="Multi-Agent HR Assistant", page_icon="🤖")

st.title("🤖 Multi-Agent HR Assistant")
st.write("Ask me about **salary** or **insurance** policies.")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Clear Chat button
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

