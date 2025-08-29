# 🤖 Multi-Agent HR Assistant

A simple **Streamlit app** built with **LangChain + Groq API** that demonstrates a **multi-agent RAG (Retrieval-Augmented Generation) system**.  

Two specialized agents share the same vector store but handle different queries:
- **Salary Agent** → answers salary-related questions.  
- **Insurance Agent** → answers insurance-related questions.  
- **Coordinator** → decides which agent should respond.  

---

## 🚀 Features
- Built with **Streamlit** for an interactive chat UI.  
- Uses **LangChain** to manage retrieval and agents.  
- Embeds HR knowledge base (`salary.txt`, `insurance.txt`) into a **FAISS vector store**.  
- **Coordinator logic** routes queries to the right agent.  
- Maintains **chat history** until cleared.  

---

## 📂 Project Structure

---

## 🛠️ Setup Instructions

### Clone the repo
```bash
git clone https://github.com/your-username/ai-agent-hr-assistant.git
cd ai-agent-hr-assistant

### Create a virtual environment
python -m venv venv
venv\Scripts\activate      # On Windows

### install dependencies
pip install -r requirements.txt

### add your groq api key
# Windows (Powershell)
setx GROQ_API_KEY "your_api_key_here"

###Run the app
streamlit run agentapp.py


