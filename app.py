import os
import tempfile
import streamlit as st
from dotenv import load_dotenv
from llm_utils import get_vectorstore, get_conversation_chain, get_general_conversation_chain
from langchain_core.messages import HumanMessage, AIMessage

load_dotenv() 

# Langsmith Tracking Setup
if os.getenv("LANGCHAIN_API_KEY"):
    os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
if os.getenv("LANGCHAIN_PROJECT"):
    os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT")
os.environ["LANGCHAIN_TRACING_V2"] = "true"

st.set_page_config(page_title="Chat with Documents", page_icon="📚", layout="wide")

st.title("📚 Conversational RAG with Gemma")

# Initialize session state for memory and vector store
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "conversation_chain" not in st.session_state:
    st.session_state.conversation_chain = get_general_conversation_chain()

with st.sidebar:
    st.header("Configuration & Data")
    st.info("Upload PDF documents to ground the AI's answers in your specific data.")
    
    uploaded_files = st.file_uploader(
        "Upload your PDFs", accept_multiple_files=True, type=["pdf"]
    )
    
    if st.button("Process Documents"):
        if uploaded_files:
            with st.spinner("Processing documents (Chunking & Embedding)..."):
                # Save uploaded files temporarily so PyPDFLoader can read them from disk
                temp_file_paths = []
                for uploaded_file in uploaded_files:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                        tmp_file.write(uploaded_file.getvalue())
                        temp_file_paths.append(tmp_file.name)
                
                # Build vectorstore and setup conversational chain
                st.session_state.vectorstore = get_vectorstore(temp_file_paths)
                st.session_state.conversation_chain = get_conversation_chain(st.session_state.vectorstore)
                
                # Clear chat history when new documents are uploaded so the AI doesn't get confused by old questions
                st.session_state.chat_history = []
                
                # Clean up temporary files
                for tmp_file_path in temp_file_paths:
                    os.remove(tmp_file_path)
                
                st.success("Documents successfully processed! You can now ask questions about them.")
        else:
            st.warning("Please upload at least one PDF file first.")
    
    st.divider()
    st.caption("Powered by LangChain, Ollama (Gemma), FAISS, and Streamlit.")

# Display Chat History
for message in st.session_state.chat_history:
    role = "user" if isinstance(message, HumanMessage) else "assistant"
    with st.chat_message(role):
        st.markdown(message.content)

# Chat Input
user_input = st.chat_input("Ask a question about your documents...")

if user_input:
    # 1. Display user message
    st.session_state.chat_history.append(HumanMessage(content=user_input))
    with st.chat_message("user"):
        st.markdown(user_input)
    
    # 2. Generate and display assistant response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            # Invoke the chain
            response = st.session_state.conversation_chain.invoke({
                "input": user_input,
                "chat_history": st.session_state.chat_history
            })
            
            answer = response["answer"]
            st.markdown(answer)
            
            # Append assistant message to chat history
            st.session_state.chat_history.append(AIMessage(content=answer))
