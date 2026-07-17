import os
from langchain_community.llms import Ollama
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter 
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_classic.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain

def get_vectorstore(pdf_files):
    """Processes uploaded PDF files and returns a FAISS vectorstore."""
    documents = []
    for pdf_file in pdf_files:
        loader = PyPDFLoader(pdf_file)
        documents.extend(loader.load())
        
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(documents)
    
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    return vectorstore

def get_conversation_chain(vectorstore):
    """Creates a RAG chain that supports conversational memory."""
    llm = Ollama(model="gemma:2b")
    
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    
    # Prompt to contextualize the question using chat history
    contextualize_q_system_prompt = (
        "Given a chat history and the latest user question "
        "which might reference context in the chat history, "
        "formulate a standalone question which can be understood "
        "without the chat history. Do NOT answer the question, "
        "just reformulate it if needed and otherwise return it as is."
    )
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )
    
    # Prompt to answer the question using the retrieved context
    qa_system_prompt = (
        "You are a helpful, intelligent assistant. Use the following pieces of retrieved context "
        "to answer the user's question. If the context is relevant, use it. "
        "If the context doesn't contain the answer or no context is provided, use your general knowledge to answer. "
        "Keep the answer concise and relevant.\n\n"
        "Context: {context}"
    )
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", qa_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    
    # Combine history aware retriever and QA chain
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
    
    return rag_chain

def get_general_conversation_chain():
    """Creates a basic conversational chain without RAG for general questions."""
    llm = Ollama(model="gemma:2b")
    
    system_prompt = "You are a helpful, intelligent assistant."
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    
    # A simple chain that takes input and chat_history, and returns 'answer'
    # to match the dict output format of the retrieval chain
    from langchain_core.runnables import RunnablePassthrough
    
    def format_output(response):
        return {"answer": response}
        
    chain = prompt | llm | format_output
    return chain
