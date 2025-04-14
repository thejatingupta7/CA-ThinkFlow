import streamlit as st
import os
import traceback
from langchain.chains import RetrievalQA
from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
import pickle
from typing import Any, Dict

def configure_page():
    """Configure Streamlit page settings"""
    st.set_page_config(
        page_title="CA-ThinkFlow",
        page_icon="💰",
        layout="wide"
    )

def get_response(query, chain, history):
    """
    Process query and manage conversation history with robust error handling
    """
    history.append({"user": query})
    
    try:
        result = chain({"query": query})
        response = (
            result.get('answer') or 
            result.get('result') or 
            "Sorry, I couldn't find a precise answer."
        )
        history[-1]["response"] = response
    except Exception as e:
        error_msg = f"An error occurred: {str(e)}"
        st.error(error_msg)
        history[-1]["response"] = "Sorry, I encountered an issue processing your query."
        st.error(traceback.format_exc())
    
    return history[-1]["response"], history

def safe_load_vector_store(storing_path: str, embedding_model: Any) -> Any:
    """
    Safely load FAISS vector store with error handling for version compatibility
    """
    try:
        # First try normal loading
        return FAISS.load_local(storing_path, embedding_model)
    except (KeyError, pickle.UnpicklingError) as e:
        st.warning("Standard load failed, attempting legacy compatibility load...")
        try:
            # Try loading with legacy support
            import faiss
            index = faiss.read_index(os.path.join(storing_path, "index.faiss"))
            with open(os.path.join(storing_path, "index.pkl"), "rb") as f:
                docstore = pickle.load(f)
            with open(os.path.join(storing_path, "index_to_id.pkl"), "rb") as f:
                index_to_id = pickle.load(f)
            
            return FAISS(embedding_model.embed_query, index, docstore, index_to_id)
        except Exception as e:
            st.error(f"Failed to load vector store: {str(e)}")
            st.error(traceback.format_exc())
            return None

def setup_llm_and_chain():
    """
    Setup Language Model and Retrieval QA Chain with robust error handling
    """
    try:
        # Initialize embeddings
        embedding_model = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}
        )
        
        # Load vector store with compatibility handling
        retriever = safe_load_vector_store(
            storing_path="vectorstore", 
            embedding_model=embedding_model
        )
        
        if retriever is None:
            raise ValueError("Failed to load vector store")
        
        # Custom prompt template
        template = """
        You are a financial expert specializing in detailed analysis of financial statements and performing a wide range of data-driven financial tasks. For every task or question presented, follow a strict step-by-step logical approach. When multiple-choice options are provided, your response must be strictly one of the given options—no explanations or justifications.

        Context: {context}
        Question: {question}
        Answer: 
        """
        
        prompt = PromptTemplate(
            template=template, 
            input_variables=["context", "question"]
        )
        
        # Initialize LLM
        llm = Ollama(
            model="gemma3:1b",  # Changed from "llama3.1" to standard "llama3"
            temperature=0.7
        )
        
        # Create Retrieval QA Chain
        chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=retriever.as_retriever(),
            chain_type="stuff",
            return_source_documents=True,
            chain_type_kwargs={
                'prompt': prompt,
                'verbose': True
            }
        )
        
        return llm, chain
    
    except Exception as e:
        st.error(f"Error setting up LLM and Chain: {e}")
        st.error(traceback.format_exc())
        return None, None

def add_custom_css():
    """Add custom CSS for enhanced UI"""
    st.markdown("""
    <style>
    .stButton button {
        width: 200px;
        height: 60px;
        font-size: 16px;
        border-radius: 10px;
        background-color: #2C3E50;
        color: white;
        transition: all 0.3s ease;
    }
    .stButton button:hover {
        background-color: #34495E;
        transform: scale(1.05);
    }
    </style>
    """, unsafe_allow_html=True)

def main():
    configure_page()
    add_custom_css()
    
    st.title("CA-ThinkFlow 🪙💰💱")
    st.subheader("Your AI Financial Consultant")
    
    llm, chain = setup_llm_and_chain()
    
    if not chain:
        st.error("Failed to initialize the AI assistant. Please check your configurations.")
        return
    
    if 'history' not in st.session_state:
        st.session_state['history'] = []
    
    # Predefined query buttons
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("Tax Benefits - Rental Income"):
            st.session_state['user_query'] = "What are the tax benefits for rental income in India?"
    
    with col2:
        if st.button("Property Tax Resolution"):
            st.session_state['user_query'] = "What are the tax implications when selling a property in India?"
    
    with col3:
        if st.button("Employment Tax"):
            st.session_state['user_query'] = "How does TDS work on salary income in India?"
    
    with col4:
        if st.button("ITR Filing Process"):
            st.session_state['user_query'] = "What is the process to file an Income Tax Return (ITR) in India?"
    
    user_query = st.text_input(
        "Enter your financial question:", 
        value=st.session_state.get('user_query', "")
    )
    
    if user_query:
        with st.spinner('Analyzing your query...'):
            response, st.session_state['history'] = get_response(
                user_query, 
                chain, 
                st.session_state['history']
            )
        st.success(response)
    
    with st.sidebar:
        st.header("Conversation History")
        for entry in reversed(st.session_state['history'][-5:]):
            st.markdown(f"*Q:* {entry['user']}")
            st.markdown(f"*A:* {entry['response']}")
            st.markdown("---")

if __name__ == "_main_":
    main()
