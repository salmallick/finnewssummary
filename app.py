import os
import streamlit as st
from typing import List, Optional
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import WebBaseLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
from dotenv import load_dotenv

def initialize_app():
    """Initialize the Streamlit app and load environment variables."""
    load_dotenv()
    st.set_page_config(page_title="News Research Tool", page_icon="📈")
    st.title("📈 News Research Tool 📈")
    
def load_urls() -> List[str]:
    """Collect URLs from the user via sidebar."""
    st.sidebar.title("📰 News Article URLs")
    urls = []
    for i in range(3):
        url = st.sidebar.text_input(f"URL {i + 1}", key=f"url_{i}")
        if url:  # Only append non-empty URLs
            urls.append(url)
    return urls

def process_urls(urls: List[str], status_placeholder) -> Optional[FAISS]:
    """Process URLs and create FAISS vector store."""
    if not urls:
        status_placeholder.error("Please provide at least one URL.")
        return None
    
    try:
        # Load data from URLs
        status_placeholder.info("Loading articles...")
        loader = WebBaseLoader(urls)
        data = loader.load()
        
        if not data:
            status_placeholder.error("No valid data found. Please check your URLs.")
            return None
            
        # Split text into chunks
        status_placeholder.info("Processing text...")
        text_splitter = RecursiveCharacterTextSplitter(
            separators=['\n\n', '\n', '.', ','],
            chunk_size=1000,
            chunk_overlap=200
        )
        docs = text_splitter.split_documents(data)
        
        # Create embeddings and vector store
        status_placeholder.info("Creating embeddings...")
        embeddings = OpenAIEmbeddings()
        vector_store = FAISS.from_documents(docs, embeddings)
        
        # Save vector store
        vector_store.save_local("faiss_store")
        status_placeholder.success("Processing complete! You can now ask questions about the articles.")
        
        return vector_store
        
    except Exception as e:
        status_placeholder.error(f"An error occurred: {str(e)}")
        return None

def setup_qa_chain(vector_store: FAISS):
    """Set up an optimized question-answering chain."""
    llm = ChatOpenAI(
        temperature=0.2,         # Lower temperature for more precise answers
        model_name="gpt-4",     # Upgraded to GPT-4
        max_tokens=2000         # Increased token limit for detailed answers
    )
    
    # Create custom prompt template
    template = """You are an expert financial analyst analyzing recent news articles. Provide a detailed step-by-step analysis:

    1. Identify the key facts and information from the articles
    2. Connect different pieces of information to form a comprehensive answer
    3. Include specific quotes and numbers when available
    4. Mention any relevant context or implications

    Context: {context}
    Question: {question}
    
    Analysis:"""

    PROMPT = PromptTemplate(
        template=template,
        input_variables=["context", "question"]
    )
    
    # Create the chain
    qa_chain = load_qa_chain(
        llm=llm,
        chain_type="stuff",
        prompt=PROMPT
    )
    
    return RetrievalQAWithSourcesChain(
        combine_documents_chain=qa_chain,
        retriever=vector_store.as_retriever(
            search_kwargs={
                "k": 5  # Retrieve more relevant chunks
            }
        )
    )

def main():
    initialize_app()
    
    # Create a status placeholder
    status_placeholder = st.empty()
    
    # Get URLs from sidebar
    urls = load_urls()
    
    # Process button
    if st.sidebar.button("Process URLs"):
        vector_store = process_urls(urls, status_placeholder)
        if vector_store:
            st.session_state['vector_store'] = vector_store
    
    # Query input
    query = st.text_input("Ask a question about the articles:")
    
    if query and 'vector_store' in st.session_state:
        try:
            with st.spinner("Generating answer..."):
                chain = setup_qa_chain(st.session_state['vector_store'])
                result = chain({"question": query}, return_only_outputs=True)
                
                st.header("Answer")
                st.write(result["answer"])
                
                # Display sources if available
                if "sources" in result and result["sources"]:
                    st.subheader("Sources")
                    st.write(result["sources"])
                    
        except Exception as e:
            st.error(f"Error generating response: {str(e)}")

if __name__ == "__main__":
    main()