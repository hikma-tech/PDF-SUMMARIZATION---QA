import streamlit as st
import fitz  # PyMuPDF
import os
import requests
import json
from typing import List, Any, Optional
import faiss
import numpy as np
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.llms.base import LLM
from langchain.schema import Generation, LLMResult
from langchain.callbacks.manager import CallbackManagerForLLMRun
import time
from pydantic import Field

# Custom OpenRouter LLM Class
class OpenRouterLLM(LLM):
    """Custom LLM wrapper for OpenRouter API"""

    api_key: str = Field(...)
    model: str = Field(default="meta-llama/llama-3.3-8b-instruct:free")
    base_url: str = Field(default="https://openrouter.ai/api/v1/chat/completions")

    @property
    def _llm_type(self) -> str:
        return "openrouter"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """Call the OpenRouter API"""
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
            "HTTP-Referer": "https://github.com/your-username/pdf-qa-tool",  # Optional
            "X-Title": "PDF Q&A Tool"  # Optional
        }
        
        payload = {
            "model": self.model,
            "messages": [
                {"role": "user", "content": prompt}
            ]
        }
        
        try:
            response = requests.post(
                self.base_url,
                headers=headers,
                json=payload,
                timeout=60
            )
            response.raise_for_status()
            
            result = response.json()
            return result["choices"][0]["message"]["content"]
            
        except requests.exceptions.RequestException as e:
            st.error(f"API request failed: {str(e)}")
            return "Error: Could not get response from OpenRouter API"
        except (KeyError, IndexError) as e:
            st.error(f"Invalid API response format: {str(e)}")
            return "Error: Invalid response from API"

# Utility Functions
@st.cache_data
def extract_text_from_pdf(pdf_file) -> str:
    """Extract text from uploaded PDF file"""
    try:
        doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
        text = ""
        for page in doc:
            text += page.get_text()
        doc.close()
        return text
    except Exception as e:
        st.error(f"Error extracting text from PDF: {str(e)}")
        return ""

def split_text(text: str) -> List[str]:
    """Split text into chunks using RecursiveCharacterTextSplitter"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
        length_function=len,
    )
    chunks = text_splitter.split_text(text)
    return chunks

@st.cache_resource
def create_vector_store(chunks: List[str], _embeddings):
    """Create FAISS vector store from text chunks"""
    try:
        vectorstore = FAISS.from_texts(chunks, _embeddings)
        return vectorstore
    except Exception as e:
        st.error(f"Error creating vector store: {str(e)}")
        return None

def get_tone_prompt(tone: str) -> str:
    """Get the appropriate prompt prefix based on selected tone"""
    tone_prompts = {
        "formal": "Please summarize this document in a formal, professional tone:",
        "casual": "Give me a relaxed, friendly summary of this document:",
        "bullet": "Summarize this document using clear, concise bullet points:"
    }
    return tone_prompts.get(tone, tone_prompts["formal"])

# Streamlit App
def main():
    st.set_page_config(
        page_title="PDF Q&A & Summarization Tool",
        page_icon="📄",
        layout="wide"
    )
    
    st.title("📄 PDF Q&A & Summarization Tool")
    st.markdown("Upload a PDF, ask questions, and get AI-powered summaries!")
    
    # Sidebar for API configuration
    with st.sidebar:
        st.header("🔧 Configuration")
        openrouter_api_key = st.text_input(
            "OpenRouter API Key",
            type="password",
            value=os.getenv("OPENROUTER_API_KEY", ""),
            help="Enter your OpenRouter API key or set OPENROUTER_API_KEY environment variable"
        )
        
        if not openrouter_api_key:
            st.warning("⚠️ Please enter your OpenRouter API key to continue")
            st.markdown("You can get your API key from [OpenRouter](https://openrouter.ai/)")
            return
        
        st.success("✅ API key configured")
        
        # Optional configuration hints
        with st.expander("ℹ️ Optional Settings"):
            st.info("The app is configured with:\n- Model: openai/gpt-4o\n- Referer: GitHub project\n- Title: PDF Q&A Tool")
    
    # Initialize LLM and SentenceTransformers embeddings
    try:
        llm = OpenRouterLLM(api_key=openrouter_api_key)
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    except Exception as e:
        st.error(f"Error initializing LLM or embeddings: {str(e)}")
        return
    
    # Main content
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📤 Upload PDF")
        uploaded_file = st.file_uploader(
            "Choose a PDF file",
            type="pdf",
            help="Upload a PDF document to analyze"
        )
        
        if uploaded_file is not None:
            st.success(f"✅ Uploaded: {uploaded_file.name}")
            
            # Extract text
            with st.spinner("🔄 Extracting text from PDF..."):
                text = extract_text_from_pdf(uploaded_file)
            
            if text:
                st.info(f"📝 Extracted {len(text)} characters from PDF")
                
                # Split text into chunks
                with st.spinner("✂️ Processing text chunks..."):
                    chunks = split_text(text)
                
                st.info(f"📊 Created {len(chunks)} text chunks")
                
                # Create vector store
                with st.spinner("🧠 Building vector database..."):
                    vectorstore = create_vector_store(chunks, embeddings)
                
                if vectorstore:
                    st.success("✅ Vector database ready!")
                    
                    # Store in session state
                    st.session_state['vectorstore'] = vectorstore
                    st.session_state['full_text'] = text
                    st.session_state['llm'] = llm
                else:
                    st.error("❌ Failed to create vector database")
            else:
                st.error("❌ Could not extract text from PDF")
    
    with col2:
        st.header("💬 Q&A & Summary")
        
        if 'vectorstore' in st.session_state:
            # Q&A Section
            st.subheader("❓ Ask Questions")
            question = st.text_input(
                "Enter your question about the document:",
                placeholder="What is the main topic of this document?"
            )
            
            if st.button("🔍 Get Answer", type="primary"):
                if question:
                    with st.spinner("🤔 Thinking..."):
                        try:
                            # Create RetrievalQA chain
                            qa_chain = RetrievalQA.from_chain_type(
                                llm=st.session_state['llm'],
                                chain_type="stuff",
                                retriever=st.session_state['vectorstore'].as_retriever(
                                    search_kwargs={"k": 3}
                                )
                            )
                            
                            answer = qa_chain.run(question)
                            
                            st.success("✅ Answer:")
                            st.markdown(f"**Q:** {question}")
                            st.markdown(f"**A:** {answer}")
                            
                        except Exception as e:
                            st.error(f"Error getting answer: {str(e)}")
                else:
                    st.warning("Please enter a question")
            
            st.divider()
            
            # Summarization Section
            st.subheader("📋 Document Summary")
            
            tone = st.selectbox(
                "Choose summary tone:",
                options=["formal", "casual", "bullet"],
                format_func=lambda x: {
                    "formal": "🎩 Formal & Professional",
                    "casual": "😊 Casual & Friendly", 
                    "bullet": "📌 Bullet Points"
                }[x]
            )
            
            if st.button("📝 Generate Summary", type="secondary"):
                with st.spinner("✍️ Generating summary..."):
                    try:
                        # Combine all text with tone-specific prompt
                        tone_prompt = get_tone_prompt(tone)
                        full_prompt = f"{tone_prompt}\n\n{st.session_state['full_text']}"
                        
                        # Generate summary
                        summary = st.session_state['llm'](full_prompt)
                        
                        st.success("✅ Summary:")
                        st.markdown(summary)
                        
                    except Exception as e:
                        st.error(f"Error generating summary: {str(e)}")
        else:
            st.info("👆 Please upload a PDF file first to start asking questions and generating summaries")
    
    # Footer
    st.divider()
    st.markdown(
        """
        <div style='text-align: center; color: gray;'>
        Built with ❤️ using Streamlit, LangChain, FAISS, and OpenRouter
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()