# 📄 PDF Q&A & Summarization Tool

An AI-powered tool for PDF document analysis, question answering, and intelligent summarization using LangChain, FAISS, and OpenRouter API.

## 🚀 Features

- **PDF Upload & Text Extraction**: Upload PDF files and extract text using PyMuPDF
- **Intelligent Q&A**: Ask questions about your documents with AI-powered answers
- **Smart Summarization**: Generate summaries in different tones (formal, casual, bullet points)
- **Vector Search**: Fast document retrieval using FAISS vector database
- **Custom LLM Integration**: Uses OpenRouter API with GPT-4o model
- **Local Embeddings**: Uses SentenceTransformers (HuggingFace) for embeddings (no OpenAI key needed)
- **Beautiful UI**: Clean, responsive Streamlit interface

## 🛠️ Installation

1. **Clone or download the project files**

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up your OpenRouter API key**:
   
   **Option A: Environment Variable (Recommended)**
   ```bash
   export OPENROUTER_API_KEY="your-api-key-here"
   ```
   
   **Option B: Enter in the app interface**
   - Get your API key from [OpenRouter](https://openrouter.ai/)
   - Enter it in the sidebar when running the app

## 🏃‍♂️ Running the Application

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

## 📋 How to Use

1. **Configure API Key**: Enter your OpenRouter API key in the sidebar
2. **Upload PDF**: Use the file uploader to select your PDF document
3. **Wait for Processing**: The app will extract text, create chunks, and build the vector database
4. **Ask Questions**: Enter questions about your document in the Q&A section
5. **Generate Summaries**: Choose a tone and generate AI summaries of your document

## 🎯 Key Components

### Custom OpenRouter LLM
- Custom LangChain LLM wrapper for OpenRouter API
- Uses `meta-llama/llama-3.3-8b-instruct:free` model by default
- Proper error handling and timeout management

### Text Processing
- **PyMuPDF** (`fitz`) for PDF text extraction
- **RecursiveCharacterTextSplitter** for intelligent text chunking
- 1000-character chunks with 100-character overlap

### Vector Database
- **FAISS** for fast similarity search
- **HuggingFace/SentenceTransformers Embeddings** for text vectorization (no OpenAI key required)
- Cached vector store creation for performance

### Q&A System
- **RetrievalQA** chain from LangChain
- Retrieves top 3 most relevant chunks
- Context-aware question answering

### Summarization
- **Tone Control**: Formal, casual, or bullet point summaries
- **Full Document Processing**: Combines all text chunks
- **Custom Prompting**: Tone-specific instructions

## 🔧 Configuration Options

The app includes several configurable aspects:

- **Model**: Currently set to `meta-llama/llama-3.3-8b-instruct:free` (can be modified in the code)
- **Chunk Size**: 1000 characters (adjustable in `split_text()`)
- **Chunk Overlap**: 100 characters (adjustable in `split_text()`)
- **Retrieval Count**: Top 3 chunks for Q&A (adjustable in RetrievalQA)

## 🚨 Troubleshooting

### Common Issues

1. **API Key Errors**: Make sure your OpenRouter API key is valid and has sufficient credits
2. **PDF Processing Errors**: Ensure your PDF is not password-protected or corrupted
3. **Memory Issues**: For very large PDFs, consider reducing chunk size or splitting the document
4. **Slow Performance**: Vector database creation can take time for large documents

### Error Handling

The app includes comprehensive error handling for:
- API request failures
- PDF processing errors
- Vector store creation issues
- Invalid file formats

## 📊 Technical Architecture

```
PDF Upload → Text Extraction → Text Chunking → Vector Embeddings → FAISS Storage
                                                                          ↓
User Question → Vector Search → Context Retrieval → LLM Processing → Answer
                                                                          ↓
Summary Request → Full Text + Tone Prompt → LLM Processing → Formatted Summary
```

## 🔐 Security Notes

- API keys are handled securely and not logged
- Environment variables are preferred for API key storage
- No user data is stored permanently
- All processing happens locally except for LLM API calls

## 📝 Requirements

- Python 3.8+
- OpenRouter API account and key
- Sufficient system memory for vector operations
- Internet connection for API calls

## 🤝 Contributing

Feel free to:
- Report bugs or issues
- Suggest new features
- Submit pull requests
- Improve documentation

## 📄 License

This project is provided as-is for educational and personal use.

---

**Happy document analysis! 🎉**