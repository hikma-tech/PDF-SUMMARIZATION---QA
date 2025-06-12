import os
import asyncio
import logging
import tempfile
import shutil
from typing import Dict, Any, Optional, List
import json
import requests
import fitz  # PyMuPDF
import numpy as np
from io import BytesIO

# Telegram imports
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup, Document
from telegram.ext import (
    Application, CommandHandler, MessageHandler, 
    CallbackQueryHandler, ContextTypes, filters
)

# LangChain and ML imports
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.llms.base import LLM
from langchain.schema import Generation, LLMResult
from langchain.callbacks.manager import CallbackManagerForLLMRun
from pydantic import Field
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Bot Configuration
TELEGRAM_TOKEN = "7275862861:AAGkByDVsjvzs-1Ez3bT2ptARDryvv_Sviw"
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")

# User sessions storage (in production, use Redis or database)
user_sessions: Dict[int, Dict[str, Any]] = {}

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
            "HTTP-Referer": "https://github.com/pdf-qa-telegram-bot",
            "X-Title": "PDF Q&A Telegram Bot"
        }
        
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 1000,
            "temperature": 0.7
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
        except Exception as e:
            logger.error(f"OpenRouter API error: {str(e)}")
            return f"❌ Error: Could not get response from AI service. Please try again."

class PDFProcessor:
    """Handle PDF processing and text extraction"""
    
    @staticmethod
    def extract_text_from_pdf(pdf_bytes: bytes) -> str:
        """Extract text from PDF bytes"""
        try:
            doc = fitz.open(stream=pdf_bytes, filetype="pdf")
            text = ""
            for page in doc:
                text += page.get_text()
            doc.close()
            return text
        except Exception as e:
            logger.error(f"PDF extraction error: {str(e)}")
            return ""
    
    @staticmethod
    def split_text(text: str) -> List[str]:
        """Split text into chunks"""
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100,
            length_function=len,
        )
        return text_splitter.split_text(text)
    
    @staticmethod
    def create_vector_store(chunks: List[str]) -> Optional[FAISS]:
        """Create FAISS vector store from text chunks"""
        try:
            embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )
            vectorstore = FAISS.from_texts(chunks, embeddings)
            return vectorstore
        except Exception as e:
            logger.error(f"Vector store creation error: {str(e)}")
            return None

# Bot Handlers
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Start command handler"""
    user_id = update.effective_user.id
    user_sessions[user_id] = {"status": "waiting_for_pdf"}
    
    welcome_text = """
🤖 **Welcome to PDF Q&A Bot!**

I can help you analyze PDF documents by:
📄 **Reading PDFs** - Upload any PDF document
❓ **Answering Questions** - Ask me anything about the content
📋 **Creating Summaries** - Get formatted summaries

**How to use:**
1. Send me a PDF file (max 20MB)
2. Wait for processing confirmation
3. Ask questions or request summaries

Ready? Send me a PDF to get started! 📎
    """
    
    await update.message.reply_text(
        welcome_text,
        parse_mode='Markdown'
    )

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Help command handler"""
    help_text = """
🆘 **Help - How to Use PDF Q&A Bot**

**Commands:**
• `/start` - Start the bot
• `/help` - Show this help message
• `/status` - Check current session status
• `/clear` - Clear current PDF session

**Features:**
📤 **Upload PDF:** Send any PDF file (up to 20MB)
❓ **Ask Questions:** Type any question about your PDF
📋 **Get Summary:** Use summary buttons or type "summarize"

**Examples:**
• "What is this document about?"
• "Summarize the main points"
• "What are the key findings?"
• "Explain the methodology"

**Tips:**
✅ Ensure your PDF has readable text
✅ Wait for "✅ Ready!" before asking questions
✅ Ask specific questions for better answers
✅ Use /clear to upload a new PDF

Need more help? Just ask! 🤝
    """
    
    await update.message.reply_text(help_text, parse_mode='Markdown')

async def status_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Status command handler"""
    user_id = update.effective_user.id
    session = user_sessions.get(user_id, {})
    
    if not session:
        status_text = "❌ No active session. Use /start to begin!"
    elif session.get("status") == "waiting_for_pdf":
        status_text = "⏳ Waiting for PDF upload. Please send a PDF file."
    elif session.get("status") == "processing":
        status_text = "🔄 Processing your PDF. Please wait..."
    elif session.get("status") == "ready":
        pdf_name = session.get("pdf_name", "Unknown")
        char_count = session.get("char_count", 0)
        chunk_count = session.get("chunk_count", 0)
        status_text = f"""
✅ **Session Ready!**

📄 **PDF:** {pdf_name}
📊 **Characters:** {char_count:,}
📦 **Chunks:** {chunk_count}

You can now ask questions or request summaries!
        """
    else:
        status_text = "❓ Unknown status. Use /start to restart."
    
    await update.message.reply_text(status_text, parse_mode='Markdown')

async def clear_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Clear command handler"""
    user_id = update.effective_user.id
    if user_id in user_sessions:
        del user_sessions[user_id]
    
    user_sessions[user_id] = {"status": "waiting_for_pdf"}
    
    await update.message.reply_text(
        "🗑️ Session cleared! Send me a new PDF to analyze.",
        parse_mode='Markdown'
    )

async def handle_document(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle PDF document uploads"""
    user_id = update.effective_user.id
    document = update.message.document
    
    # Check if it's a PDF
    if not document.mime_type == 'application/pdf':
        await update.message.reply_text(
            "❌ Please send a PDF file only. Other formats are not supported."
        )
        return
    
    # Check file size (20MB limit)
    if document.file_size > 20 * 1024 * 1024:
        await update.message.reply_text(
            "❌ File too large! Please send a PDF smaller than 20MB."
        )
        return
    
    # Update session status
    user_sessions[user_id] = {"status": "processing", "pdf_name": document.file_name}
    
    processing_msg = await update.message.reply_text(
        "🔄 **Processing your PDF...**\n\n"
        "⏳ Downloading file...",
        parse_mode='Markdown'
    )
    
    try:
        # Download the file
        file = await context.bot.get_file(document.file_id)
        pdf_bytes = await file.download_as_bytearray()
        
        await processing_msg.edit_text(
            "🔄 **Processing your PDF...**\n\n"
            "✅ Downloaded\n⏳ Extracting text...",
            parse_mode='Markdown'
        )
        
        # Extract text
        text = PDFProcessor.extract_text_from_pdf(bytes(pdf_bytes))
        
        if not text.strip():
            await processing_msg.edit_text(
                "❌ **Error:** Could not extract text from PDF.\n"
                "Make sure your PDF contains readable text (not just images)."
            )
            user_sessions[user_id] = {"status": "waiting_for_pdf"}
            return
        
        await processing_msg.edit_text(
            "🔄 **Processing your PDF...**\n\n"
            "✅ Downloaded\n✅ Text extracted\n⏳ Creating chunks...",
            parse_mode='Markdown'
        )
        
        # Split into chunks
        chunks = PDFProcessor.split_text(text)
        
        await processing_msg.edit_text(
            "🔄 **Processing your PDF...**\n\n"
            "✅ Downloaded\n✅ Text extracted\n✅ Chunks created\n⏳ Building vector database...",
            parse_mode='Markdown'
        )
        
        # Create vector store
        vectorstore = PDFProcessor.create_vector_store(chunks)
        
        if not vectorstore:
            await processing_msg.edit_text(
                "❌ **Error:** Could not create vector database. Please try again."
            )
            user_sessions[user_id] = {"status": "waiting_for_pdf"}
            return
        
        # Initialize LLM
        if not OPENROUTER_API_KEY:
            await processing_msg.edit_text(
                "❌ **Error:** OpenRouter API key not configured. Please contact the administrator."
            )
            return
        
        llm = OpenRouterLLM(api_key=OPENROUTER_API_KEY)
        
        # Update session with all data
        user_sessions[user_id] = {
            "status": "ready",
            "pdf_name": document.file_name,
            "char_count": len(text),
            "chunk_count": len(chunks),
            "full_text": text,
            "vectorstore": vectorstore,
            "llm": llm
        }
        
        # Create action buttons
        keyboard = [
            [InlineKeyboardButton("❓ Ask Question", callback_data="ask_question")],
            [
                InlineKeyboardButton("📋 Formal Summary", callback_data="summary_formal"),
                InlineKeyboardButton("😊 Casual Summary", callback_data="summary_casual")
            ],
            [InlineKeyboardButton("📌 Bullet Points", callback_data="summary_bullet")]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        success_text = f"""
✅ **PDF Ready for Analysis!**

📄 **File:** {document.file_name}
📊 **Characters:** {len(text):,}
📦 **Chunks:** {len(chunks)}

**What would you like to do?**
        """
        
        await processing_msg.edit_text(
            success_text,
            parse_mode='Markdown',
            reply_markup=reply_markup
        )
        
    except Exception as e:
        logger.error(f"Document processing error: {str(e)}")
        await processing_msg.edit_text(
            f"❌ **Error processing PDF:** {str(e)}\n\nPlease try again with a different file."
        )
        user_sessions[user_id] = {"status": "waiting_for_pdf"}

async def handle_question(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle user questions"""
    user_id = update.effective_user.id
    session = user_sessions.get(user_id, {})
    
    if session.get("status") != "ready":
        await update.message.reply_text(
            "❌ Please upload a PDF first using /start"
        )
        return
    
    question = update.message.text.strip()
    
    # Check for summary request
    if any(word in question.lower() for word in ['summary', 'summarize', 'sum up']):
        await handle_summary_request(update, context, "formal")
        return
    
    thinking_msg = await update.message.reply_text(
        f"🤔 **Question:** {question}\n\n⏳ Searching for answer..."
    )
    
    try:
        # Create RetrievalQA chain
        qa_chain = RetrievalQA.from_chain_type(
            llm=session['llm'],
            chain_type="stuff",
            retriever=session['vectorstore'].as_retriever(search_kwargs={"k": 3})
        )
        
        # Get answer
        answer = qa_chain.run(question)
        
        response_text = f"""
❓ **Question:** {question}

💡 **Answer:**
{answer}

---
💬 Ask another question or use /clear for a new PDF
        """
        
        await thinking_msg.edit_text(response_text, parse_mode='Markdown')
        
    except Exception as e:
        logger.error(f"Question handling error: {str(e)}")
        await thinking_msg.edit_text(
            f"❌ Error getting answer: {str(e)}\n\nPlease try rephrasing your question."
        )

async def handle_summary_request(update: Update, context: ContextTypes.DEFAULT_TYPE, tone: str) -> None:
    """Handle summary requests"""
    user_id = update.effective_user.id
    session = user_sessions.get(user_id, {})
    
    if session.get("status") != "ready":
        if update.callback_query:
            await update.callback_query.answer("Please upload a PDF first!")
        return
    
    tone_prompts = {
        "formal": "Please provide a formal, professional summary of this document:",
        "casual": "Give me a friendly, easy-to-understand summary of this document:",
        "bullet": "Summarize this document using clear, concise bullet points:"
    }
    
    tone_emojis = {
        "formal": "🎩",
        "casual": "😊", 
        "bullet": "📌"
    }
    
    if update.callback_query:
        await update.callback_query.answer()
        thinking_msg = await update.callback_query.edit_message_text(
            f"{tone_emojis[tone]} **Generating {tone} summary...**\n\n⏳ Please wait..."
        )
    else:
        thinking_msg = await update.message.reply_text(
            f"{tone_emojis[tone]} **Generating {tone} summary...**\n\n⏳ Please wait..."
        )
    
    try:
        # Create prompt
        prompt = f"{tone_prompts[tone]}\n\n{session['full_text'][:4000]}..."  # Limit text length
        
        # Generate summary
        summary = session['llm'](prompt)
        
        response_text = f"""
{tone_emojis[tone]} **{tone.title()} Summary:**

{summary}

---
💬 Ask a question or request another summary type
        """
        
        await thinking_msg.edit_text(response_text, parse_mode='Markdown')
        
    except Exception as e:
        logger.error(f"Summary generation error: {str(e)}")
        await thinking_msg.edit_text(
            f"❌ Error generating summary: {str(e)}\n\nPlease try again."
        )

async def button_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle button callbacks"""
    query = update.callback_query
    data = query.data
    
    if data == "ask_question":
        await query.answer()
        await query.edit_message_text(
            "❓ **Ask me anything about your PDF!**\n\n"
            "Just type your question in the chat.\n\n"
            "**Examples:**\n"
            "• What is this document about?\n"
            "• What are the main conclusions?\n"
            "• Explain the methodology\n"
            "• Who are the key people mentioned?",
            parse_mode='Markdown'
        )
    elif data.startswith("summary_"):
        tone = data.replace("summary_", "")
        await handle_summary_request(update, context, tone)

async def handle_unknown(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle unknown messages"""
    user_id = update.effective_user.id
    session = user_sessions.get(user_id, {})
    
    if session.get("status") == "waiting_for_pdf":
        await update.message.reply_text(
            "📎 Please send me a PDF file to analyze.\n\n"
            "Use /help if you need assistance!"
        )
    elif session.get("status") == "ready":
        # Treat as question
        await handle_question(update, context)
    else:
        await update.message.reply_text(
            "❓ I didn't understand that. Use /help for available commands!"
        )

async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Log errors"""
    logger.error(f"Exception while handling an update: {context.error}")

def main():
    """Start the bot"""
    if not TELEGRAM_TOKEN:
        logger.error("TELEGRAM_TOKEN not provided!")
        return
    
    if not OPENROUTER_API_KEY:
        logger.error("OPENROUTER_API_KEY not found in environment variables!")
        return
    
    # Create application
    application = Application.builder().token(TELEGRAM_TOKEN).build()
    
    # Add handlers
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(CommandHandler("status", status_command))
    application.add_handler(CommandHandler("clear", clear_command))
    
    # Document handler (PDF files)
    application.add_handler(MessageHandler(filters.Document.PDF, handle_document))
    
    # Button callback handler
    application.add_handler(CallbackQueryHandler(button_callback))
    
    # Text message handler (questions)
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_unknown))
    
    # Error handler
    application.add_error_handler(error_handler)
    
    # Start the bot
    logger.info("Starting PDF Q&A Telegram Bot...")
    application.run_polling(allowed_updates=Update.ALL_TYPES)

if __name__ == "__main__":
    main()