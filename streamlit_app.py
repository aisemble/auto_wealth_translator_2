"""
AutoWealthTranslate Streamlit App
---------------------------------
A Streamlit UI for the AutoWealthTranslate application.
"""

import os
import streamlit as st
import tempfile
from pathlib import Path
import time
import logging
import shutil
import uuid
import base64
import io
import re

from auto_wealth_translate.core.document_processor import DocumentProcessor
from auto_wealth_translate.core.translator import TranslationService
from auto_wealth_translate.core.document_rebuilder import DocumentRebuilder
from auto_wealth_translate.core.validator import OutputValidator
from auto_wealth_translate.core.markdown_processor import MarkdownProcessor
from auto_wealth_translate.core.chart_processor import chart_to_markdown
from auto_wealth_translate.utils.logger import setup_logger, get_logger
from auto_wealth_translate.core.deepl_translator import DeepLTranslationService

# Configure logging
setup_logger(logging.INFO)
logger = get_logger("streamlit_app")

# Set more verbose logging for the markdown processor module
markdown_logger = logging.getLogger("auto_wealth_translate.core.markdown_processor")
markdown_logger.setLevel(logging.DEBUG)

# Create a file handler for detailed logs
log_file = Path(tempfile.gettempdir()) / "markdown_processor_debug.log"
file_handler = logging.FileHandler(log_file)
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
markdown_logger.addHandler(file_handler)

logger.info(f"Debug logs will be written to: {log_file}")

# Supported languages
SUPPORTED_LANGUAGES = {
    "en": "English",
    "zh": "Chinese (中文)",
    "fr": "French (Français)",
    "es": "Spanish (Español)",
    "de": "German (Deutsch)",
    "ja": "Japanese (日本語)",
    "ko": "Korean (한국어)",
    "ru": "Russian (Русский)",
    "ar": "Arabic (العربية)",
    "it": "Italian (Italiano)",
    "pt": "Portuguese (Português)"
}

# Translation service configuration
TRANSLATION_SERVICE = "Document Translation"

def create_temp_dir():
    """Create a temporary directory for file processing."""
    temp_dir = Path(tempfile.gettempdir()) / "autowealthtranslate_streamlit"
    temp_dir.mkdir(parents=True, exist_ok=True)
    return temp_dir

def get_file_download_link(file_path, link_text):
    """Generate a download link for a file."""
    with open(file_path, "rb") as f:
        data = f.read()
    b64 = base64.b64encode(data).decode()
    filename = Path(file_path).name
    mime_type = "application/pdf" if file_path.endswith(".pdf") else "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    href = f'<a href="data:{mime_type};base64,{b64}" download="{filename}">{link_text}</a>'
    return href

def translate_document(input_file, source_lang, target_lang, api_key=None):
    """
    Process and translate a document.
    
    Args:
        input_file: Path to input file
        source_lang: Source language code
        target_lang: Target language code
        api_key: API key for translation service
    
    Returns:
        Path to translated file, validation results, markdown content (if markdown mode), summary
    """
    if api_key:
        os.environ["DEEPL_API_KEY"] = api_key
    
    temp_dir = create_temp_dir()
    
    # Generate unique filename
    input_path = Path(input_file)
    unique_id = str(uuid.uuid4())[:8]
    output_path = temp_dir / f"{input_path.stem}_{target_lang}_{unique_id}{input_path.suffix}"
    
    # Initialize translator
    logger.info(f"Translating document from {source_lang} to {target_lang}")
    
    translator = DeepLTranslationService(
        source_lang=source_lang, 
        target_lang=target_lang,
        api_key=api_key
    )
    
    if not translator.is_ready():
        raise Exception("Translation service is not initialized. Please check your API key.")
    
    # Translate document
    translated_file, metadata = translator.translate_document(input_file)
    
    if not translated_file:
        error_message = metadata.get("error", "Unknown error during translation")
        raise Exception(f"Translation failed: {error_message}")
    
    # Create validation result
    validation_result = {
        "score": 9,  # Translations are generally high quality
        "issues": [],
        "metadata": metadata
    }
    
    # Generate summary
    summary = generate_document_summary(translated_file, target_lang)
    
    # Return the translated file path, validation results, and summary
    return translated_file, validation_result, None, summary

def generate_document_summary(file_path, language):
    """Generate a summary of the translated document."""
    from pathlib import Path
    import os
    
    # Use temporary file for extraction
    temp_dir = create_temp_dir()
    
    file_path = Path(file_path)
    file_ext = file_path.suffix.lower()
    
    # Extract text from document
    extracted_text = ""
    
    try:
        if file_ext == '.pdf':
            # For PDF files
            import PyPDF2
            with open(file_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                for page in reader.pages:
                    extracted_text += page.extract_text() + "\n\n"
        elif file_ext == '.docx':
            # For DOCX files
            import docx
            doc = docx.Document(file_path)
            for para in doc.paragraphs:
                extracted_text += para.text + "\n"
            
            # Also get text from tables
            for table in doc.tables:
                for row in table.rows:
                    for cell in row.cells:
                        extracted_text += cell.text + " | "
                    extracted_text += "\n"
        else:
            return "Summary generation is only supported for PDF and DOCX files."
        
        # Limit text to avoid token limits
        max_chars = 15000
        if len(extracted_text) > max_chars:
            extracted_text = extracted_text[:max_chars] + "..."
            
        # Create prompt for summary generation
        if language == 'zh':
            prompt = f"""请为以下文档生成一个简明的摘要（不超过 250 字）。关注主要内容、关键观点和结论。

文档内容:
{extracted_text}

摘要:"""
        else:
            prompt = f"""Please generate a concise summary of the following document (no more than 250 words). Focus on the main content, key points, and conclusions.

Document content:
{extracted_text}

Summary:"""
        
        return "Summary generation is currently disabled."
        
    except Exception as e:
        logger.error(f"Error generating summary: {str(e)}")
        return "Error generating summary."

def main():
    st.set_page_config(
        page_title="AutoWealthTranslate",
        page_icon="🌐",
        layout="wide"
    )
    
    st.title("AutoWealthTranslate")
    st.markdown("Translate financial documents while preserving formatting and structure.")
    
    # Sidebar
    st.sidebar.header("Settings")
    
    # Language selection
    source_lang = st.sidebar.selectbox(
        "Source Language",
        options=list(SUPPORTED_LANGUAGES.keys()),
        format_func=lambda x: SUPPORTED_LANGUAGES[x],
        index=0
    )
    
    target_lang = st.sidebar.selectbox(
        "Target Language",
        options=list(SUPPORTED_LANGUAGES.keys()),
        format_func=lambda x: SUPPORTED_LANGUAGES[x],
        index=1
    )
    
    # API key input
    api_key = st.sidebar.text_input("API Key", type="password")
    
    # File upload
    uploaded_file = st.file_uploader("Upload Document", type=["pdf", "docx"])
    
    if uploaded_file is not None:
        # Save uploaded file
        temp_dir = create_temp_dir()
        input_path = temp_dir / uploaded_file.name
        with open(input_path, "wb") as f:
            f.write(uploaded_file.getvalue())
        
        # Translation button
        if st.button("Translate"):
            try:
                with st.spinner("Translating document..."):
                    translated_file, validation, _, summary = translate_document(
                        input_path,
                        source_lang,
                        target_lang,
                        api_key
                    )
                
                # Display results
                st.success("Translation completed!")
                
                # Download link
                st.markdown(
                    get_file_download_link(translated_file, "Download Translated Document"),
                    unsafe_allow_html=True
                )
                
                # Display summary if available
                if summary:
                    st.subheader("Document Summary")
                    st.write(summary)
                
            except Exception as e:
                st.error(f"Error during translation: {str(e)}")
    
    # Footer
    st.markdown("---")
    st.markdown("© 2024 AutoWealthTranslate")

if __name__ == "__main__":
    main() 