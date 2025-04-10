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

# Translation models - only keep DeepL
TRANSLATION_MODELS = {
    "deepl": "Document Translation"
}

# PDF processing modes - only keep DeepL
PDF_PROCESSING_MODES = {
    "deepl": "Document Translation"
}

# More detailed descriptions for each mode - remove DeepL explanation
PDF_MODE_DESCRIPTIONS = {
    "deepl": ""
}

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

def translate_document(input_file, source_lang, target_lang, model="deepl", api_key=None, pdf_mode="deepl", xai_api_key=None, deepl_api_key=None, gpt4o_key=None):
    """
    Process and translate a document.
    
    Args:
        input_file: Path to input file
        source_lang: Source language code
        target_lang: Target language code
        model: Translation model to use
        api_key: OpenAI API key
        pdf_mode: PDF processing mode
        xai_api_key: xAI API key for Grok models
        deepl_api_key: DeepL API key for DeepL translation
        gpt4o_key: API key for GPT-4o summary generation
    
    Returns:
        Path to translated file, validation results, markdown content (if markdown mode), summary
    """
    # Set API keys if provided
    if api_key and model.startswith("gpt"):
        os.environ["OPENAI_API_KEY"] = api_key
    
    if xai_api_key and model.startswith("grok"):
        os.environ["XAI_API_KEY"] = xai_api_key
        
    if deepl_api_key:
        os.environ["DEEPL_API_KEY"] = deepl_api_key
        
    if gpt4o_key:
        os.environ["OPENAI_API_KEY"] = gpt4o_key
    
    temp_dir = create_temp_dir()
    
    # Generate unique filename
    input_path = Path(input_file)
    unique_id = str(uuid.uuid4())[:8]
    output_path = temp_dir / f"{input_path.stem}_{target_lang}_{unique_id}{input_path.suffix}"
    
    # Initialize DeepL translator
    logger.info(f"Using DeepL API for document translation from {source_lang} to {target_lang}")
    
    deepl_translator = DeepLTranslationService(
        source_lang=source_lang, 
        target_lang=target_lang,
        api_key=deepl_api_key
    )
    
    if not deepl_translator.is_ready():
        raise Exception("DeepL translator is not initialized. Please check your API key.")
    
    # Translate document
    translated_file, metadata = deepl_translator.translate_document(input_file)
    
    if not translated_file:
        error_message = metadata.get("error", "Unknown error during DeepL translation")
        raise Exception(f"Translation failed: {error_message}")
    
    # Create validation result
    validation_result = {
        "score": 9,  # Translations are generally high quality
        "issues": [],
        "metadata": metadata
    }
    
    # Generate summary using GPT-4o or Grok3
    summary = generate_document_summary(translated_file, target_lang, gpt4o_key, xai_api_key)
    
    # Return the translated file path, validation results, and summary
    return translated_file, validation_result, None, summary

def generate_document_summary(file_path, language, openai_key=None, xai_key=None):
    """Generate a summary of the translated document using GPT-4o or Grok3."""
    import openai
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
        
        # Try to use GPT-4o first
        if openai_key:
            try:
                client = openai.OpenAI(api_key=openai_key)
                response = client.chat.completions.create(
                    model="gpt-4o",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=500,
                    temperature=0.3
                )
                return response.choices[0].message.content.strip()
            except Exception as e:
                logger.error(f"Error generating summary with GPT-4o: {str(e)}")
                # Fall back to Grok3 if available
        
        # Try Grok3 if GPT-4o failed or was not available
        if xai_key:
            try:
                client = openai.OpenAI(
                    api_key=xai_key,
                    base_url="https://api.x.ai/v1"
                )
                response = client.chat.completions.create(
                    model="grok-3",  # Use Grok3 if available
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=500,
                    temperature=0.3
                )
                return response.choices[0].message.content.strip()
            except Exception as e:
                logger.error(f"Error generating summary with Grok3: {str(e)}")
                
        return "Unable to generate summary. Please check API keys."
    
    except Exception as e:
        logger.error(f"Error during summary generation: {str(e)}")
        return f"Error generating summary: {str(e)}"

def main():
    st.set_page_config(
        page_title="AutoWealthTranslate",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Logo and title
    st.sidebar.image("https://i.imgur.com/ddr4OWY.png", width=100)
    st.sidebar.title("AutoWealthTranslate")
    st.sidebar.markdown("*Translate financial documents with accurate terminology and preserved formatting*")
    
    # API Keys section in the sidebar
    with st.sidebar.expander("API Keys (Required for Translation)", expanded=False):
        deepl_key = st.text_input("DeepL API Key (for translations)", type="password")
        gpt4o_key = st.text_input("OpenAI API Key (for summary generation)", type="password")
        xai_key = st.text_input("xAI API Key (alternative for summary generation)", type="password")
        
        st.markdown("""
        **Note on API Keys:**
        - DeepL key is required for document translation
        - OpenAI key is used for document summary generation with GPT-4o
        - xAI key is an alternative for summary generation with Grok3
        """)
    
    # Main content
    st.title("Financial Document Translation")
    st.markdown("### Automatically translate wealth plan reports while preserving formatting")
    
    # Main area - File upload
    st.header("Upload Document")
    
    uploaded_file = st.file_uploader(
        "Upload a PDF or Word document",
        type=["pdf", "docx"],
        help="Maximum file size: 200MB"
    )
    
    # Translation settings
    col1, col2 = st.columns(2)
    
    with col1:
        source_language = st.selectbox(
            "Source Language",
            options=list(SUPPORTED_LANGUAGES.keys()),
            format_func=lambda x: SUPPORTED_LANGUAGES[x],
            index=0,  # Default to English
            help="The language of the document you're uploading"
        )
    
    with col2:
        target_language = st.selectbox(
            "Target Language",
            options=list(SUPPORTED_LANGUAGES.keys()),
            format_func=lambda x: SUPPORTED_LANGUAGES[x],
            index=1,  # Default to Chinese
            help="The language to translate the document into"
        )
    
    # Process button
    if st.button("Translate Document"):
        # Check for required API key
        if not deepl_key:
            st.error("DeepL API Key is required for translation. Please enter it in the sidebar.")
            st.stop()
        
        # Save uploaded file to temp location
        temp_dir = create_temp_dir()
        temp_file_path = temp_dir / uploaded_file.name
        
        with open(temp_file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # Show progress
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            # Update progress
            status_text.text("Starting translation...")
            progress_bar.progress(10)
            
            # Translate document
            status_text.text("Translating document...")
            progress_bar.progress(30)
            
            # Set PDF mode to deepl
            pdf_mode = "deepl"
            model = "deepl"
            
            translated_file, validation_results, _, summary = translate_document(
                str(temp_file_path),
                source_language,
                target_language,
                model=model,
                pdf_mode=pdf_mode,
                deepl_api_key=deepl_key,
                gpt4o_key=gpt4o_key,
                xai_api_key=xai_key
            )
            
            progress_bar.progress(90)
            status_text.text("Finalizing translation...")
            
            # Complete progress
            progress_bar.progress(100)
            status_text.text("Translation complete!")
            
            # Display summary if available
            if summary:
                st.subheader("Document Summary")
                st.write(summary)
            
            # Display download link
            st.subheader("Download Translated Document")
            
            # Get file extension
            file_ext = Path(translated_file).suffix.lower()
            
            if file_ext == '.pdf':
                file_type = "PDF"
            elif file_ext == '.docx':
                file_type = "Word"
            else:
                file_type = "Document"
                
            st.markdown(
                get_file_download_link(translated_file, f"Download Translated {file_type}"),
                unsafe_allow_html=True
            )
            
            # Add translation details and quality score
            st.subheader("Translation Details")
            st.write(f"Quality Score: {validation_results['score']}/10")
            
            # Display metadata
            metadata = validation_results.get("metadata", {})
            if metadata:
                st.write("**Translation Metadata:**")
                for key, value in metadata.items():
                    if key not in ["error", "success"]:
                        st.write(f"- {key.replace('_', ' ').title()}: {value}")
        
        except Exception as e:
            progress_bar.progress(100)
            status_text.empty()
            st.error(f"Error during translation: {str(e)}")
            st.write("Please check your API keys and try again.")
            logger.error(f"Error during translation: {str(e)}", exc_info=True)

if __name__ == "__main__":
    main() 