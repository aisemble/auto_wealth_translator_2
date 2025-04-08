"""
AutoWealthTranslate Streamlit Cloud App
---------------------------------------
Entry point for Streamlit Cloud deployment.
"""

import streamlit as st
import os
import logging
import time
import sys
from importlib import import_module

# Configure basic logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("streamlit_cloud")

# Set environment variables for cloud deployment
def setup_environment():
    """Configure environment variables for the app."""
    # Check for and set API keys from Streamlit secrets if available
    if hasattr(st, "secrets"):
        # OpenAI API key
        if "openai" in st.secrets and "api_key" in st.secrets["openai"]:
            os.environ["OPENAI_API_KEY"] = st.secrets["openai"]["api_key"]
            logger.info("OpenAI API key configured from Streamlit secrets")
            
        # DeepL API key
        if "deepl" in st.secrets and "api_key" in st.secrets["deepl"]:
            os.environ["DEEPL_API_KEY"] = st.secrets["deepl"]["api_key"]
            logger.info("DeepL API key configured from Streamlit secrets")
            
        # xAI API key
        if "xai" in st.secrets and "api_key" in st.secrets["xai"]:
            os.environ["XAI_API_KEY"] = st.secrets["xai"]["api_key"]
            logger.info("xAI API key configured from Streamlit secrets")
    
    # Configure Tesseract for OCR
    try:
        import pytesseract
        # Set Tesseract path for Linux environments (like Streamlit Cloud)
        if os.path.exists('/usr/bin/tesseract'):
            pytesseract.pytesseract.tesseract_cmd = r'/usr/bin/tesseract'
            logger.info("Tesseract OCR path configured")
        else:
            logger.warning("Tesseract OCR not found at expected path")
    except ImportError:
        logger.warning("pytesseract module not found, OCR functionality may be limited")
    
    # Configure other system dependencies
    os.environ["PYTHONIOENCODING"] = "utf-8"  # Ensure proper encoding
    
    # For WeasyPrint and Cairo
    os.environ["PANGOCAIRO_BACKEND"] = "fontconfig"
    
    # Create temporary directories if needed
    temp_dirs = [
        "temp_uploads",
        "temp_output",
        "temp_images"
    ]
    
    for dir_name in temp_dirs:
        os.makedirs(dir_name, exist_ok=True)
        logger.info(f"Created temporary directory: {dir_name}")
    
    # Log platform and Python version information
    logger.info(f"Python version: {sys.version}")
    logger.info(f"Platform: {sys.platform}")

# Set page configuration
def configure_page():
    """Configure the Streamlit page."""
    st.set_page_config(
        page_title="AutoWealthTranslate",
        page_icon="📄",
        layout="wide",
        initial_sidebar_state="expanded",
    )

if __name__ == "__main__":
    try:
        start_time = time.time()
        logger.info("Starting AutoWealthTranslate Streamlit Cloud app")
        
        # Setup environment
        setup_environment()
        
        # Configure page
        configure_page()
        
        # Import and run the main Streamlit app
        streamlit_app = import_module("streamlit_app")
        streamlit_app.main()
        
        logger.info(f"App initialized in {time.time() - start_time:.2f} seconds")
    except Exception as e:
        logger.error(f"Error starting app: {str(e)}", exc_info=True)
        st.error("😢 An error occurred while starting the application.")
        st.error(f"Error details: {str(e)}")
        st.info("Please check the logs for more information or contact support.") 