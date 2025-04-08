"""
DeepL translation module for AutoWealthTranslate.

This module provides functionality to translate entire documents using the DeepL API.
"""

import os
import time
import logging
import tempfile
import shutil
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import deepl

from auto_wealth_translate.utils.logger import get_logger

logger = get_logger(__name__)

class DeepLTranslationService:
    """
    Service for translating documents using the DeepL API.
    
    This service allows for direct document translation, preserving the
    original format of PDF, DOCX, PPTX, XLSX, HTML, and TXT files.
    """
    
    def __init__(self, source_lang: str = None, target_lang: str = "ZH", api_key: str = None):
        """
        Initialize the DeepL translation service.
        
        Args:
            source_lang: Source language code (e.g., 'EN', 'FR'). If None, DeepL will auto-detect.
            target_lang: Target language code (e.g., 'ZH', 'FR')
            api_key: DeepL API key. If None, will try to get from DEEPL_API_KEY env var.
            
        Note:
            DeepL language codes are uppercase (e.g., 'EN', 'ZH') unlike our internal lowercase codes.
            This class handles the conversion internally.
        """
        self.source_lang = self._convert_to_deepl_lang_code(source_lang) if source_lang else None
        self.target_lang = self._convert_to_deepl_lang_code(target_lang)
        
        # Get API key from param or environment
        self.api_key = api_key or os.environ.get("DEEPL_API_KEY")
        
        if not self.api_key:
            logger.warning("DeepL API key not found. Please set DEEPL_API_KEY environment variable or provide it in the constructor.")
            self.client = None
        else:
            try:
                # Initialize DeepL client
                self.client = deepl.Translator(self.api_key)
                logger.info("Initialized DeepL translator client")
                
                # Verify API connection
                usage = self.client.get_usage()
                logger.info(f"DeepL API connected. Character usage: {usage.character.count}/{usage.character.limit}")
            except Exception as e:
                logger.error(f"Failed to initialize DeepL client: {str(e)}")
                self.client = None
    
    def _convert_to_deepl_lang_code(self, lang_code: str) -> str:
        """Convert internal language code to DeepL format."""
        # DeepL uses uppercase language codes
        if lang_code:
            # Handle special cases like Chinese
            if lang_code.lower() == "zh":
                return "ZH"
            # Convert from lowercase to uppercase
            return lang_code.upper()
        return None
    
    def _convert_from_deepl_lang_code(self, lang_code: str) -> str:
        """Convert DeepL language code to internal format."""
        if lang_code:
            return lang_code.lower()
        return None
    
    def translate_document(self, input_file: str, output_file: str = None, 
                          formality: str = "default") -> Tuple[str, Dict[str, Any]]:
        """
        Translate a document using DeepL's document translation API.
        
        Args:
            input_file: Path to the document to translate
            output_file: Path where to save the translated document. If None, a path will be generated.
            formality: Formality level ('default', 'more', 'less', 'prefer_more', 'prefer_less')
            
        Returns:
            Tuple of (output_file_path, metadata)
        """
        if not self.client:
            logger.error("DeepL client not initialized. Cannot translate document.")
            return None, {"error": "DeepL client not initialized"}
            
        input_path = Path(input_file)
        
        # Check if file exists
        if not input_path.exists():
            logger.error(f"Input file not found: {input_file}")
            return None, {"error": f"Input file not found: {input_file}"}
            
        # Check file type
        supported_extensions = ['.pdf', '.docx', '.pptx', '.xlsx', '.html', '.htm', '.txt']
        if input_path.suffix.lower() not in supported_extensions:
            logger.error(f"Unsupported file type: {input_path.suffix}. Supported types: {', '.join(supported_extensions)}")
            return None, {"error": f"Unsupported file type: {input_path.suffix}"}
            
        # Generate output path if not provided
        if not output_file:
            # Create temp directory if needed
            temp_dir = Path(tempfile.gettempdir()) / "autowealthtranslate_deepl"
            temp_dir.mkdir(parents=True, exist_ok=True)
            
            # For PDF files, we can specify output format as DOCX
            output_suffix = input_path.suffix
            if input_path.suffix.lower() == '.pdf':
                output_suffix = '.docx'  # DeepL can convert PDF to DOCX
                
            output_file = temp_dir / f"{input_path.stem}_{self.target_lang.lower()}{output_suffix}"
        
        try:
            # Log translation attempt
            logger.info(f"Translating document with DeepL: {input_file} -> {output_file}")
            
            # Set translation options
            translator_options = {
                "target_lang": self.target_lang,
                "formality": formality
            }
            
            # Add source_lang if specified
            if self.source_lang:
                translator_options["source_lang"] = self.source_lang
                
            # For PDF files, we can specify output format as DOCX
            if input_path.suffix.lower() == '.pdf':
                translator_options["output_format"] = "docx"
                
            # Translate the document
            logger.info(f"Starting document translation with options: {translator_options}")
            
            # Ensure output directory exists
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # DeepL's translate_document method requires both input and output files
            with open(input_path, "rb") as in_file:
                # Create output file object for translate_document
                with open(output_path, "wb") as out_file:
                    result = self.client.translate_document(
                        in_file,
                        out_file,
                        **translator_options
                    )
            
            # Calculate file size
            file_size = output_path.stat().st_size
            
            # Return success and metadata
            return str(output_path), {
                "source_lang": self.source_lang or result.detected_source_lang,
                "target_lang": self.target_lang,
                "formality": formality,
                "file_size": file_size,
                "success": True
            }
            
        except deepl.DeepLException as e:
            logger.error(f"DeepL API error: {str(e)}")
            return None, {"error": f"DeepL API error: {str(e)}"}
        except Exception as e:
            logger.error(f"Error translating document: {str(e)}")
            return None, {"error": f"Error: {str(e)}"}
            
    def is_ready(self) -> bool:
        """Check if the DeepL client is ready for translation."""
        return self.client is not None 