#!/usr/bin/env python3
"""
AutoWealthTranslate CLI
-----------------------
Command-line interface for the AutoWealthTranslate application.
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import List, Optional
import time

from auto_wealth_translate.core.document_processor import DocumentProcessor
from auto_wealth_translate.core.translator import TranslationService
from auto_wealth_translate.core.document_rebuilder import DocumentRebuilder
from auto_wealth_translate.core.validator import OutputValidator
from auto_wealth_translate.utils.logger import setup_logger, get_logger

# Supported languages
SUPPORTED_LANGUAGES = {
    "en": "English",
    "zh": "Chinese",
    "fr": "French",
    "es": "Spanish",
    "de": "German",
    "ja": "Japanese",
    "ko": "Korean",
    "ru": "Russian",
    "ar": "Arabic",
    "it": "Italian",
    "pt": "Portuguese"
}

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="AutoWealthTranslate - Automatically translate wealth plan reports while preserving formatting."
    )
    
    parser.add_argument(
        "--input", "-i", 
        required=True,
        help="Input file path (PDF or DOCX)"
    )
    
    parser.add_argument(
        "--output", "-o",
        help="Output file path. If not specified, uses the input filename with language suffix."
    )
    
    parser.add_argument(
        "--lang", "-l",
        required=True,
        choices=list(SUPPORTED_LANGUAGES.keys()),
        help=f"Target language: {', '.join([f'{k} ({v})' for k, v in SUPPORTED_LANGUAGES.items()])}"
    )
    
    parser.add_argument(
        "--batch", "-b",
        action="store_true",
        help="Process a directory of files instead of a single file"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    
    parser.add_argument(
        "--model",
        default="gpt-4",
        choices=["gpt-4", "gpt-3.5-turbo", "llama-2", "palm", "custom"],
        help="Translation model to use"
    )
    
    parser.add_argument(
        "--log-file",
        help="Path to log file. If not specified, logs to console only."
    )
    
    parser.add_argument(
        "--max-files",
        type=int,
        default=100,
        help="Maximum number of files to process in batch mode"
    )
    
    return parser.parse_args()

def validate_input(input_path: str, is_batch: bool) -> bool:
    """
    Validate that the input file or directory exists.
    
    Args:
        input_path: Path to input file or directory
        is_batch: Whether to process as batch
        
    Returns:
        True if valid, False otherwise
    """
    path = Path(input_path)
    logger = get_logger()
    
    if is_batch:
        if not path.is_dir():
            logger.error(f"Batch mode specified but {input_path} is not a directory")
            return False
        if not any(p.suffix.lower() in ('.pdf', '.docx') for p in path.iterdir()):
            logger.error(f"No PDF or DOCX files found in {input_path}")
            return False
    else:
        if not path.exists():
            logger.error(f"Input file {input_path} does not exist")
            return False
        if path.suffix.lower() not in ('.pdf', '.docx'):
            logger.error(f"Input file {input_path} is not a PDF or DOCX file")
            return False
    
    return True

def process_file(input_path: str, output_path: Optional[str], target_lang: str, model: str) -> bool:
    """
    Process a single file.
    
    Args:
        input_path: Path to input file
        output_path: Path to output file
        target_lang: Target language code
        model: Translation model to use
        
    Returns:
        True if successful, False otherwise
    """
    logger = get_logger()
    
    try:
        input_path = Path(input_path)
        
        # Auto-generate output path if not specified
        if not output_path:
            output_dir = input_path.parent
            output_filename = f"{input_path.stem}_{target_lang}{input_path.suffix}"
            output_path = str(output_dir / output_filename)
        
        logger.info(f"Processing {input_path} to {output_path} in {SUPPORTED_LANGUAGES[target_lang]}")
        
        start_time = time.time()
        
        # Initialize core components
        logger.debug("Initializing document processor")
        doc_processor = DocumentProcessor(str(input_path))
        
        logger.debug("Initializing translation service")
        translation_service = TranslationService(target_lang=target_lang, model=model)
        
        logger.debug("Initializing document rebuilder")
        doc_rebuilder = DocumentRebuilder()
        
        logger.debug("Initializing validator")
        validator = OutputValidator()
        
        # Process document
        logger.info("Extracting document components")
        doc_components = doc_processor.process()
        
        # Translate components
        logger.info("Translating document components")
        translated_components = translation_service.translate(doc_components)
        
        # Rebuild document
        logger.info("Rebuilding document with translated content")
        rebuilt_doc = doc_rebuilder.rebuild(translated_components, output_format=input_path.suffix[1:])
        
        # Validate output
        logger.info("Validating translation")
        validation_result = validator.validate(doc_components, rebuilt_doc)
        if not validation_result['success']:
            logger.warning(f"Validation issues: {validation_result['issues']}")
        
        # Save output
        logger.info(f"Saving output to {output_path}")
        rebuilt_doc.save(output_path)
        
        elapsed_time = time.time() - start_time
        logger.info(f"Successfully translated to {output_path} in {elapsed_time:.1f} seconds")
        logger.info(f"Validation score: {validation_result['score']:.2f}/10")
        
        return True
    
    except Exception as e:
        logger.error(f"Error processing file {input_path}: {str(e)}", exc_info=True)
        return False

def process_batch(input_dir: str, target_lang: str, model: str, max_files: int) -> List[str]:
    """
    Process a batch of files in a directory.
    
    Args:
        input_dir: Path to input directory
        target_lang: Target language code
        model: Translation model to use
        max_files: Maximum number of files to process
        
    Returns:
        List of successfully processed file paths
    """
    logger = get_logger()
    input_dir = Path(input_dir)
    successful_files = []
    
    # Get all PDF and DOCX files
    files = [p for p in input_dir.iterdir() if p.suffix.lower() in ('.pdf', '.docx')]
    
    if len(files) > max_files:
        logger.warning(f"Found {len(files)} files, but will only process the first {max_files}")
        files = files[:max_files]
    
    logger.info(f"Found {len(files)} files to process")
    
    for i, file_path in enumerate(files):
        logger.info(f"Processing file {i+1}/{len(files)}: {file_path}")
        output_path = str(input_dir / f"{file_path.stem}_{target_lang}{file_path.suffix}")
        
        if process_file(str(file_path), output_path, target_lang, model):
            successful_files.append(str(file_path))
        else:
            logger.error(f"Failed to process {file_path}")
    
    return successful_files

def main():
    """Main entry point for the CLI."""
    args = parse_args()
    
    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    setup_logger(log_level, args.log_file)
    logger = get_logger()
    
    logger.info("AutoWealthTranslate starting")
    logger.info(f"Using translation model: {args.model}")
    
    # Validate input
    if not validate_input(args.input, args.batch):
        sys.exit(1)
    
    # Process files
    try:
        if args.batch:
            logger.info(f"Processing batch from directory: {args.input}")
            successful_files = process_batch(args.input, args.lang, args.model, args.max_files)
            total_files = len([p for p in Path(args.input).iterdir() 
                              if p.suffix.lower() in ('.pdf', '.docx')])
            
            logger.info(f"Successfully processed {len(successful_files)}/{min(total_files, args.max_files)} files")
            
            if len(successful_files) != min(total_files, args.max_files):
                logger.warning("Some files failed to process. Check the log for details.")
                sys.exit(1)
        else:
            if not process_file(args.input, args.output, args.lang, args.model):
                logger.error("Failed to process file. Check the log for details.")
                sys.exit(1)
        
        logger.info("All tasks completed successfully")
    
    except KeyboardInterrupt:
        logger.info("Process interrupted by user")
        sys.exit(130)
    
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
