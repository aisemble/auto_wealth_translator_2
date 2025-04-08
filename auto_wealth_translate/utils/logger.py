"""
Logging utilities for AutoWealthTranslate.
"""

import logging
import colorlog
import sys
import os
from pathlib import Path

# Create logs directory if it doesn't exist
logs_dir = Path(__file__).parents[2] / "logs"
logs_dir.mkdir(exist_ok=True)

# Configure default logging
DEFAULT_LOG_LEVEL = logging.INFO
DEFAULT_LOG_FORMAT = "%(log_color)s%(asctime)s [%(levelname)s] %(name)s: %(message)s%(reset)s"

def setup_logger(level=DEFAULT_LOG_LEVEL, log_to_file=True):
    """
    Set up the root logger with color output.
    
    Args:
        level: Logging level (default: INFO)
        log_to_file: Whether to log to a file (default: True)
    """
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    
    # Clear existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Set up console handler with color formatting
    console_handler = colorlog.StreamHandler(stream=sys.stdout)
    console_handler.setLevel(level)
    
    color_formatter = colorlog.ColoredFormatter(
        DEFAULT_LOG_FORMAT,
        log_colors={
            'DEBUG': 'cyan',
            'INFO': 'green',
            'WARNING': 'yellow',
            'ERROR': 'red',
            'CRITICAL': 'red,bg_white',
        }
    )
    
    console_handler.setFormatter(color_formatter)
    root_logger.addHandler(console_handler)
    
    # Add file handler if requested
    if log_to_file:
        log_file = logs_dir / "auto_wealth_translate.log"
        file_handler = logging.FileHandler(str(log_file))
        file_handler.setLevel(level)
        file_formatter = logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
        )
        file_handler.setFormatter(file_formatter)
        root_logger.addHandler(file_handler)
    
    # Suppress overly verbose loggers
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    logging.getLogger('PIL').setLevel(logging.WARNING)
    logging.getLogger('pdfminer').setLevel(logging.WARNING)
    
    return root_logger

def get_logger(name):
    """
    Get a logger instance with the given name.
    
    Args:
        name: Name of the logger
        
    Returns:
        Logger instance
    """
    return logging.getLogger(name)
