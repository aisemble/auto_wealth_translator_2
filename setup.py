from setuptools import setup, find_packages

setup(
    name="auto_wealth_translate",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        # PDF Processing
        "PyMuPDF>=1.21.1",
        "pdfplumber>=0.7.6",
        "pdf2image>=1.16.3",
        "reportlab>=4.0.4",
        "WeasyPrint>=59.0",
        
        # DOCX Processing
        "python-docx>=0.8.11",
        
        # OCR
        "pytesseract>=0.3.10",
        "opencv-python>=4.7.0.72",
        
        # Image and Chart Processing
        "numpy>=1.24.3",
        "matplotlib>=3.7.1",
        
        # Translation and NLP
        "langchain>=0.0.267",
        "openai>=1.1.1",
        "tiktoken>=0.5.1",
        
        # API
        "fastapi>=0.95.1",
        "uvicorn>=0.22.0",
        "python-multipart>=0.0.6",
        
        # Streamlit Interface
        "streamlit>=1.25.0",
        "watchdog>=3.0.0",
        
        # Utilities
        "tqdm>=4.65.0",
        "colorlog>=6.7.0",
    ],
    entry_points={
        "console_scripts": [
            "auto-wealth-translate=auto_wealth_translate.cli:main",
        ],
    },
    author="AutoWealthTranslate Team",
    author_email="support@autowealthtranslate.com",
    description="Automatically translate wealth plan reports while preserving formatting",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/autowealthtranslate/auto-wealth-translate",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Financial and Insurance Industry",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.8",
)
