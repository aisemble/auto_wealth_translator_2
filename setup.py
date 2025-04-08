from setuptools import setup, find_packages

setup(
    name="auto_wealth_translate",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "streamlit>=1.32.0",
        "deepl>=1.15.0",
        "python-docx>=1.1.0",
        "PyPDF2>=3.0.0",
        "python-magic>=0.4.27",
        "python-magic-bin>=0.4.14; sys_platform == 'win32'",
        "PyMuPDF>=1.21.1",
        "pdfplumber>=0.7.6",
        "pdf2image>=1.16.3",
        "reportlab>=4.0.4",
        "WeasyPrint>=59.0",
        "pytesseract>=0.3.10",
        "opencv-python>=4.7.0.72",
        "numpy>=1.24.3",
        "matplotlib>=3.7.1",
        "Pillow>=9.5.0",
        "langchain>=0.0.267",
        "openai>=1.1.1",
        "tiktoken>=0.5.1",
        "fastapi>=0.95.1",
        "uvicorn>=0.22.0",
        "python-multipart>=0.0.6",
        "tqdm>=4.65.0",
        "colorlog>=6.7.0",
    ],
    setup_requires=[
        "setuptools>=69.0.0",
        "wheel>=0.42.0",
        "build>=1.0.3",
    ],
    python_requires=">=3.8",
)
