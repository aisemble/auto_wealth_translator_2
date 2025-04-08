from setuptools import setup, find_packages

setup(
    name="auto_wealth_translate",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "streamlit>=1.24.0",
        "deepl>=1.15.0",
        "python-docx>=0.8.11",
        "PyPDF2>=3.0.0",
        "python-magic>=0.4.24",
        "python-magic-bin>=0.4.14; sys_platform == 'win32'",
        "PyMuPDF==1.21.1",
        "pdfplumber>=0.7.0,<0.8.0",
        "pdf2image>=1.16.0",
        "reportlab>=3.6.0",
        "WeasyPrint>=54.0",
        "pytesseract>=0.3.8",
        "opencv-python-headless>=4.6.0",
        "numpy>=1.21.0,<2.0.0",
        "matplotlib>=3.5.0",
        "Pillow==9.5.0",
        "langchain>=0.0.235",
        "openai>=1.0.0",
        "tiktoken>=0.4.0",
        "fastapi>=0.95.0",
        "uvicorn>=0.20.0",
        "python-multipart>=0.0.6",
        "tqdm>=4.60.0",
        "colorlog>=6.7.0",
    ],
    setup_requires=[
        "setuptools>=42.0.0",
        "wheel>=0.36.0",
        "build>=0.7.0",
    ],
    python_requires=">=3.9, <3.13",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
