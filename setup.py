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
    ],
    python_requires=">=3.8",
)
