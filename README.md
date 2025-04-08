# AutoWealthTranslate

Automatically translate wealth plan reports (PDF/Word) into client-preferred languages while preserving formatting, charts, and tables.

## Overview

AutoWealthTranslate is designed for wealth managers, financial advisors, and private banking institutions who need to translate complex financial documents while maintaining the integrity of financial terminology and document structure.

## Features

- **Accurate Translation**: Maintains financial terminology integrity
- **Layout Preservation**: Replicates original document structure (headers, footers, tables, charts)
- **Multi-Format Support**: Processes PDF and Word documents
- **Scalability**: Handles batch processing of multiple reports
- **User-Friendly**: Provides CLI, Web UI and API interfaces

## Installation

```bash
# Clone the repository
git clone https://github.com/autowealthtranslate/auto-wealth-translate.git
cd auto-wealth-translate

# Install the package
pip install -e .
```

## Usage

### Streamlit Web Interface

```bash
# Run the Streamlit app
streamlit run streamlit_app.py
```

This will open a web browser with a user-friendly interface where you can:
- Upload PDF or Word documents
- Select the target language
- Choose a translation model
- Download translated documents

### Command Line Interface

```bash
# Translate a single file
auto-wealth-translate --input report.pdf --lang fr --output report_fr.pdf

# Batch process a directory
auto-wealth-translate --input reports_dir/ --lang zh --batch
```

### Python API

```python
from auto_wealth_translate.core.document_processor import DocumentProcessor
from auto_wealth_translate.core.translator import TranslationService
from auto_wealth_translate.core.document_rebuilder import DocumentRebuilder

# Process document
doc_processor = DocumentProcessor("report.pdf")
doc_components = doc_processor.process()

# Translate components
translation_service = TranslationService(target_lang="fr", model="gpt-4")
translated_components = translation_service.translate(doc_components)

# Rebuild document
doc_rebuilder = DocumentRebuilder()
output_doc = doc_rebuilder.rebuild(translated_components, output_format="pdf")
output_doc.save("report_fr.pdf")
```

### Web API

The application also provides a RESTful API that can be accessed through HTTP requests:

```bash
# Start the API server
uvicorn auto_wealth_translate.api:app --host 0.0.0.0 --port 8000
```

Then you can make requests to the API:

```bash
curl -X POST "http://localhost:8000/translate" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@report.pdf" \
  -F "target_lang=zh" \
  -o translated_report.pdf
```

## Supported Languages

AutoWealthTranslate currently supports the following languages:

- English (en)
- Chinese (zh)
- French (fr)
- Spanish (es)
- German (de)
- Japanese (ja)
- Korean (ko)
- Russian (ru)
- Arabic (ar)
- Italian (it)
- Portuguese (pt)

## Technical Details

### Document Processing Flow

1. **Document Processing**: Extract text, tables, images, and layout information
2. **Translation**: Chunk text and translate while preserving financial terminology
3. **Document Rebuilding**: Reconstruct the document with translated content
4. **Validation**: Ensure completeness and formatting accuracy

### Key Components

- **Document Processor**: Extracts content from PDFs and DOCXs
- **Translation Service**: Handles translation using LLMs
- **Document Rebuilder**: Reconstructs documents with translated content
- **Validator**: Ensures translation quality and layout preservation

## Deployment Options

### Local Deployment

Run the application locally with Streamlit:

```bash
streamlit run streamlit_app.py
```

### Cloud Deployment

The Streamlit app can be deployed to Streamlit Cloud, Heroku, or any other platform that supports Python applications.

For Streamlit Cloud, simply push the code to a GitHub repository and connect it to your Streamlit Cloud account.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
