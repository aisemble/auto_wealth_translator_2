#!/bin/bash

# Install dependencies if not already installed
if ! command -v streamlit &> /dev/null
then
    echo "Installing dependencies..."
    pip install -e .
fi

# Set environment variables if needed
# export OPENAI_API_KEY=your_api_key_here

# Run the Streamlit app
echo "Starting AutoWealthTranslate Streamlit app..."
streamlit run streamlit_app.py 