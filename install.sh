#!/bin/bash

# Exit on error
set -e

# Function to handle errors
handle_error() {
    echo "Error occurred on line $1"
    echo "Error message: $2"
    exit 1
}

trap 'handle_error $LINENO "$BASH_COMMAND"' ERR

# Update package lists
echo "Updating package lists..."
sudo apt-get update

# Install system dependencies
echo "Installing system dependencies..."
while read -r package; do
    if [ -n "$package" ] && [[ ! "$package" =~ ^# ]]; then
        echo "Installing $package..."
        sudo apt-get install -y "$package" || {
            echo "Failed to install $package, trying to continue..."
        }
    fi
done < packages.txt

# Install Python dependencies
echo "Installing Python dependencies..."
pip install --upgrade pip
pip install -r requirements-streamlit.txt || {
    echo "Failed to install Python dependencies"
    exit 1
}

# Verify installation
echo "Verifying installation..."
python -c "import fitz; import PIL; import streamlit" || {
    echo "Failed to verify installation"
    exit 1
}

echo "Installation completed successfully!" 