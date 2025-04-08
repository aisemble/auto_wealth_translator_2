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

# Create and activate virtual environment
echo "Creating Python virtual environment..."
python3 -m venv venv
source venv/bin/activate || {
    echo "Failed to create/activate virtual environment"
    exit 1
}

# Upgrade pip to latest version
echo "Upgrading pip..."
python -m pip install --upgrade pip || {
    echo "Failed to upgrade pip"
    exit 1
}

# Update package lists
echo "Updating package lists..."
if [[ "$OSTYPE" == "darwin"* ]]; then
    # macOS specific commands
    brew update || {
        echo "Failed to update Homebrew"
        exit 1
    }
else
    # Linux specific commands
    sudo apt-get update || {
        echo "Failed to update package lists"
        exit 1
    }
fi

# Install system dependencies
echo "Installing system dependencies..."
while read -r package; do
    if [ -n "$package" ] && [[ ! "$package" =~ ^# ]]; then
        echo "Installing $package..."
        if [[ "$OSTYPE" == "darwin"* ]]; then
            brew install "$package" || {
                echo "Failed to install $package, trying to continue..."
            }
        else
            sudo apt-get install -y "$package" || {
                echo "Failed to install $package, trying to continue..."
            }
        fi
    fi
done < packages.txt

# Install Python dependencies with specific versions
echo "Installing Python dependencies..."
# First uninstall potentially conflicting packages
pip uninstall -y fitz pymupdf || true

# Install core dependencies first
pip install wheel setuptools || {
    echo "Failed to install core dependencies"
    exit 1
}

# Install PyMuPDF and its dependencies with specific versions
pip install "PyMuPDF==1.23.3" || {
    echo "Failed to install PyMuPDF"
    exit 1
}

pip install -r requirements-streamlit.txt || {
    echo "Failed to install remaining Python dependencies"
    exit 1
}

# Verify installation
echo "Verifying installation..."
python -c "import pymupdf; import PIL; import streamlit" || {
    echo "Failed to verify installation. Try importing packages individually to identify the problem:"
    echo "Testing PyMuPDF..."
    python -c "import pymupdf" || echo "PyMuPDF import failed"
    echo "Testing Pillow..."
    python -c "import PIL" || echo "Pillow import failed"
    echo "Testing Streamlit..."
    python -c "import streamlit" || echo "Streamlit import failed"
    exit 1
}

echo "Installation completed successfully!" 