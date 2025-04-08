#!/bin/bash

# Exit on error
set -e

# Function to handle errors
handle_error() {
    echo "Error occurred on line $1"
    echo "Error message: $2"
    exit 1
}

# Function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to verify Python package installation
verify_python_package() {
    python3 -c "import $1" 2>/dev/null
    return $?
}

trap 'handle_error $LINENO "$BASH_COMMAND"' ERR

# Check for required commands
echo "Checking system requirements..."
for cmd in python3 pip3 brew; do
    if ! command_exists "$cmd"; then
        echo "Error: $cmd is not installed or not in PATH"
        if [ "$cmd" = "brew" ]; then
            echo "Please install Homebrew first:"
            echo "/bin/bash -c \"\$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)\""
        fi
        exit 1
    fi
done

# Update Homebrew
echo "Updating Homebrew..."
for i in {1..3}; do
    if brew update; then
        break
    fi
    if [ $i -eq 3 ]; then
        echo "Failed to update Homebrew after 3 attempts"
        exit 1
    fi
    echo "Retrying Homebrew update..."
    sleep 5
done

# Install system dependencies with better error handling
echo "Installing system dependencies..."
while read -r package; do
    if [ -n "$package" ] && [[ ! "$package" =~ ^# ]]; then
        echo "Installing $package..."
        for i in {1..3}; do
            if brew install "$package"; then
                break
            fi
            if [ $i -eq 3 ]; then
                echo "Failed to install $package after 3 attempts"
                echo "Continuing with other packages..."
            fi
            echo "Retrying installation of $package..."
            sleep 5
        done
    fi
done < packages.txt

# Create and activate virtual environment
echo "Creating virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Upgrade pip with retry
echo "Upgrading pip..."
for i in {1..3}; do
    if pip install --upgrade pip; then
        break
    fi
    if [ $i -eq 3 ]; then
        echo "Failed to upgrade pip after 3 attempts"
        exit 1
    fi
    echo "Retrying pip upgrade..."
    sleep 5
done

# Install Python dependencies with retry
echo "Installing Python dependencies..."
for i in {1..3}; do
    if pip install -r requirements-streamlit.txt; then
        break
    fi
    if [ $i -eq 3 ]; then
        echo "Failed to install Python dependencies after 3 attempts"
        exit 1
    fi
    echo "Retrying Python dependencies installation..."
    sleep 5
done

# Verify critical package installations
echo "Verifying installation..."
for package in fitz PIL streamlit; do
    if ! verify_python_package "$package"; then
        echo "Error: Failed to verify $package installation"
        exit 1
    fi
done

echo "Installation completed successfully!"
echo "To activate the virtual environment, run: source venv/bin/activate" 