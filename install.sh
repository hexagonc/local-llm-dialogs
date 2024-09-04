#!/bin/bash

set -e

# Function to check the status of the last command and exit if it failed
check_status() {
    if [ $? -ne 0 ]; then
        echo "Error: $1"
        exit 1
    fi
}

# Check if 'venv' directory exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
    check_status "Failed to create virtual environment."
fi

# Activate the virtual environment if not already activated
if [ -z "$VIRTUAL_ENV" ]; then
    echo "Activating virtual environment..."
    source venv/bin/activate
    check_status "Failed to activate virtual environment."
fi

# Check if requirements.txt exists
if [ ! -f "requirements.txt" ]; then
    echo "requirements.txt not found."
    exit 1
fi

# Upgrade pip to avoid warnings
echo "Upgrading pip..."
python3 -m pip install --upgrade pip
check_status "Failed to upgrade pip."

# Check installed packages and install missing ones
echo "Checking installed packages..."
pip freeze > installed_packages.txt
check_status "Failed to list installed packages."

# Temporarily disable 'set -e' to handle the return status manually
set +e
python3 check_requirements.py
REQUIREMENTS_STATUS=$?
set -e

if [ $REQUIREMENTS_STATUS -ne 0 ]; then
    echo "Installing missing packages..."
    pip install -r requirements.txt
    check_status "Failed to install required packages."
    echo "Packages installed successfully."
else
    echo "All required packages are already installed."
fi

# Clean up
rm installed_packages.txt
check_status "Failed to remove installed_packages.txt."
echo "Stop tracking changes to config.json so that it can be updated with local overrides"
git update-index --assume-unchanged config.json
echo "Setup complete."