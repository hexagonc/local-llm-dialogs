#!/bin/bash

# Check for --help flag
if [[ "$1" == "--help" ]]; then
    cat help.txt
    exit 0
fi

# Activate the virtual environment
source venv/bin/activate

# Run the chat runner script
python chat_runner.py