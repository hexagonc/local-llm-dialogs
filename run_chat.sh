#!/bin/bash

# Check for --help flag
if [[ "$1" == "--help" ]]; then
    cat help.txt
    exit 0
fi

# Activate the virtual environment
source venv/bin/activate

# Run the chat runner script
# Default command to run the chat runner script
CMD="python chat_runner.py"

# Check for --dialog-pattern-file argument
if [[ "$1" == --dialog-pattern-file=* ]]; then
    # Extract the file path from the argument
    FILE_PATH="${1#*=}"
    CMD="$CMD --dialog-pattern-file $FILE_PATH"
elif [[ "$1" == "--dialog-pattern-file" && -n "$2" ]]; then
    CMD="$CMD --dialog-pattern-file $2"
fi

# Run the command
$CMD