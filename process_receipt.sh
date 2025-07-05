#!/bin/bash
# Receipt Processing Script
# Automatically sets credentials and runs the pipeline

# Get the directory where this script is located
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# Set the Google Application Credentials
export GOOGLE_APPLICATION_CREDENTIALS="$DIR/splendid-yeti-464913-j2-83b565740502.json"

# Check if user wants to run the chatbot
if [ "$1" = "--chatbot" ] || [ "$1" = "-c" ]; then
    echo "🤖 Starting Receipt Chatbot..."
    python3 "$DIR/gemini.py"
    exit $?
fi

# Check if user wants to test the chatbot
if [ "$1" = "--test-chatbot" ] || [ "$1" = "-t" ]; then
    echo "🧪 Testing Receipt Chatbot..."
    python3 "$DIR/test_chatbot.py"
    exit $?
fi

# Check if user wants to show all results
if [ "$1" = "--show-all" ] || [ "$1" = "-a" ]; then
    python3 "$DIR/receipt_pipeline.py" --show-all "${@:2}"
    exit $?
fi

# Check if input file is provided
if [ $# -eq 0 ]; then
    echo "Usage: $0 <receipt_image> [options]"
    echo "       $0 --show-all [options]"
    echo "       $0 --chatbot"
    echo "       $0 --test-chatbot"
    echo ""
    echo "Examples:"
    echo "  $0 33.jpg                    # Process receipt"
    echo "  $0 33.jpg --output my.json   # Process with custom output"
    echo "  $0 --show-all                # Show all stored receipts"
    echo "  $0 -a                        # Show all (short form)"
    echo "  $0 --chatbot                 # Start chatbot server"
    echo "  $0 -c                        # Start chatbot (short form)"
    echo "  $0 --test-chatbot            # Test chatbot functionality"
    echo "  $0 -t                        # Test chatbot (short form)"
    exit 1
fi

# Run the pipeline
python3 "$DIR/receipt_pipeline.py" --input "$1" "${@:2}" 