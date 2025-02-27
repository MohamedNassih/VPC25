#!/bin/bash

##############################################
# DO NOT MODIFY THIS FILE
##############################################

# Script for running the evaluation of your anonymization algorithm
##############################################

run() {
    echo "Running evaluation ..."

    # Determine the correct Python command
    PYTHON_CMD=$(command -v python3.12 || command -v python || command -v python3)
    if [ -z "$PYTHON_CMD" ]; then
        echo "Python not found! Please install Python 3.12"
        exit 1
    fi

    # Check if virtual environment exists
    if [ ! -d ".venv" ]; then
        echo "Virtual environment not found! Creating one..."
        $PYTHON_CMD -m venv .venv
    fi

    # Activate virtual environment (different paths for Windows and Unix)
    if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
        source .venv/Scripts/activate
    else
        source .venv/Scripts/activate

    fi

    # Ensure required packages are installed
    python -m pip install --upgrade pip
    pip install -r requirements.txt -q

    # Set evaluation directory (default: "evaluation_data/")
    EVAL_DIR=${1:-"evaluation_data/"}

    # Set second argument (default: "default_value")
    ASR_MODEL_ID=${2:-"facebook/wav2vec2-base-960h"}

    # Run the evaluation with the specified directory
    python evaluation.py "$EVAL_DIR" "$ASR_MODEL_ID"

    echo "... end of evaluation run"
}

run "$@"
