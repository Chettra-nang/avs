#!/bin/bash
# Convenient script to watch corner and circle scenarios with venv activated

# Get the script's directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/../.." && pwd )"

# Activate virtual environment
source "$PROJECT_ROOT/avs_venv/bin/activate"

# Run the Python script with all arguments passed through
python3 "$SCRIPT_DIR/watch_corner_and_circle.py" "$@"

# Deactivate when done
deactivate
