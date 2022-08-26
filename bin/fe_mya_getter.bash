#!/bin/env bash

# Get the directory containing this script
DIR="$( cd "$( dirname "$(readlink -f "${BASH_SOURCE[0]}")" )" >/dev/null 2>&1 && pwd )"

# Load the virtual environment
source ${DIR}/../venv/bin/activate

# Run the app passing along all of the args
python3 ${DIR}/../src/fe_mya_getter.py "$@"

# Unload the virtual environment
deactivate