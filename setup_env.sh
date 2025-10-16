#!/usr/bin/env bash

#This script sets up a Python virtual environment for the project and installs required dependencies.
#Exit if any command fails
set -e

# Create virtual environment in the 'venv' folder
python3 -m venv gnn_venv

# Activate the virtual environment
source gnn_venv/bin/activate

# Upgrade pip to latest version
pip install --upgrade pip

# Install requirements
pip install -r requirements.txt

echo "Virtual environment is set up and required packages are installed."