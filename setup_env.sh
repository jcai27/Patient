#!/bin/bash
# Setup script for the persona chatbot project

# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Create necessary directories
mkdir -p persona
mkdir -p data
mkdir -p eval/datasets

echo "Setup complete! Don't forget to:"
echo "1. Create a .env file with your API keys"
echo "2. Run 'source venv/bin/activate' to activate the virtual environment"
echo "3. Run 'python run_server.py' to start the server"

