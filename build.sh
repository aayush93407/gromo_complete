#!/usr/bin/env bash

# Render custom build script
echo "Installing system packages..."
apt-get update && apt-get install -y ffmpeg

echo "Installing Python dependencies..."
pip install -r requirements.txt

echo "Downloading SpaCy model..."
python -m spacy download en_core_web_sm
