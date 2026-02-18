#!/bin/sh
set -e

# Start the Ollama server in the background
echo "Starting Ollama server..."
ollama serve &

# Optionally, pull or load a model at startup (uncomment if needed)
# echo "Pulling llama2 model..."
# ollama pull llama2

# Wait for background processes
wait
