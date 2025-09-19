#!/bin/bash
set -e

# Change to the app directory
cd /app

# Train the model if it doesn't exist
if [ ! -d "models" ] || [ -z "$(ls -A models)" ]; then
    echo "No model found. Training a new model..."
    rasa train --quiet
fi

# Start the Rasa server
echo "Starting Rasa server..."
exec rasa run --enable-api --cors "*" --debug --endpoints endpoints.yml
