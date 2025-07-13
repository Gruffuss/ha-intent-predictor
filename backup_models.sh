#!/bin/bash

# Automatic model backup script
# Run this after training to backup models to GitHub

echo "Backing up models to GitHub..."

# Check if models directory exists and has content
if [ -d "models" ] && [ "$(ls -A models)" ]; then
    # Add and commit model files
    git add models/
    git add -A  # Add any other changes too
    
    # Create commit with timestamp
    timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    git commit -m "Backup trained models - $timestamp

🤖 Automated model backup

Co-Authored-By: Claude <noreply@anthropic.com>"
    
    # Push to GitHub
    git push origin master
    
    echo "✅ Models backed up successfully at $timestamp"
else
    echo "❌ No models found in models/ directory"
fi