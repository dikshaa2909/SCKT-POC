#!/bin/bash

# Setup script for ScanCode ML Required Phrase Marking PoC
# This script helps integrate the standalone PoC into your local scancode-toolkit fork.

echo "🚀 Setting up ScanCode ML PoC..."

# 1. Verify we are in a ScanCode Toolkit clone
if [ ! -d "src/licensedcode" ]; then
    echo "❌ Error: Please run this script from the root of your scancode-toolkit clone."
    exit 1
fi

# 2. Create the target directory if it doesn't exist
mkdir -p src/licensedcode/ml_required_phrases/

# 3. Copy the ML module files
echo "📦 Copying ML module files to src/licensedcode/ml_required_phrases/..."
cp -r ../gsoc-ml-poc/ml_required_phrases/* src/licensedcode/ml_required_phrases/

# 4. Install dependencies
echo "🐍 Installing ML dependencies..."
pip install -r ../gsoc-ml-poc/requirements.txt

echo "✅ Setup complete! You can now run the pipeline using:"
echo "python3 src/licensedcode/ml_required_phrases/run_pipeline.py --help"
