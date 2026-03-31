#!/bin/bash

# Setup script for ScanCode ML Required Phrase Marking PoC
# Integrates the standalone PoC into your local scancode-toolkit clone.

# Resolve the absolute path of this PoC repository
POC_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

if [ -z "$1" ]; then
    echo "Usage: ./setup_local.sh /path/to/scancode-toolkit"
    exit 1
fi

SCANCODE_DIR="$1"

if [ ! -d "$SCANCODE_DIR/src/licensedcode" ]; then
    echo "❌ Error: The provided path does not seem to be a valid scancode-toolkit clone."
    exit 1
fi

echo "🚀 Setting up ScanCode ML PoC..."

echo "📦 Copying ML module files into scancode-toolkit source..."
mkdir -p "$SCANCODE_DIR/src/licensedcode/ml_required_phrases/"
cp -r "$POC_DIR/ml_required_phrases/"* "$SCANCODE_DIR/src/licensedcode/ml_required_phrases/"

echo "📦 Copying demo rules to ST root..."
cp -r "$POC_DIR/demo_rules" "$SCANCODE_DIR/"

echo "🐍 Checking python environment..."
if [ -f "$SCANCODE_DIR/venv/bin/pip" ]; then
    echo "   Using standard ScanCode configure venv..."
    "$SCANCODE_DIR/venv/bin/pip" install -r "$POC_DIR/requirements.txt"
else
    echo "   Assuming virtualenv is active via pip..."
    pip install -r "$POC_DIR/requirements.txt"
fi

echo "✅ Setup complete!"
echo ""
echo "You can now navigate to your ScanCode Toolkit directory and run the model."
echo "cd $SCANCODE_DIR"
echo "python3 src/licensedcode/ml_required_phrases/run_pipeline.py --help"
