#!/bin/bash

# Diffusion Model Project - Installation Script
# This script automates the installation process

echo "=================================="
echo "Diffusion Model Project Installer"
echo "=================================="
echo ""

# Check Python version
echo "Checking Python installation..."
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
    echo "✓ Python $PYTHON_VERSION found"
else
    echo "✗ Python 3 not found. Please install Python 3.8 or newer."
    exit 1
fi

# Check pip
echo ""
echo "Checking pip installation..."
if command -v pip3 &> /dev/null; then
    PIP_VERSION=$(pip3 --version 2>&1 | awk '{print $2}')
    echo "✓ pip $PIP_VERSION found"
else
    echo "✗ pip not found. Installing pip..."
    python3 -m ensurepip --default-pip
fi

# Install requirements
echo ""
echo "Installing Python dependencies..."
echo "(This may take 2-3 minutes)"
echo ""
pip3 install -r requirements.txt

# Verify installation
echo ""
echo "Verifying installation..."
python3 -c "import numpy, matplotlib, pandas, jupyter" 2>&1

if [ $? -eq 0 ]; then
    echo ""
    echo "=================================="
    echo "✓ Installation successful!"
    echo "=================================="
    echo ""
    echo "Next steps:"
    echo "1. Run: jupyter notebook diffusion_model_demo.ipynb"
    echo "2. In the browser, click: Cell → Run All"
    echo "3. Wait for generation to complete"
    echo "4. Find your art in the outputs/ folder"
    echo ""
    echo "For help, read:"
    echo "- QUICK_START.md (for beginners)"
    echo "- README.md (for general info)"
    echo "- TECHNICAL_DOCS.md (for details)"
    echo ""
else
    echo ""
    echo "✗ Installation verification failed"
    echo "Please check error messages above"
    exit 1
fi
