#!/bin/bash

# TaxoClass Training Script

echo "=========================================="
echo "   TaxoClass Framework - Training"
echo "=========================================="

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install requirements
echo "Installing requirements..."
pip install -r requirements.txt

# Run main pipeline
echo ""
echo "Starting TaxoClass pipeline..."
echo ""

python main.py

echo ""
echo "=========================================="
echo "   Training Complete!"
echo "=========================================="
echo ""
echo "Results saved to:"
echo "  - Models: ./saved_models/"
echo "  - Metrics: ./outputs/"
echo "  - Cache: ./cache/"

