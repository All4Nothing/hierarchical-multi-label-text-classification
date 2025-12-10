#!/bin/bash

# TELEClass Pipeline Execution Script
# This script runs the complete pipeline for hierarchical multi-label classification

echo "=========================================="
echo "TELEClass Pipeline Execution"
echo "=========================================="

# Check if requirements are installed
echo "Checking dependencies..."
python -c "import torch; import transformers; import sentence_transformers; import networkx" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "Installing required packages..."
    pip install -r requirements_teleclass.txt
fi

# Set environment variables
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0  # Adjust GPU ID as needed

# Run pipeline
echo ""
echo "Starting pipeline..."
python pipeline_teleclass.py

echo ""
echo "=========================================="
echo "Pipeline execution complete!"
echo "Check outputs/ directory for results:"
echo "  - outputs/models/best_model/      : Best trained model"
echo "  - outputs/submission.csv          : Kaggle submission file"
echo "  - outputs/intermediate/           : Intermediate results"
echo "=========================================="
