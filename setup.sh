#!/bin/bash

# VehicleBERT Quick Setup Script
# This script sets up the VehicleBERT project from scratch

set -e  # Exit on error

echo "=========================================="
echo "  VehicleBERT Setup Script"
echo "=========================================="
echo ""

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Check Python version
echo -e "${BLUE}Checking Python version...${NC}"
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "Python version: $python_version"

if ! python3 -c 'import sys; assert sys.version_info >= (3,8)' 2>/dev/null; then
    echo "Error: Python 3.8 or higher is required"
    exit 1
fi
echo -e "${GREEN}✓ Python version OK${NC}"
echo ""

# Create virtual environment
echo -e "${BLUE}Creating virtual environment...${NC}"
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo -e "${GREEN}✓ Virtual environment created${NC}"
else
    echo -e "${GREEN}✓ Virtual environment already exists${NC}"
fi
echo ""

# Activate virtual environment
echo -e "${BLUE}Activating virtual environment...${NC}"
source venv/bin/activate
echo -e "${GREEN}✓ Virtual environment activated${NC}"
echo ""

# Upgrade pip
echo -e "${BLUE}Upgrading pip...${NC}"
pip install --upgrade pip --quiet
echo -e "${GREEN}✓ Pip upgraded${NC}"
echo ""

# Install requirements
echo -e "${BLUE}Installing dependencies...${NC}"
echo "This may take a few minutes..."
pip install -r requirements.txt --quiet
echo -e "${GREEN}✓ Dependencies installed${NC}"
echo ""

# Install in development mode
echo -e "${BLUE}Installing VehicleBERT package...${NC}"
pip install -e . --quiet
echo -e "${GREEN}✓ Package installed${NC}"
echo ""

# Create necessary directories
echo -e "${BLUE}Creating directories...${NC}"
mkdir -p data/annotated data/raw models
echo -e "${GREEN}✓ Directories created${NC}"
echo ""

# Generate sample data
echo -e "${BLUE}Generating sample dataset...${NC}"
echo "Creating 1,000 sample sentences for quick testing..."
python scripts/prepare_data.py --num-sentences 1000
echo -e "${GREEN}✓ Sample data generated${NC}"
echo ""

# Run tests
echo -e "${BLUE}Running tests...${NC}"
if command -v pytest &> /dev/null; then
    pytest tests/test_model.py -v
    echo -e "${GREEN}✓ Tests passed${NC}"
else
    echo "pytest not found, skipping tests"
fi
echo ""

# Summary
echo "=========================================="
echo -e "${GREEN}  Setup Complete!${NC}"
echo "=========================================="
echo ""
echo "Quick Start Commands:"
echo ""
echo "1. Train a model (quick test - 3 epochs):"
echo "   python scripts/train.py --epochs 3 --batch-size 8"
echo ""
echo "2. Run inference:"
echo "   python scripts/inference.py --model-path models/vehiclebert/best --text \"Replace the O2 sensor\""
echo ""
echo "3. Evaluate model:"
echo "   python scripts/evaluate.py --model-path models/vehiclebert/best --test-file data/test.json"
echo ""
echo "4. Open Jupyter notebook:"
echo "   jupyter notebook notebooks/demo_notebook.ipynb"
echo ""
echo "Documentation:"
echo "  - README.md        - Full documentation"
echo "  - QUICKSTART.md    - Quick start guide"
echo "  - GITHUB_SETUP.md  - GitHub setup instructions"
echo "  - DEPLOYMENT.md    - Deployment guide"
echo ""
echo "For production training with 5,000 sentences:"
echo "   python scripts/prepare_data.py --num-sentences 5000"
echo "   python scripts/train.py --epochs 10"
echo ""
echo "Happy coding! 🚗"
