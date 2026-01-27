#!/bin/bash
# =============================================================================
# Setup Script for Nano ML Training Environment
# =============================================================================
#
# This script creates a Python virtual environment and installs all
# necessary dependencies for training the model.
#
# Usage:
#   chmod +x setup.sh
#   ./setup.sh
#
# After setup, activate the environment with:
#   source venv/bin/activate
#
# =============================================================================

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo -e "${BLUE}=============================================${NC}"
echo -e "${BLUE}  Nano ML Training Environment Setup${NC}"
echo -e "${BLUE}=============================================${NC}"
echo ""

# -----------------------------------------------------------------------------
# Check Python version
# -----------------------------------------------------------------------------
echo -e "${YELLOW}[1/7] Checking Python version...${NC}"

PYTHON_CMD=""
for cmd in python3.11 python3.10 python3.9 python3; do
    if command -v $cmd &> /dev/null; then
        version=$($cmd --version 2>&1 | awk '{print $2}')
        major=$(echo $version | cut -d. -f1)
        minor=$(echo $version | cut -d. -f2)
        if [ "$major" -ge 3 ] && [ "$minor" -ge 9 ]; then
            PYTHON_CMD=$cmd
            break
        fi
    fi
done

if [ -z "$PYTHON_CMD" ]; then
    echo -e "${RED}Error: Python 3.9+ is required but not found.${NC}"
    echo "Please install Python 3.9 or later and try again."
    exit 1
fi

echo -e "${GREEN}Found $PYTHON_CMD (version $($PYTHON_CMD --version 2>&1 | awk '{print $2}'))${NC}"

# -----------------------------------------------------------------------------
# Create virtual environment
# -----------------------------------------------------------------------------
echo ""
echo -e "${YELLOW}[2/7] Creating virtual environment...${NC}"

if [ -d "venv" ]; then
    echo -e "${YELLOW}Virtual environment already exists. Removing old one...${NC}"
    rm -rf venv
fi

$PYTHON_CMD -m venv venv
echo -e "${GREEN}Virtual environment created successfully.${NC}"

# -----------------------------------------------------------------------------
# Activate virtual environment
# -----------------------------------------------------------------------------
echo ""
echo -e "${YELLOW}[3/7] Activating virtual environment...${NC}"

source venv/bin/activate
echo -e "${GREEN}Virtual environment activated.${NC}"

# -----------------------------------------------------------------------------
# Upgrade pip
# -----------------------------------------------------------------------------
echo ""
echo -e "${YELLOW}[4/7] Upgrading pip...${NC}"

pip install --upgrade pip setuptools wheel
echo -e "${GREEN}pip upgraded successfully.${NC}"

# -----------------------------------------------------------------------------
# Install PyTorch (with CUDA support if available)
# -----------------------------------------------------------------------------
echo ""
echo -e "${YELLOW}[5/7] Installing PyTorch...${NC}"

# Check for CUDA
if command -v nvidia-smi &> /dev/null; then
    echo "CUDA detected. Installing PyTorch with CUDA support..."
    # Get CUDA version
    CUDA_VERSION=$(nvidia-smi | grep "CUDA Version" | awk '{print $9}' | cut -d. -f1-2)
    echo "Detected CUDA version: $CUDA_VERSION"

    # Install appropriate PyTorch version
    if [[ "$CUDA_VERSION" == "12"* ]]; then
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    elif [[ "$CUDA_VERSION" == "11"* ]]; then
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    else
        echo "Unknown CUDA version, installing default PyTorch..."
        pip install torch torchvision torchaudio
    fi
else
    echo "CUDA not detected. Installing CPU-only PyTorch..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
fi

echo -e "${GREEN}PyTorch installed successfully.${NC}"

# -----------------------------------------------------------------------------
# Install other dependencies
# -----------------------------------------------------------------------------
echo ""
echo -e "${YELLOW}[6/7] Installing dependencies from requirements.txt...${NC}"

pip install -r requirements.txt

echo -e "${GREEN}Dependencies installed successfully.${NC}"

# -----------------------------------------------------------------------------
# Create directory structure
# -----------------------------------------------------------------------------
echo ""
echo -e "${YELLOW}[7/7] Creating directory structure...${NC}"

mkdir -p models datasets checkpoints scripts logs

echo -e "${GREEN}Directory structure created.${NC}"

# -----------------------------------------------------------------------------
# Setup Hugging Face authentication
# -----------------------------------------------------------------------------
echo ""
echo -e "${BLUE}=============================================${NC}"
echo -e "${BLUE}  Setup Complete!${NC}"
echo -e "${BLUE}=============================================${NC}"
echo ""
echo -e "To activate the environment, run:"
echo -e "  ${GREEN}source venv/bin/activate${NC}"
echo ""
echo -e "To set your Hugging Face token, run:"
echo -e "  ${GREEN}export HF_TOKEN='your_token_here'${NC}"
echo ""
echo -e "To start training, run:"
echo -e "  ${GREEN}python train.py${NC}"
echo ""

# -----------------------------------------------------------------------------
# Optional: Run a quick test
# -----------------------------------------------------------------------------
echo -e "${YELLOW}Running quick import test...${NC}"

python -c "
import torch
import transformers
import datasets
import accelerate
from huggingface_hub import HfApi

print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA device: {torch.cuda.get_device_name(0)}')
print(f'Transformers version: {transformers.__version__}')
print(f'Datasets version: {datasets.__version__}')
print(f'Accelerate version: {accelerate.__version__}')
print('All imports successful!')
"

if [ $? -eq 0 ]; then
    echo -e "${GREEN}All tests passed! Environment is ready.${NC}"
else
    echo -e "${RED}Some tests failed. Please check the installation.${NC}"
    exit 1
fi
