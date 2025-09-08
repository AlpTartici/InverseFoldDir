#!/bin/bash
# Installation script for inv_fold_dir environment
# Exactly matches Docker_full_pipeline_fixed installation steps

set -e  # Exit on any error

echo "Setting up inv_fold_dir environment..."

# Check if conda/mamba is available
if command -v mamba &> /dev/null; then
    CONDA_CMD="mamba"
    echo "Using mamba for faster installation"
elif command -v conda &> /dev/null; then
    CONDA_CMD="conda"
    echo "Using conda"
else
    echo "Error: Neither conda nor mamba found. Please install Anaconda or Miniconda first."
    exit 1
fi

# Create base environment
echo "Creating base environment..."
$CONDA_CMD env create -f inv_fold_dir_environment_minimal.yml

# Activate environment
echo "Activating environment..."
eval "$(conda shell.bash hook)"
conda activate inv_fold_dir

# Install PyTorch with CUDA 12.6 support (exactly as in Docker)
echo "Installing PyTorch with CUDA 12.6 support..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126

# Install PyTorch Geometric (exactly as in Docker)
echo "Installing PyTorch Geometric..."
pip install --no-cache-dir torch-geometric

# Install PyTorch Geometric extension libraries (exactly as in Docker)
echo "Installing PyTorch Geometric extensions..."
pip install --no-cache-dir torch-scatter torch-sparse torch-cluster torch-spline-conv

# Verify installation
echo "Verifying installation..."
python -c "
import torch
print('PyTorch version:', torch.__version__)
print('CUDA available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('CUDA version:', torch.version.cuda)
    print('GPU device:', torch.cuda.get_device_name(0))
else:
    print('Warning: CUDA not available. This environment requires GPU support.')

try:
    import torch_geometric
    print('PyTorch Geometric version:', torch_geometric.__version__)
except ImportError:
    print('Error: PyTorch Geometric not installed properly')
    exit(1)

try:
    import torch_scatter
    import torch_sparse
    import torch_cluster
    print('All PyTorch Geometric extensions installed successfully')
except ImportError as e:
    print('Error: PyTorch Geometric extensions not installed properly:', e)
    exit(1)

try:
    import esm
    print('ESM (fair-esm) imported successfully')
except ImportError:
    print('Error: ESM not installed properly')

try:
    import e3nn
    print('E3NN imported successfully')
except ImportError:
    print('Error: E3NN not installed properly')

try:
    import jupyter
    print('Jupyter installed successfully')
except ImportError:
    print('Error: Jupyter not installed properly')
"

echo ""
echo "Installation complete!"
echo "To activate the environment, run: conda activate inv_fold_dir"
echo "To start Jupyter Lab, run: jupyter lab"
echo "To start Jupyter Notebook, run: jupyter notebook"
