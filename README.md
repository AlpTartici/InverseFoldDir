# Inverse Folding: Structure-Conditioned Protein Sequence Sampling

A generative protein sequence design framework using **Dirichlet Flow Matching (DFM)** for structure-conditioned sequence generation. This repository implements deep generative models that learn to predict amino acid sequences from protein backbone structures.

## Quick Start

```bash
# Clone repository
git clone <your-repo-url>
cd inverse-folding

# Option 1: Automated Setup (Easiest)
./install_inv_fold_dir.sh

# Option 2: Manual Setup
conda env create -f inv_fold_dir_environment_minimal.yml
conda activate inv_fold_dir
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
pip install --no-cache-dir torch-geometric torch-scatter torch-sparse torch-cluster torch-spline-conv

# Start Jupyter for interactive notebooks
jupyter lab  # or jupyter notebook

# Interactive notebook for easy experimentation
jupyter notebook run_inference.ipynb

# Run inference with a PDB structure
cd training
python sample.py --pdb_input 1fcd.C --steps 20 --flow_temp 0.2

# Run inpainting on specific positions
cd training
python inpainting.py --pdb_input 1abc --mask-positions "10,25,40" --steps 20

```

---

## Table of Contents

1. [Installation](#installation)
2. [Quick Usage Examples](#quick-usage-examples)
3. [Sampling Modes](#sampling-modes)
4. [Example Scripts](#example-scripts)
5. [Input Formats](#input-formats)
6. [Advanced Configuration](#advanced-configuration)
7. [Output Formats](#output-formats)
8. [Training Your Own Models](#training-your-own-models)
9. [Troubleshooting](#troubleshooting)

---

## Installation

### Requirements

- **Python**: 3.9
- **CUDA**: 11.8+ (for GPU acceleration)
- **Memory**: 8GB+ RAM, 6GB+ VRAM recommended

### Option 1: Automated Installation (Recommended)

The easiest way to set up the environment. This script handles everything automatically and matches the Docker environment exactly.

```bash
# Make script executable and run
chmod +x install_inv_fold_dir.sh
./install_inv_fold_dir.sh
```

The script will:

- Create the conda environment
- Install PyTorch with CUDA 12.6 support
- Install PyTorch Geometric and all extensions
- Verify the installation

### Option 2: Manual Installation

Step-by-step installation if you prefer manual control or need to troubleshoot.

```bash
# Create base environment
conda env create -f inv_fold_dir_environment_minimal.yml
conda activate inv_fold_dir

# Install PyTorch with CUDA 12.6 support (matching Docker)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126

# Install PyTorch Geometric
pip install --no-cache-dir torch-geometric

# Install PyTorch Geometric extensions
pip install --no-cache-dir torch-scatter torch-sparse torch-cluster torch-spline-conv

# Verify installation
python -c "
import torch, torch_geometric
print('PyTorch version:', torch.__version__)
print('CUDA available:', torch.cuda.is_available())
print('PyTorch Geometric version:', torch_geometric.__version__)
print('Setup complete!')
"
```

### Alternative Methods (Legacy)

#### Method A: Cross-platform environment

```bash
# Create environment from the exact tested configuration
conda env create -f inv_fold_dir_environment_cross_platform.yml
conda activate inv_fold_dir

# Verify installation
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
```

#### Option B: Alternative environment files

```bash
# Use the main environment configuration
conda env create -f inv_fold_environment.yml
conda activate inv_fold


# Or use the training-specific environment
conda env create -f training/inv_fold_environment.yml
conda activate inv_fold
```

#### Option C: Cross-platform compatible installation

```bash
# For systems where the exact environment export doesn't work
conda env create -f inv_fold_dir_environment.yml
conda activate inv_fold_dir

# If you encounter platform-specific issues, create manually:
conda create -n inv_fold_dir python=3.9
conda activate inv_fold_dir
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
conda install pyg -c pyg
conda install biopython numpy scipy matplotlib pandas jupyter -c conda-forge
pip install -r requirements_inv_fold_dir.txt
```

### Method 2: Pip Installation

```bash
# Create virtual environment
python -m venv inv_fold_env
source inv_fold_env/bin/activate  # Linux/Mac
# or
inv_fold_env\Scripts\activate     # Windows

# Install core dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install torch-geometric

# Install remaining dependencies from requirements file
pip install -r requirements_inv_fold_dir.txt

# Install additional dependencies for structure processing (if needed)
pip install biopython spyrmsd

# Verify installation
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
```

### Environment Files Available

The repository provides several environment configuration options:

- **`inv_fold_dir_environment_minimal.yml`** - Minimal environment matching Docker (Recommended)
- **`inv_fold_dir_environment_cross_platform.yml`** - Tested production environment
- **`inv_fold_dir_environment.yml`** - Full environment with build strings (Linux-specific)
- **`requirements_inv_fold_dir.txt`** - Key Python packages for pip installation
- **`inv_fold_environment.yml`** - Original environment file

---

## Running Jupyter Notebooks

The environment includes Jupyter Lab and Jupyter Notebook for interactive experimentation:

```bash
# Activate environment
conda activate inv_fold_dir

# Start Jupyter Lab (recommended)
jupyter lab

# Or start classic Jupyter Notebook
jupyter notebook

# Run a specific notebook
jupyter notebook run_inference.ipynb
```

### Available Notebooks

The repository includes several interactive notebooks:

- **`run_inference.ipynb`** - Main inference notebook for sampling and inpainting
- **`analysis_scripts/zero_shot_eval_proteingym.ipynb`** - Protein evaluation and analysis
- Other analysis notebooks in the `analysis_scripts/` and `notebooks/` directories

---

## Quick Usage Examples

### Full Sequence Sampling

Generate complete sequences conditioned on protein backbone structure:

```bash
# Navigate to training directory
cd training

# Using PDB ID
python sample.py --pdb_input 1abc --steps 20 --flow_temp 0.2

# Using local PDB file
python sample.py --pdb_input /path/to/protein.pdb --steps 30

# Multiple structures with ensemble sampling
python sample.py --pdb_input 1fcd.C --ensemble_size 5 --steps 20
```

### Sequence Inpainting

Predict specific amino acids while keeping others fixed:

```bash
# Navigate to training directory (if not already there)
cd training

# Mask specific positions (with validation)
python inpainting.py --pdb_input 1abc --mask-positions "D45,Y67,K89"

# Mask positions without validation
python inpainting.py --pdb_input 1abc --mask-positions "45,67,89"

# Template-based inpainting
python inpainting.py --pdb_input 1abc --template-sequence "ACDEFXHIKLXNPQXSTVWY"

# Random masking
python inpainting.py --pdb_input 1abc --mask-ratio 0.15
```

---

## Sampling Modes

### 1. Full Sequence Sampling

- **Purpose**: Generate complete amino acid sequences that fold to given structures
- **Use Cases**: De novo protein design, sequence optimization
- **Command**: `cd training && python sample.py`

### 2. Sequence Inpainting

- **Purpose**: Predict masked positions while conditioning on known residues
- **Use Cases**: Variant effect prediction, protein completion, mutation design
- **Command**: `cd training && python inpainting.py`

### 3. Ensemble Sampling

- **Purpose**: Generate multiple structural variants for robust predictions
- **Use Cases**: Uncertainty quantification, consensus design
- **Parameters**: `--ensemble_size`, `--ensemble_consensus_strength`

---

## Example Scripts

Ready-to-use shell scripts are provided in `example_scripts_for_prediction/`:

### 1. Full Sequence Sampling

```bash
#!/bin/bash
# examples_scripts_for_prediction/full_sampling.sh

# Navigate to training directory
cd training

# Full sequence sampling using PDB file path
python sample.py \
    --pdb_input /path/to/protein.pdb \
    --steps 20 \
    --flow_temp 0.2 \
    --ensemble_size 3 \
    --output_dir ./results/full_sampling

echo "Full sequence sampling completed. Results saved to ./results/full_sampling"
```

### 2. Inpainting with Position+Validation Format

```bash
#!/bin/bash
# examples_scripts_for_prediction/inpainting_validated.sh

# Navigate to training directory
cd training

# Inpainting with position validation (D45 means position 45 should have D)
python inpainting.py \
    --pdb_input 1abc \
    --mask-positions "D45,Y67,K89" \
    --steps 20 \
    --flow_temp 0.3 \
    --output_dir ./results/inpainting_validated

echo "Validated inpainting completed. Results saved to ./results/inpainting_validated"
```

### 3. Inpainting with Position-Only Format

```bash
#!/bin/bash
# examples_scripts_for_prediction/inpainting_positions.sh

# Navigate to training directory
cd training

# Inpainting without validation (just mask positions 16, 42)
python inpainting.py \
    --pdb_input 1fcd.C \
    --mask-positions "16,42" \
    --steps 20 \
    --flow_temp 0.3 \
    --ensemble_size 2 \
    --output_dir ./results/inpainting_positions

echo "Position-based inpainting completed. Results saved to ./results/inpainting_positions"
```

### 4. Template-Based Inpainting

```bash
#!/bin/bash
# examples_scripts_for_prediction/template_inpainting.sh

# Navigate to training directory
cd training

# Template-based inpainting using sequence with 'X' for unknowns
python inpainting.py \
    --pdb_input 1abc \
    --template-sequence "MVQPQVQHPIQXIKLMNPQRXTVWYX" \
    --steps 25 \
    --flow_temp 0.25 \
    --output_dir ./results/template_inpainting

echo "Template inpainting completed. Results saved to ./results/template_inpainting"
```

### 5. Batch Processing from CSV

```bash
#!/bin/bash
# examples_scripts_for_prediction/batch_processing.sh

# Navigate to training directory
cd training

# Batch processing from CSV file with custom batch size
python inpainting.py \
    --list_csv mutations.csv \
    --steps 20 \
    --flow_temp 0.3 \
    --batch_size 8 \
    --output_dir ./results/batch_processing

echo "Batch processing completed. Results saved to ./results/batch_processing"
```

---

## Input Formats

### Mask Position Formats

#### 1. Position Only Format

```bash
--mask-positions "45,67,89"
```

- **Usage**: Mask these positions without validation
- **Format**: Comma-separated position numbers (0-indexed)
- **Example**: Position 45, 67, and 89 will be masked

#### 2. Position + Validation Format

```bash
--mask-positions "D45,Y67,K89"
```

- **Usage**: Mask positions 45, 67, 89 but first verify amino acids
- **Format**: `{amino_acid}{position}` format
- **Validation**: Position 45 must have D, position 67 must have Y, position 89 must have K
- **Safety**: Program errors out if validation fails (perfect for variant effect studies)

### CSV File Format

For batch processing with `--list_csv`:

```csv
mutant,mutated_sequence,DMS_score,DMS_score_bin,mask-positions,protein
A1V,MVQPQVQHPIQ...,-2.1,low,1,PIN1_HUMAN
L2P,MPQPQVQHPIQ...,0.5,medium,2,PIN1_HUMAN
G3A,MVQAQVQHPIQ...,1.2,high,3,PIN1_HUMAN
D45A,MVQPQVQHPIQXIKLMNPQ...,0.8,medium,D45,PIN1_HUMAN
```

**Required columns:**

- `mutant`: Mutation identifier
- `mutated_sequence`: Sequence with mutations (optional for validation)
- `DMS_score`: Deep Mutational Scanning score
- `DMS_score_bin`: Score category (low/medium/high)
- `mask-positions`: Positions to mask (same format as `--mask-positions`)
- `protein`: UniProt ID (corresponds to `--uniprot` argument)

### Structure Input Formats

```bash
# Local PDB file
--pdb_input /path/to/protein.pdb

# Local CIF file
--pdb_input /path/to/protein.cif

# PDB ID from RCSB
--pdb_input 1abc

# PDB ID with specific chain
--pdb_input 1fcd.C
```

---

## Advanced Configuration

### Sampling Parameters

```bash
# Temperature control - Trade-off between diversity and structural recovery
--flow_temp 0.2          # Conservative sampling (better structure recovery, less diversity)
--flow_temp 1.0          # Balanced sampling
--flow_temp 2.0          # Diverse sampling (higher diversity, may compromise structure recovery)

# Integration steps (more = higher quality, slower)
--steps 10               # Fast sampling
--steps 20               # Balanced quality/speed
--steps 50               # High quality

# Time range
--T 8.0                  # Maximum noise level
--t_min 0.0              # Minimum noise level

# Initial distribution - Controls noise variance and diversity
--dirichlet_concentration 1.0     # Standard value (balanced noise variance)
--dirichlet_concentration 20.0    # Higher concentration (reduced noise variance, less diversity, slight improvement in structural recovery)
--dirichlet_concentration 0.5     # Lower concentration (higher noise variance, more diversity)
```

#### Parameter Guidelines:

**Temperature (`--flow_temp`)**:

- **Lower values (0.1-0.3)**: More conservative predictions with better average structural recovery but reduced sequence diversity
- **Higher values (1.0-2.0)**: More diverse predictions but may compromise average structural recovery
- **Recommended**: Start with 0.2-0.3 for most applications

**Dirichlet Concentration (`--dirichlet_concentration`)**:

- **Standard value**: 1.0 (default balanced setting)
- **Higher values (>1.0)**: Reduces noise variance, leading to less diversity but slight improvement in structural recovery
- **Lower values (<1.0)**: Increases noise variance, leading to more diversity but potentially lower structural recovery
- **Recommended**: Use 1.0 as baseline, increase to 5.0-20.0 for more focused sampling

### Ensemble Sampling

```bash
# Ensemble size (1-10 replicas)
--ensemble_size 5

# Consensus strength (0=independent, 1=full consensus)
--ensemble_consensus_strength 0.3

# Ensemble method
--ensemble_method arithmetic     # Mean in probability space
--ensemble_method geometric      # Mean in log space

# Structure noise for diversity
--structure_noise_mag_std 1.0    # Standard deviation (Angstroms)
--uncertainty_struct_noise_scaling  # Scale by B-factors
```

### Output Control

```bash
# Output directory
--output_dir ./results/my_experiment

# Output prefix
--output_prefix my_protein_design

# File formats
--save_probabilities         # Save probability distributions
--no_probabilities          # Skip probabilities (faster)
--detailed_json             # Generate time-step trajectories
```

---

## Output Formats

### 1. Sequence Files (CSV)

```csv
structure_idx,structure_name,length,predicted_sequence,true_sequence,accuracy
0,1abc,150,MKFLVLLFNISCV...,MKFLVLLFNISCV...,94.67
```

### 2. Probability Distributions (NPZ)

```python
import numpy as np

# Load results
data = np.load('results_probabilities.npz')

# Access probabilities for structure 0
probs = data['struct_0_probabilities']  # Shape: [seq_len, 21]
true_indices = data['struct_0_true_indices']  # Ground truth
predicted = data['struct_0_predicted_indices']  # Predictions
```

### 3. Trajectory Analysis (JSON)

```json
{
  "1abc": {
    "0": {
      "trajectory": [
        {
          "time_point": 0.0,
          "most_likely_amino_acid": "M",
          "current_probability": 0.456789,
          "amino_acid_breakdown": {
            "A": { "predicted_prob": 0.023, "current_prob": 0.045 },
            "M": { "predicted_prob": 0.678, "current_prob": 0.456 }
          }
        }
      ],
      "final_prediction": "M",
      "ground_truth": "M"
    }
  }
}
```

### 4. Metadata Files (TXT)

```
PROTEIN SEQUENCE SAMPLING METADATA
==================================================

Timestamp: 20241204_143022
Generated: 2024-12-04 14:30:22

SAMPLING PARAMETERS:
  Model: best_model.pt
  Dataset split: validation
  Sampling steps: 20
  Max time (T): 8.0

RESULTS SUMMARY:
  Total structures: 5
  Successful: 5
  Failed: 0
  Average accuracy: 39.45%
```

---

## Training Your Own Models

### 1. Data Preparation

```bash
# Process PDB files
python data/pdb_processor.py --input_dir /path/to/pdbs --output_file processed_data.pkl

# Create dataset splits
python data/create_splits.py --data_file processed_data.pkl --output_dir ./splits/
```

### 2. Training Configuration

```bash
# Navigate to training directory
cd training

# Start training with default parameters
python train.py --config configs/default_training.yaml

# Training with custom parameters
python train.py \
    --batch_size 32 \
    --learning_rate 1e-4 \
    --num_layers_gvp 6 \
    --hidden_dim 256 \
    --steps 50 \
    --epochs 100
```

````

---

## Troubleshooting

### Common Issues

#### 1. CUDA Out of Memory
```bash
# Reduce batch size
--batch_size 2

# Use ensemble instead of large batches
--ensemble_size 3 --batch_size 1

# Reduce integration steps
--steps 10
````

#### 2. Model Loading Errors

```bash
# Navigate to training directory
cd training

# Check model path
ls -la /path/to/model.pt

# Try auto-discovery
python sample.py --pdb_input 1abc  # Will find *best*.pt files

# Verify model architecture
python sample.py --pdb_input 1abc --verbose
```

#### 3. PDB Download Issues

```bash
# Use local files instead
--pdb_input /path/to/local/file.pdb

# Check internet connection
curl -I https://files.rcsb.org/download/1abc.pdb

# Try different PDB ID format
--pdb_input 1ABC  # Sometimes case matters
```

#### 4. Dependency Conflicts

```bash
# Start with the recommended tested environment
conda env create -f inv_fold_dir_environment_cross_platform.yml

# If that fails, try the full environment export
conda env create -f inv_fold_dir_environment.yml

# For cross-platform issues, create manually:
conda create -n inv_fold_dir python=3.9
conda activate inv_fold_dir
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
conda install pyg -c pyg
conda install biopython numpy scipy matplotlib pandas jupyter -c conda-forge

# If conda environments don't work, try pip:
python -m venv inv_fold_env
source inv_fold_env/bin/activate
pip install -r requirements_inv_fold_dir.txt

# Check package versions
conda list | grep torch
pip list | grep torch

# Environment cleanup (if needed)
conda env remove -n inv_fold_dir
conda env remove -n inv_fold
```

#### 5. Performance Issues

```bash
# Enable GPU acceleration
nvidia-smi  # Check GPU availability

# Optimize parameters
--batch_size 4     # Balance memory/speed
--steps 20         # Reasonable quality/speed
--flow_temp 0.3    # Faster convergence
```

### Debug Mode

```bash
# Enable verbose output
--verbose

# Test with simple structure
--pdb_input 1abc --steps 5 --verbose

# Check model parameters
python -c "
import torch
checkpoint = torch.load('model.pt', map_location='cpu')
print('Available keys:', checkpoint.keys())
if 'args' in checkpoint:
    print('Model args:', checkpoint['args'])
"
```

### Performance Benchmarks

| Configuration   | Speed (seq/min) | Memory (GB) | Quality   |
| --------------- | --------------- | ----------- | --------- |
| CPU, steps=10   | 5               | 2           | Basic     |
| GPU, steps=20   | 50              | 4           | Good      |
| GPU, ensemble=5 | 25              | 8           | Excellent |

---

## Citation

If you use this code in your research, please cite:

```bibtex
@article{your_paper_2025,
  title={Inverse FoldDir: Flexibility Aware Structure-Conditioned Protein Sequence Design with Dirichlet Flow Matching},
  author={Tartici et al.},
  journal={TBD},
  year={2025}
}
```

---

## License

TBD

---

## Contributing

This project welcomes contributions and suggestions. Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit [Contributor License Agreements](https://cla.opensource.microsoft.com).

When you submit a pull request, a CLA bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

## Trademarks

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft
trademarks or logos is subject to and must follow
[Microsoft's Trademark & Brand Guidelines](https://www.microsoft.com/legal/intellectualproperty/trademarks/usage/general).
Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship.
Any use of third-party trademarks or logos are subject to those third-party's policies.

---

## Support

- **Issues**: [GitHub Issues](https://github.com/your-username/inverse-folding/issues)

---
