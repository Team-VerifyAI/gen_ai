#!/bin/bash
#SBATCH --job-name=install_hunyuan
#SBATCH --output=/data/YOUR_USERNAME/repos/gen_image/logs/install_%j.out
#SBATCH --error=/data/YOUR_USERNAME/repos/gen_image/logs/install_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --partition=YOUR_GPU_PARTITION  # e.g., batch_ugrad, gpu, etc.
#SBATCH --time=23:59:00

set -e

echo "=========================================="
echo "HunyuanImage-2.1 Installation"
echo "=========================================="

# IMPORTANT: Update this to your repos directory
REPOS_DIR="/data/YOUR_USERNAME/repos"

# Navigate to repos directory
cd $REPOS_DIR

# Clone repository
if [ ! -d "HunyuanImage-2.1" ]; then
    echo "Cloning HunyuanImage-2.1 repository..."
    git clone https://github.com/Tencent-Hunyuan/HunyuanImage-2.1.git
else
    echo "Repository already exists, skipping clone..."
fi

cd HunyuanImage-2.1

# IMPORTANT: Update these variables
DATA_DIR="/data/YOUR_USERNAME"
CONDA_DIR="$DATA_DIR/anaconda3"  # or miniconda3

# Environment variables - MUST SET BEFORE CONDA
export HF_HOME=$DATA_DIR/.cache/huggingface
export TMPDIR=$DATA_DIR/.tmp
export PIP_CACHE_DIR=$DATA_DIR/.cache/pip
export CONDA_PKGS_DIRS=$DATA_DIR/.cache/conda/pkgs
export CONDA_ENVS_PATH=$CONDA_DIR/envs
export HF_TOKEN=YOUR_HF_TOKEN_HERE  # Get from https://huggingface.co/settings/tokens

# Create cache directories
mkdir -p $DATA_DIR/.cache/{huggingface,pip,conda/pkgs}
mkdir -p $DATA_DIR/.tmp

# Create conda environment
echo ""
echo "Creating conda environment 'hunyuan'..."
source $CONDA_DIR/etc/profile.d/conda.sh

if conda env list | grep -q "^hunyuan "; then
    echo "Environment 'hunyuan' already exists"
    conda activate hunyuan
else
    # Create in NAS location, not home
    conda create -p $CONDA_DIR/envs/hunyuan python=3.10 -y
    conda activate $CONDA_DIR/envs/hunyuan
fi

# Install dependencies
echo ""
echo "Installing dependencies..."
pip install --upgrade pip

echo "Installing requirements.txt..."
pip install -r requirements.txt

echo ""
echo "Installing flash-attn (this will take 10-20 minutes)..."
pip install flash-attn==2.7.3 --no-build-isolation

# Download models
echo ""
echo "=========================================="
echo "Downloading models (~173GB, 1-2 hours)"
echo "=========================================="

mkdir -p ./ckpts

echo "Starting model download..."
huggingface-cli download tencent/HunyuanImage-2.1 --local-dir ./ckpts --resume-download

echo ""
echo "=========================================="
echo "Installation Complete!"
echo "=========================================="
echo "Conda environment: hunyuan"
echo "Model location: $REPOS_DIR/HunyuanImage-2.1/ckpts"
echo ""
echo "Next steps:"
echo "1. Update paths in gen_image scripts (see QUICKSTART.md)"
echo "2. Test with: python scripts/test_hunyuan.py"
echo "3. Batch generation: sbatch run_hunyuan.sh"
