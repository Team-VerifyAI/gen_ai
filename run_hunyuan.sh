#!/bin/bash
#SBATCH --job-name=hunyuan_gen
#SBATCH --output=/data/YOUR_USERNAME/repos/gen_image/logs/hunyuan_%j.out
#SBATCH --error=/data/YOUR_USERNAME/repos/gen_image/logs/hunyuan_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --partition=YOUR_GPU_PARTITION  # e.g., batch_ugrad, gpu, etc.
#SBATCH --time=23:59:00

cd /data/YOUR_USERNAME/repos/gen_image

# Environment setup
export HF_HOME=/data/YOUR_USERNAME/.cache/huggingface
export TMPDIR=/data/YOUR_USERNAME/.tmp
export HF_TOKEN=YOUR_HF_TOKEN_HERE  # Get from https://huggingface.co/settings/tokens
export PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True'

# Activate conda
source /data/YOUR_USERNAME/anaconda3/etc/profile.d/conda.sh  # or miniconda3
conda activate hunyuan

echo "=========================================="
echo "HunyuanImage-2.1 Continuous Generation"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Start: $(date)"
echo "Time Limit: 23:59:00"
echo "=========================================="

# Create directories
mkdir -p logs outputs/hunyuan_dataset

# Run generation
python scripts/generate_hunyuan_dataset.py

echo ""
echo "=========================================="
echo "Generation Complete"
echo "=========================================="
echo "End: $(date)"
echo "Total images: $(ls outputs/hunyuan_dataset/*.png 2>/dev/null | wc -l)"
echo "=========================================="
