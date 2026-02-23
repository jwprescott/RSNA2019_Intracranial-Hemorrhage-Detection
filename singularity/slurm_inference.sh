#!/bin/bash
#SBATCH --job-name=rsna2019_inference
#SBATCH --output=rsna2019_%j.out
#SBATCH --error=rsna2019_%j.err
#SBATCH --time=04:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

# ============================================================
# RSNA 2019 Intracranial Hemorrhage Detection - SLURM Job Script
# ============================================================
#
# Usage:
#   sbatch slurm_inference.sh /path/to/data /path/to/output.csv
#
# Or with environment variables:
#   DATA_DIR=/path/to/data OUTPUT_FILE=predictions.csv sbatch slurm_inference.sh
#
# ============================================================

# Get arguments or use environment variables
DATA_DIR="${1:-${DATA_DIR:-/path/to/your/data}}"
OUTPUT_FILE="${2:-${OUTPUT_FILE:-predictions.csv}}"

# Project directory (adjust this to your setup)
PROJECT_DIR="${PROJECT_DIR:-$(dirname $(dirname $(realpath $0)))}"

# Singularity image path
SIF_IMAGE="${SIF_IMAGE:-${PROJECT_DIR}/singularity/rsna2019.sif}"

# Model directory
MODEL_DIR="${MODEL_DIR:-${PROJECT_DIR}/models}"
CACHE_DIR="${CACHE_DIR:-${PROJECT_DIR}/.cache/model_backbones}"

echo "============================================================"
echo "RSNA 2019 Hemorrhage Detection - Inference Job"
echo "============================================================"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURM_NODELIST}"
echo "Start time: $(date)"
echo ""
echo "Configuration:"
echo "  Project dir: ${PROJECT_DIR}"
echo "  Data dir: ${DATA_DIR}"
echo "  Output file: ${OUTPUT_FILE}"
echo "  Model dir: ${MODEL_DIR}"
echo "  Cache dir: ${CACHE_DIR}"
echo "  SIF image: ${SIF_IMAGE}"
echo "============================================================"

# Check if Singularity image exists
if [ ! -f "${SIF_IMAGE}" ]; then
    echo "ERROR: Singularity image not found at ${SIF_IMAGE}"
    echo "Build it first with: singularity build rsna2019.sif rsna2019.def"
    exit 1
fi

# Check if data directory exists
if [ ! -d "${DATA_DIR}" ]; then
    echo "ERROR: Data directory not found: ${DATA_DIR}"
    exit 1
fi

mkdir -p "${CACHE_DIR}/torch" "${CACHE_DIR}/pretrainedmodels"

# Load modules (adjust for your cluster)
# module load singularity
# module load cuda/11.6

# Run inference with Singularity
# --nv: Enable NVIDIA GPU support
# --bind: Mount directories into the container
singularity exec --nv \
    --bind "${PROJECT_DIR}:/workspace" \
    --bind "${DATA_DIR}:/data" \
    --bind "${MODEL_DIR}:/models" \
    --bind "${CACHE_DIR}:/model_cache" \
    --pwd /workspace \
    --env "TORCH_HOME=/model_cache/torch" \
    --env "PRETRAINEDMODELS_HOME=/model_cache/pretrainedmodels" \
    "${SIF_IMAGE}" \
    python inference/run_inference.py \
        --data_dir /data \
        --model_dir /models \
        --output "${OUTPUT_FILE}" \
        --batch_size 16

# Check exit status
if [ $? -eq 0 ]; then
    echo ""
    echo "============================================================"
    echo "Inference completed successfully!"
    echo "Output saved to: ${OUTPUT_FILE}"
    echo "End time: $(date)"
    echo "============================================================"
else
    echo ""
    echo "============================================================"
    echo "ERROR: Inference failed!"
    echo "Check the error log: rsna2019_${SLURM_JOB_ID}.err"
    echo "============================================================"
    exit 1
fi
