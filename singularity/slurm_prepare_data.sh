#!/bin/bash
#SBATCH --job-name=rsna2019_prep
#SBATCH --output=rsna2019_prep_%j.out
#SBATCH --error=rsna2019_prep_%j.err
#SBATCH --time=02:00:00
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G

# ============================================================
# RSNA 2019 - DICOM Data Preparation SLURM Job
# ============================================================
#
# Usage:
#   sbatch slurm_prepare_data.sh /path/to/dicoms /path/to/output
#
# ============================================================

# Get arguments
DICOM_DIR="${1:-${DICOM_DIR:-/path/to/dicoms}}"
OUTPUT_DIR="${2:-${OUTPUT_DIR:-/path/to/processed}}"

# Project directory
PROJECT_DIR="${PROJECT_DIR:-$(dirname $(dirname $(realpath $0)))}"
SIF_IMAGE="${SIF_IMAGE:-${PROJECT_DIR}/singularity/rsna2019.sif}"

echo "============================================================"
echo "RSNA 2019 - DICOM Data Preparation"
echo "============================================================"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURM_NODELIST}"
echo "Start time: $(date)"
echo ""
echo "Configuration:"
echo "  DICOM dir: ${DICOM_DIR}"
echo "  Output dir: ${OUTPUT_DIR}"
echo "============================================================"

# Check inputs
if [ ! -d "${DICOM_DIR}" ]; then
    echo "ERROR: DICOM directory not found: ${DICOM_DIR}"
    exit 1
fi

if [ ! -f "${SIF_IMAGE}" ]; then
    echo "ERROR: Singularity image not found: ${SIF_IMAGE}"
    exit 1
fi

# Create output directory
mkdir -p "${OUTPUT_DIR}"

# Run data preparation
singularity exec \
    --bind "${PROJECT_DIR}:/workspace" \
    --bind "${DICOM_DIR}:/input" \
    --bind "${OUTPUT_DIR}:/output" \
    --pwd /workspace \
    "${SIF_IMAGE}" \
    python scripts/prepare_custom_data.py \
        --input_dir /input \
        --output_dir /output \
        --n_jobs ${SLURM_CPUS_PER_TASK:-8}

if [ $? -eq 0 ]; then
    echo ""
    echo "============================================================"
    echo "Data preparation completed!"
    echo "Processed data saved to: ${OUTPUT_DIR}"
    echo ""
    echo "Next step - run inference:"
    echo "  sbatch singularity/slurm_inference.sh ${OUTPUT_DIR} predictions.csv"
    echo "============================================================"
else
    echo "ERROR: Data preparation failed!"
    exit 1
fi
