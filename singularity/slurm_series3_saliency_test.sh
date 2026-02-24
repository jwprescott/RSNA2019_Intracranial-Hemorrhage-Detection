#!/bin/bash
#SBATCH --job-name=rsna-s3-sal
#SBATCH --output=rsna_s3_sal_%j.out
#SBATCH --error=rsna_s3_sal_%j.err
#SBATCH --time=02:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

set -euo pipefail

module load singularity/4.3.4

PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
SIF_IMAGE="${SIF_IMAGE:-${PROJECT_DIR}/singularity/rsna2019.sif}"

# Input subset copied locally from stage_2_test (no writes to /mnt source dir).
DICOM_INPUT="${DICOM_INPUT:-${PROJECT_DIR}/tmp/stage2_test_series3_dicoms}"
RUN_ROOT="${RUN_ROOT:-${PROJECT_DIR}/tmp/series3_saliency_run_${SLURM_JOB_ID}}"
PROCESSED_DIR="${RUN_ROOT}/processed"
OUTPUT_DIR="${RUN_ROOT}/output"

# Isolated dependency path for container python.
PYDEPS="${PROJECT_DIR}/.cache/singularity_pydeps"
CACHE_DIR="${CACHE_DIR:-${PROJECT_DIR}/.cache/model_backbones}"
SALIENCY_METHOD="${SALIENCY_METHOD:-input_grad}"

echo "============================================================"
echo "RSNA series3 saliency test"
echo "============================================================"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURM_NODELIST}"
echo "Project: ${PROJECT_DIR}"
echo "Input DICOM dir: ${DICOM_INPUT}"
echo "Run root: ${RUN_ROOT}"
echo "SIF: ${SIF_IMAGE}"
echo "PYDEPS: ${PYDEPS}"
echo "Cache dir: ${CACHE_DIR}"
echo "Saliency method: ${SALIENCY_METHOD}"
echo "============================================================"

if [ ! -f "${SIF_IMAGE}" ]; then
    echo "ERROR: Missing container image: ${SIF_IMAGE}"
    exit 1
fi

if [ ! -d "${DICOM_INPUT}" ]; then
    echo "ERROR: Missing input subset: ${DICOM_INPUT}"
    exit 1
fi

if [ ! -d "${PYDEPS}" ]; then
    echo "ERROR: Missing dependency dir: ${PYDEPS}"
    exit 1
fi

mkdir -p "${PROCESSED_DIR}" "${OUTPUT_DIR}"
mkdir -p "${CACHE_DIR}/torch" "${CACHE_DIR}/pretrainedmodels"

SING_COMMON=(
    singularity exec --nv
    --bind "${PROJECT_DIR}:/workspace"
    --bind "${CACHE_DIR}:/model_cache"
    --pwd /workspace
    --env PYTHONNOUSERSITE=1
    --env "PYTHONPATH=/workspace/.cache/singularity_pydeps"
    --env "TORCH_HOME=/model_cache/torch"
    --env "PRETRAINEDMODELS_HOME=/model_cache/pretrainedmodels"
    "${SIF_IMAGE}"
)

echo "[1/3] Preparing DICOM subset -> processed PNG/CSV"
"${SING_COMMON[@]}" python scripts/prepare_custom_data.py \
    --input_dir "${DICOM_INPUT}" \
    --output_dir "${PROCESSED_DIR}" \
    --group_by series \
    --n_jobs "${SLURM_CPUS_PER_TASK:-8}"

echo "[2/3] Running RSNA inference"
"${SING_COMMON[@]}" python inference/run_inference.py \
    --data_dir "${PROCESSED_DIR}" \
    --model_dir /workspace/models \
    --output "${OUTPUT_DIR}/predictions.csv" \
    --batch_size 16 \
    --num_workers 0

echo "[3/3] Exporting 3-panel saliency images"
"${SING_COMMON[@]}" python inference/export_saliency_panels.py \
    --data_dir "${PROCESSED_DIR}" \
    --output_dir "${OUTPUT_DIR}/saliency_panels" \
    --model_dir /workspace/models \
    --saliency_mode ensemble \
    --saliency_method "${SALIENCY_METHOD}" \
    --ensemble_slice_csv "${OUTPUT_DIR}/predictions_per_slice.csv" \
    --image_size_override 224

echo "============================================================"
echo "Done"
echo "Predictions: ${OUTPUT_DIR}/predictions.csv"
echo "Per-slice:   ${OUTPUT_DIR}/predictions_per_slice.csv"
echo "Saliency:    ${OUTPUT_DIR}/saliency_panels/"
echo "============================================================"
