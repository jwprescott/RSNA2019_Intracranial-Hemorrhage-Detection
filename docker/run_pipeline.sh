#!/usr/bin/env bash
set -euo pipefail

# Run full RSNA inference + saliency export in Docker with NVIDIA GPUs.
#
# Example:
#   bash docker/run_pipeline.sh \
#     --dicom_dir /path/to/dicoms \
#     --run_root "$PWD/tmp/docker_run_$(date +%Y%m%d_%H%M%S)"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

IMAGE="${IMAGE:-rsna2019-ich:gpu}"
DICOM_INPUT="${DICOM_INPUT:-}"
RUN_ROOT="${RUN_ROOT:-${PROJECT_DIR}/tmp/docker_run_$(date +%Y%m%d_%H%M%S)}"
MODEL_DIR="${MODEL_DIR:-${PROJECT_DIR}/models}"
CACHE_DIR="${CACHE_DIR:-${PROJECT_DIR}/.cache/model_backbones}"
N_JOBS="${N_JOBS:-8}"
BATCH_SIZE="${BATCH_SIZE:-16}"
NUM_WORKERS="${NUM_WORKERS:-0}"
IMAGE_SIZE_OVERRIDE="${IMAGE_SIZE_OVERRIDE:-224}"
AUTO_BATCH_SIZE="${AUTO_BATCH_SIZE:-0}"
AUTO_BATCH_SIZE_MAX="${AUTO_BATCH_SIZE_MAX:-128}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --image)
            IMAGE="$2"
            shift 2
            ;;
        --dicom_dir)
            DICOM_INPUT="$2"
            shift 2
            ;;
        --run_root)
            RUN_ROOT="$2"
            shift 2
            ;;
        --model_dir)
            MODEL_DIR="$2"
            shift 2
            ;;
        --cache_dir)
            CACHE_DIR="$2"
            shift 2
            ;;
        --n_jobs)
            N_JOBS="$2"
            shift 2
            ;;
        --batch_size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --num_workers)
            NUM_WORKERS="$2"
            shift 2
            ;;
        --auto_batch_size)
            AUTO_BATCH_SIZE=1
            shift
            ;;
        --auto_batch_size_max)
            AUTO_BATCH_SIZE_MAX="$2"
            shift 2
            ;;
        --image_size_override)
            IMAGE_SIZE_OVERRIDE="$2"
            shift 2
            ;;
        -h|--help)
            cat <<EOF
Usage: $0 --dicom_dir /path/to/dicoms [options]

Options:
  --image TAG               Docker image tag (default: ${IMAGE})
  --dicom_dir PATH          Input DICOM root (required)
  --run_root PATH           Output root on host (default: ${RUN_ROOT})
  --model_dir PATH          Host model dir (default: ${MODEL_DIR})
  --cache_dir PATH          Host backbone-cache dir (default: ${CACHE_DIR})
  --n_jobs N                DICOM preprocessing workers (default: ${N_JOBS})
  --batch_size N            Inference batch size (default: ${BATCH_SIZE})
  --num_workers N           DataLoader workers (default: ${NUM_WORKERS})
  --auto_batch_size         Auto-select batch size on available GPUs
  --auto_batch_size_max N   Upper bound for auto batch size search (default: ${AUTO_BATCH_SIZE_MAX})
  --image_size_override N   Saliency input size (default: ${IMAGE_SIZE_OVERRIDE})
EOF
            exit 0
            ;;
        *)
            echo "Unknown option: $1" >&2
            exit 1
            ;;
    esac
done

if ! command -v docker >/dev/null 2>&1; then
    echo "ERROR: docker not found in PATH." >&2
    exit 1
fi

if [[ -z "${DICOM_INPUT}" ]]; then
    echo "ERROR: --dicom_dir is required." >&2
    exit 1
fi

if [[ ! -d "${DICOM_INPUT}" ]]; then
    echo "ERROR: DICOM input directory does not exist: ${DICOM_INPUT}" >&2
    exit 1
fi

if [[ ! -d "${MODEL_DIR}" ]]; then
    echo "ERROR: Model directory does not exist: ${MODEL_DIR}" >&2
    exit 1
fi

mkdir -p "${CACHE_DIR}/torch"
mkdir -p "${RUN_ROOT}/processed" "${RUN_ROOT}/output"

echo "============================================================"
echo "RSNA Docker GPU pipeline"
echo "============================================================"
echo "Image:       ${IMAGE}"
echo "DICOM input: ${DICOM_INPUT}"
echo "Run root:    ${RUN_ROOT}"
echo "Model dir:   ${MODEL_DIR}"
echo "Cache dir:   ${CACHE_DIR}"
echo "============================================================"

docker run --rm --gpus all --ipc=host \
    -v "${PROJECT_DIR}:/workspace" \
    -v "${DICOM_INPUT}:/input:ro" \
    -v "${RUN_ROOT}:/run" \
    -v "${MODEL_DIR}:/models:ro" \
    -v "${CACHE_DIR}:/model_cache" \
    -w /workspace \
    -e TORCH_HOME=/model_cache/torch \
    -e PRETRAINEDMODELS_HOME=/model_cache/pretrainedmodels \
    -e N_JOBS="${N_JOBS}" \
    -e BATCH_SIZE="${BATCH_SIZE}" \
    -e NUM_WORKERS="${NUM_WORKERS}" \
    -e AUTO_BATCH_SIZE="${AUTO_BATCH_SIZE}" \
    -e AUTO_BATCH_SIZE_MAX="${AUTO_BATCH_SIZE_MAX}" \
    -e IMAGE_SIZE_OVERRIDE="${IMAGE_SIZE_OVERRIDE}" \
    "${IMAGE}" \
    bash -lc '
set -euo pipefail

python - <<'"'"'PY'"'"'
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available inside Docker container.")
print(f"CUDA device count: {torch.cuda.device_count()}")
PY

echo "[1/3] Preparing DICOM data (SeriesInstanceUID grouping)"
python scripts/prepare_custom_data.py \
    --input_dir /input \
    --output_dir /run/processed \
    --group_by series \
    --n_jobs "${N_JOBS}"

echo "[2/3] Running ensemble inference"
INFER_ARGS=(
    --data_dir /run/processed
    --model_dir /models
    --output /run/output/predictions.csv
    --batch_size "${BATCH_SIZE}"
    --num_workers "${NUM_WORKERS}"
)
if [[ "${AUTO_BATCH_SIZE}" == "1" ]]; then
    INFER_ARGS+=(--auto_batch_size --auto_batch_size_max "${AUTO_BATCH_SIZE_MAX}")
fi
python inference/run_inference.py "${INFER_ARGS[@]}"

echo "[3/3] Exporting saliency panels"
SALIENCY_ARGS=(
    --data_dir /run/processed
    --output_dir /run/output/saliency_panels
    --model_dir /models
    --saliency_mode ensemble
    --ensemble_slice_csv /run/output/predictions_per_slice.csv
)
if [[ -n "${IMAGE_SIZE_OVERRIDE}" ]]; then
    SALIENCY_ARGS+=(--image_size_override "${IMAGE_SIZE_OVERRIDE}")
fi
python inference/export_saliency_panels.py "${SALIENCY_ARGS[@]}"
'

echo "============================================================"
echo "Done"
echo "Predictions: ${RUN_ROOT}/output/predictions.csv"
echo "Per-slice:   ${RUN_ROOT}/output/predictions_per_slice.csv"
echo "Saliency:    ${RUN_ROOT}/output/saliency_panels/"
echo "============================================================"
