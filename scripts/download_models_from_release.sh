#!/bin/bash
# Download RSNA 2D model weights from GitHub release assets.
#
# Usage:
#   bash scripts/download_models_from_release.sh
#   bash scripts/download_models_from_release.sh --tag model-weights-v1 --force
#
# Optional environment variables:
#   RSNA_GH_REPO   (default: jwprescott/RSNA2019_Intracranial-Hemorrhage-Detection)
#   RSNA_REL_TAG   (default: model-weights-v1)
#   GITHUB_TOKEN or GITHUB_TOKEN_RSNA_ICH_DETECTION_REPO (optional for private access)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

REPO="${RSNA_GH_REPO:-jwprescott/RSNA2019_Intracranial-Hemorrhage-Detection}"
TAG="${RSNA_REL_TAG:-model-weights-v1}"
FORCE=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --tag)
            TAG="$2"
            shift 2
            ;;
        --repo)
            REPO="$2"
            shift 2
            ;;
        --force)
            FORCE=1
            shift
            ;;
        -h|--help)
            cat <<EOF
Usage: $0 [--tag TAG] [--repo OWNER/REPO] [--force]

Downloads 2D model weights from GitHub release assets and extracts to:
  models/2DNet/

Defaults:
  --repo  ${REPO}
  --tag   ${TAG}
EOF
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

MODELS_DIR="${PROJECT_DIR}/models/2DNet"
STAGING_DIR="${PROJECT_DIR}/tmp/release_download_${TAG}"
mkdir -p "${MODELS_DIR}" "${STAGING_DIR}"

D121_DIR="${MODELS_DIR}/DenseNet121_change_avg_512"
D169_DIR="${MODELS_DIR}/DenseNet169_change_avg_256"
SRX_DIR="${MODELS_DIR}/se_resnext101_32x4d_256"

if [[ ${FORCE} -eq 0 ]] \
  && [[ -f "${D121_DIR}/model_epoch_best_0.pth" ]] \
  && [[ -f "${D169_DIR}/model_epoch_best_0.pth" ]] \
  && [[ -f "${SRX_DIR}/model_epoch_best_0.pth" ]]; then
    echo "Model weights already present. Use --force to re-download."
    exit 0
fi

TOKEN="${GITHUB_TOKEN:-${GITHUB_TOKEN_RSNA_ICH_DETECTION_REPO:-}}"
AUTH_HEADER=()
if [[ -n "${TOKEN}" ]]; then
    AUTH_HEADER=(-H "Authorization: Bearer ${TOKEN}")
fi

BASE_URL="https://github.com/${REPO}/releases/download/${TAG}"

FILES=(
    "rsna-ich-2d-densenet121-change-avg-512.tar.gz"
    "rsna-ich-2d-densenet169-change-avg-256.tar.gz"
    "rsna-ich-2d-se-resnext101-32x4d-256.tar.gz"
    "rsna-ich-2d-weights-sha256.txt"
)

echo "Downloading release assets from ${REPO} tag ${TAG}"
for f in "${FILES[@]}"; do
    url="${BASE_URL}/${f}"
    out="${STAGING_DIR}/${f}"
    echo "  -> ${f}"
    curl -fL "${AUTH_HEADER[@]}" -o "${out}" "${url}"
done

echo "Verifying checksums..."
(
    cd "${STAGING_DIR}"
    sha256sum -c rsna-ich-2d-weights-sha256.txt
)

echo "Extracting model archives..."
tar -C "${MODELS_DIR}" -xzf "${STAGING_DIR}/rsna-ich-2d-densenet121-change-avg-512.tar.gz"
tar -C "${MODELS_DIR}" -xzf "${STAGING_DIR}/rsna-ich-2d-densenet169-change-avg-256.tar.gz"
tar -C "${MODELS_DIR}" -xzf "${STAGING_DIR}/rsna-ich-2d-se-resnext101-32x4d-256.tar.gz"

echo "Done. Model weights installed in ${MODELS_DIR}"
ls -lh "${D121_DIR}"/*.pth | head -n 1
ls -lh "${D169_DIR}"/*.pth | head -n 1
ls -lh "${SRX_DIR}"/*.pth | head -n 1
