#!/bin/bash
# Download pretrained models for RSNA 2019 Intracranial Hemorrhage Detection
#
# Usage: bash scripts/download_models.sh [--all|--2d|--sequence|--csv]
#
# Options:
#   --all       Download everything (default)
#   --2d        Download only 2D CNN models
#   --sequence  Download only sequence model weights
#   --csv       Download only CSV data files

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Base directories
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
MODELS_DIR="${PROJECT_DIR}/models"
DATA_DIR="${PROJECT_DIR}/data"

# Google Drive file IDs
# 2D Models
SERESNEXT101_ID="18Py5eW1E4hSbTT6658IAjQjJGS28grdx"
DENSENET169_ID="1vCsX12pMZxBmuGGNVnjFFiZ-5u5vD-h6"
DENSENET121_ID="1o0ok-6I2hY1ygSWdZOKmSD84FsEpgDaa"

# CSV Data
DATA_ZIP_ID="1buISR_b3HQDU4KeNc_DmvKTYJ1gvj5-3"
CSV_ZIP_ID="1qYi4k-DuOLJmyZ7uYYrnomU2U7MrYRBV"

# Create directories
mkdir -p "${MODELS_DIR}/2DNet/se_resnext101_32x4d_256"
mkdir -p "${MODELS_DIR}/2DNet/DenseNet169_change_avg_256"
mkdir -p "${MODELS_DIR}/2DNet/DenseNet121_change_avg_512"
mkdir -p "${MODELS_DIR}/SequenceModel"
mkdir -p "${DATA_DIR}"

echo -e "${GREEN}======================================${NC}"
echo -e "${GREEN}RSNA 2019 Model Downloader${NC}"
echo -e "${GREEN}======================================${NC}"

# Check if gdown is installed
if ! command -v gdown &> /dev/null; then
    echo -e "${YELLOW}Installing gdown...${NC}"
    pip install gdown
fi

download_2d_models() {
    echo -e "\n${GREEN}Downloading 2D CNN Models...${NC}"

    # SE-ResNeXt101
    echo -e "${YELLOW}[1/3] Downloading SE-ResNeXt101 (256x256)...${NC}"
    if [ ! -f "${MODELS_DIR}/2DNet/se_resnext101_32x4d_256/model_epoch_best_0.pth" ]; then
        gdown --id "${SERESNEXT101_ID}" -O "${MODELS_DIR}/2DNet/se_resnext101_32x4d.zip" --fuzzy || {
            echo -e "${RED}Failed to download SE-ResNeXt101. Try downloading manually from:${NC}"
            echo "https://drive.google.com/open?id=${SERESNEXT101_ID}"
        }
        if [ -f "${MODELS_DIR}/2DNet/se_resnext101_32x4d.zip" ]; then
            unzip -o "${MODELS_DIR}/2DNet/se_resnext101_32x4d.zip" -d "${MODELS_DIR}/2DNet/se_resnext101_32x4d_256/"
            rm "${MODELS_DIR}/2DNet/se_resnext101_32x4d.zip"
        fi
    else
        echo -e "${GREEN}SE-ResNeXt101 already downloaded${NC}"
    fi

    # DenseNet169
    echo -e "${YELLOW}[2/3] Downloading DenseNet169 (256x256)...${NC}"
    if [ ! -f "${MODELS_DIR}/2DNet/DenseNet169_change_avg_256/model_epoch_best_0.pth" ]; then
        gdown --id "${DENSENET169_ID}" -O "${MODELS_DIR}/2DNet/densenet169.zip" --fuzzy || {
            echo -e "${RED}Failed to download DenseNet169. Try downloading manually from:${NC}"
            echo "https://drive.google.com/open?id=${DENSENET169_ID}"
        }
        if [ -f "${MODELS_DIR}/2DNet/densenet169.zip" ]; then
            unzip -o "${MODELS_DIR}/2DNet/densenet169.zip" -d "${MODELS_DIR}/2DNet/DenseNet169_change_avg_256/"
            rm "${MODELS_DIR}/2DNet/densenet169.zip"
        fi
    else
        echo -e "${GREEN}DenseNet169 already downloaded${NC}"
    fi

    # DenseNet121 (512x512 resolution - different from the other models!)
    echo -e "${YELLOW}[3/3] Downloading DenseNet121 (512x512)...${NC}"
    if [ ! -f "${MODELS_DIR}/2DNet/DenseNet121_change_avg_512/model_epoch_best_0.pth" ]; then
        gdown --id "${DENSENET121_ID}" -O "${MODELS_DIR}/2DNet/densenet121.zip" --fuzzy || {
            echo -e "${RED}Failed to download DenseNet121. Try downloading manually from:${NC}"
            echo "https://drive.google.com/open?id=${DENSENET121_ID}"
        }
        if [ -f "${MODELS_DIR}/2DNet/densenet121.zip" ]; then
            unzip -o "${MODELS_DIR}/2DNet/densenet121.zip" -d "${MODELS_DIR}/2DNet/DenseNet121_change_avg_512/"
            rm "${MODELS_DIR}/2DNet/densenet121.zip"
        fi
    else
        echo -e "${GREEN}DenseNet121 already downloaded${NC}"
    fi
}

download_csv_data() {
    echo -e "\n${GREEN}Downloading CSV Data Files...${NC}"

    # Data CSV
    echo -e "${YELLOW}[1/2] Downloading data.zip...${NC}"
    if [ ! -d "${DATA_DIR}/fold_5_by_study" ]; then
        gdown --id "${DATA_ZIP_ID}" -O "${DATA_DIR}/data.zip" --fuzzy || {
            echo -e "${RED}Failed to download data.zip. Try downloading manually from:${NC}"
            echo "https://drive.google.com/open?id=${DATA_ZIP_ID}"
        }
        if [ -f "${DATA_DIR}/data.zip" ]; then
            unzip -o "${DATA_DIR}/data.zip" -d "${DATA_DIR}/"
            rm "${DATA_DIR}/data.zip"
        fi
    else
        echo -e "${GREEN}data.zip already downloaded${NC}"
    fi

    # CSV for sequence model
    echo -e "${YELLOW}[2/2] Downloading csv.zip (for sequence model)...${NC}"
    if [ ! -d "${DATA_DIR}/csv" ]; then
        gdown --id "${CSV_ZIP_ID}" -O "${DATA_DIR}/csv.zip" --fuzzy || {
            echo -e "${RED}Failed to download csv.zip. Try downloading manually from:${NC}"
            echo "https://drive.google.com/open?id=${CSV_ZIP_ID}"
        }
        if [ -f "${DATA_DIR}/csv.zip" ]; then
            unzip -o "${DATA_DIR}/csv.zip" -d "${DATA_DIR}/"
            rm "${DATA_DIR}/csv.zip"
        fi
    else
        echo -e "${GREEN}csv.zip already downloaded${NC}"
    fi
}

# Parse command line arguments
DOWNLOAD_ALL=true
DOWNLOAD_2D=false
DOWNLOAD_SEQUENCE=false
DOWNLOAD_CSV=false

for arg in "$@"; do
    case $arg in
        --all)
            DOWNLOAD_ALL=true
            ;;
        --2d)
            DOWNLOAD_ALL=false
            DOWNLOAD_2D=true
            ;;
        --sequence)
            DOWNLOAD_ALL=false
            DOWNLOAD_SEQUENCE=true
            ;;
        --csv)
            DOWNLOAD_ALL=false
            DOWNLOAD_CSV=true
            ;;
        -h|--help)
            echo "Usage: $0 [--all|--2d|--sequence|--csv]"
            echo ""
            echo "Options:"
            echo "  --all       Download everything (default)"
            echo "  --2d        Download only 2D CNN models"
            echo "  --sequence  Download only sequence model weights"
            echo "  --csv       Download only CSV data files"
            exit 0
            ;;
        *)
            echo "Unknown option: $arg"
            exit 1
            ;;
    esac
done

# Execute downloads
if $DOWNLOAD_ALL || $DOWNLOAD_2D; then
    download_2d_models
fi

if $DOWNLOAD_ALL || $DOWNLOAD_CSV; then
    download_csv_data
fi

echo -e "\n${GREEN}======================================${NC}"
echo -e "${GREEN}Download complete!${NC}"
echo -e "${GREEN}======================================${NC}"
echo -e "\nModel locations:"
echo -e "  2D Models:       ${MODELS_DIR}/2DNet/"
echo -e "  Sequence Model:  ${MODELS_DIR}/SequenceModel/"
echo -e "  Data files:      ${DATA_DIR}/"
echo -e "\n${YELLOW}Note: Sequence model weights are generated by training."
echo -e "For inference with pre-trained sequence model, you may need to"
echo -e "train first or obtain weights from the original authors.${NC}"
