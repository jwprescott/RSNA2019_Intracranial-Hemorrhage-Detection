#!/bin/bash
# ============================================================
# Build Singularity container for RSNA 2019 Hemorrhage Detection
# ============================================================
#
# This script builds the Singularity container using one of several methods:
#   1. Pull from Docker Hub (no privileges needed) - DEFAULT
#   2. Build with --fakeroot (if available)
#   3. Build with sudo (if available)
#
# Usage:
#   bash singularity/build_container.sh [--pull|--fakeroot|--sudo]
#
# ============================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

echo "============================================================"
echo "Building Singularity container for RSNA 2019"
echo "============================================================"

cd "${SCRIPT_DIR}"

# Parse arguments
BUILD_METHOD="${1:-pull}"

case "$BUILD_METHOD" in
    --pull|-p|pull)
        BUILD_METHOD="pull"
        ;;
    --fakeroot|-f|fakeroot)
        BUILD_METHOD="fakeroot"
        ;;
    --sudo|-s|sudo)
        BUILD_METHOD="sudo"
        ;;
    --help|-h)
        echo "Usage: $0 [--pull|--fakeroot|--sudo]"
        echo ""
        echo "Methods:"
        echo "  --pull     Pull and convert Docker image (default, no privileges needed)"
        echo "  --fakeroot Build from .def file using fakeroot"
        echo "  --sudo     Build from .def file using sudo"
        exit 0
        ;;
    *)
        echo "Unknown option: $BUILD_METHOD"
        echo "Use --help for usage information"
        exit 1
        ;;
esac

# Method 1: Pull from Docker Hub (RECOMMENDED - no privileges needed)
if [ "$BUILD_METHOD" = "pull" ]; then
    echo ""
    echo "Method: Pulling from Docker Hub and installing packages"
    echo "This requires no special privileges."
    echo ""

    # First, pull the base PyTorch image
    echo "Step 1/2: Pulling PyTorch base image..."
    singularity pull --force pytorch_base.sif docker://pytorch/pytorch:1.13.1-cuda11.6-cudnn8-runtime

    echo ""
    echo "Step 2/2: Creating overlay with additional packages..."

    # Create a wrapper script that installs packages at runtime
    cat > "${PROJECT_DIR}/singularity/setup_env.sh" << 'SETUP_EOF'
#!/bin/bash
# Setup script - run once inside the container to install packages
pip install --user --no-cache-dir \
    pretrainedmodels>=0.7.4 \
    efficientnet-pytorch>=0.7.1 \
    pydicom>=2.3.0 \
    albumentations>=1.3.0 \
    opencv-contrib-python-headless>=4.5.0 \
    scikit-image>=0.19.0 \
    scikit-learn>=1.0.0 \
    pandas>=1.3.0 \
    "numpy>=1.21.0,<2.0.0" \
    Pillow>=9.0.0 \
    tqdm>=4.64.0 \
    joblib>=1.1.0 \
    gdown>=4.6.0

echo "Setup complete! Packages installed to ~/.local"
SETUP_EOF
    chmod +x "${PROJECT_DIR}/singularity/setup_env.sh"

    # Rename to final name
    mv pytorch_base.sif rsna2019.sif

    echo ""
    echo "============================================================"
    echo "SUCCESS: Container created at ${SCRIPT_DIR}/rsna2019.sif"
    echo "============================================================"
    echo ""
    echo "IMPORTANT: First-time setup required!"
    echo "Run this ONCE to install Python packages:"
    echo ""
    echo "  singularity exec rsna2019.sif bash singularity/setup_env.sh"
    echo ""
    echo "This installs packages to ~/.local (persists across runs)."
    echo ""
    exit 0
fi

# Method 2: Build with fakeroot
if [ "$BUILD_METHOD" = "fakeroot" ]; then
    echo ""
    echo "Method: Building with --fakeroot"
    echo ""

    if [ ! -f "rsna2019.def" ]; then
        echo "ERROR: rsna2019.def not found"
        exit 1
    fi

    singularity build --fakeroot rsna2019.sif rsna2019.def
fi

# Method 3: Build with sudo
if [ "$BUILD_METHOD" = "sudo" ]; then
    echo ""
    echo "Method: Building with sudo"
    echo ""

    if [ ! -f "rsna2019.def" ]; then
        echo "ERROR: rsna2019.def not found"
        exit 1
    fi

    sudo singularity build rsna2019.sif rsna2019.def
fi

echo ""
echo "============================================================"
echo "SUCCESS: Container built at ${SCRIPT_DIR}/rsna2019.sif"
echo "============================================================"
echo ""
echo "Next steps:"
echo "  1. Download pretrained models:"
echo "     bash scripts/download_models.sh"
echo ""
echo "  2. Prepare your DICOM data:"
echo "     singularity exec --nv rsna2019.sif python scripts/prepare_custom_data.py \\"
echo "         --input_dir /path/to/dicoms --output_dir /path/to/processed"
echo ""
echo "  3. Run inference:"
echo "     sbatch singularity/slurm_inference.sh /path/to/processed predictions.csv"
