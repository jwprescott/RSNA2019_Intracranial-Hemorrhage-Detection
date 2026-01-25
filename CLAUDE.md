# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is the **first-place solution** for the RSNA2019 Intracranial Hemorrhage Detection Challenge - a medical AI system for detecting acute intracranial hemorrhages in head CT scans across 6 categories (any, epidural, intraparenchymal, intraventricular, subarachnoid, subdural).

## Current State

The project has been adapted for **inference on custom CT data** using **Singularity containers** for HPC/SLURM clusters. The 3DNet component is skipped (incomplete in this repo) - using 2DNet ensemble only.

## Architecture

The solution uses a **two-stage pipeline** (for inference):

1. **2DNet** - 2D CNN feature extraction from individual CT slices (3 models ensemble)
2. **SequenceModel** - LSTM-based temporal modeling and final ensemble (optional)

Data flows: DICOM → PNG → 2D CNN ensemble → Final predictions

### Pretrained Models
| Model | Resolution | Directory |
|-------|------------|-----------|
| SE-ResNeXt101 | 256x256 | `models/2DNet/se_resnext101_32x4d_256/` |
| DenseNet169 | 256x256 | `models/2DNet/DenseNet169_change_avg_256/` |
| DenseNet121 | 512x512 | `models/2DNet/DenseNet121_change_avg_512/` |

**Note:** DenseNet121 uses 512x512 images, others use 256x256.

## Running Inference (Singularity)

### Quick Start
```bash
# Build container
cd singularity && bash build_container.sh

# Install packages (once)
singularity exec rsna2019.sif bash singularity/setup_env.sh

# Run inference
singularity exec --nv rsna2019.sif python inference/run_inference.py \
    --data_dir /path/to/processed --output predictions.csv
```

### SLURM Jobs
```bash
# Data preparation (CPU)
sbatch singularity/slurm_prepare_data.sh /path/to/dicoms /path/to/processed

# Inference (GPU)
sbatch singularity/slurm_inference.sh /path/to/processed predictions.csv
```

## Key Files

### Infrastructure (New)
- `singularity/rsna2019.def` - Singularity container definition
- `singularity/build_container.sh` - Container build script (uses `singularity pull`)
- `singularity/setup_env.sh` - Installs Python packages to ~/.local
- `singularity/slurm_inference.sh` - SLURM GPU job script
- `singularity/slurm_prepare_data.sh` - SLURM CPU job script

### Inference Pipeline (New)
- `inference/run_inference.py` - Main inference entry point
- `scripts/prepare_custom_data.py` - DICOM preprocessing for custom data
- `scripts/download_models.sh` - Downloads pretrained weights from Google Drive

### Original Code
- `2DNet/src/train.py` - 2D model training
- `2DNet/src/predict.py` - 2D model prediction
- `2DNet/src/net/models.py` - Model architectures
- `SequenceModel/main.py` - Sequence model training

## Dependencies

Container uses PyTorch 1.13.1 + CUDA 11.6. Key packages in `requirements-inference.txt`:
- pretrainedmodels, efficientnet-pytorch
- pydicom, albumentations, opencv-contrib-python
- scikit-image, scikit-learn, pandas, numpy

## Output Format

Predictions for 6 hemorrhage types (probabilities 0-1):
- `any`, `epidural`, `intraparenchymal`, `intraventricular`, `subarachnoid`, `subdural`

Output files:
- `predictions.csv` - Kaggle submission format
- `predictions_per_slice.csv` - Per-slice analysis format

## Remote Development

Connect to cluster compute node via VS Code tunnel:
```bash
# On cluster: request GPU node and start tunnel
srun --partition=gpu --gres=gpu:1 --mem=32G --time=04:00:00 --pty bash
~/code tunnel

# In VS Code: Remote-Tunnels: Connect to Tunnel...
```

See README.md for detailed setup instructions.

## Medical AI Context

This is a competition-winning medical imaging solution. When modifying code, consider:
- Model architectures are specifically tuned for hemorrhage detection patterns
- Cross-validation and ensemble strategies are critical for medical AI robustness
- DenseNet121 was trained at 512x512, other models at 256x256