# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is the **first-place solution** for the RSNA2019 Intracranial Hemorrhage Detection Challenge - a medical AI system for detecting acute intracranial hemorrhages in head CT scans across 6 categories (any, epidural, intraparenchymal, intraventricular, subarachnoid, subdural).

## Architecture

The solution uses a **three-stage pipeline**:

1. **2DNet** - 2D CNN feature extraction from individual CT slices
2. **3DNet** - 3D ConvNet processing of volumetric sequences  
3. **SequenceModel** - LSTM-based temporal modeling and final ensemble

Data flows: DICOM → PNG → 2D CNN features → 3D processing → Sequence modeling → Final predictions

## Development Commands

### Data Preparation
```bash
# Convert DICOM to PNG (required first step)
cd 2DNet/src
python3 prepare_data.py -dcm_path stage_1_train_images -png_path train_png
```

### 2D Model Training
```bash
cd 2DNet/src
# DenseNet models
python3 train.py -backbone DenseNet121_change_avg -img_size 256 -tbs 256 -vbs 128 -save_path DenseNet121_change_avg_256
python3 train.py -backbone DenseNet169_change_avg -img_size 256 -tbs 256 -vbs 128 -save_path DenseNet169_change_avg_256

# SE-ResNeXt model  
python3 train.py -backbone se_resnext101_32x4d -img_size 256 -tbs 80 -vbs 40 -save_path se_resnext101_32x4d_256

# Generate predictions
python3 predict.py -backbone DenseNet121_change_avg -img_size 256 -tbs 4 -vbs 4 -spth DenseNet121_change_avg_256
```

### 3D Model Training
```bash
cd 3DNet
python train_RSNA19.py
```

### Sequence Model Training
```bash
cd SequenceModel
CUDA_VISIBLE_DEVICES=0 python main.py
```

## Key Dependencies

- PyTorch 1.1.0 + torchvision 0.3.0
- pandas 0.25.0, numpy 1.22.0
- albumentations, opencv-contrib-python
- scikit-image 0.14.2

## Important Files

- **Entry Points**: `2DNet/src/train.py`, `3DNet/train_RSNA19.py`, `SequenceModel/main.py`
- **Settings**: Each module has its own `settings.py` with hyperparameters
- **Data Processing**: `2DNet/src/prepare_data.py` for DICOM→PNG conversion
- **Model Definitions**: `2DNet/src/net/models.py`, `3DNet/model.py`, `SequenceModel/seq_model.py`

## Output Structure

- 2D model outputs: `./SingleModelOutput/`
- Final predictions: `./FinalSubmission/final_version/submission_tta.csv`
- Uses 5-fold cross-validation with test-time augmentation (TTA)

## Medical AI Context

This is a competition-winning medical imaging solution. When modifying code, consider:
- Model architectures are specifically tuned for hemorrhage detection patterns
- The three-stage pipeline captures different scales of medical features
- Cross-validation and ensemble strategies are critical for medical AI robustness