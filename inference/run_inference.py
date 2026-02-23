#!/usr/bin/env python3
"""
RSNA 2019 Intracranial Hemorrhage Detection - Inference Pipeline

This script runs the complete inference pipeline on custom CT images:
1. DICOM to PNG conversion (if needed)
2. 2D CNN feature extraction
3. Sequence model prediction
4. Final ensemble output

Usage:
    # From raw DICOM files:
    python inference/run_inference.py --dicom_dir /path/to/dicoms --output predictions.csv

    # From pre-processed data:
    python inference/run_inference.py --data_dir /path/to/processed --output predictions.csv
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import cv2
import albumentations

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / '2DNet' / 'src'))
sys.path.insert(0, str(PROJECT_ROOT / 'SequenceModel'))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Hemorrhage types
HEMORRHAGE_TYPES = ['any', 'epidural', 'intraparenchymal', 'intraventricular', 'subarachnoid', 'subdural']


class InferenceDataset(Dataset):
    """Dataset for inference on PNG images."""

    def __init__(self, image_paths: List[str], image_size: int = 256):
        self.image_paths = image_paths
        self.image_size = image_size
        self.transform = albumentations.Compose([
            albumentations.Resize(image_size, image_size),
            albumentations.Normalize(
                mean=(0.456, 0.456, 0.456),
                std=(0.224, 0.224, 0.224),
                max_pixel_value=255.0
            )
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        filename = os.path.basename(img_path)

        # Load grayscale image
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Could not load image: {img_path}")

        # Resize to 512x512 first (as in original code)
        img = cv2.resize(img, (512, 512))

        # Create 3-channel image (simulating adjacent slices context)
        # In inference without adjacent slices, we just replicate
        img_3ch = np.stack([img, img, img], axis=-1)

        # Apply center crop (ratio=0.8) as in original validation
        h, w = img_3ch.shape[:2]
        target_h, target_w = int(h * 0.8), int(w * 0.8)
        start_x = (w - target_w) // 2
        start_y = (h - target_h) // 2
        img_3ch = img_3ch[start_y:start_y+target_h, start_x:start_x+target_w, :]
        img_3ch = cv2.resize(img_3ch, (w, h))

        # Apply normalization
        augmented = self.transform(image=img_3ch)
        img_tensor = augmented['image'].transpose(2, 0, 1)

        return filename, torch.FloatTensor(img_tensor)


def load_2d_model(model_name: str, model_path: str, device: torch.device) -> nn.Module:
    """Load a pretrained 2D CNN model."""
    from net.models import DenseNet121_change_avg, DenseNet169_change_avg, se_resnext101_32x4d

    model_classes = {
        'DenseNet121_change_avg': DenseNet121_change_avg,
        'DenseNet169_change_avg': DenseNet169_change_avg,
        'se_resnext101_32x4d': se_resnext101_32x4d,
    }

    if model_name not in model_classes:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(model_classes.keys())}")

    # In inference we load full RSNA checkpoints right after construction, so
    # ImageNet backbone download is unnecessary and can be skipped.
    model = model_classes[model_name](use_imagenet_pretrained=False)
    use_dp = device.type == 'cuda' and torch.cuda.device_count() > 1
    if use_dp:
        model = nn.DataParallel(model)

    if os.path.exists(model_path):
        state = torch.load(model_path, map_location=device)
        state_dict = state['state_dict'] if isinstance(state, dict) and 'state_dict' in state else state
        try:
            model.load_state_dict(state_dict)
        except RuntimeError:
            # Handle common DP/non-DP key mismatch by stripping "module." prefix.
            stripped_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith('module.'):
                    stripped_state_dict[k[len('module.'):]] = v
                else:
                    stripped_state_dict[k] = v
            model.load_state_dict(stripped_state_dict, strict=False)
        logger.info(f"Loaded {model_name} from {model_path}")
    else:
        logger.warning(f"Model weights not found: {model_path}")

    model = model.to(device)
    model.eval()
    return model


def extract_features_2d(
    model: nn.Module,
    model_name: str,
    dataloader: DataLoader,
    device: torch.device
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """
    Extract features and predictions from a 2D CNN model.

    Returns:
        features: Dict mapping filename to feature vector
        predictions: Dict mapping filename to prediction probabilities
    """
    features = {}
    predictions = {}

    model.eval()
    with torch.no_grad():
        for filenames, inputs in tqdm(dataloader, desc=f"Extracting {model_name}"):
            inputs = inputs.to(device)
            net = model.module if hasattr(model, 'module') else model

            # Get features based on model type
            if 'DenseNet121' in model_name:
                feat = net.densenet121(inputs)
                feat = net.relu(feat)
                feat = net.avgpool(feat)
                feat = feat.view(feat.size(0), -1)
                out = net.mlp(feat)
            elif 'DenseNet169' in model_name:
                feat = net.densenet169(inputs)
                feat = net.relu(feat)
                feat = net.avgpool(feat)
                feat = feat.view(feat.size(0), -1)
                out = net.mlp(feat)
            elif 'se_resnext' in model_name:
                feat = net.model_ft.layer0(inputs)
                feat = net.model_ft.layer1(feat)
                feat = net.model_ft.layer2(feat)
                feat = net.model_ft.layer3(feat)
                feat = net.model_ft.layer4(feat)
                feat = net.model_ft.avg_pool(feat)
                feat = feat.view(feat.size(0), -1)
                out = net.model_ft.last_linear(feat)
            else:
                # Generic forward pass
                out = model(inputs)
                feat = out

            probs = torch.sigmoid(out)

            for i, fname in enumerate(filenames):
                features[fname] = feat[i].cpu().numpy()
                predictions[fname] = probs[i].cpu().numpy()

    return features, predictions


def run_inference(
    data_dir: str,
    model_dir: str,
    output_path: str,
    batch_size: int = 16,
    num_workers: int = 0,
    models_to_use: Optional[List[str]] = None,
    device: Optional[torch.device] = None
):
    """
    Run the complete inference pipeline.

    Args:
        data_dir: Directory with processed data (png/ and csv/ subdirs)
        model_dir: Directory containing pretrained model weights
        output_path: Path for output CSV file
        batch_size: Batch size for inference
        num_workers: DataLoader worker processes
        models_to_use: List of model names to use (default: all available)
        device: PyTorch device (default: auto-detect)
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {device}")

    data_dir = Path(data_dir)
    model_dir = Path(model_dir)

    # Find PNG files
    png_dir = data_dir / 'png'
    if not png_dir.exists():
        # Try data_dir directly
        png_dir = data_dir
        if not any(png_dir.glob('*.png')):
            raise ValueError(f"No PNG files found in {data_dir}")

    image_paths = sorted(list(png_dir.glob('*.png')))
    logger.info(f"Found {len(image_paths)} PNG files")

    if not image_paths:
        raise ValueError(f"No PNG files found in {png_dir}")

    # Available models with their configurations
    # Note: DenseNet121 uses 512x512, others use 256x256
    model_configs = {
        'DenseNet121_change_avg': {
            'subdir': 'DenseNet121_change_avg_512',
            'image_size': 512
        },
        'DenseNet169_change_avg': {
            'subdir': 'DenseNet169_change_avg_256',
            'image_size': 256
        },
        'se_resnext101_32x4d': {
            'subdir': 'se_resnext101_32x4d_256',
            'image_size': 256
        },
    }

    if models_to_use is None:
        models_to_use = list(model_configs.keys())

    # Collect predictions from all models
    all_predictions = {}

    # Create dataloaders for each image size (to avoid recreating for each model)
    dataloaders = {}
    for model_name in models_to_use:
        img_size = model_configs[model_name]['image_size']
        if img_size not in dataloaders:
            dataset = InferenceDataset([str(p) for p in image_paths], image_size=img_size)
            dataloaders[img_size] = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=(device.type == 'cuda')
            )
            logger.info(f"Created dataloader for {img_size}x{img_size} images")

    for model_name in models_to_use:
        config = model_configs[model_name]
        model_subdir = config['subdir']
        img_size = config['image_size']
        dataloader = dataloaders[img_size]
        model_weights_dir = model_dir / '2DNet' / model_subdir

        # Look for model weights (any fold)
        weight_files = list(model_weights_dir.glob('model_epoch_best_*.pth'))
        if not weight_files:
            weight_files = list(model_weights_dir.glob('*.pth'))

        if not weight_files:
            logger.warning(f"No weights found for {model_name} in {model_weights_dir}")
            continue

        # Use first available weight file
        weight_path = weight_files[0]
        logger.info(f"Loading {model_name} from {weight_path}")

        try:
            model = load_2d_model(model_name, str(weight_path), device)
            features, predictions = extract_features_2d(model, model_name, dataloader, device)
            all_predictions[model_name] = predictions

            # Free GPU memory
            del model
            torch.cuda.empty_cache()

        except Exception as e:
            logger.error(f"Error loading {model_name}: {e}")
            continue

    if not all_predictions:
        raise ValueError("No models were successfully loaded")

    # Ensemble predictions (simple average)
    logger.info("Ensembling predictions...")
    ensemble_predictions = {}

    all_filenames = list(next(iter(all_predictions.values())).keys())
    for fname in all_filenames:
        preds = [all_predictions[m][fname] for m in all_predictions if fname in all_predictions[m]]
        ensemble_predictions[fname] = np.mean(preds, axis=0)

    # Create output DataFrame
    results = []
    for fname in sorted(ensemble_predictions.keys()):
        pred = ensemble_predictions[fname]
        # Remove extension for ID
        slice_id = fname.replace('.png', '').replace('.dcm', '')

        for i, htype in enumerate(HEMORRHAGE_TYPES):
            results.append({
                'ID': f"{slice_id}_{htype}",
                'Label': float(pred[i])
            })

    df_results = pd.DataFrame(results)
    df_results.to_csv(output_path, index=False)
    logger.info(f"Saved predictions to {output_path}")

    # Also save per-slice predictions for easier analysis
    slice_results = []
    for fname in sorted(ensemble_predictions.keys()):
        pred = ensemble_predictions[fname]
        slice_id = fname.replace('.png', '').replace('.dcm', '')
        row = {'ID': slice_id, 'filename': fname}
        for i, htype in enumerate(HEMORRHAGE_TYPES):
            row[htype] = float(pred[i])
        slice_results.append(row)

    df_slices = pd.DataFrame(slice_results)
    slice_output = output_path.replace('.csv', '_per_slice.csv')
    df_slices.to_csv(slice_output, index=False)
    logger.info(f"Saved per-slice predictions to {slice_output}")

    return df_results, df_slices


def main():
    parser = argparse.ArgumentParser(
        description='RSNA 2019 Intracranial Hemorrhage Detection - Inference',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Input options (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        '--dicom_dir',
        type=str,
        help='Directory containing DICOM files (will be preprocessed)'
    )
    input_group.add_argument(
        '--data_dir',
        type=str,
        help='Directory with preprocessed data (png/ and csv/ subdirs)'
    )

    parser.add_argument(
        '--output', '-o',
        type=str,
        default='predictions.csv',
        help='Output CSV file path'
    )
    parser.add_argument(
        '--model_dir',
        type=str,
        default='./models',
        help='Directory containing pretrained model weights'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=16,
        help='Batch size for inference'
    )
    parser.add_argument(
        '--num_workers',
        type=int,
        default=0,
        help='DataLoader worker processes'
    )
    parser.add_argument(
        '--models',
        type=str,
        nargs='+',
        choices=['DenseNet121_change_avg', 'DenseNet169_change_avg', 'se_resnext101_32x4d'],
        help='Models to use for ensemble (default: all available)'
    )
    parser.add_argument(
        '--cpu',
        action='store_true',
        help='Force CPU inference'
    )

    args = parser.parse_args()

    # Determine device
    if args.cpu:
        device = torch.device('cpu')
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Handle DICOM input
    if args.dicom_dir:
        from scripts.prepare_custom_data import prepare_data
        logger.info(f"Preprocessing DICOM files from {args.dicom_dir}")

        # Create temporary processed directory
        processed_dir = Path(args.dicom_dir).parent / 'processed_data'
        prepare_data(args.dicom_dir, str(processed_dir), group_by='series')
        data_dir = str(processed_dir)
    else:
        data_dir = args.data_dir

    # Run inference
    run_inference(
        data_dir=data_dir,
        model_dir=args.model_dir,
        output_path=args.output,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        models_to_use=args.models,
        device=device
    )

    print("\n" + "="*60)
    print("Inference complete!")
    print("="*60)
    print(f"\nOutput files:")
    print(f"  Submission format: {args.output}")
    print(f"  Per-slice format:  {args.output.replace('.csv', '_per_slice.csv')}")
    print("\nPrediction columns:")
    print("  any, epidural, intraparenchymal, intraventricular, subarachnoid, subdural")
    print("  Values are probabilities (0-1) for each hemorrhage type")
    print("="*60)


if __name__ == '__main__':
    main()
