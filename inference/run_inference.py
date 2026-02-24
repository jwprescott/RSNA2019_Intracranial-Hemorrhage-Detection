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
import json
import hashlib
from datetime import datetime, timezone
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
torch.backends.cudnn.benchmark = True

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
        logger.info(
            "Enabled DataParallel for %s across %d visible GPUs",
            model_name,
            torch.cuda.device_count()
        )

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


def _is_oom_error(exc: RuntimeError) -> bool:
    msg = str(exc).lower()
    oom_markers = (
        'out of memory',
        'cuda error: out of memory',
        'cudnn_status_alloc_failed',
        'cudnn_status_not_supported',
        'cudnn_status_internal_error',
    )
    return any(marker in msg for marker in oom_markers)


def _forward_features_and_logits(
    model: nn.Module,
    model_name: str,
    inputs: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    net = model.module if hasattr(model, 'module') else model

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
        out = model(inputs)
        feat = out
    return feat, out


def _can_run_batch_size(
    model: nn.Module,
    model_name: str,
    device: torch.device,
    image_size: int,
    batch_size: int
) -> bool:
    try:
        with torch.no_grad():
            x = torch.zeros((batch_size, 3, image_size, image_size), device=device)
            _, out = _forward_features_and_logits(model, model_name, x)
            # Force evaluation so kernel launches are realized.
            _ = out.mean().item()
        del x
        del out
        if device.type == 'cuda':
            torch.cuda.synchronize()
        return True
    except RuntimeError as exc:
        if _is_oom_error(exc):
            if device.type == 'cuda':
                torch.cuda.empty_cache()
            return False
        raise


def _auto_select_batch_size_for_model(
    model_name: str,
    weight_path: Path,
    image_size: int,
    device: torch.device,
    max_batch_size: int
) -> int:
    model = load_2d_model(model_name, str(weight_path), device)
    try:
        if not _can_run_batch_size(model, model_name, device, image_size, 1):
            raise RuntimeError(
                f"GPU OOM for {model_name} even at batch_size=1 (image_size={image_size})."
            )

        if _can_run_batch_size(model, model_name, device, image_size, max_batch_size):
            return max_batch_size

        low = 1
        high = max_batch_size
        while low < high:
            mid = (low + high + 1) // 2
            if _can_run_batch_size(model, model_name, device, image_size, mid):
                low = mid
            else:
                high = mid - 1
        return low
    finally:
        del model
        if device.type == 'cuda':
            torch.cuda.empty_cache()


def _auto_select_batch_size(
    model_configs: Dict[str, Dict[str, object]],
    model_weight_files: Dict[str, List[Path]],
    models_to_use: List[str],
    device: torch.device,
    max_batch_size: int
) -> int:
    per_model_limits = []
    for model_name in models_to_use:
        if model_name not in model_weight_files or not model_weight_files[model_name]:
            continue
        image_size = int(model_configs[model_name]['image_size'])
        first_weight = model_weight_files[model_name][0]
        logger.info(
            "Auto-tuning batch size for %s (%dx%d) using %s",
            model_name,
            image_size,
            image_size,
            first_weight.name
        )
        model_limit = _auto_select_batch_size_for_model(
            model_name=model_name,
            weight_path=first_weight,
            image_size=image_size,
            device=device,
            max_batch_size=max_batch_size
        )
        logger.info("Auto-tuned max batch size for %s: %d", model_name, model_limit)
        per_model_limits.append(model_limit)

    if not per_model_limits:
        raise ValueError("Auto batch size could not be determined: no valid model checkpoints found.")

    final_bs = max(1, min(per_model_limits))
    if device.type == 'cuda' and torch.cuda.device_count() > 1 and final_bs < torch.cuda.device_count():
        logger.warning(
            "Auto-selected batch size (%d) is smaller than visible GPU count (%d); "
            "DataParallel may not fully saturate all GPUs.",
            final_bs,
            torch.cuda.device_count()
        )
    logger.info("Auto-selected global batch size (min across backbones): %d", final_bs)
    return final_bs


def _now_iso_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _auto_batch_cache_key(
    models_to_use: List[str],
    model_configs: Dict[str, Dict[str, object]],
    model_weight_files: Dict[str, List[Path]],
    max_folds_per_model: Optional[int],
    auto_batch_size_max: int,
    device: torch.device
) -> str:
    gpu_signature = []
    if device.type == 'cuda':
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            gpu_signature.append({
                'index': i,
                'name': props.name,
                'total_memory': int(props.total_memory),
            })

    payload = {
        'version': 1,
        'gpus': gpu_signature,
        'models': sorted(models_to_use),
        'image_sizes': {m: int(model_configs[m]['image_size']) for m in sorted(models_to_use)},
        'fold_counts': {
            m: len(model_weight_files.get(m, []))
            for m in sorted(models_to_use)
        },
        'max_folds_per_model': max_folds_per_model,
        'auto_batch_size_max': int(auto_batch_size_max),
    }
    raw = json.dumps(payload, sort_keys=True).encode('utf-8')
    return hashlib.sha256(raw).hexdigest()


def _load_auto_batch_cache(cache_path: Path) -> Dict[str, dict]:
    if not cache_path.exists():
        return {}
    try:
        data = json.loads(cache_path.read_text())
        if isinstance(data, dict):
            return data
    except Exception as exc:
        logger.warning("Could not parse auto-batch cache at %s: %s", cache_path, exc)
    return {}


def _write_auto_batch_cache(cache_path: Path, cache_data: Dict[str, dict]) -> None:
    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = cache_path.with_suffix(cache_path.suffix + '.tmp')
        tmp_path.write_text(json.dumps(cache_data, indent=2, sort_keys=True))
        tmp_path.replace(cache_path)
    except Exception as exc:
        logger.warning("Failed to write auto-batch cache at %s: %s", cache_path, exc)


def _get_cached_auto_batch_size(cache_path: Path, cache_key: str) -> Optional[int]:
    cache = _load_auto_batch_cache(cache_path)
    entry = cache.get(cache_key)
    if not isinstance(entry, dict):
        return None
    value = entry.get('batch_size')
    if isinstance(value, int) and value > 0:
        return value
    return None


def _set_cached_auto_batch_size(
    cache_path: Path,
    cache_key: str,
    batch_size: int,
    note: str
) -> None:
    cache = _load_auto_batch_cache(cache_path)
    cache[cache_key] = {
        'batch_size': int(batch_size),
        'updated_at': _now_iso_utc(),
        'note': note,
    }
    _write_auto_batch_cache(cache_path, cache)


def _create_inference_dataloader(
    image_paths: List[str],
    image_size: int,
    batch_size: int,
    num_workers: int,
    device: torch.device
) -> DataLoader:
    dataset = InferenceDataset(image_paths, image_size=image_size)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == 'cuda')
    )


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
            feat, out = _forward_features_and_logits(model, model_name, inputs)

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
    max_folds_per_model: Optional[int] = None,
    auto_batch_size: bool = False,
    auto_batch_size_max: int = 128,
    auto_batch_size_cache_path: Optional[str] = None,
    disable_auto_batch_size_cache: bool = False,
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
        max_folds_per_model: Max fold checkpoints to use per backbone (default: all available)
        auto_batch_size: Auto-tune batch size for GPU inference
        auto_batch_size_max: Upper bound used for auto batch size search
        auto_batch_size_cache_path: Optional JSON cache path for auto batch size reuse
        disable_auto_batch_size_cache: Disable reading/writing auto batch cache
        device: PyTorch device (default: auto-detect)
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {device}")

    if max_folds_per_model is not None and max_folds_per_model <= 0:
        raise ValueError("max_folds_per_model must be > 0 when provided")
    if auto_batch_size_max <= 0:
        raise ValueError("auto_batch_size_max must be > 0")

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

    if device.type == 'cuda':
        gpu_count = torch.cuda.device_count()
        gpu_names = [torch.cuda.get_device_name(i) for i in range(gpu_count)]
        logger.info("Visible CUDA GPUs (%d): %s", gpu_count, ", ".join(gpu_names))

    model_weight_files: Dict[str, List[Path]] = {}
    for model_name in models_to_use:
        config = model_configs[model_name]
        model_subdir = config['subdir']
        model_weights_dir = model_dir / '2DNet' / model_subdir

        # Look for model weights across all available folds.
        weight_files = sorted(model_weights_dir.glob('model_epoch_best_*.pth'))
        if not weight_files:
            weight_files = sorted(model_weights_dir.glob('*.pth'))

        if not weight_files:
            logger.warning(f"No weights found for {model_name} in {model_weights_dir}")
            continue

        if max_folds_per_model is not None and len(weight_files) > max_folds_per_model:
            logger.info(
                "Limiting %s folds from %d to %d (via --max_folds_per_model)",
                model_name,
                len(weight_files),
                max_folds_per_model
            )
            weight_files = weight_files[:max_folds_per_model]

        model_weight_files[model_name] = weight_files

    auto_batch_cache_path: Optional[Path] = None
    auto_batch_cache_key: Optional[str] = None

    if auto_batch_size:
        if device.type != 'cuda':
            logger.warning("Ignoring --auto_batch_size on non-CUDA device; using provided batch_size=%d", batch_size)
        else:
            auto_batch_cache_path = Path(
                auto_batch_size_cache_path
                if auto_batch_size_cache_path is not None
                else (PROJECT_ROOT / '.cache' / 'auto_batch_size_cache.json')
            )
            auto_batch_cache_key = _auto_batch_cache_key(
                models_to_use=models_to_use,
                model_configs=model_configs,
                model_weight_files=model_weight_files,
                max_folds_per_model=max_folds_per_model,
                auto_batch_size_max=auto_batch_size_max,
                device=device,
            )

            cached_batch_size = None
            if not disable_auto_batch_size_cache:
                cached_batch_size = _get_cached_auto_batch_size(
                    auto_batch_cache_path, auto_batch_cache_key
                )

            if cached_batch_size is not None:
                batch_size = min(cached_batch_size, auto_batch_size_max)
                logger.info(
                    "Using cached auto batch size: %d (cache=%s)",
                    batch_size,
                    auto_batch_cache_path,
                )
            else:
                batch_size = _auto_select_batch_size(
                    model_configs=model_configs,
                    model_weight_files=model_weight_files,
                    models_to_use=models_to_use,
                    device=device,
                    max_batch_size=auto_batch_size_max
                )
                if not disable_auto_batch_size_cache:
                    _set_cached_auto_batch_size(
                        auto_batch_cache_path,
                        auto_batch_cache_key,
                        batch_size,
                        note='auto_tuned'
                    )
    logger.info("Using inference batch size: %d", batch_size)

    # Collect predictions from all models
    all_predictions = {}
    image_path_strs = [str(p) for p in image_paths]
    current_batch_size = batch_size

    for model_name in models_to_use:
        config = model_configs[model_name]
        model_subdir = config['subdir']
        img_size = config['image_size']
        model_weights_dir = model_dir / '2DNet' / model_subdir
        weight_files = model_weight_files.get(model_name, [])
        if not weight_files:
            logger.warning(f"No weights found for {model_name} in {model_weights_dir}")
            continue

        logger.info(
            "Loading %d fold checkpoint(s) for %s from %s",
            len(weight_files),
            model_name,
            model_weights_dir
        )

        fold_predictions = []
        for fold_idx, weight_path in enumerate(weight_files):
            logger.info(
                "  -> Fold checkpoint %d/%d: %s",
                fold_idx + 1,
                len(weight_files),
                weight_path.name
            )
            fold_batch_size = current_batch_size
            fold_succeeded = False

            while not fold_succeeded:
                model = None
                dataloader = None
                try:
                    dataloader = _create_inference_dataloader(
                        image_paths=image_path_strs,
                        image_size=img_size,
                        batch_size=fold_batch_size,
                        num_workers=num_workers,
                        device=device,
                    )
                    model = load_2d_model(model_name, str(weight_path), device)
                    _, predictions = extract_features_2d(model, model_name, dataloader, device)
                    fold_predictions.append(predictions)
                    fold_succeeded = True

                    if fold_batch_size < current_batch_size:
                        logger.warning(
                            "Reducing global inference batch size from %d to %d after OOM retry",
                            current_batch_size,
                            fold_batch_size
                        )
                        current_batch_size = fold_batch_size
                        if auto_batch_size and device.type == 'cuda' and not disable_auto_batch_size_cache \
                                and auto_batch_cache_path is not None and auto_batch_cache_key is not None:
                            _set_cached_auto_batch_size(
                                auto_batch_cache_path,
                                auto_batch_cache_key,
                                current_batch_size,
                                note='reduced_after_runtime_oom'
                            )
                except RuntimeError as exc:
                    if device.type == 'cuda' and _is_oom_error(exc):
                        if fold_batch_size <= 1:
                            raise RuntimeError(
                                f"OOM for {model_name} ({weight_path.name}) even at batch_size=1"
                            ) from exc
                        next_batch_size = max(1, fold_batch_size // 2)
                        logger.warning(
                            "CUDA OOM for %s (%s) at batch size %d. Retrying with batch size %d.",
                            model_name,
                            weight_path.name,
                            fold_batch_size,
                            next_batch_size
                        )
                        fold_batch_size = next_batch_size
                    else:
                        logger.error(
                            "Error loading %s fold from %s: %s",
                            model_name,
                            weight_path,
                            exc
                        )
                        break
                except Exception as exc:
                    logger.error(
                        "Error loading %s fold from %s: %s",
                        model_name,
                        weight_path,
                        exc
                    )
                    break
                finally:
                    if dataloader is not None:
                        del dataloader
                    if model is not None:
                        del model
                    if device.type == 'cuda':
                        torch.cuda.empty_cache()

        if not fold_predictions:
            logger.warning("No valid fold checkpoints were loaded for %s", model_name)
            continue

        # Stage 1 ensemble: average predictions across folds for each backbone.
        model_predictions = {}
        model_filenames = sorted({fname for preds in fold_predictions for fname in preds.keys()})
        for fname in model_filenames:
            preds = [preds_by_fold[fname] for preds_by_fold in fold_predictions if fname in preds_by_fold]
            model_predictions[fname] = np.mean(preds, axis=0)

        all_predictions[model_name] = model_predictions
        logger.info(
            "Averaged %d fold(s) for %s over %d slice(s)",
            len(fold_predictions),
            model_name,
            len(model_predictions)
        )

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
        '--max_folds_per_model',
        type=int,
        default=None,
        help='Maximum fold checkpoints to use per backbone (default: all available)'
    )
    parser.add_argument(
        '--auto_batch_size',
        action='store_true',
        help='Automatically select the largest safe batch size on visible GPUs'
    )
    parser.add_argument(
        '--auto_batch_size_max',
        type=int,
        default=128,
        help='Upper bound for auto batch size search'
    )
    parser.add_argument(
        '--auto_batch_size_cache_path',
        type=str,
        default=None,
        help='Optional JSON cache path for auto batch size reuse'
    )
    parser.add_argument(
        '--disable_auto_batch_size_cache',
        action='store_true',
        help='Disable reading/writing auto batch size cache'
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
        max_folds_per_model=args.max_folds_per_model,
        auto_batch_size=args.auto_batch_size,
        auto_batch_size_max=args.auto_batch_size_max,
        auto_batch_size_cache_path=args.auto_batch_size_cache_path,
        disable_auto_batch_size_cache=args.disable_auto_batch_size_cache,
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
