#!/usr/bin/env python3
"""
Export per-slice saliency 3-panel images for RSNA 2D models.

Panel layout:
1) Brain-window slice (grayscale)
2) Gradient saliency overlay
3) Class probability bars
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

# Add project root to path for imports from inference.run_inference
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from inference.run_inference import HEMORRHAGE_TYPES, load_2d_model

logger = logging.getLogger(__name__)
MODEL_CHOICES = ["DenseNet121_change_avg", "DenseNet169_change_avg", "se_resnext101_32x4d"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export 3-panel saliency images for preprocessed RSNA data.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data_dir", type=str, required=True, help="Processed data directory")
    parser.add_argument(
        "--output_dir", type=str, required=True, help="Directory to save saliency panel images"
    )
    parser.add_argument(
        "--saliency_mode",
        type=str,
        choices=["single", "ensemble"],
        default="ensemble",
        help="How to compute saliency overlays",
    )
    parser.add_argument(
        "--model_dir", type=str, default="./models", help="Directory containing model weights"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="DenseNet121_change_avg",
        choices=MODEL_CHOICES,
        help="Backbone to use for saliency",
    )
    parser.add_argument(
        "--ensemble_models",
        type=str,
        nargs="+",
        choices=MODEL_CHOICES,
        default=MODEL_CHOICES,
        help="Backbones to include for ensemble saliency mode",
    )
    parser.add_argument(
        "--weight_path",
        type=str,
        default=None,
        help="Optional explicit model weight .pth path",
    )
    parser.add_argument(
        "--target_class",
        type=int,
        default=None,
        help="Optional fixed class index (0-5). If omitted, use predicted class per slice.",
    )
    parser.add_argument(
        "--max_slices_per_series",
        type=int,
        default=None,
        help="Optional cap of slices per series (center-cropped by order).",
    )
    parser.add_argument(
        "--image_size_override",
        type=int,
        default=None,
        help="Optional model input size override for saliency memory control.",
    )
    parser.add_argument(
        "--ensemble_slice_csv",
        type=str,
        default=None,
        help="Optional run_inference predictions_per_slice.csv for ensemble bar probabilities.",
    )
    parser.add_argument("--cpu", action="store_true", help="Force CPU inference")
    return parser.parse_args()


def _model_config(model_name: str) -> Tuple[str, int]:
    configs = {
        "DenseNet121_change_avg": ("DenseNet121_change_avg_512", 512),
        "DenseNet169_change_avg": ("DenseNet169_change_avg_256", 256),
        "se_resnext101_32x4d": ("se_resnext101_32x4d_256", 256),
    }
    return configs[model_name]


def _find_weight_file(model_dir: Path, model_name: str, explicit_path: Optional[str]) -> Path:
    if explicit_path:
        p = Path(explicit_path)
        if not p.exists():
            raise FileNotFoundError(f"Weight path does not exist: {p}")
        return p

    subdir, _ = _model_config(model_name)
    weights_dir = model_dir / "2DNet" / subdir
    weight_files = sorted(weights_dir.glob("model_epoch_best_*.pth"))
    if not weight_files:
        weight_files = sorted(weights_dir.glob("*.pth"))
    if not weight_files:
        raise FileNotFoundError(f"No weights found in {weights_dir}")
    return weight_files[0]


def _forward_logits(model: torch.nn.Module, model_name: str, x: torch.Tensor) -> torch.Tensor:
    net = model.module if hasattr(model, "module") else model
    if "DenseNet121" in model_name:
        feat = net.densenet121(x)
        feat = net.relu(feat)
        feat = net.avgpool(feat)
        feat = feat.view(feat.size(0), -1)
        return net.mlp(feat)
    if "DenseNet169" in model_name:
        feat = net.densenet169(x)
        feat = net.relu(feat)
        feat = net.avgpool(feat)
        feat = feat.view(feat.size(0), -1)
        return net.mlp(feat)
    if "se_resnext" in model_name:
        feat = net.model_ft.layer0(x)
        feat = net.model_ft.layer1(feat)
        feat = net.model_ft.layer2(feat)
        feat = net.model_ft.layer3(feat)
        feat = net.model_ft.layer4(feat)
        feat = net.model_ft.avg_pool(feat)
        feat = feat.view(feat.size(0), -1)
        return net.model_ft.last_linear(feat)
    return model(x)


def _normalize_map(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    x -= float(x.min())
    max_v = float(x.max())
    if max_v > 0:
        x /= max_v
    return x


def _normalize_prob_vector(probs: np.ndarray) -> np.ndarray:
    probs = probs.astype(np.float32)
    s = float(np.sum(probs))
    if s > 0:
        return probs / s
    return probs


def _preprocess_slice_for_model(png_path: Path, image_size: int) -> Tuple[np.ndarray, torch.Tensor]:
    img = cv2.imread(str(png_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Could not load image: {png_path}")

    # Match run_inference.py behavior
    img = cv2.resize(img, (512, 512))
    img_3ch = np.stack([img, img, img], axis=-1)

    h, w = img_3ch.shape[:2]
    target_h, target_w = int(h * 0.8), int(w * 0.8)
    start_x = (w - target_w) // 2
    start_y = (h - target_h) // 2
    img_3ch = img_3ch[start_y : start_y + target_h, start_x : start_x + target_w, :]
    img_3ch = cv2.resize(img_3ch, (w, h))

    # Display image for panel rendering
    display_gray = img_3ch[:, :, 0].astype(np.float32) / 255.0

    # Model input size + normalization
    model_img = cv2.resize(img_3ch, (image_size, image_size)).astype(np.float32) / 255.0
    mean = np.array([0.456, 0.456, 0.456], dtype=np.float32)
    std = np.array([0.224, 0.224, 0.224], dtype=np.float32)
    model_img = (model_img - mean) / std
    model_img = np.transpose(model_img, (2, 0, 1))
    tensor = torch.from_numpy(model_img).float().unsqueeze(0)
    return display_gray, tensor


def _overlay(brain_gray: np.ndarray, saliency: np.ndarray) -> np.ndarray:
    sal = _normalize_map(saliency)
    base_rgb = np.stack([brain_gray, brain_gray, brain_gray], axis=-1)
    heat_rgb = plt.get_cmap("jet")(sal)[..., :3]
    alpha = 0.65 * sal
    out = (1.0 - alpha[..., None]) * base_rgb + alpha[..., None] * heat_rgb
    return np.clip(out, 0.0, 1.0)


def _resolve_series_csv_dir(data_dir: Path) -> Path:
    series_csv_dir = data_dir / "csv" / "series_csv"
    if series_csv_dir.exists():
        logger.info("Using series metadata at %s", series_csv_dir)
        return series_csv_dir

    study_csv_dir = data_dir / "csv" / "study_csv"
    if study_csv_dir.exists():
        logger.warning(
            "Using legacy study_csv metadata at %s. "
            "For SeriesInstanceUID grouping, regenerate data with scripts/prepare_custom_data.py.",
            study_csv_dir,
        )
        return study_csv_dir

    raise FileNotFoundError(
        f"Missing metadata directory. Expected one of: {series_csv_dir}, {study_csv_dir}"
    )


def _sorted_series_rows(data_dir: Path) -> List[Tuple[str, pd.DataFrame]]:
    series_csv_dir = _resolve_series_csv_dir(data_dir)

    rows: List[Tuple[str, pd.DataFrame]] = []
    for csv_path in sorted(series_csv_dir.glob("*.csv")):
        series_id = csv_path.stem
        df = pd.read_csv(csv_path)
        if "Position2" in df.columns:
            df = df.sort_values("Position2", kind="mergesort").reset_index(drop=True)
        rows.append((series_id, df))
    return rows


def _load_ensemble_slice_probs(csv_path: Optional[str]) -> Dict[str, np.ndarray]:
    if csv_path is None:
        return {}

    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"--ensemble_slice_csv does not exist: {path}")

    df = pd.read_csv(path)
    required_cols = {"filename", *HEMORRHAGE_TYPES}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(
            f"--ensemble_slice_csv missing required columns: {sorted(missing)}"
        )

    mapping: Dict[str, np.ndarray] = {}
    for _, row in df.iterrows():
        filename = str(row["filename"])
        mapping[filename] = np.array(
            [float(row[htype]) for htype in HEMORRHAGE_TYPES], dtype=np.float32
        )
    return mapping


def _render_panel(
    out_path: Path,
    series_id: str,
    seq_idx: int,
    filename: str,
    brain_gray: np.ndarray,
    overlay_rgb: np.ndarray,
    bar_probs: np.ndarray,
    bar_pred_idx: int,
    bar_source_label: str,
    saliency_source_label: str,
    saliency_score: float,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)

    axes[0].imshow(brain_gray, cmap="gray", vmin=0.0, vmax=1.0)
    axes[0].set_title(f"Series {series_id}\nSlice {seq_idx} ({filename})")
    axes[0].axis("off")

    axes[1].imshow(overlay_rgb)
    axes[1].set_title(f"Saliency overlay\nscore={saliency_score:.5f}")
    axes[1].axis("off")

    bars = axes[2].bar(np.arange(len(HEMORRHAGE_TYPES)), bar_probs, color="#6aa4ff")
    bars[bar_pred_idx].set_color("#e45d5d")
    axes[2].set_xticks(np.arange(len(HEMORRHAGE_TYPES)))
    axes[2].set_xticklabels(HEMORRHAGE_TYPES, rotation=45, ha="right")
    axes[2].set_ylim(0.0, 1.0)
    axes[2].set_title(
        f"Class probabilities ({bar_source_label})\n"
        f"Pred={HEMORRHAGE_TYPES[bar_pred_idx]} ({bar_probs[bar_pred_idx]:.3f})"
    )
    axes[2].grid(axis="y", alpha=0.2)
    fig.text(
        0.5,
        0.015,
        f"Saliency source: {saliency_source_label}. "
        f"Bar chart source: {bar_source_label}.",
        ha="center",
        va="bottom",
        fontsize=9,
    )

    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    model_dir = Path(args.model_dir)
    png_dir = data_dir / "png"
    output_dir.mkdir(parents=True, exist_ok=True)

    if not png_dir.exists():
        raise FileNotFoundError(f"Missing PNG directory: {png_dir}")

    if args.target_class is not None and not (0 <= args.target_class < len(HEMORRHAGE_TYPES)):
        raise ValueError("--target_class must be in [0, 5]")

    device = torch.device("cpu" if args.cpu else ("cuda" if torch.cuda.is_available() else "cpu"))
    logger.info("Using device: %s", device)

    if args.saliency_mode == "single":
        weight_path = _find_weight_file(
            model_dir=model_dir, model_name=args.model, explicit_path=args.weight_path
        )
        _, image_size = _model_config(args.model)
        if args.image_size_override is not None:
            image_size = int(args.image_size_override)
        logger.info("Loading single-model saliency %s from %s", args.model, weight_path)
        logger.info("Saliency input size: %d", image_size)
        saliency_models: Dict[str, Dict[str, object]] = {
            args.model: {
                "model": load_2d_model(args.model, str(weight_path), device),
                "image_size": image_size,
                "weight_path": weight_path,
            }
        }
        saliency_source_label = f"single model {args.model} ({weight_path.name})"
    else:
        if args.weight_path is not None:
            logger.warning("--weight_path is ignored in ensemble saliency mode.")
        if args.model != "DenseNet121_change_avg":
            logger.warning("--model is ignored in ensemble saliency mode.")
        saliency_models = {}
        for model_name in args.ensemble_models:
            weight_path = _find_weight_file(
                model_dir=model_dir, model_name=model_name, explicit_path=None
            )
            _, image_size = _model_config(model_name)
            if args.image_size_override is not None:
                image_size = int(args.image_size_override)
            logger.info(
                "Loading ensemble saliency component %s from %s (input size %d)",
                model_name,
                weight_path,
                image_size,
            )
            saliency_models[model_name] = {
                "model": load_2d_model(model_name, str(weight_path), device),
                "image_size": image_size,
                "weight_path": weight_path,
            }
        saliency_source_label = (
            "equal-weight normalized per-model outputs + normalized saliency maps"
        )

    ensemble_slice_probs = _load_ensemble_slice_probs(args.ensemble_slice_csv)
    if ensemble_slice_probs:
        logger.info(
            "Loaded ensemble per-slice probabilities for %d slices from %s",
            len(ensemble_slice_probs),
            args.ensemble_slice_csv,
        )
    else:
        logger.warning(
            "No ensemble per-slice CSV provided. Bar chart will use single-model probabilities."
        )

    series_rows = _sorted_series_rows(data_dir)
    logger.info("Found %d series", len(series_rows))

    run_manifest: List[Dict[str, object]] = []
    all_slice_rows: List[Dict[str, object]] = []

    for series_id, df_series in series_rows:
        if "filename" not in df_series.columns:
            logger.warning("Skipping series %s: missing filename column", series_id)
            continue

        if args.max_slices_per_series is not None and len(df_series) > args.max_slices_per_series:
            start = (len(df_series) - args.max_slices_per_series) // 2
            df_series = df_series.iloc[start : start + args.max_slices_per_series].reset_index(drop=True)

        series_dir = output_dir / series_id
        series_dir.mkdir(parents=True, exist_ok=True)

        slice_rows: List[Dict[str, object]] = []
        series_probs = np.zeros(len(HEMORRHAGE_TYPES), dtype=np.float64)
        success_count = 0

        for seq_idx, row in df_series.reset_index(drop=True).iterrows():
            filename = str(row["filename"])
            png_path = png_dir / filename
            if not png_path.exists():
                logger.warning("Missing PNG for series %s: %s", series_id, png_path)
                continue

            try:
                if args.saliency_mode == "single":
                    model_name = args.model
                    model_bundle = saliency_models[model_name]
                    model = model_bundle["model"]
                    image_size = int(model_bundle["image_size"])

                    brain_gray, inp = _preprocess_slice_for_model(
                        png_path=png_path, image_size=image_size
                    )
                    inp = inp.to(device)
                    inp.requires_grad_(True)
                    model.zero_grad(set_to_none=True)

                    logits = _forward_logits(model, model_name, inp)
                    model_probs = torch.sigmoid(logits)[0].detach().cpu().numpy().astype(np.float32)
                    model_pred_idx = int(np.argmax(model_probs))
                    target_idx = (
                        int(args.target_class) if args.target_class is not None else model_pred_idx
                    )

                    logits[0, target_idx].backward()
                    saliency = inp.grad.detach().abs().mean(dim=1)[0].cpu().numpy()
                    saliency = cv2.resize(saliency, (brain_gray.shape[1], brain_gray.shape[0]))
                    saliency = _normalize_map(saliency)
                    saliency_score = float(np.mean(np.abs(saliency)))
                    overlay_rgb = _overlay(brain_gray, saliency)

                    combined_probs = model_probs
                    component_probs = {model_name: model_probs}
                    component_probs_normalized = {model_name: _normalize_prob_vector(model_probs)}
                else:
                    component_probs: Dict[str, np.ndarray] = {}
                    component_probs_normalized: Dict[str, np.ndarray] = {}
                    brain_gray = None

                    # Pass 1: get per-model probabilities (equal weighted + normalized).
                    for model_name, model_bundle in saliency_models.items():
                        model = model_bundle["model"]
                        image_size = int(model_bundle["image_size"])
                        brain_gray_this, inp = _preprocess_slice_for_model(
                            png_path=png_path, image_size=image_size
                        )
                        if brain_gray is None:
                            brain_gray = brain_gray_this
                        inp = inp.to(device)
                        with torch.no_grad():
                            logits = _forward_logits(model, model_name, inp)
                            probs = torch.sigmoid(logits)[0].detach().cpu().numpy().astype(np.float32)
                        component_probs[model_name] = probs
                        component_probs_normalized[model_name] = _normalize_prob_vector(probs)

                    norm_prob_stack = np.stack(
                        [component_probs_normalized[m] for m in sorted(component_probs_normalized.keys())],
                        axis=0,
                    )
                    combined_probs = np.mean(norm_prob_stack, axis=0)
                    model_pred_idx = int(np.argmax(combined_probs))
                    target_idx = int(args.target_class) if args.target_class is not None else model_pred_idx

                    # Pass 2: get per-model saliency, normalize each map, then average equally.
                    saliency_maps: List[np.ndarray] = []
                    for model_name, model_bundle in saliency_models.items():
                        model = model_bundle["model"]
                        image_size = int(model_bundle["image_size"])
                        _, inp = _preprocess_slice_for_model(png_path=png_path, image_size=image_size)
                        inp = inp.to(device)
                        inp.requires_grad_(True)
                        model.zero_grad(set_to_none=True)
                        logits = _forward_logits(model, model_name, inp)
                        logits[0, target_idx].backward()
                        saliency = inp.grad.detach().abs().mean(dim=1)[0].cpu().numpy()
                        saliency = cv2.resize(saliency, (brain_gray.shape[1], brain_gray.shape[0]))
                        saliency_maps.append(_normalize_map(saliency))

                    saliency = np.mean(np.stack(saliency_maps, axis=0), axis=0)
                    saliency_score = float(np.mean(np.abs(saliency)))
                    overlay_rgb = _overlay(brain_gray, saliency)

                bar_probs = combined_probs
                bar_source = (
                    "equal-weight normalized model ensemble probabilities"
                    if args.saliency_mode == "ensemble"
                    else "single model probabilities"
                )
                if filename in ensemble_slice_probs:
                    bar_probs = ensemble_slice_probs[filename]
                    bar_source = "ensemble probabilities"
                elif args.ensemble_slice_csv is not None:
                    logger.warning(
                        "Slice %s missing in ensemble CSV; using computed saliency probabilities for bar chart.",
                        filename,
                    )
                bar_pred_idx = int(np.argmax(bar_probs))

                stem = Path(filename).stem
                out_name = (
                    f"{seq_idx:03d}_pred-{HEMORRHAGE_TYPES[bar_pred_idx]}_"
                    f"p{float(bar_probs[bar_pred_idx]):.3f}_s{saliency_score:.5f}_{stem}.png"
                )
                out_path = series_dir / out_name

                _render_panel(
                    out_path=out_path,
                    series_id=series_id,
                    seq_idx=int(seq_idx),
                    filename=filename,
                    brain_gray=brain_gray,
                    overlay_rgb=overlay_rgb,
                    bar_probs=bar_probs,
                    bar_pred_idx=bar_pred_idx,
                    bar_source_label=bar_source,
                    saliency_source_label=saliency_source_label,
                    saliency_score=saliency_score,
                )

                row_payload = {
                    "series_id": series_id,
                    "seq_idx": int(seq_idx),
                    "filename": filename,
                    "saved_png": out_name,
                    "predicted_class_index": bar_pred_idx,
                    "predicted_class_name": HEMORRHAGE_TYPES[bar_pred_idx],
                    "bar_prob_source": bar_source,
                    "saliency_mode": args.saliency_mode,
                    "saliency_model_name": (
                        args.model if args.saliency_mode == "single" else "ensemble"
                    ),
                    "saliency_model_weight": (
                        str(saliency_models[args.model]["weight_path"].name)
                        if args.saliency_mode == "single"
                        else ",".join(
                            f"{m}:{saliency_models[m]['weight_path'].name}"
                            for m in sorted(saliency_models.keys())
                        )
                    ),
                    "saliency_model_predicted_class_index": model_pred_idx,
                    "saliency_model_predicted_class_name": HEMORRHAGE_TYPES[model_pred_idx],
                    "target_class_index": target_idx,
                    "saliency_score": saliency_score,
                }
                for i, cls_name in enumerate(HEMORRHAGE_TYPES):
                    row_payload[f"prob_{cls_name}"] = float(bar_probs[i])
                    row_payload[f"saliency_model_prob_{cls_name}"] = float(combined_probs[i])
                    if args.saliency_mode == "ensemble":
                        for model_name in sorted(component_probs.keys()):
                            row_payload[f"{model_name}_prob_{cls_name}"] = float(
                                component_probs[model_name][i]
                            )
                            row_payload[f"{model_name}_prob_norm_{cls_name}"] = float(
                                component_probs_normalized[model_name][i]
                            )
                slice_rows.append(row_payload)
                all_slice_rows.append(row_payload)

                series_probs += bar_probs.astype(np.float64)
                success_count += 1

            except Exception as exc:
                logger.exception("Failed saliency on %s (%s): %s", series_id, filename, exc)

        if success_count == 0:
            logger.warning("No slices exported for series %s", series_id)
            continue

        mean_probs = series_probs / float(success_count)
        pred_idx = int(np.argmax(mean_probs))
        series_payload = {
            "series_id": series_id,
            "num_slices_exported": int(success_count),
            "predicted_class_index": pred_idx,
            "predicted_class_name": HEMORRHAGE_TYPES[pred_idx],
            "probabilities_mean": {cls: float(mean_probs[i]) for i, cls in enumerate(HEMORRHAGE_TYPES)},
        }
        (series_dir / "series_prediction.json").write_text(json.dumps(series_payload, indent=2))
        pd.DataFrame(slice_rows).to_csv(series_dir / "slice_saliency_scores.csv", index=False)
        run_manifest.append(series_payload)
        logger.info("Saved %d panel images for series %s", success_count, series_id)

    (output_dir / "run_manifest.json").write_text(json.dumps(run_manifest, indent=2))
    if all_slice_rows:
        pd.DataFrame(all_slice_rows).to_csv(output_dir / "all_slice_saliency_scores.csv", index=False)

    logger.info("Wrote run manifest to %s", output_dir / "run_manifest.json")


if __name__ == "__main__":
    main()
