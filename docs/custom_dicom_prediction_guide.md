# Custom CT DICOM Prediction Guide (Docker GPU + SLURM/Singularity)

This guide documents the current inference workflow in this repository for running predictions on your own head CT DICOM files.

## What This Pipeline Produces

- `predictions.csv`: Kaggle-format slice predictions (`ID`, `Label`)
- `predictions_per_slice.csv`: one row per slice with probabilities for:
  - `any`, `epidural`, `intraparenchymal`, `intraventricular`, `subarachnoid`, `subdural`
- Saliency 3-panel PNGs per slice:
  1. grayscale CT slice
  2. saliency overlay
  3. class-probability bar chart

Important model behavior:
- Final prediction CSVs use two-stage ensembling:
  1. average across all available fold checkpoints (`model_epoch_best_0..4.pth`) within each backbone
  2. average across the 3 backbones (`DenseNet121_change_avg`, `DenseNet169_change_avg`, `se_resnext101_32x4d`)
- Saliency overlays are from an equal-weight ensemble of all three backbones, using
  normalized per-model class outputs and normalized per-model saliency maps.
- The bar chart in saliency panels uses ensemble probabilities (from `predictions_per_slice.csv`).
- Each panel includes text noting: ensemble saliency source + bar source.
- Saliency method is configurable via `--saliency_method {input_grad,gradcam,gradcampp}`.

## Key Files

- Docker image definition: `docker/Dockerfile`
- Docker end-to-end runner: `docker/run_pipeline.sh`
- Main SLURM workflow: `singularity/slurm_series3_saliency_test.sh`
- DICOM preprocessing: `scripts/prepare_custom_data.py`
- Ensemble inference: `inference/run_inference.py`
- Saliency panel export: `inference/export_saliency_panels.py`

## Prerequisites

From repo root:

1. Pretrained model weights exist:
- `models/2DNet/DenseNet121_change_avg_512/model_epoch_best_*.pth`
- `models/2DNet/DenseNet169_change_avg_256/model_epoch_best_*.pth`
- `models/2DNet/se_resnext101_32x4d_256/model_epoch_best_*.pth`

2. For Docker runs: Docker + NVIDIA Container Toolkit installed (`docker run --gpus all ...` works).

3. For Singularity runs: Python deps are available at:
- `.cache/singularity_pydeps`

If `.cache/singularity_pydeps` is missing, create it once:

```bash
cd /home/jwp84/lab/RSNA2019_Intracranial-Hemorrhage-Detection
mkdir -p .cache/singularity_pydeps

singularity exec \
  --bind "$PWD:/workspace" \
  --pwd /workspace \
  singularity/rsna2019.sif \
  python -m pip install --upgrade --no-cache-dir \
  --target /workspace/.cache/singularity_pydeps \
  -r requirements-inference.txt matplotlib
```

## Download Model Weights from GitHub Release

Release tag:
- `model-weights-v1`

Release page:
- `https://github.com/jwprescott/RSNA2019_Intracranial-Hemorrhage-Detection/releases/tag/model-weights-v1`

Recommended command:

```bash
cd /home/jwp84/lab/RSNA2019_Intracranial-Hemorrhage-Detection
bash scripts/download_models_from_release.sh --tag model-weights-v1
```

If release access is private, export either token variable first:
- `GITHUB_TOKEN`
- `GITHUB_TOKEN_RSNA_ICH_DETECTION_REPO`

## Recommended: Run with Docker + GPU

Build the image once:

```bash
cd /path/to/RSNA2019_Intracranial-Hemorrhage-Detection
docker build -t rsna2019-ich:gpu -f docker/Dockerfile .
```

Run full pipeline (preprocess + inference + saliency):

```bash
cd /path/to/RSNA2019_Intracranial-Hemorrhage-Detection

bash docker/run_pipeline.sh \
  --dicom_dir /path/to/your/dicom/root \
  --run_root "$PWD/tmp/docker_run_$(date +%Y%m%d_%H%M%S)" \
  --auto_batch_size
```

The Docker runner performs:
1. `scripts/prepare_custom_data.py` with `--group_by series`
2. `inference/run_inference.py` (fold-averaged per backbone + cross-backbone ensemble)
3. `inference/export_saliency_panels.py` (3-panel outputs using ensemble bars)

Notes:
- Inference loads local RSNA checkpoints directly and does not require ImageNet backbone downloads.
- Docker runner supports `--cache_dir` to persist PyTorch/pretrainedmodels caches on the host.
- For non-SLURM workstations, `--auto_batch_size` can maximize throughput automatically; tune with `--auto_batch_size_max`.
- Auto batch size is cached and reused by default; override via `--auto_batch_size_cache_path` or disable via `--disable_auto_batch_size_cache`.
- If runtime CUDA OOM occurs, inference automatically retries with smaller batch sizes.

## Recommended: Run Your Own DICOM Folder with SLURM

Set your input and run locations:

```bash
cd /home/jwp84/lab/RSNA2019_Intracranial-Hemorrhage-Detection

export DICOM_INPUT="/path/to/your/dicom/root"
export RUN_ROOT="$PWD/tmp/custom_run_$(date +%Y%m%d_%H%M%S)"
```

Submit the job:

```bash
sbatch --export=ALL,DICOM_INPUT="$DICOM_INPUT",RUN_ROOT="$RUN_ROOT" \
  singularity/slurm_series3_saliency_test.sh
```

Monitor:

```bash
squeue -u "$USER"
sacct -j <JOB_ID> --format=JobID,State,ExitCode,Elapsed,NodeList -P
```

The script runs:
1. `scripts/prepare_custom_data.py` (DICOM -> PNG + series CSVs)
2. `inference/run_inference.py` (fold-averaged per backbone + cross-backbone ensemble predictions)
3. `inference/export_saliency_panels.py` (3-panel saliency images, ensemble bar chart)

## Output Layout

Under `${RUN_ROOT}/output`:

- `predictions.csv`
- `predictions_per_slice.csv`
- `saliency_panels/run_manifest.json`
- `saliency_panels/all_slice_saliency_scores.csv`
- `saliency_panels/<series_id>/*.png`

Notes:
- Output is written under `${RUN_ROOT}` only.
- Input DICOM directory is read-only in practice (no writes required there).

## Run Without SLURM (Manual Container Commands)

```bash
cd /home/jwp84/lab/RSNA2019_Intracranial-Hemorrhage-Detection
module load singularity/4.3.4

export PROJECT_DIR="$PWD"
export SIF_IMAGE="$PROJECT_DIR/singularity/rsna2019.sif"
export PYTHONPATH="$PROJECT_DIR/.cache/singularity_pydeps"
export PYTHONNOUSERSITE=1

export DICOM_INPUT="/path/to/your/dicom/root"
export RUN_ROOT="$PROJECT_DIR/tmp/manual_run_$(date +%Y%m%d_%H%M%S)"
export PROCESSED_DIR="$RUN_ROOT/processed"
export OUTPUT_DIR="$RUN_ROOT/output"

mkdir -p "$PROCESSED_DIR" "$OUTPUT_DIR"

singularity exec --nv \
  --bind "$PROJECT_DIR:/workspace" \
  --pwd /workspace \
  --env PYTHONNOUSERSITE=1 \
  --env "PYTHONPATH=/workspace/.cache/singularity_pydeps" \
  "$SIF_IMAGE" \
  python scripts/prepare_custom_data.py \
  --input_dir "$DICOM_INPUT" \
  --output_dir "$PROCESSED_DIR" \
  --group_by series \
  --n_jobs 8

singularity exec --nv \
  --bind "$PROJECT_DIR:/workspace" \
  --pwd /workspace \
  --env PYTHONNOUSERSITE=1 \
  --env "PYTHONPATH=/workspace/.cache/singularity_pydeps" \
  "$SIF_IMAGE" \
  python inference/run_inference.py \
  --data_dir "$PROCESSED_DIR" \
  --model_dir /workspace/models \
  --output "$OUTPUT_DIR/predictions.csv" \
  --batch_size 16 \
  --num_workers 0

singularity exec --nv \
  --bind "$PROJECT_DIR:/workspace" \
  --pwd /workspace \
  --env PYTHONNOUSERSITE=1 \
  --env "PYTHONPATH=/workspace/.cache/singularity_pydeps" \
  "$SIF_IMAGE" \
  python inference/export_saliency_panels.py \
  --data_dir "$PROCESSED_DIR" \
  --output_dir "$OUTPUT_DIR/saliency_panels" \
  --model_dir /workspace/models \
  --saliency_mode ensemble \
  --saliency_method input_grad \
  --ensemble_slice_csv "$OUTPUT_DIR/predictions_per_slice.csv" \
  --image_size_override 224
```

## Practical Caveats

- `prepare_custom_data.py` now groups slices by `SeriesInstanceUID` by default.
- The script also writes compatibility files under `csv/study_csv/` for older workflows.
- If DICOM metadata is malformed, you may see pydicom warnings; inference can still succeed.
- If a slice is missing from `predictions_per_slice.csv`, saliency export falls back to single-model bars for that slice.
