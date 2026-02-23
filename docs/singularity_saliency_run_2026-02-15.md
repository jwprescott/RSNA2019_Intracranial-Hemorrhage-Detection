# Singularity GPU Saliency Test Run (2026-02-15)

Reusable run instructions are documented in:
- `docs/custom_dicom_prediction_guide.md`

## Summary
- Goal: run RSNA ICH inference on custom head CT DICOM series using Singularity on a SLURM GPU node, with saliency-map 3-panel outputs.
- Source DICOM root (read-only): `/mnt/vstor/SOM_RAD_JWP84/lab/rsna-intracranial-hemorrhage-detection/rsna-intracranial-hemorrhage-detection/stage_2_test`
- Series selection basis: existing ICHApp stage-2 series index outputs under `/home/jwp84/Projects/ICHApp/tmp/...`
- A small subset of 3 series was copied into repo-local temp space for testing:
  - `tmp/stage2_test_series3_dicoms/ID_69c6805eee`
  - `tmp/stage2_test_series3_dicoms/ID_7540f20589`
  - `tmp/stage2_test_series3_dicoms/ID_76f3cb832e`
- No files were written into the `/mnt/.../stage_2_test` directory.

## Execution
- SLURM script: `singularity/slurm_series3_saliency_test.sh`
- Job ID: `3142659`
- Final job state: `COMPLETED`
- Runtime node: `gput067`
- Container runtime: Singularity module `4.3.4`

## Output Locations
- Main run directory:
  - `tmp/series3_saliency_run_3142659/`
- Predictions:
  - `tmp/series3_saliency_run_3142659/output/predictions.csv`
  - `tmp/series3_saliency_run_3142659/output/predictions_per_slice.csv`
- Saliency outputs:
  - `tmp/series3_saliency_run_3142659/output/saliency_panels/run_manifest.json`
  - `tmp/series3_saliency_run_3142659/output/saliency_panels/all_slice_saliency_scores.csv`
  - `tmp/series3_saliency_run_3142659/output/saliency_panels/<series_id>/*.png` (3-panel images)

## Download Archive
- Zip archive created for easy transfer:
  - `tmp/series3_saliency_run_3142659_output.zip`
- Archive size at creation: ~35 MB

## Update: Ensemble Bars + Single-Model Saliency (2026-02-15)
- Updated saliency exporter to use ensemble per-slice probabilities in panel-3 bar charts via `--ensemble_slice_csv`.
- Each generated panel now includes an explicit note:
  - saliency source = single model (`DenseNet169_change_avg` checkpoint),
  - bar chart source = ensemble probabilities (`predictions_per_slice.csv` from `run_inference.py`).
- Re-ran SLURM job with updated script:
  - Job ID: `3142679`
  - State: `COMPLETED`
  - Node: `gput052`
- New output root:
  - `tmp/series3_saliency_run_3142679/output/`
- New zip archive:
  - `tmp/series3_saliency_run_3142679_output.zip`
  - Size at creation: ~36 MB
