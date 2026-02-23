# RSNA Intracranial Hemorrhage Detection
This is the source code for the first place solution to the [RSNA2019 Intracranial Hemorrhage Detection Challenge](https://www.kaggle.com/c/rsna-intracranial-hemorrhage-detection).

## Citation
```
@article{wang2021deep,
  title={A deep learning algorithm for automatic detection and classification of acute intracranial hemorrhages in head CT scans},
  author={Wang, Xiyue and Shen, Tao and Yang, Sen and Lan, Jun and Xu, Yanming and Wang, Minghui and Zhang, Jing and Han, Xiao},
  journal={NeuroImage: Clinical},
  volume={32},
  pages={102785},
  year={2021},
  publisher={Elsevier}
}
```

Solution write up: [Link](https://www.kaggle.com/c/rsna-intracranial-hemorrhage-detection/discussion/117210#latest-682640).

## Solutuoin Overview
![image](https://github.com/SeuTao/RSNA2019_1st_place_solution/blob/master/docs/overview.png)

#### Dependencies
- opencv-python==3.4.2
- scikit-image==0.14.0
- scikit-learn==0.19.1
- scipy==1.1.0
- torch==1.1.0
- torchvision==0.2.1

### CODE
- 2DNet
- 3DNet
- SequenceModel

# 2D CNN Classifier

## Pretrained models
- seresnext101_256*256 [\[seresnext101\]](https://drive.google.com/open?id=18Py5eW1E4hSbTT6658IAjQjJGS28grdx)
- densenet169_256*256 [\[densenet169\]](https://drive.google.com/open?id=1vCsX12pMZxBmuGGNVnjFFiZ-5u5vD-h6)
- densenet121_512*512 [\[densenet121\]](https://drive.google.com/open?id=1o0ok-6I2hY1ygSWdZOKmSD84FsEpgDaa)

## Preprocessing
![image](https://github.com/SeuTao/RSNA2019_1st_place_solution/blob/master/docs/preprocessing.png)

Prepare csv file:

download data.zip:  https://drive.google.com/open?id=1buISR_b3HQDU4KeNc_DmvKTYJ1gvj5-3

1. convert dcm to png
```
python3 prepare_data.py -dcm_path stage_1_train_images -png_path train_png
python3 prepare_data.py -dcm_path stage_1_test_images -png_path train_png
python3 prepare_data.py -dcm_path stage_2_test_images -png_path test_png
```

2. train

```
python3 train_model.py -backbone DenseNet121_change_avg -img_size 256 -tbs 256 -vbs 128 -save_path DenseNet121_change_avg_256
python3 train_model.py -backbone DenseNet169_change_avg -img_size 256 -tbs 256 -vbs 128 -save_path DenseNet169_change_avg_256
python3 train_model.py -backbone se_resnext101_32x4d -img_size 256 -tbs 80 -vbs 40 -save_path se_resnext101_32x4d_256
```

3. predict
```
python3 predict.py -backbone DenseNet121_change_avg -img_size 256 -tbs 4 -vbs 4 -spth DenseNet121_change_avg_256
python3 predict.py -backbone DenseNet169_change_avg -img_size 256 -tbs 4 -vbs 4 -spth DenseNet169_change_avg_256
python3 predict.py -backbone se_resnext101_32x4d -img_size 256 -tbs 4 -vbs 4 -spth se_resnext101_32x4d_256
```

After single models training,  the oof files will be saved in ./SingleModelOutput(three folders for three pipelines). 

After training the sequence model, the final submission will be ./FinalSubmission/final_version/submission_tta.csv

# Sequence Models

## Sequence Model 1
![image](https://github.com/SeuTao/RSNA2019_1st_place_solution/blob/master/docs/s1.png)

## Sequence Model 2
![image](https://github.com/SeuTao/RSNA2019_1st_place_solution/blob/master/docs/s2.png)

#### Path Setup
Set data path in ./setting.py

#### download 

download [\[csv.zip\]](https://drive.google.com/open?id=1qYi4k-DuOLJmyZ7uYYrnomU2U7MrYRBV)

download [\[feature samples\]](https://drive.google.com/open?id=1lJgzZoHFu6HI4JBktkGY3qMk--28IUkC)

#### Sequence Model Training
```
CUDA_VISIBLE_DEVICES=0 python main.py
```
The final submissions are in the folder ../FinalSubmission/version2/submission_tta.csv

## Final Submission
### Private Leaderboard:
- 0.04383

---

# Running Inference on Custom CT Data

This section describes how to run the pretrained models on your own DICOM CT images using Docker with NVIDIA GPUs or Singularity (for HPC/SLURM clusters).

## Quick Start (Docker + GPU)

```bash
# 1. Build Docker image
docker build -t rsna2019-ich:gpu -f docker/Dockerfile .

# 2. Download pretrained models
bash scripts/download_models_from_release.sh --tag model-weights-v1

# 3. Run preprocessing + inference + saliency export
bash docker/run_pipeline.sh \
    --dicom_dir /path/to/dicoms \
    --run_root "$PWD/tmp/docker_run_$(date +%Y%m%d_%H%M%S)"
```

The preprocessing step groups slices by `SeriesInstanceUID` by default (`--group_by series`).
Inference now skips ImageNet backbone downloads and loads only your local RSNA checkpoints.
Optional: set `--cache_dir /path/to/cache` to persist PyTorch/pretrainedmodels caches for other workflows.

## Quick Start (Singularity)

```bash
# 1. Build the Singularity container
cd singularity/
bash build_container.sh

# 2. Install Python packages (run once)
singularity exec rsna2019.sif bash singularity/setup_env.sh

# 3. Download pretrained models (or manually place them)
singularity exec rsna2019.sif bash -c 'export PATH="$HOME/.local/bin:$PATH" && bash scripts/download_models.sh'

# 4. Prepare your DICOM data
singularity exec rsna2019.sif python scripts/prepare_custom_data.py \
    --input_dir /path/to/dicoms --output_dir /path/to/processed --group_by series

# 5. Run inference
singularity exec --nv rsna2019.sif python inference/run_inference.py \
    --data_dir /path/to/processed --output predictions.csv
```

## Pretrained Model Setup

### Automatic Download
```bash
bash scripts/download_models.sh
```

### Manual Download
If automatic download fails, download manually and extract to:

| Model | Resolution | Download Link | Extract To |
|-------|------------|---------------|------------|
| SE-ResNeXt101 | 256x256 | [Download](https://drive.google.com/open?id=18Py5eW1E4hSbTT6658IAjQjJGS28grdx) | `models/2DNet/se_resnext101_32x4d_256/` |
| DenseNet169 | 256x256 | [Download](https://drive.google.com/open?id=1vCsX12pMZxBmuGGNVnjFFiZ-5u5vD-h6) | `models/2DNet/DenseNet169_change_avg_256/` |
| DenseNet121 | 512x512 | [Download](https://drive.google.com/open?id=1o0ok-6I2hY1ygSWdZOKmSD84FsEpgDaa) | `models/2DNet/DenseNet121_change_avg_512/` |
| data.zip | - | [Download](https://drive.google.com/open?id=1buISR_b3HQDU4KeNc_DmvKTYJ1gvj5-3) | `data/` |
| csv.zip | - | [Download](https://drive.google.com/open?id=1qYi4k-DuOLJmyZ7uYYrnomU2U7MrYRBV) | `data/` |

Expected directory structure:
```
models/
└── 2DNet/
    ├── se_resnext101_32x4d_256/
    │   └── model_epoch_best_*.pth (folds 0-4)
    ├── DenseNet169_change_avg_256/
    │   └── model_epoch_best_*.pth (folds 0-4)
    └── DenseNet121_change_avg_512/
        └── model_epoch_best_*.pth (folds 0-4)
```

## Singularity Container

### Building the Container

```bash
cd singularity/
bash build_container.sh          # Default: pull from Docker Hub (no privileges needed)
bash build_container.sh --fakeroot  # Alternative: build from .def file
```

### First-Time Setup
After building, install Python packages to `~/.local`:
```bash
singularity exec rsna2019.sif bash singularity/setup_env.sh
```

### SLURM Job Scripts

**Data Preparation (CPU job):**
```bash
sbatch singularity/slurm_prepare_data.sh /path/to/dicoms /path/to/processed
```

**Inference (GPU job):**
```bash
sbatch singularity/slurm_inference.sh /path/to/processed predictions.csv
```

### Interactive Usage
```bash
# Interactive shell with GPU
singularity shell --nv singularity/rsna2019.sif

# Single command with GPU
singularity exec --nv singularity/rsna2019.sif python inference/run_inference.py \
    --data_dir /path/to/data --output predictions.csv
```

## Output Format

The inference script outputs predictions for 6 hemorrhage types:

| Type | Description |
|------|-------------|
| `any` | Any type of hemorrhage present |
| `epidural` | Epidural hemorrhage |
| `intraparenchymal` | Intraparenchymal hemorrhage |
| `intraventricular` | Intraventricular hemorrhage |
| `subarachnoid` | Subarachnoid hemorrhage |
| `subdural` | Subdural hemorrhage |

Output files:
- `predictions.csv` - Kaggle submission format (ID, Label)
- `predictions_per_slice.csv` - Per-slice probabilities for analysis

## Infrastructure Files

| File | Description |
|------|-------------|
| `singularity/rsna2019.def` | Singularity container definition |
| `singularity/build_container.sh` | Container build script |
| `singularity/setup_env.sh` | Python package installer |
| `singularity/slurm_inference.sh` | SLURM GPU job script |
| `singularity/slurm_prepare_data.sh` | SLURM CPU job script |
| `docker/Dockerfile` | Docker GPU image definition |
| `docker/run_pipeline.sh` | Docker end-to-end inference + saliency runner |
| `scripts/download_models.sh` | Model download script |
| `scripts/prepare_custom_data.py` | DICOM preprocessing |
| `inference/run_inference.py` | Main inference pipeline |
| `requirements-inference.txt` | Python dependencies |

---

# Remote Development with VS Code

This section describes how to develop on the cluster using VS Code Remote Tunnels.

## Setup VS Code Tunnel on Cluster

The `code` CLI tool is located in the home directory on the cluster (`~/code`).

### 1. Start an Interactive Session on a Compute Node

```bash
# SSH to the cluster
ssh jwp84@pioneer.case.edu

# Request an interactive session with GPU (adjust resources as needed)
srun --partition=gpu --gres=gpu:1 --cpus-per-task=8 --mem=32G --time=04:00:00 --pty bash
```

### 2. Start the VS Code Tunnel

```bash
# On the compute node, start the tunnel
~/code tunnel

# First time: You'll be prompted to authenticate with GitHub
# Follow the URL and enter the code shown in the terminal
```

### 3. Connect from Local VS Code

1. Install the **Remote - Tunnels** extension in VS Code
2. Open Command Palette (`Ctrl+Shift+P` / `Cmd+Shift+P`)
3. Select **Remote-Tunnels: Connect to Tunnel...**
4. Sign in with GitHub (same account used on cluster)
5. Select the tunnel (usually named after the compute node hostname)

### 4. Open the Project

Once connected, open the folder:
```
/home/jwp84/lab/RSNA2019_Intracranial-Hemorrhage-Detection
```

## Running Commands in VS Code Terminal

When connected via tunnel, the VS Code terminal runs on the compute node. You can:

```bash
# Load Singularity module if needed
module load singularity

# Run commands inside the container
singularity exec --nv singularity/rsna2019.sif python inference/run_inference.py \
    --data_dir /path/to/data --output predictions.csv

# Or get an interactive shell
singularity shell --nv singularity/rsna2019.sif
```

## Tips for Remote Development

- **Keep tunnel running**: The tunnel stays active as long as your compute session is alive
- **Reconnect**: If disconnected, just reconnect from VS Code - the tunnel persists
- **Background jobs**: For long jobs, use `sbatch` scripts instead of running in terminal
- **File sync**: Files are on the cluster filesystem - no sync needed
- **Extensions**: Install Python/Jupyter extensions for better editing experience

## Alternative: SSH Remote

If tunnels don't work, use SSH Remote extension:

1. Install **Remote - SSH** extension
2. Add host to `~/.ssh/config`:
   ```
   Host pioneer
       HostName pioneer.case.edu
       User jwp84
   ```
3. Connect via **Remote-SSH: Connect to Host...**
4. Note: This connects to login node, not compute node
