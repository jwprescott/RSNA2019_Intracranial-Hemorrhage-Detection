"""
Settings for 2DNet component.

Paths can be configured via environment variables or by modifying defaults below.
"""
import os

# Data directories - configurable via environment variables
train_png_dir = os.environ.get('RSNA_TRAIN_PNG_DIR', '')
test_png_dir = os.environ.get('RSNA_TEST_PNG_DIR', '')

# Model directory
model_dir = os.environ.get('RSNA_MODEL_DIR', './models/2DNet')

# Data directory (for CSV files)
data_dir = os.environ.get('RSNA_DATA_DIR', '../data')


def set_paths(train_png=None, test_png=None, models=None, data=None):
    """
    Programmatically set paths for the pipeline.

    Args:
        train_png: Path to training PNG directory
        test_png: Path to test PNG directory
        models: Path to model directory
        data: Path to data directory (CSV files)
    """
    global train_png_dir, test_png_dir, model_dir, data_dir
    if train_png is not None:
        train_png_dir = train_png
    if test_png is not None:
        test_png_dir = test_png
    if models is not None:
        model_dir = models
    if data is not None:
        data_dir = data
