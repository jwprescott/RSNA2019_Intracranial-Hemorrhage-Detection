"""
Settings for SequenceModel component.

Paths can be configured via environment variables or by modifying defaults below.
"""
import os

# Default paths (original hardcoded paths as fallback)
_DEFAULT_CSV_ROOT = r'/home/sean/Projects/RSNA_data/csv'
_DEFAULT_FEATURE_PATH = r'/home/sean/Projects/RSNA_data/features'
_DEFAULT_OUTPUT_PATH = r'/home/sean/Projects/RSNA_data/FinalSubmission'

# Configurable via environment variables
csv_root = os.environ.get('RSNA_CSV_ROOT', _DEFAULT_CSV_ROOT)
feature_path = os.environ.get('RSNA_FEATURE_PATH', _DEFAULT_FEATURE_PATH)
final_output_path = os.environ.get('RSNA_OUTPUT_PATH', _DEFAULT_OUTPUT_PATH)

# Model directory for sequence model weights
model_dir = os.environ.get('RSNA_SEQ_MODEL_DIR', './models/SequenceModel')


def set_paths(csv=None, features=None, output=None, models=None):
    """
    Programmatically set paths for the sequence model pipeline.

    Args:
        csv: Path to CSV root directory
        features: Path to features directory
        output: Path to output directory
        models: Path to sequence model weights directory
    """
    global csv_root, feature_path, final_output_path, model_dir
    if csv is not None:
        csv_root = csv
    if features is not None:
        feature_path = features
    if output is not None:
        final_output_path = output
    if models is not None:
        model_dir = models


def get_paths():
    """Return current path configuration as a dictionary."""
    return {
        'csv_root': csv_root,
        'feature_path': feature_path,
        'final_output_path': final_output_path,
        'model_dir': model_dir
    }
