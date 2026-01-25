#!/usr/bin/env python3
"""
Prepare custom DICOM data for RSNA 2019 Hemorrhage Detection inference.

This script:
1. Reads DICOM files from an input directory
2. Extracts study/series metadata
3. Converts DICOM to PNG with proper windowing
4. Generates required CSV files for the pipeline

Usage:
    python scripts/prepare_custom_data.py --input_dir /path/to/dicoms --output_dir /path/to/output
"""

import os
import argparse
import logging
from pathlib import Path
from glob import glob
from collections import defaultdict

import numpy as np
import pandas as pd
import pydicom
from PIL import Image
from tqdm import tqdm
from joblib import Parallel, delayed

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def get_first_of_dicom_field_as_int(x):
    """Extract first value from a DICOM field that may be a MultiValue."""
    if isinstance(x, pydicom.multival.MultiValue):
        return int(x[0])
    return int(x)


def get_metadata_from_dicom(dcm):
    """Extract windowing parameters from DICOM file."""
    try:
        metadata = {
            "window_center": dcm.WindowCenter if hasattr(dcm, 'WindowCenter') else 40,
            "window_width": dcm.WindowWidth if hasattr(dcm, 'WindowWidth') else 80,
            "intercept": dcm.RescaleIntercept if hasattr(dcm, 'RescaleIntercept') else 0,
            "slope": dcm.RescaleSlope if hasattr(dcm, 'RescaleSlope') else 1,
        }
        return {k: get_first_of_dicom_field_as_int(v) for k, v in metadata.items()}
    except Exception as e:
        logger.warning(f"Could not extract metadata: {e}, using defaults")
        return {"window_center": 40, "window_width": 80, "intercept": 0, "slope": 1}


def window_image(img, window_center, window_width, intercept, slope):
    """Apply windowing to CT image (brain window by default)."""
    img = img * slope + intercept
    img_min = window_center - window_width // 2
    img_max = window_center + window_width // 2
    img = np.clip(img, img_min, img_max)
    return img


def normalize_minmax(img):
    """Normalize image to 0-1 range."""
    mi, ma = img.min(), img.max()
    if ma - mi > 0:
        return (img - mi) / (ma - mi)
    return img - mi


def process_dicom(dcm_path, output_dir):
    """
    Process a single DICOM file: extract metadata and convert to PNG.

    Returns:
        dict: Metadata for this slice, or None if processing failed
    """
    try:
        dcm = pydicom.dcmread(dcm_path, force=True)

        # Get unique identifier
        sop_instance_uid = str(dcm.SOPInstanceUID) if hasattr(dcm, 'SOPInstanceUID') else Path(dcm_path).stem
        study_instance_uid = str(dcm.StudyInstanceUID) if hasattr(dcm, 'StudyInstanceUID') else "unknown_study"
        series_instance_uid = str(dcm.SeriesInstanceUID) if hasattr(dcm, 'SeriesInstanceUID') else "unknown_series"

        # Get slice position for ordering
        try:
            if hasattr(dcm, 'ImagePositionPatient'):
                position_z = float(dcm.ImagePositionPatient[2])
            elif hasattr(dcm, 'SliceLocation'):
                position_z = float(dcm.SliceLocation)
            elif hasattr(dcm, 'InstanceNumber'):
                position_z = float(dcm.InstanceNumber)
            else:
                position_z = 0.0
        except:
            position_z = 0.0

        # Get windowing parameters and convert to PNG
        metadata = get_metadata_from_dicom(dcm)

        # Get pixel array
        pixel_array = dcm.pixel_array.astype(float)

        # Apply windowing
        img = window_image(pixel_array, **metadata)

        # Normalize and convert to 8-bit
        img = normalize_minmax(img) * 255
        img = img.astype(np.uint8)

        # Save as PNG
        img_pil = Image.fromarray(img, mode="L")
        png_filename = f"ID_{sop_instance_uid}.png"
        png_path = os.path.join(output_dir, 'png', png_filename)
        img_pil.save(png_path)

        return {
            'filename': png_filename,
            'sop_instance_uid': sop_instance_uid,
            'StudyInstance': study_instance_uid,
            'SeriesInstance': series_instance_uid,
            'Position2': position_z,
            'dcm_path': dcm_path
        }

    except Exception as e:
        logger.error(f"Error processing {dcm_path}: {e}")
        return None


def prepare_data(input_dir, output_dir, n_jobs=-1):
    """
    Prepare custom DICOM data for inference.

    Args:
        input_dir: Directory containing DICOM files (can be nested)
        output_dir: Output directory for PNG files and CSV metadata
        n_jobs: Number of parallel jobs (-1 for all CPUs)
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    # Create output directories
    png_dir = output_dir / 'png'
    csv_dir = output_dir / 'csv'
    study_csv_dir = csv_dir / 'study_csv'

    png_dir.mkdir(parents=True, exist_ok=True)
    csv_dir.mkdir(parents=True, exist_ok=True)
    study_csv_dir.mkdir(parents=True, exist_ok=True)

    # Find all DICOM files (common extensions)
    dcm_files = []
    for pattern in ['**/*.dcm', '**/*.DCM', '**/*.dicom', '**/*']:
        for f in input_dir.glob(pattern):
            if f.is_file() and not f.suffix.lower() in ['.png', '.jpg', '.csv', '.txt', '.json']:
                # Try to check if it's a DICOM file
                try:
                    with open(f, 'rb') as fp:
                        fp.seek(128)
                        magic = fp.read(4)
                        if magic == b'DICM' or f.suffix.lower() in ['.dcm', '.dicom']:
                            dcm_files.append(str(f))
                except:
                    pass

    dcm_files = list(set(dcm_files))  # Remove duplicates

    if not dcm_files:
        logger.error(f"No DICOM files found in {input_dir}")
        return None

    logger.info(f"Found {len(dcm_files)} DICOM files")

    # Process all DICOM files in parallel
    logger.info("Processing DICOM files...")
    results = Parallel(n_jobs=n_jobs)(
        delayed(process_dicom)(dcm_path, str(output_dir))
        for dcm_path in tqdm(dcm_files, desc="Converting DICOM to PNG")
    )

    # Filter out failed results
    results = [r for r in results if r is not None]

    if not results:
        logger.error("No DICOM files were successfully processed")
        return None

    logger.info(f"Successfully processed {len(results)} DICOM files")

    # Create main metadata DataFrame
    df = pd.DataFrame(results)

    # Sort by study and position
    df = df.sort_values(['StudyInstance', 'Position2']).reset_index(drop=True)

    # Add slice index within each study
    df['slice_idx'] = df.groupby('StudyInstance').cumcount()
    df['slice_id'] = df['StudyInstance'] + '_' + df['slice_idx'].astype(str)

    # Save main metadata CSV (compatible with train_meta_id_seriser.csv format)
    main_csv_path = csv_dir / 'test_meta_id_seriser.csv'
    df[['filename', 'StudyInstance', 'Position2', 'slice_id']].to_csv(main_csv_path, index=False)
    logger.info(f"Saved main metadata to {main_csv_path}")

    # Create per-study CSV files
    studies = df['StudyInstance'].unique()
    logger.info(f"Creating CSV files for {len(studies)} studies...")

    for study_id in tqdm(studies, desc="Creating study CSVs"):
        study_df = df[df['StudyInstance'] == study_id].copy()
        study_df = study_df.sort_values('Position2').reset_index(drop=True)

        # Add columns expected by the pipeline
        study_df['index'] = range(len(study_df))

        study_csv_path = study_csv_dir / f'{study_id}.csv'
        study_df.to_csv(study_csv_path, index=False)

    # Create a summary file
    summary = {
        'total_files': len(results),
        'total_studies': len(studies),
        'studies': list(studies),
        'output_dir': str(output_dir)
    }

    import json
    with open(csv_dir / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    # Create standard.csv format (for compatibility)
    # This would normally have labels, but for inference we set them to 0
    standard_df = df[['filename']].copy()
    standard_df['any'] = 0
    standard_df['epidural'] = 0
    standard_df['intraparenchymal'] = 0
    standard_df['intraventricular'] = 0
    standard_df['subarachnoid'] = 0
    standard_df['subdural'] = 0
    standard_df.to_csv(csv_dir / 'standard_test.csv', index=False)

    logger.info(f"\nData preparation complete!")
    logger.info(f"  PNG files: {png_dir}")
    logger.info(f"  CSV files: {csv_dir}")
    logger.info(f"  Total slices: {len(results)}")
    logger.info(f"  Total studies: {len(studies)}")

    return {
        'png_dir': str(png_dir),
        'csv_dir': str(csv_dir),
        'num_slices': len(results),
        'num_studies': len(studies),
        'studies': list(studies)
    }


def main():
    parser = argparse.ArgumentParser(
        description='Prepare custom DICOM data for RSNA 2019 Hemorrhage Detection',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--input_dir', '-i',
        type=str,
        required=True,
        help='Directory containing DICOM files (can be nested)'
    )
    parser.add_argument(
        '--output_dir', '-o',
        type=str,
        required=True,
        help='Output directory for processed data'
    )
    parser.add_argument(
        '--n_jobs', '-j',
        type=int,
        default=-1,
        help='Number of parallel jobs (-1 for all CPUs)'
    )

    args = parser.parse_args()

    result = prepare_data(args.input_dir, args.output_dir, args.n_jobs)

    if result:
        print("\n" + "="*60)
        print("Data preparation successful!")
        print("="*60)
        print(f"\nNext steps:")
        print(f"  1. Run inference:")
        print(f"     python inference/run_inference.py \\")
        print(f"         --data_dir {args.output_dir} \\")
        print(f"         --output predictions.csv")
        print("="*60)


if __name__ == '__main__':
    main()
