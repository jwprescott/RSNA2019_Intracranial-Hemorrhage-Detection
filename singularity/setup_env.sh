#!/bin/bash
# Setup script - run once inside the container to install packages
pip install --user --no-cache-dir \
    pretrainedmodels>=0.7.4 \
    efficientnet-pytorch>=0.7.1 \
    pydicom>=2.3.0 \
    albumentations>=1.3.0 \
    opencv-contrib-python-headless>=4.5.0 \
    scikit-image>=0.19.0 \
    scikit-learn>=1.0.0 \
    pandas>=1.3.0 \
    "numpy>=1.21.0,<2.0.0" \
    Pillow>=9.0.0 \
    tqdm>=4.64.0 \
    joblib>=1.1.0 \
    gdown>=4.6.0

echo "Setup complete! Packages installed to ~/.local"
