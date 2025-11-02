#!/usr/bin/env python3
"""Simple script to download test images from Kaggle Human Detection Dataset"""

import os
import shutil
from pathlib import Path

try:
    import kagglehub
except ImportError:
    print("Installing kagglehub...")
    os.system("pip install kagglehub")
    import kagglehub

def main():
    # Download dataset
    print("Downloading Human Detection Dataset...")
    dataset_path = kagglehub.dataset_download("constantinwerner/human-detection-dataset")
    print(f"Downloaded to: {dataset_path}")
    
    # Copy a few test images to models/
    dataset_dir = Path(dataset_path)
    models_dir = Path(__file__).parent.parent / "models"
    
    # Find image files
    image_files = list(dataset_dir.rglob("*.jpg")) + list(dataset_dir.rglob("*.png"))
    
    if image_files:
        # Copy first image as test_image.jpg
        shutil.copy2(image_files[0], models_dir / "test_image.jpg")
        print(f"Copied test image: {image_files[0].name} -> models/test_image.jpg")
        
        # Copy a few more as numbered test images
        for i, img in enumerate(image_files[1:6], 1):  # Copy 5 more images
            dest = models_dir / f"test_image_{i}.jpg"
            shutil.copy2(img, dest)
            print(f"Copied: {img.name} -> {dest.name}")
        
        print(f"\nTest with: ./build/app/shell-app test_image")
    else:
        print("No image files found in dataset!")

if __name__ == "__main__":
    main()
