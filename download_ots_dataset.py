#!/usr/bin/env python3
"""
Download OTS (Outdoor Training Set) from RESIDE dataset using Kaggle API
and organize it into a proper structure for training.
"""

import os
import shutil
from pathlib import Path
import kagglehub


def download_and_organize_ots(target_dir: str = "datasets/SOTS"):
    """
    Download OTS dataset from Kaggle and organize into clean structure.
    
    Args:
        target_dir: Local directory to save organized dataset
    """
    print("=" * 60)
    print("Downloading OTS Dataset from Kaggle")
    print("=" * 60)
    
    # Download dataset
    print("📥 Downloading dataset...")
    try:
        kaggle_path = kagglehub.dataset_download("brunobelloni/outdoor-training-set-ots-reside")
        print(f"✅ Downloaded to: {kaggle_path}")
    except Exception as e:
        print(f"❌ Download failed: {e}")
        print("Make sure you have:")
        print("1. pip install kagglehub")
        print("2. Kaggle API credentials configured")
        return
    
    # Create target directories
    target_path = Path(target_dir)
    outdoor_hazy_dir = target_path / "outdoor" / "hazy"
    outdoor_gt_dir = target_path / "outdoor" / "gt"
    
    outdoor_hazy_dir.mkdir(parents=True, exist_ok=True)
    outdoor_gt_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"📁 Created target directories:")
    print(f"   - {outdoor_hazy_dir}")
    print(f"   - {outdoor_gt_dir}")
    
    # Explore kaggle download structure
    kaggle_path = Path(kaggle_path)
    print(f"\n🔍 Exploring download structure:")
    
    def explore_directory(path, prefix="", max_depth=3, current_depth=0):
        if current_depth >= max_depth:
            return
        
        items = list(path.iterdir()) if path.is_dir() else []
        for i, item in enumerate(items[:10]):  # Show first 10 items
            is_last = i == len(items[:10]) - 1
            current_prefix = "└── " if is_last else "├── "
            print(f"{prefix}{current_prefix}{item.name}")
            
            if item.is_dir() and current_depth < max_depth - 1:
                next_prefix = prefix + ("    " if is_last else "│   ")
                explore_directory(item, next_prefix, max_depth, current_depth + 1)
        
        if len(items) > 10:
            print(f"{prefix}    ... and {len(items) - 10} more items")
    
    explore_directory(kaggle_path)
    
    # Common OTS structure patterns to look for
    possible_paths = [
        kaggle_path / "outdoor",
        kaggle_path / "OTS",
        kaggle_path / "SOTS" / "outdoor",
        kaggle_path / "reside" / "outdoor",
        kaggle_path,  # Fallback: search in root
    ]
    
    hazy_patterns = ["hazy", "hazed", "input", "degraded"]
    gt_patterns = ["gt", "clear", "target", "clean"]
    
    print(f"\n🔄 Organizing files...")
    
    copied_hazy = 0
    copied_gt = 0
    
    # Search for hazy and gt directories
    for base_path in possible_paths:
        if not base_path.exists():
            continue
            
        print(f"   Searching in: {base_path}")
        
        # Look for hazy images
        for hazy_pattern in hazy_patterns:
            hazy_search_dir = base_path / hazy_pattern
            if hazy_search_dir.exists():
                print(f"   Found hazy dir: {hazy_search_dir}")
                for img_file in hazy_search_dir.glob("*"):
                    if img_file.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
                        dest_file = outdoor_hazy_dir / img_file.name
                        if not dest_file.exists():
                            shutil.copy2(img_file, dest_file)
                            copied_hazy += 1
        
        # Look for gt/clean images
        for gt_pattern in gt_patterns:
            gt_search_dir = base_path / gt_pattern
            if gt_search_dir.exists():
                print(f"   Found gt dir: {gt_search_dir}")
                for img_file in gt_search_dir.glob("*"):
                    if img_file.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
                        dest_file = outdoor_gt_dir / img_file.name
                        if not dest_file.exists():
                            shutil.copy2(img_file, dest_file)
                            copied_gt += 1
        
        # If we found files, break
        if copied_hazy > 0 or copied_gt > 0:
            break
    
    # Alternative: search recursively if structure is nested
    if copied_hazy == 0 and copied_gt == 0:
        print("   📂 Searching recursively...")
        
        all_images = list(kaggle_path.rglob("*"))
        image_files = [f for f in all_images if f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']]
        
        print(f"   Found {len(image_files)} total image files")
        
        for img_file in image_files:
            # Heuristic: classify based on parent directory name or filename
            parent_name = img_file.parent.name.lower()
            file_name = img_file.name.lower()
            
            is_hazy = any(pattern in parent_name or pattern in file_name 
                         for pattern in hazy_patterns)
            is_gt = any(pattern in parent_name or pattern in file_name 
                       for pattern in gt_patterns)
            
            if is_hazy:
                dest_file = outdoor_hazy_dir / img_file.name
                if not dest_file.exists():
                    shutil.copy2(img_file, dest_file)
                    copied_hazy += 1
            elif is_gt:
                dest_file = outdoor_gt_dir / img_file.name
                if not dest_file.exists():
                    shutil.copy2(img_file, dest_file)
                    copied_gt += 1
    
    print(f"\n📊 Summary:")
    print(f"   ✅ Copied {copied_hazy} hazy images to {outdoor_hazy_dir}")
    print(f"   ✅ Copied {copied_gt} gt images to {outdoor_gt_dir}")
    print(f"   📁 Dataset organized in: {target_path}")
    
    # Create a simple validation script
    validation_script = target_path / "validate_dataset.py"
    validation_code = f'''#!/usr/bin/env python3
"""Validate the downloaded OTS dataset structure."""

from pathlib import Path
from PIL import Image

def validate_dataset():
    base_dir = Path(__file__).parent
    hazy_dir = base_dir / "outdoor" / "hazy"
    gt_dir = base_dir / "outdoor" / "gt"
    
    print("🔍 Dataset Validation")
    print("=" * 40)
    
    hazy_files = list(hazy_dir.glob("*"))
    gt_files = list(gt_dir.glob("*"))
    
    print(f"Hazy images: {{len(hazy_files)}}")
    print(f"GT images: {{len(gt_files)}}")
    
    # Sample a few images to check they load properly
    for i, img_path in enumerate(hazy_files[:3]):
        try:
            with Image.open(img_path) as img:
                print(f"✅ {{img_path.name}}: {{img.size}} {{img.mode}}")
        except Exception as e:
            print(f"❌ {{img_path.name}}: {{e}}")
    
    print("\\n🎯 To use in your training:")
    print(f'data_root = "{{base_dir.absolute()}}"')

if __name__ == "__main__":
    validate_dataset()
'''
    
    with open(validation_script, 'w') as f:
        f.write(validation_code)
    
    print(f"   📝 Created validation script: {validation_script}")
    print(f"\n🚀 Run validation: python {validation_script}")
    
    return str(target_path)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Download and organize OTS dataset")
    parser.add_argument("--target_dir", type=str, default="datasets/SOTS",
                       help="Target directory to save organized dataset")
    
    args = parser.parse_args()
    
    result_path = download_and_organize_ots(args.target_dir)
    print(f"\n✨ All done! Dataset ready at: {result_path}")
















