#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Download PrivScreen dataset from HuggingFace.
Dataset: https://huggingface.co/datasets/fyzzzzzz/PrivScreen
"""

import os
import argparse
import sys
# Optional: set HF_ENDPOINT for mirror, e.g. export HF_ENDPOINT=https://hf-mirror.com
if os.environ.get("HF_ENDPOINT"):
    pass  # already set
else:
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

def download_with_huggingface_hub(repo_id, target_dir):
    """Download dataset using huggingface_hub."""
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("Error: huggingface_hub is required")
        print("Run: pip install huggingface_hub")
        return False
    
    print(f"Downloading dataset from HuggingFace: {repo_id}")
    print(f"Target directory: {target_dir}")
    
    try:
        # Skip macOS junk; some HF mirrors return 403 on .DS_Store and abort the whole snapshot.
        snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            local_dir=target_dir,
            ignore_patterns=["**/.DS_Store", ".DS_Store", "**/.ds_store"],
        )
        print(f"\n✓ Download complete.")
        print(f"Dataset location: {target_dir}")
        return True
    except Exception as e:
        print(f"Download failed: {e}")
        return False


def download_with_git(repo_id, target_dir):
    """Download dataset using git clone (fallback)."""
    import subprocess
    
    repo_url = f"https://huggingface.co/datasets/{repo_id}"
    
    print(f"Downloading dataset via git: {repo_url}")
    print(f"Target directory: {target_dir}")
    
    try:
        if os.path.exists(target_dir):
            print(f"Warning: target directory already exists: {target_dir}")
            response = input("Remove and re-download? (y/n): ")
            if response.lower() == 'y':
                import shutil
                shutil.rmtree(target_dir)
            else:
                print("Download cancelled.")
                return False
        
        cmd = ["git", "clone", repo_url, target_dir]
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print(f"\n✓ Download complete.")
            print(f"Dataset location: {target_dir}")
            print("\nNote: If the dataset uses Git LFS, you may need:")
            print("  sudo apt-get install git-lfs  # Ubuntu/Debian")
            print("  git lfs install")
            print("  cd {} && git lfs pull".format(target_dir))
            return True
        else:
            print(f"Git clone failed: {result.stderr}")
            return False
    except FileNotFoundError:
        print("Error: git not found. Please install git.")
        return False
    except Exception as e:
        print(f"Download failed: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Download PrivScreen dataset")
    parser.add_argument(
        '--target',
        type=str,
        default='./data',
        help='Target data directory (default: ./data)'
    )
    parser.add_argument(
        '--repo-id',
        type=str,
        default='fyzzzzzz/PrivScreen',
        help='HuggingFace dataset repo ID (default: fyzzzzzz/PrivScreen)'
    )
    parser.add_argument(
        '--method',
        type=str,
        choices=['auto', 'huggingface_hub', 'git'],
        default='auto',
        help='Download method: auto, huggingface_hub (recommended), or git'
    )
    
    args = parser.parse_args()
    target_dir = os.path.abspath(args.target)
    
    print("="*60)
    print("PrivScreen dataset download")
    print("="*60)
    print(f"Dataset: {args.repo_id}")
    print(f"Target: {target_dir}")
    print("="*60 + "\n")
    
    if os.path.exists(target_dir) and os.listdir(target_dir):
        print(f"Warning: target directory exists and is non-empty: {target_dir}")
        response = input("Continue? (y/n): ")
        if response.lower() != 'y':
            print("Download cancelled.")
            return 1
    
    if args.method == 'auto':
        try:
            import huggingface_hub
            method = 'huggingface_hub'
        except ImportError:
            print("huggingface_hub not installed, using git.")
            method = 'git'
    else:
        method = args.method
    
    if method == 'huggingface_hub':
        success = download_with_huggingface_hub(args.repo_id, target_dir)
    else:
        success = download_with_git(args.repo_id, target_dir)
    
    if success:
        print("\n" + "="*60)
        print("Dataset structure:")
        print("  data/")
        print("    {app}/")
        print("      images/")
        print("        *.png")
        print("      privacy_qa.json")
        print("      normal_qa.json")
        print("="*60)
        return 0
    else:
        print("\nDownload failed. Check the messages above.")
        return 1


if __name__ == "__main__":
    exit(main())
