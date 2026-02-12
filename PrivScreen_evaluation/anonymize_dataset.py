#!/usr/bin/env python3
"""
Dataset image anonymization script for PrivScreen reproduction.

- Walk all image files under a directory.
- Anonymize with PrivacyProtectionLayer (from AndLab_protected).
- Write results to a new directory preserving structure.
- Copy non-image files as-is.

Depends on parent directory AndLab_protected (utils_mobile.privacy_protection).
"""

import os
import shutil
from pathlib import Path
from typing import List, Tuple, Dict

# Import privacy layer from gui_privacy_protection/AndLab_protected
import sys
_here = os.path.dirname(os.path.abspath(__file__))
_parent = os.path.abspath(os.path.join(_here, '..'))
if _parent not in sys.path:
    sys.path.insert(0, _parent)
from AndLab_protected.utils_mobile.privacy_protection import PrivacyProtectionLayer


IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff', '.tif', '.webp'}


def is_image_file(file_path: str) -> bool:
    """Return True if the file is an image by extension."""
    return Path(file_path).suffix.lower() in IMAGE_EXTENSIONS


def get_all_files(source_dir: str) -> List[Tuple[str, str]]:
    """
    List all files under source_dir (including subdirs).
    Returns:
        List of (relative_path, absolute_path) tuples.
    """
    files = []
    source_path = Path(source_dir)
    if not source_path.exists():
        raise ValueError(f"Source directory does not exist: {source_dir}")
    for root, dirs, filenames in os.walk(source_dir):
        for filename in filenames:
            abs_path = os.path.join(root, filename)
            rel_path = os.path.relpath(abs_path, source_dir)
            files.append((rel_path, abs_path))
    return files


def anonymize_dataset(
    source_dir: str,
    output_dir: str,
    privacy_layer: PrivacyProtectionLayer = None
) -> None:
    """Anonymize all images under source_dir and write to output_dir."""
    if privacy_layer is None:
        print("[AnonymizeDataset] Initializing privacy protection layer...")
        privacy_layer = PrivacyProtectionLayer(enabled=True)
        print("[AnonymizeDataset] Privacy layer ready.")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"[AnonymizeDataset] Scanning source: {source_dir}")
    all_files = get_all_files(source_dir)
    print(f"[AnonymizeDataset] Found {len(all_files)} files")

    image_count = 0
    copied_count = 0
    failed_count = 0
    timing_stats = {'ocr_times': [], 'ner_times': [], 'total_times': []}

    for idx, (rel_path, abs_path) in enumerate(all_files, 1):
        output_file_path = output_path / rel_path
        output_file_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"[AnonymizeDataset] [{idx}/{len(all_files)}] Processing: {rel_path}")

        if is_image_file(abs_path):
            try:
                image_count += 1
                print(f"  -> Image file, anonymizing...")
                (masked_image_path, tokens), timing = privacy_layer.identify_and_mask_screenshot_with_timing(abs_path)
                timing_stats['ocr_times'].append(timing['ocr_time'])
                timing_stats['ner_times'].append(timing['ner_time'])
                timing_stats['total_times'].append(timing['total_time'])
                print(f"  -> Timing: OCR={timing['ocr_time']:.3f}s, NER={timing['ner_time']:.3f}s, total={timing['total_time']:.3f}s")

                if masked_image_path == abs_path:
                    print(f"  -> No sensitive info detected, copying original")
                    shutil.copy2(abs_path, output_file_path)
                else:
                    if os.path.exists(masked_image_path):
                        shutil.copy2(masked_image_path, output_file_path)
                        print(f"  -> Anonymized, saved to: {output_file_path}")
                        if tokens:
                            print(f"  -> Detected {len(tokens)} new token(s)")
                        if masked_image_path != abs_path and os.path.exists(masked_image_path):
                            try:
                                os.remove(masked_image_path)
                            except Exception as e:
                                print(f"  -> Warning: could not remove temp file {masked_image_path}: {e}")
                    else:
                        print(f"  -> Warning: masked file missing, copying original")
                        shutil.copy2(abs_path, output_file_path)
            except Exception as e:
                failed_count += 1
                print(f"  -> Error processing image: {e}")
                try:
                    shutil.copy2(abs_path, output_file_path)
                    print(f"  -> Copied original as fallback")
                except Exception as e2:
                    print(f"  -> Error copying original: {e2}")
        else:
            try:
                copied_count += 1
                shutil.copy2(abs_path, output_file_path)
                print(f"  -> Non-image file, copied")
            except Exception as e:
                failed_count += 1
                print(f"  -> Error copying file: {e}")

    avg_ocr_time = sum(timing_stats['ocr_times']) / len(timing_stats['ocr_times']) if timing_stats['ocr_times'] else 0.0
    avg_ner_time = sum(timing_stats['ner_times']) / len(timing_stats['ner_times']) if timing_stats['ner_times'] else 0.0
    avg_total_time = sum(timing_stats['total_times']) / len(timing_stats['total_times']) if timing_stats['total_times'] else 0.0

    print("\n" + "="*60)
    print("[AnonymizeDataset] Done.")
    print(f"  Total: {len(all_files)}  Images: {image_count}  Copied: {copied_count}  Failed: {failed_count}")
    print(f"  Output: {output_dir}")
    print("="*60)
    if timing_stats['ocr_times']:
        print("\n[AnonymizeDataset] Average timing")
        print(f"  OCR: {avg_ocr_time:.3f}s  NER: {avg_ner_time:.3f}s  Total: {avg_total_time:.3f}s")
        print("="*60)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Anonymize dataset images with PrivacyProtectionLayer (PrivScreen reproduction)")
    parser.add_argument("--source", type=str, default="./data", help="Source dataset directory (default: ./data)")
    parser.add_argument("--output", type=str, default="./data_anonymized/privscreen", help="Output directory (default: ./data_anonymized/privscreen)")
    args = parser.parse_args()
    source_dir = os.path.abspath(args.source)
    output_dir = os.path.abspath(args.output)

    print("="*60)
    print("Dataset anonymization (PrivacyProtectionLayer)")
    print("="*60)
    print(f"Source: {source_dir}")
    print(f"Output: {output_dir}")
    print("="*60)

    if os.path.exists(output_dir):
        response = input(f"\nOutput directory exists: {output_dir}\nContinue? (y/n): ")
        if response.lower() != 'y':
            print("Cancelled.")
            return 0

    try:
        anonymize_dataset(source_dir, output_dir)
    except Exception as e:
        print(f"\n[AnonymizeDataset] Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    return 0


if __name__ == "__main__":
    exit(main())
