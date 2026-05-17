#!/usr/bin/env python3
"""Re-save joblib model files with compression to reduce disk/download size."""

import argparse
from pathlib import Path
import joblib


def compress_ucbshift_weights(ucbshift_weights_dir: Path, compress: int = 3) -> None:
    sav_files = sorted(ucbshift_weights_dir.glob("*.sav"))
    if not sav_files:
        raise FileNotFoundError(f"No .sav files found in {ucbshift_weights_dir}")

    total_before = sum(f.stat().st_size for f in sav_files)
    print(f"Found {len(sav_files)} model files — {total_before / 1e9:.2f} GB total\n")

    for sav in sav_files:
        before = sav.stat().st_size
        model = joblib.load(sav)
        joblib.dump(model, sav, compress=compress)
        after = sav.stat().st_size
        ratio = (1 - after / before) * 100
        print(f"  {sav.name:20s}  {before/1e6:6.1f} MB → {after/1e6:6.1f} MB  ({ratio:.0f}% smaller)")

    total_after = sum(f.stat().st_size for f in sav_files)
    ratio = (1 - total_after / total_before) * 100
    print(f"\nTotal: {total_before/1e9:.2f} GB → {total_after/1e9:.2f} GB  ({ratio:.0f}% smaller)")
    print("\nDone. Re-package with:  tar -czf ucbshift_weights.tgz ucbshift_weights/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compress UCBShift .sav model files in-place.")
    parser.add_argument("--ucbshift_weights", default="ucbshift_weights", help="Path to ucbshift_weights directory (default: ./ucbshift_weights)")
    parser.add_argument("--compress", type=int, default=3, choices=range(1, 10),
                        metavar="1-9", help="Compression level (default: 3)")
    args = parser.parse_args()
    compress_ucbshift_weights(Path(args.ucbshift_weights), compress=args.compress)
