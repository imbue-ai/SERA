#!/usr/bin/env python3
"""Download a trained model from the Modal sera-models volume.

Usage:
    uv run python scripts/download_model.py flask-specialist
    uv run python scripts/download_model.py flask-specialist --output-dir ./models/flask
    uv run python scripts/download_model.py --list  # list available runs
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys


def list_runs() -> None:
    """List available runs on the sera-models volume."""
    subprocess.run(["modal", "volume", "ls", "sera-models"], check=True)


def download_model(run_name: str, output_dir: str, variant: str = "converted") -> None:
    """Download a trained model from the Modal volume to local disk."""
    remote_path = f"{run_name}/{variant}"
    os.makedirs(output_dir, exist_ok=True)
    print(f"Downloading sera-models:/{remote_path} -> {output_dir}")
    subprocess.run(
        ["modal", "volume", "get", "sera-models", remote_path, output_dir],
        check=True,
    )
    print(f"Done! Model saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Download trained SERA models from Modal")
    parser.add_argument("run_name", nargs="?", help="Name of the training run to download")
    parser.add_argument("--output-dir", type=str, default="./models", help="Local output directory")
    parser.add_argument("--raw", action="store_true", help="Download raw checkpoint (before conversion)")
    parser.add_argument("--list", action="store_true", help="List available runs")
    args = parser.parse_args()

    if args.list:
        list_runs()
        return

    if not args.run_name:
        print("Error: run_name is required (or use --list)")
        sys.exit(1)

    variant = "raw" if args.raw else "converted"
    output = os.path.join(args.output_dir, args.run_name)
    download_model(args.run_name, output, variant=variant)


if __name__ == "__main__":
    main()
