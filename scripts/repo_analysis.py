#!/usr/bin/env python3
"""Analyze repos in SERA/repos/ for complexity and test coverage proxies.

Outputs a CSV with per-repo metrics and prints summary statistics.

Usage:
    python scripts/repo_analysis.py [--repos-dir repos/] [--output repo_analysis.csv]
"""

import argparse
import ast
import csv
import os
import sys
from dataclasses import dataclass, fields
from pathlib import Path
from statistics import mean, median


@dataclass
class RepoMetrics:
    repo: str
    py_files: int = 0
    py_loc: int = 0  # non-blank, non-comment lines
    total_functions: int = 0
    total_classes: int = 0
    test_files: int = 0
    test_functions: int = 0
    test_fn_ratio: float = 0.0


def is_test_path(path: Path) -> bool:
    """Check if a Python file is test-related based on its path."""
    parts = path.parts
    name = path.stem
    for part in parts:
        p = part.lower()
        if p in ("test", "tests", "testing", "test_utils"):
            return True
    if name.startswith("test_") or name.endswith("_test") or name == "conftest":
        return True
    return False


def is_test_function(name: str) -> bool:
    """Check if a function name looks test-related."""
    return name.startswith("test_") or name.startswith("test") and (
        len(name) == 4 or name[4].isupper()
    )


def count_loc(source: str) -> int:
    """Count non-blank, non-comment lines."""
    count = 0
    for line in source.splitlines():
        stripped = line.strip()
        if stripped and not stripped.startswith("#"):
            count += 1
    return count


def extract_functions(source: str) -> list[tuple[str, bool]]:
    """Parse source and return list of (function_name, is_method) tuples."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return [], []

    functions = []
    classes = []

    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            classes.append(node.name)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            functions.append(node.name)

    return functions, classes


def analyze_repo(repo_path: Path) -> RepoMetrics:
    """Collect metrics for a single repository."""
    metrics = RepoMetrics(repo=repo_path.name)

    for root, dirs, files in os.walk(repo_path):
        # Skip hidden dirs, __pycache__, .git, venvs
        dirs[:] = [
            d
            for d in dirs
            if not d.startswith(".")
            and d != "__pycache__"
            and d not in ("venv", ".venv", "node_modules", ".tox", ".eggs")
        ]

        for fname in files:
            if not fname.endswith(".py"):
                continue

            fpath = Path(root) / fname
            rel_path = fpath.relative_to(repo_path)
            metrics.py_files += 1

            if is_test_path(rel_path):
                metrics.test_files += 1

            try:
                source = fpath.read_text(encoding="utf-8", errors="replace")
            except (OSError, PermissionError):
                continue

            metrics.py_loc += count_loc(source)
            functions, classes = extract_functions(source)
            metrics.total_classes += len(classes)
            metrics.total_functions += len(functions)

            for fn_name in functions:
                if is_test_function(fn_name):
                    metrics.test_functions += 1

    if metrics.total_functions > 0:
        metrics.test_fn_ratio = round(
            metrics.test_functions / metrics.total_functions, 4
        )

    return metrics


def print_summary(all_metrics: list[RepoMetrics]):
    """Print summary statistics across all repos."""
    numeric_fields = [
        "py_files",
        "py_loc",
        "total_functions",
        "total_classes",
        "test_files",
        "test_functions",
        "test_fn_ratio",
    ]

    print("\n" + "=" * 70)
    print("SUMMARY STATISTICS")
    print("=" * 70)
    print(f"Total repos analyzed: {len(all_metrics)}")
    print()

    header = f"{'metric':<20} {'min':>10} {'max':>10} {'median':>10} {'mean':>10}"
    print(header)
    print("-" * len(header))

    for field in numeric_fields:
        values = [getattr(m, field) for m in all_metrics]
        if not values:
            continue
        fmt = ".4f" if field == "test_fn_ratio" else ".0f"
        print(
            f"{field:<20} {min(values):>10{fmt}} {max(values):>10{fmt}} "
            f"{median(values):>10{fmt}} {mean(values):>10{fmt}}"
        )

    # Top/bottom 5 by LOC
    by_loc = sorted(all_metrics, key=lambda m: m.py_loc, reverse=True)
    print(f"\nTop 5 by Python LOC:")
    for m in by_loc[:5]:
        print(f"  {m.repo:<30} {m.py_loc:>8} LOC, {m.total_functions:>6} fns")

    print(f"\nBottom 5 by Python LOC:")
    for m in by_loc[-5:]:
        print(f"  {m.repo:<30} {m.py_loc:>8} LOC, {m.total_functions:>6} fns")

    # Top/bottom 5 by test ratio
    with_fns = [m for m in all_metrics if m.total_functions > 0]
    by_ratio = sorted(with_fns, key=lambda m: m.test_fn_ratio, reverse=True)
    print(f"\nTop 5 by test function ratio:")
    for m in by_ratio[:5]:
        print(
            f"  {m.repo:<30} {m.test_fn_ratio:.4f} "
            f"({m.test_functions}/{m.total_functions})"
        )

    print(f"\nBottom 5 by test function ratio (with >0 functions):")
    for m in by_ratio[-5:]:
        print(
            f"  {m.repo:<30} {m.test_fn_ratio:.4f} "
            f"({m.test_functions}/{m.total_functions})"
        )


def main():
    parser = argparse.ArgumentParser(description="Analyze SERA repos")
    parser.add_argument(
        "--repos-dir",
        type=Path,
        default=Path("repos"),
        help="Directory containing cloned repos",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("repo_analysis.csv"),
        help="Output CSV path",
    )
    args = parser.parse_args()

    if not args.repos_dir.is_dir():
        print(f"Error: {args.repos_dir} is not a directory", file=sys.stderr)
        sys.exit(1)

    repo_dirs = sorted(
        [d for d in args.repos_dir.iterdir() if d.is_dir() and not d.name.startswith(".")]
    )
    print(f"Found {len(repo_dirs)} repos in {args.repos_dir}")

    all_metrics = []
    for i, repo_dir in enumerate(repo_dirs, 1):
        print(f"  [{i}/{len(repo_dirs)}] Analyzing {repo_dir.name}...", end="", flush=True)
        metrics = analyze_repo(repo_dir)
        all_metrics.append(metrics)
        print(f" {metrics.py_loc} LOC, {metrics.total_functions} fns, {metrics.test_functions} test fns")

    # Write CSV
    fieldnames = [f.name for f in fields(RepoMetrics)]
    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for m in all_metrics:
            writer.writerow({f.name: getattr(m, f.name) for f in fields(RepoMetrics)})

    print(f"\nCSV written to {args.output}")
    print_summary(all_metrics)


if __name__ == "__main__":
    main()
