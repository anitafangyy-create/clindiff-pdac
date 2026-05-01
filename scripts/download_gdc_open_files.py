#!/usr/bin/env python3
"""Download open-access GDC files by file_id."""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path


GDC_DATA_URL = "https://api.gdc.cancer.gov/data/"


def load_manifest(path: Path) -> dict[str, dict[str, str]]:
    if path.suffix.lower() == ".csv":
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            return {
                row["file_id"]: row
                for row in reader
                if row.get("file_id")
            }

    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)

    hits = payload.get("data", {}).get("hits", [])
    return {row["file_id"]: row for row in hits if row.get("file_id")}


def download_file(file_id: str, file_name: str, output_dir: Path, timeout: int) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / file_name
    request = urllib.request.Request(f"{GDC_DATA_URL}{file_id}")

    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            destination.write_bytes(response.read())
    except Exception:
        subprocess.run(
            [
                "curl",
                "-L",
                "--fail",
                "--max-time",
                str(timeout),
                "-o",
                str(destination),
                f"{GDC_DATA_URL}{file_id}",
            ],
            check=True,
        )

    return destination


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        required=True,
        help="Path to a GDC manifest in JSON or CSV format with file_id/file_name fields.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory to save downloaded files into.",
    )
    parser.add_argument(
        "--file-id",
        action="append",
        default=[],
        help="Specific GDC file_id to download. Can be passed multiple times.",
    )
    parser.add_argument(
        "--priority",
        choices=["high", "medium", "low"],
        help="If manifest is CSV and has a priority column, download all rows at this priority.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional cap on how many files to download.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=180,
        help="Per-request timeout in seconds.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest_path = Path(args.manifest)
    output_dir = Path(args.output_dir)
    manifest = load_manifest(manifest_path)

    requested_ids = list(args.file_id)
    if args.priority:
        for row in manifest.values():
            if row.get("priority") == args.priority and row.get("file_id"):
                requested_ids.append(row["file_id"])

    if not requested_ids:
        print("No files selected. Pass --file-id or --priority.", file=sys.stderr)
        return 2

    seen: set[str] = set()
    ordered_ids: list[str] = []
    for file_id in requested_ids:
        if file_id and file_id not in seen:
            seen.add(file_id)
            ordered_ids.append(file_id)

    if args.limit is not None:
        ordered_ids = ordered_ids[: args.limit]

    failures = 0
    for file_id in ordered_ids:
        row = manifest.get(file_id)
        if not row:
            print(f"[skip] missing manifest row for {file_id}", file=sys.stderr)
            failures += 1
            continue

        file_name = row.get("file_name") or file_id
        try:
            dest = download_file(file_id, file_name, output_dir, args.timeout)
            print(f"[ok] {file_id} -> {dest}")
        except urllib.error.HTTPError as exc:
            failures += 1
            print(f"[error] {file_id} HTTP {exc.code}: {file_name}", file=sys.stderr)
        except Exception as exc:  # noqa: BLE001
            failures += 1
            print(f"[error] {file_id} {type(exc).__name__}: {exc}", file=sys.stderr)

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
