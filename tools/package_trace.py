#!/usr/bin/env python3
"""Maintainer utility: make the original CUDA trace portable and compressed.

This utility is not part of the tutorial runtime.  Run it once in an
environment that can load the original CUDA-backed ``obj_2d.pkl``.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import pickle
import shutil
import subprocess
from pathlib import Path
from typing import Any

import numpy as np


TRACE_NAMES = ("obj", "rel", "obj_2d", "rel_2d")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def numpy_only(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: numpy_only(item) for key, item in value.items()}
    if isinstance(value, list):
        return [numpy_only(item) for item in value]
    if isinstance(value, tuple):
        return tuple(numpy_only(item) for item in value)
    if isinstance(value, np.ndarray):
        return value
    if hasattr(value, "detach") and hasattr(value, "cpu"):
        return value.detach().cpu().numpy()
    return value


def compact_relations(value: list[Any]) -> tuple[list[dict[str, np.ndarray]], int]:
    """Keep exactly the two dense-relation values consumed by visualization."""
    compact: list[dict[str, np.ndarray]] = []
    maximum_score = 0
    for index, relation in enumerate(value):
        dense = np.asarray(relation)
        if dense.ndim != 3 or dense.shape[0] != dense.shape[1] or dense.shape[2] != 50:
            raise ValueError(f"Invalid dense relation shape at frame {index}: {dense.shape}")
        predicates = dense.argmax(-1).astype(np.uint8, copy=False)
        scores = np.take_along_axis(dense, predicates[..., None], axis=-1)[..., 0]
        if scores.size:
            maximum_score = max(maximum_score, int(scores.max()))
        if np.any(scores < 0) or maximum_score > np.iinfo(np.uint32).max:
            raise ValueError("Relation evidence cannot be represented exactly as uint32")
        compact.append(
            {
                "predicates": predicates,
                "scores": scores.astype(np.uint32, copy=False),
            }
        )
    return compact, maximum_score


def gzip_copy(source: Path, destination: Path) -> None:
    """Compress a NumPy-only pickle without materializing it in memory."""
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    destination.parent.mkdir(parents=True, exist_ok=True)
    pigz = shutil.which("pigz")
    if pigz is not None:
        with temporary.open("wb") as output:
            process = subprocess.run(
                [pigz, "-6", "-n", "-c", str(source)],
                stdout=output,
                stderr=subprocess.PIPE,
                text=False,
            )
        if process.returncode != 0:
            temporary.unlink(missing_ok=True)
            message = process.stderr.decode("utf-8", errors="replace")
            raise RuntimeError(f"pigz failed for {source}: {message}")
    else:
        with source.open("rb") as input_stream, temporary.open("wb") as raw_output:
            with gzip.GzipFile(
                filename="", mode="wb", fileobj=raw_output, compresslevel=6, mtime=0
            ) as output:
                shutil.copyfileobj(input_stream, output, length=1024 * 1024)
    os.replace(temporary, destination)


def package_file(
    source: Path,
    destination: Path,
    conversion: str,
) -> dict[str, Any]:
    source_hash = sha256_file(source)
    if conversion == "copy":
        gzip_copy(source, destination)
        return {
            "sha256": sha256_file(destination),
            "compressed_bytes": destination.stat().st_size,
            "pickle_sha256": source_hash,
            "pickle_bytes": source.stat().st_size,
            "source_sha256": source_hash,
            "source_bytes": source.stat().st_size,
            "numpy_only": True,
        }

    with source.open("rb") as stream:
        value = pickle.load(stream)
    representation = "numpy"
    if conversion == "tensors":
        portable = numpy_only(value)
    elif conversion == "relations":
        portable, maximum_score = compact_relations(value)
        representation = "argmax-predicate+max-evidence"
        del value
    else:
        raise ValueError(f"Unknown conversion: {conversion}")
    serialized = pickle.dumps(portable, protocol=4)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with temporary.open("wb") as raw_stream:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw_stream, compresslevel=6, mtime=0) as stream:
            stream.write(serialized)
    os.replace(temporary, destination)
    return {
        "sha256": sha256_file(destination),
        "compressed_bytes": destination.stat().st_size,
        "pickle_sha256": sha256_bytes(serialized),
        "pickle_bytes": len(serialized),
        "source_sha256": source_hash,
        "source_bytes": source.stat().st_size,
        "frames": len(portable),
        "numpy_only": True,
        "representation": representation,
        **({"maximum_evidence": maximum_score} if conversion == "relations" else {}),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--scene", type=str)
    args = parser.parse_args()

    source_dir = args.source_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    records = {}
    for name in TRACE_NAMES:
        source = source_dir / f"{name}.pkl"
        if not source.is_file():
            raise FileNotFoundError(source)
        destination = output_dir / f"{name}.pkl.gz"
        print(f"Packaging {source.name} -> {destination.name}")
        records[destination.name] = package_file(
            source,
            destination,
            conversion=(
                "tensors"
                if name == "obj_2d"
                else "relations"
                if name == "rel"
                else "copy"
            ),
        )

    frame_count = records["obj_2d.pkl.gz"]["frames"]
    for record in records.values():
        record["frames"] = frame_count

    manifest = {
        "format": "deworldsg-temporal-trace",
        "format_version": 1,
        "scene": args.scene or source_dir.name,
        "frames": frame_count,
        "description": (
            "Original visualization trace; obj_2d CUDA tensors were converted "
            "losslessly to NumPy, and dense relations retain the exact argmax "
            "predicate and max evidence used by the renderer."
        ),
        "files": records,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(f"Wrote {output_dir / 'manifest.json'}")


if __name__ == "__main__":
    main()
