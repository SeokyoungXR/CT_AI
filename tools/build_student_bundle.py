#!/usr/bin/env python3
"""Build the Google Drive ZIP used by the two-scene student tutorial."""

from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path


SCENES = ("office_1", "apartment_1")


def scene_files(scene_dir: Path) -> list[tuple[Path, Path]]:
    sequence = scene_dir / "sequence"
    required = [scene_dir / "mesh.ply", sequence / "_info.txt"]
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)
    images = sorted(sequence.glob("frame-*.color.jpg"))
    if not images:
        images = sorted(sequence.glob("frame-*.color.png"))
    poses = sorted(
        path
        for path in sequence.glob("frame-*.pose.txt")
        if not path.name.endswith(".slam.pose.txt")
    )
    if len(images) != len(poses):
        raise ValueError(
            f"RGB/pose count mismatch for {scene_dir.name}: {len(images)} vs {len(poses)}"
        )
    image_stems = [path.name.rsplit(".color.", 1)[0] for path in images]
    pose_stems = [path.name[: -len(".pose.txt")] for path in poses]
    if image_stems != pose_stems:
        raise ValueError(f"RGB/pose frame names do not align for {scene_dir.name}")
    files = [
        (scene_dir / "mesh.ply", Path("data") / scene_dir.name / "mesh.ply"),
        (sequence / "_info.txt", Path("data") / scene_dir.name / "sequence" / "_info.txt"),
    ]
    files.extend(
        (path, Path("data") / scene_dir.name / "sequence" / path.name)
        for path in images + poses
    )
    return files


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--replica-data-root", type=Path, required=True)
    parser.add_argument("--office-trace-dir", type=Path, required=True)
    parser.add_argument("--apartment-trace-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    replica_root = args.replica_data_root.expanduser().resolve()
    trace_dirs = {
        "office_1": args.office_trace_dir.expanduser().resolve(),
        "apartment_1": args.apartment_trace_dir.expanduser().resolve(),
    }
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    bundle_files: list[tuple[Path, Path]] = []
    scene_counts = {}
    for scene in SCENES:
        files = scene_files(replica_root / scene)
        bundle_files.extend(files)
        scene_counts[scene] = {
            "rgb_pose_pairs": (len(files) - 2) // 2,
            "files": len(files),
        }
        print(f"Collected {scene_counts[scene]['rgb_pose_pairs']} RGB/pose pairs for {scene}")

    trace_names = ("obj.pkl.gz", "rel.pkl.gz", "obj_2d.pkl.gz", "rel_2d.pkl.gz", "manifest.json")
    for scene, trace_dir in trace_dirs.items():
        for name in trace_names:
            path = trace_dir / name
            if not path.is_file():
                raise FileNotFoundError(path)
            bundle_files.append(
                (path, Path("assets") / "traces" / scene / name)
            )

    bundle_manifest = {
        "format": "ct-ai-student-assets",
        "format_version": 1,
        "extract_into": "CT_AI project root",
        "scenes": scene_counts,
        "includes": [
            "minimal ReplicaSSG RGB/pose/mesh data for office_1 and apartment_1",
            "NumPy-only compressed temporal traces for office_1 and apartment_1",
        ],
        "note": "The archive is self-contained for both tutorial scenes",
    }

    temporary = output.with_suffix(output.suffix + ".tmp")
    with zipfile.ZipFile(
        temporary,
        mode="w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=6,
        allowZip64=True,
    ) as archive:
        for index, (source, archive_path) in enumerate(bundle_files, start=1):
            archive.write(source, archive_path.as_posix())
            if index % 1000 == 0 or index == len(bundle_files):
                print(f"Added {index}/{len(bundle_files)} files")
        archive.writestr(
            "STUDENT_ASSETS.json",
            json.dumps(bundle_manifest, indent=2) + "\n",
        )
    temporary.replace(output)
    print(f"Bundle complete: {output}")


if __name__ == "__main__":
    main()
