"""Load the bundled temporal trace and align it with ReplicaSSG frames.

The trace files are outputs of the original DeWorldSG inference pipeline.  They
are stored as gzip-compressed, NumPy-only pickle files so the visualization can
run on macOS without PyTorch or CUDA.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np


TRACE_NAMES = ("obj", "rel", "obj_2d", "rel_2d")


@dataclass(frozen=True)
class SceneInputs:
    scene_dir: Path
    sequence_dir: Path
    mesh_path: Path
    image_paths: tuple[Path, ...]
    pose_paths: tuple[Path, ...]

    @property
    def frame_count(self) -> int:
        return len(self.image_paths)


@dataclass(frozen=True)
class TraceBundle:
    obj: list[dict[str, Any]]
    rel: list[Any]
    obj_2d: list[dict[str, Any]]
    rel_2d: list[dict[str, Any]]

    @property
    def frame_count(self) -> int:
        return len(self.obj)


@dataclass(frozen=True)
class CameraIntrinsic:
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float


def clean_path(value: str) -> Path:
    """Make paths copied from a terminal robust to surrounding whitespace."""
    cleaned = value.strip()
    if not cleaned:
        raise ValueError("path cannot be empty")
    return Path(cleaned).expanduser()


def _frame_stem(path: Path) -> str:
    for suffix in (".color.jpg", ".color.png"):
        if path.name.endswith(suffix):
            return path.name[: -len(suffix)]
    raise ValueError(f"Not a Replica color frame: {path}")


def find_scene_inputs(scene_or_sequence: Path) -> SceneInputs:
    """Find the mesh, RGB frames, and ground-truth poses for one scene."""
    supplied = scene_or_sequence.expanduser().resolve()
    if (supplied / "sequence" / "_info.txt").is_file():
        scene_dir = supplied
        sequence_dir = supplied / "sequence"
    elif supplied.name == "sequence" and (supplied / "_info.txt").is_file():
        scene_dir = supplied.parent
        sequence_dir = supplied
    else:
        raise FileNotFoundError(
            f"Replica scene was not found: {supplied}\n"
            "Pass either <scene> or <scene>/sequence; _info.txt must exist."
        )

    mesh_path = scene_dir / "mesh.ply"
    if not mesh_path.is_file():
        raise FileNotFoundError(f"Scene mesh is missing: {mesh_path}")

    image_paths = sorted(sequence_dir.glob("frame-*.color.jpg"))
    if not image_paths:
        image_paths = sorted(sequence_dir.glob("frame-*.color.png"))
    if not image_paths:
        raise FileNotFoundError(f"No frame-*.color.jpg/png files in {sequence_dir}")

    pose_by_stem = {
        path.name[: -len(".pose.txt")]: path
        for path in sequence_dir.glob("frame-*.pose.txt")
        if not path.name.endswith(".slam.pose.txt")
    }
    missing = [_frame_stem(path) for path in image_paths if _frame_stem(path) not in pose_by_stem]
    if missing:
        raise FileNotFoundError(
            f"Ground-truth pose is missing for {missing[0]} in {sequence_dir}"
        )
    pose_paths = tuple(pose_by_stem[_frame_stem(path)] for path in image_paths)

    return SceneInputs(
        scene_dir=scene_dir,
        sequence_dir=sequence_dir,
        mesh_path=mesh_path,
        image_paths=tuple(image_paths),
        pose_paths=pose_paths,
    )


def trace_path(trace_dir: Path, name: str) -> Path:
    if name not in TRACE_NAMES:
        raise ValueError(f"Unknown trace name: {name}")
    for filename in (f"{name}.pkl.gz", f"{name}.pkl"):
        candidate = trace_dir / filename
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"Missing {name}.pkl.gz (or {name}.pkl) in trace directory: {trace_dir}"
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_packaged_file(trace_dir: Path, path: Path) -> None:
    manifest_path = trace_dir / "manifest.json"
    if not manifest_path.is_file():
        return
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    record = manifest.get("files", {}).get(path.name)
    if not isinstance(record, dict) or "sha256" not in record:
        raise ValueError(f"Trace manifest has no checksum for {path.name}")
    actual = _sha256(path)
    if actual != record["sha256"]:
        raise ValueError(
            f"Trace checksum mismatch for {path.name}: expected {record['sha256']}, got {actual}"
        )


def load_pickle(path: Path) -> Any:
    """Load one trusted local pickle, with transparent gzip support."""
    opener = gzip.open if path.suffix == ".gz" else Path.open
    try:
        with opener(path, "rb") as stream:  # type: ignore[arg-type]
            return pickle.load(stream)
    except ModuleNotFoundError as error:
        if error.name == "torch":
            raise RuntimeError(
                f"{path} contains PyTorch tensors. Use the bundled NumPy-only trace files."
            ) from error
        raise


def _as_numpy(value: Any, name: str) -> np.ndarray:
    if isinstance(value, np.ndarray):
        return value
    if hasattr(value, "detach") and hasattr(value, "cpu"):
        return value.detach().cpu().numpy()
    try:
        return np.asarray(value)
    except Exception as error:  # pragma: no cover - defensive error message
        raise TypeError(f"{name} cannot be converted to a NumPy array") from error


def _validate_trace(bundle: TraceBundle) -> None:
    lengths = {
        "obj": len(bundle.obj),
        "rel": len(bundle.rel),
        "obj_2d": len(bundle.obj_2d),
        "rel_2d": len(bundle.rel_2d),
    }
    if len(set(lengths.values())) != 1:
        raise ValueError(f"Trace lengths do not match: {lengths}")
    if bundle.frame_count == 0:
        raise ValueError("Trace is empty")

    sample_indices = sorted({0, bundle.frame_count // 2, bundle.frame_count - 1})
    for index in sample_indices:
        obj = bundle.obj[index]
        for key in ("classes", "means", "covs"):
            if key not in obj:
                raise ValueError(f"obj[{index}] has no '{key}' field")
        classes = _as_numpy(obj["classes"], f"obj[{index}].classes")
        means = _as_numpy(obj["means"], f"obj[{index}].means")
        covs = _as_numpy(obj["covs"], f"obj[{index}].covs")
        count = len(classes)
        if classes.ndim not in (1, 2) or means.shape != (count, 3) or covs.shape != (count, 3, 3):
            raise ValueError(f"Invalid 3D object shapes at trace frame {index}")
        relation_value = bundle.rel[index]
        if isinstance(relation_value, dict):
            predicates = _as_numpy(
                relation_value.get("predicates"), f"rel[{index}].predicates"
            )
            scores = _as_numpy(relation_value.get("scores"), f"rel[{index}].scores")
            if predicates.shape != (count, count) or scores.shape != (count, count):
                raise ValueError(f"Invalid compact 3D relation shape at trace frame {index}")
            if predicates.size and (predicates.min() < 0 or predicates.max() >= 50):
                raise ValueError(f"Invalid compact predicate id at trace frame {index}")
        else:
            relations = _as_numpy(relation_value, f"rel[{index}]")
            if relations.ndim != 3 or relations.shape[:2] != (count, count):
                raise ValueError(
                    f"Invalid 3D relation shape at trace frame {index}: {relations.shape}"
                )

        obj_2d = bundle.obj_2d[index]
        rel_2d = bundle.rel_2d[index]
        boxes = _as_numpy(obj_2d.get("bboxes"), f"obj_2d[{index}].bboxes")
        labels = _as_numpy(obj_2d.get("classes"), f"obj_2d[{index}].classes")
        scores = _as_numpy(obj_2d.get("scores"), f"obj_2d[{index}].scores")
        if boxes.shape != (len(labels), 4) or scores.ndim != 1:
            raise ValueError(f"Invalid 2D object shapes at trace frame {index}")
        pairs = _as_numpy(rel_2d.get("rels"), f"rel_2d[{index}].rels")
        predicates = _as_numpy(rel_2d.get("rel_classes"), f"rel_2d[{index}].rel_classes")
        if pairs.shape != (len(predicates), 2):
            raise ValueError(f"Invalid 2D relation shapes at trace frame {index}")


def load_trace(
    trace_dir: Path,
    verify_checksums: bool = True,
    expected_scene: str | None = None,
) -> TraceBundle:
    """Load and validate the four temporal visualization pickle files."""
    directory = trace_dir.expanduser().resolve()
    manifest_path = directory / "manifest.json"
    if expected_scene is not None and manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        packaged_scene = manifest.get("scene")
        if packaged_scene is not None and packaged_scene != expected_scene:
            raise ValueError(
                f"Trace scene is '{packaged_scene}', but the Replica folder is '{expected_scene}'"
            )
    paths = {name: trace_path(directory, name) for name in TRACE_NAMES}
    if verify_checksums:
        for path in paths.values():
            _verify_packaged_file(directory, path)
    values = {name: load_pickle(path) for name, path in paths.items()}
    if not all(isinstance(values[name], list) for name in TRACE_NAMES):
        raise ValueError("Every trace pickle must contain a top-level list")
    bundle = TraceBundle(
        obj=values["obj"],
        rel=values["rel"],
        obj_2d=values["obj_2d"],
        rel_2d=values["rel_2d"],
    )
    _validate_trace(bundle)
    return bundle


def select_frame_indices(
    frame_count: int,
    start_frame: int = 0,
    frame_stride: int = 1,
    max_frames: int = 0,
) -> list[int]:
    """Return original indices so sampled RGB and trace entries stay aligned."""
    if frame_count < 1:
        raise ValueError("frame_count must be positive")
    if start_frame < 0 or start_frame >= frame_count:
        raise ValueError(f"--start-frame must be in [0, {frame_count - 1}]")
    if frame_stride < 1:
        raise ValueError("--frame-stride must be at least 1")
    if max_frames < 0:
        raise ValueError("--max-frames must be zero or greater")
    indices = list(range(start_frame, frame_count, frame_stride))
    return indices if max_frames == 0 else indices[:max_frames]


def load_pose(path: Path) -> np.ndarray:
    pose = np.loadtxt(path, dtype=np.float64)
    if pose.shape != (4, 4) or not np.isfinite(pose).all():
        raise ValueError(f"Camera pose must be a finite 4x4 matrix: {path}")
    return pose


def parse_intrinsics(info_path: Path) -> CameraIntrinsic:
    values: dict[str, str] = {}
    for line in info_path.read_text(encoding="utf-8").splitlines():
        if " = " in line:
            key, value = line.split(" = ", 1)
            values[key.strip()] = value.strip()
    required = ("m_colorWidth", "m_colorHeight", "m_calibrationColorIntrinsic")
    missing = [key for key in required if key not in values]
    if missing:
        raise ValueError(f"Missing camera metadata in {info_path}: {', '.join(missing)}")
    matrix = [float(value) for value in values["m_calibrationColorIntrinsic"].split()]
    if len(matrix) != 16:
        raise ValueError(f"Color intrinsic matrix must have 16 values: {info_path}")
    return CameraIntrinsic(
        width=int(values["m_colorWidth"]),
        height=int(values["m_colorHeight"]),
        fx=matrix[0],
        fy=matrix[5],
        cx=matrix[2],
        cy=matrix[6],
    )


def ensure_trace_matches_scene(trace: TraceBundle, scene: SceneInputs) -> None:
    if trace.frame_count != scene.frame_count:
        raise ValueError(
            "Temporal trace and Replica sequence must contain the same frames: "
            f"trace={trace.frame_count}, RGB/pose={scene.frame_count}. "
            f"Use the complete {scene.scene_dir.name} sequence, not a stride-sampled copy."
        )
    stems = frame_stems(scene.image_paths)
    expected = [f"frame-{index:06d}" for index in range(scene.frame_count)]
    if stems != expected:
        mismatch = next(
            (index for index, (actual, wanted) in enumerate(zip(stems, expected)) if actual != wanted),
            0,
        )
        raise ValueError(
            "Replica frames must be the original contiguous sequence: "
            f"index {mismatch} is {stems[mismatch]!r}, expected {expected[mismatch]!r}"
        )


def frame_stems(paths: Sequence[Path]) -> list[str]:
    return [_frame_stem(path) for path in paths]
