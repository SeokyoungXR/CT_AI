"""Convert DeWorldSG temporal traces into VirtualME spatial evidence records."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from typing import Any

import numpy as np

from scripts.classes import CLASSES, REL_CLASSES
from scripts.trace_io import TraceBundle


@dataclass(frozen=True)
class SpatialObject:
    object_id: str
    class_id: int
    label: str
    position: tuple[float, float, float]
    uncertainty: float


@dataclass(frozen=True)
class SpatialRelation:
    subject_id: str
    predicate_id: int
    predicate: str
    object_id: str
    score: float


@dataclass(frozen=True)
class SpatialEvidenceRecord:
    record_id: str
    episode_id: str
    source: str
    evidence_class: str
    frame_index: int
    timestamp: str | None
    objects: tuple[SpatialObject, ...]
    relations: tuple[SpatialRelation, ...]
    features: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _array(value: Any) -> np.ndarray:
    if isinstance(value, np.ndarray):
        return value
    if hasattr(value, "detach") and hasattr(value, "cpu"):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _class_ids(value: Any) -> np.ndarray:
    classes = _array(value)
    if classes.ndim == 1:
        return classes.astype(int, copy=False)
    if classes.ndim == 2 and classes.shape[1] > 0:
        return np.argmax(classes, axis=1).astype(int, copy=False)
    raise ValueError(f"Object classes must be a non-empty 1D or 2D array: {classes.shape}")


def _predicate_entries(value: Any, count: int) -> list[tuple[int, int, int, float]]:
    if isinstance(value, dict):
        predicates = _array(value.get("predicates"))
        scores = _array(value.get("scores"))
        if predicates.shape != (count, count) or scores.shape != (count, count):
            raise ValueError("Compact relation predicates and scores must be square")
        return [
            (subject, int(predicates[subject, target]), target, float(scores[subject, target]))
            for subject in range(count)
            for target in range(count)
            if subject != target and float(scores[subject, target]) > 0
        ]

    relations = _array(value)
    if relations.shape[:2] != (count, count) or relations.ndim != 3:
        raise ValueError("Dense relations must have shape (objects, objects, predicates)")
    predicate_ids = np.argmax(relations, axis=2)
    scores = np.max(relations, axis=2)
    return [
        (subject, int(predicate_ids[subject, target]), target, float(scores[subject, target]))
        for subject in range(count)
        for target in range(count)
        if subject != target and float(scores[subject, target]) > 0
    ]


def _timestamp(start: datetime | None, frame_index: int, fps: float) -> str | None:
    if start is None:
        return None
    if fps <= 0:
        raise ValueError("fps must be positive")
    return (start + timedelta(seconds=frame_index / fps)).isoformat()


def build_spatial_records(
    trace: TraceBundle,
    scene_name: str,
    *,
    start: datetime | None = None,
    fps: float = 15.0,
    source: str = "ct_ai_scene_graph",
) -> list[SpatialEvidenceRecord]:
    """Build one evidence record per temporal frame without inferring user intent."""
    if not scene_name.strip():
        raise ValueError("scene_name cannot be empty")
    if fps <= 0:
        raise ValueError("fps must be positive")
    records: list[SpatialEvidenceRecord] = []
    previous_positions: np.ndarray | None = None

    for frame_index, (objects_value, relations_value) in enumerate(zip(trace.obj, trace.rel)):
        positions = _array(objects_value["means"]).astype(float, copy=False)
        covariances = _array(objects_value["covs"]).astype(float, copy=False)
        class_ids = _class_ids(objects_value["classes"])
        if positions.shape != (len(class_ids), 3) or covariances.shape != (len(class_ids), 3, 3):
            raise ValueError(f"Invalid object data at frame {frame_index}")

        objects = tuple(
            SpatialObject(
                object_id=f"{scene_name}_f{frame_index:06d}_o{object_index:04d}",
                class_id=int(class_id),
                label=CLASSES[int(class_id)] if 0 <= class_id < len(CLASSES) else "unknown",
                position=tuple(float(coordinate) for coordinate in position),
                uncertainty=float(np.sqrt(max(float(np.trace(covariance) / 3), 0.0))),
            )
            for object_index, (class_id, position, covariance) in enumerate(
                zip(class_ids, positions, covariances)
            )
        )
        object_ids = [item.object_id for item in objects]
        relations = tuple(
            SpatialRelation(
                subject_id=object_ids[subject],
                predicate_id=predicate_id,
                predicate=(REL_CLASSES[predicate_id] if 0 <= predicate_id < len(REL_CLASSES) else "unknown"),
                object_id=object_ids[target],
                score=score,
            )
            for subject, predicate_id, target, score in _predicate_entries(relations_value, len(objects))
            if 0 <= subject < len(objects) and 0 <= target < len(objects)
        )
        displacement = 0.0
        if previous_positions is not None and len(previous_positions) == len(positions) and len(positions):
            displacement = float(np.linalg.norm(positions - previous_positions, axis=1).mean())
        previous_positions = positions.copy()
        count = len(objects)
        records.append(
            SpatialEvidenceRecord(
                record_id=f"{scene_name}_frame_{frame_index:06d}",
                episode_id=f"{scene_name}_episode_{frame_index:06d}",
                source=source,
                evidence_class="contextual",
                frame_index=frame_index,
                timestamp=_timestamp(start, frame_index, fps),
                objects=objects,
                relations=relations,
                features={
                    "object_count": float(count),
                    "object_diversity": float(len({item.class_id for item in objects})),
                    "relation_count": float(len(relations)),
                    "relation_density": float(len(relations) / max(count * (count - 1), 1)),
                    "spatial_change_rate": displacement,
                },
            )
        )
    return records