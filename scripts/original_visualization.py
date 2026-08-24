"""Mac-compatible port of the original DeWorldSG visualization code.

This module is derived from the Apache-2.0 files in
``FROSS/Merging/Visualization``.  Path handling, NumPy-only trace loading, and
single-scene orchestration were added; the camera, Gaussian, annotation, and
Matplotlib drawing rules are intentionally kept unchanged.
"""

from __future__ import annotations

import io
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch
from PIL import Image

from scripts.classes import (
    CLASSES,
    REL_CLASSES,
    obj_colors,
    rel_colors,
    valid_class,
    valid_rel_class,
)
from scripts.trace_io import CameraIntrinsic


PANEL_WIDTH = 960
PANEL_HEIGHT = 540


def as_numpy(value: Any) -> np.ndarray:
    if isinstance(value, np.ndarray):
        return value
    if hasattr(value, "detach") and hasattr(value, "cpu"):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def curvature_sequence(count: int, base: float = 0.12) -> list[float]:
    sequence: list[float] = []
    multiplier = 2
    sign = -1
    while len(sequence) < count:
        sequence.append(sign * multiplier * base)
        sign *= -1
        if sign == -1:
            multiplier += 2
    return sequence


def arc_midpoint(p0: tuple[float, float], p1: tuple[float, float], radius: float) -> np.ndarray:
    start = np.asarray(p0, dtype=float)
    end = np.asarray(p1, dtype=float)
    chord = end - start
    distance = np.linalg.norm(chord)
    if distance == 0:
        return start
    midpoint = (start + end) / 2
    if radius == 0:
        return midpoint
    perpendicular = np.asarray([-chord[1], chord[0]]) / distance
    return midpoint + perpendicular * (radius * distance / 2)


def _text_color(background: np.ndarray) -> str:
    luminance = (
        0.2126 * background[0] ** 2.2
        + 0.7152 * background[1] ** 2.2
        + 0.0722 * background[2] ** 2.2
    )
    return "black" if luminance > 0.55 else "white"


def _figure_to_rgb(figure: plt.Figure, expected_size: tuple[int, int]) -> np.ndarray:
    stream = io.BytesIO()
    figure.savefig(stream, format="png", dpi=300, bbox_inches="tight", pad_inches=0)
    stream.seek(0)
    with Image.open(stream) as image:
        result = np.asarray(image.convert("RGB")).copy()
    plt.close(figure)
    expected_width, expected_height = expected_size
    if result.shape != (expected_height, expected_width, 3):
        raise RuntimeError(
            "Matplotlib produced an unexpected panel size: "
            f"{result.shape[1]}x{result.shape[0]} (expected {expected_width}x{expected_height})"
        )
    return result


def _draw_2d_boxes(
    axis: plt.Axes,
    boxes: np.ndarray,
    labels: np.ndarray,
    scores: np.ndarray,
) -> None:
    # This filtering order is preserved from the author's renderer.
    filtered_scores = scores[scores > 0.7]
    if len(filtered_scores) < len(boxes):
        raise ValueError(
            "The 2D trace has fewer >0.7 scores than boxes; it is not compatible with the original renderer"
        )
    for index in range(len(boxes)):
        cx, cy, width, height = boxes[index]
        x1, y1 = cx - width / 2, cy - height / 2
        class_id = int(labels[index])
        if class_id not in valid_class:
            continue
        color = obj_colors[valid_class.index(class_id)]
        axis.add_patch(
            plt.Rectangle(
                (x1, y1),
                width,
                height,
                fill=False,
                edgecolor=color,
                linewidth=0.75,
            )
        )
        axis.text(
            x1 + 8,
            y1 + 8,
            f"{CLASSES[class_id]}: {filtered_scores[index]:.2f}",
            color=_text_color(color),
            fontsize=3,
            fontfamily="sans-serif",
            ha="left",
            va="top",
            bbox=dict(
                boxstyle="round,pad=0.25,rounding_size=0.2",
                facecolor=color,
                edgecolor=color,
                alpha=1,
            ),
        )


def _draw_2d_relations(
    axis: plt.Axes,
    pairs: np.ndarray,
    predicates: np.ndarray,
    boxes: np.ndarray,
    labels: np.ndarray,
) -> None:
    grouped: defaultdict[tuple[int, int], list[int]] = defaultdict(list)
    for pair, predicate in zip(pairs, predicates):
        grouped[(int(pair[0]), int(pair[1]))].append(int(predicate))

    for (subject, object_), relation_ids in grouped.items():
        curvatures = curvature_sequence(len(relation_ids))
        for relation_index, predicate in enumerate(relation_ids):
            if (
                predicate not in valid_rel_class
                or subject == object_
                or int(labels[subject]) not in valid_class
                or int(labels[object_]) not in valid_class
            ):
                continue
            sx, sy = boxes[subject][:2]
            ox, oy = boxes[object_][:2]
            curvature = curvatures[relation_index]
            color = rel_colors[valid_rel_class.index(predicate)]
            axis.add_patch(
                FancyArrowPatch(
                    (sx, sy),
                    (ox, oy),
                    connectionstyle=f"arc3,rad={curvature}",
                    arrowstyle="-|>",
                    mutation_scale=6,
                    linewidth=0.5,
                    color=color,
                )
            )
            lx, ly = arc_midpoint((sx, sy), (ox, oy), curvature)
            axis.text(
                lx,
                ly,
                REL_CLASSES[predicate],
                ha="center",
                va="center",
                fontsize=2.5,
                color=_text_color(color),
                fontfamily="sans-serif",
                bbox=dict(
                    boxstyle="round,pad=0.25,rounding_size=0.2",
                    facecolor=color,
                    edgecolor=color,
                ),
            )


def render_2d_panel(image_path: Path, obj_2d: dict[str, Any], rel_2d: dict[str, Any]) -> np.ndarray:
    with Image.open(image_path) as source:
        image = source.convert("RGB")
        width, height = image.size
        figure = plt.figure()
        axis = figure.add_subplot(111)
        axis.imshow(image)
        boxes = as_numpy(obj_2d["bboxes"])
        labels = as_numpy(obj_2d["classes"])
        scores = as_numpy(obj_2d["scores"])
        pairs = as_numpy(rel_2d["rels"])
        predicates = as_numpy(rel_2d["rel_classes"])
        _draw_2d_relations(axis, pairs, predicates, boxes, labels)
        _draw_2d_boxes(axis, boxes, labels, scores)
        axis.axis("off")
        figure.set_size_inches(width / 300, height / 300)
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
        return _figure_to_rgb(figure, (width, height))


def object_classes(obj: dict[str, Any]) -> tuple[np.ndarray, bool]:
    classes = as_numpy(obj["classes"])
    is_v2 = classes.ndim > 1
    if is_v2:
        classes = classes.argmax(-1)
    return classes.astype(np.int64, copy=False), is_v2


def render_3d_panel(scene_mesh: Any, pose: np.ndarray, obj: dict[str, Any]) -> np.ndarray:
    """Render one frame using the author's original Gaussian construction."""
    try:
        import pyvista as pv
        from scipy.stats import multivariate_normal
    except ModuleNotFoundError as error:  # pragma: no cover - integration dependency
        raise RuntimeError(
            "PyVista/VTK and SciPy are required. Update the conda environment from environment.yml."
        ) from error

    camera_rotation = pose[:3, :3]
    camera_translation = pose[:3, 3]
    classes, is_v2 = object_classes(obj)
    means = as_numpy(obj["means"])
    covariances = as_numpy(obj["covs"])

    plotter = pv.Plotter(off_screen=True, window_size=(PANEL_WIDTH, PANEL_HEIGHT))
    try:
        plotter.camera.view_angle = 58.4
        plotter.camera.focal_point = (camera_translation + camera_rotation[:, 2]).tolist()
        plotter.camera.position = camera_translation.tolist()
        plotter.camera.up = (-camera_rotation[:, 1]).tolist()
        plotter.add_mesh(scene_mesh, rgb=True)

        for index, class_id_value in enumerate(classes):
            class_id = int(class_id_value)
            if class_id not in valid_class:
                continue
            color = obj_colors[valid_class.index(class_id)]
            mean = means[index]
            covariance = covariances[index]
            if is_v2:
                covariance = covariance * 1.5

            eigenvalues, eigenvectors = np.linalg.eigh(covariance)
            if is_v2 and np.min(eigenvalues) < 0.005:
                flat_axis = int(np.argmin(eigenvalues))
                eigenvalues[flat_axis] = 0.005
                covariance = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
            radii = np.sqrt(eigenvalues)
            scale = 2 if is_v2 else np.sqrt(2 * np.log(2))
            half_extent = float((radii * scale).max())
            step = max(50, 2 * half_extent / 0.1) * 1j

            if np.all(np.abs(camera_translation - mean) < half_extent):
                continue

            x, y, z = np.mgrid[
                mean[0] - half_extent : mean[0] + half_extent : step,
                mean[1] - half_extent : mean[1] + half_extent : step,
                mean[2] - half_extent : mean[2] + half_extent : step,
            ]
            positions = np.empty(x.shape + (3,))
            positions[:, :, :, 0] = x
            positions[:, :, :, 1] = y
            positions[:, :, :, 2] = z
            distribution = multivariate_normal(mean, covariance)
            density = distribution.pdf(positions)
            density *= ((2 * math.pi) ** 3 * np.linalg.det(covariance)) ** 0.5

            grid = pv.StructuredGrid(x, y, z)
            grid.point_data["pdf"] = density.flatten(order="F")
            iso_range = np.arange(0.1, 1.0, 0.02) if is_v2 else np.arange(0.05, 1.0, 0.01)
            opacity_scale = 0.15 if is_v2 else 0.1
            for iso in iso_range:
                # ``extract_surface(algorithm=None)`` is PyVista's direct
                # replacement for the original deprecated extract_geometry().
                surface = grid.contour([iso]).extract_surface(algorithm=None)
                if surface.n_points == 0:
                    continue
                plotter.add_mesh(
                    surface,
                    opacity=iso**2 * opacity_scale,
                    color=color,
                    lighting=False,
                )

        plotter.render()
        rendered = np.asarray(plotter.screenshot(return_img=True))
        if rendered.shape[-1] == 4:
            rendered = rendered[:, :, :3]
        if rendered.shape != (PANEL_HEIGHT, PANEL_WIDTH, 3):
            raise RuntimeError(f"PyVista produced an unexpected image shape: {rendered.shape}")
        return rendered.astype(np.uint8, copy=False)
    finally:
        plotter.close()


def projected_object_means(
    obj: dict[str, Any],
    pose: np.ndarray,
    intrinsic: CameraIntrinsic,
) -> tuple[np.ndarray, np.ndarray]:
    classes, _ = object_classes(obj)
    means = as_numpy(obj["means"])
    covariances = as_numpy(obj["covs"])
    camera_rotation_inverse = np.linalg.inv(pose[:3, :3])
    camera_translation = pose[:3, 3]
    projected = np.full((len(classes), 2), -99999.0)

    for index, class_id_value in enumerate(classes):
        class_id = int(class_id_value)
        if class_id not in valid_class:
            continue
        mean = means[index]
        covariance = covariances[index]
        eigenvalues = np.linalg.eigvalsh(covariance)
        radii = np.sqrt(eigenvalues)
        half_extent = float((radii * np.sqrt(2 * np.log(2))).max())
        if np.all(np.abs(camera_translation - mean) < half_extent):
            continue

        mean_camera = camera_rotation_inverse @ (mean - camera_translation)
        if mean_camera[2] <= 0:
            continue
        normalized = mean_camera[:2] / mean_camera[2]
        mean_2d = np.asarray(
            [
                normalized[0] * intrinsic.fx + intrinsic.cx,
                normalized[1] * intrinsic.fy + intrinsic.cy,
            ]
        )
        if 0 <= mean_2d[0] < intrinsic.width and 0 <= mean_2d[1] < intrinsic.height:
            projected[index] = mean_2d
    return classes, projected


def ranked_3d_relations(relations: Any, top_k: int = 10) -> list[tuple[int, int, int]]:
    """Preserve the original dense-relation top-candidate selection."""
    if isinstance(relations, dict):
        predicates = as_numpy(relations["predicates"])
        maximum = as_numpy(relations["scores"])
        if predicates.shape != maximum.shape or predicates.ndim != 2:
            raise ValueError("Compact relation predicates/scores must be matching square matrices")
    else:
        dense = as_numpy(relations)
        predicates = dense.argmax(-1)
        count = dense.shape[0]
        maximum = dense[
            np.arange(count)[:, None],
            np.arange(count)[None, :],
            predicates,
        ]
    count = maximum.shape[0]
    candidate_count = min(top_k * 5, count * count - 1)
    if count == 0 or candidate_count <= 0:
        return []
    flattened = maximum.ravel()
    selected = np.argpartition(-flattened, candidate_count)[:candidate_count]
    selected = selected[flattened[selected] > 0]
    selected = selected[np.argsort(-flattened[selected])]
    subjects, objects = np.unravel_index(selected, maximum.shape)
    grouped: dict[tuple[int, int], int] = {}
    for subject, object_ in zip(subjects, objects):
        grouped[(int(subject), int(object_))] = int(predicates[subject, object_])
    return [(subject, object_, predicate) for (subject, object_), predicate in grouped.items()]


def render_3d_text_panel(
    panel: np.ndarray,
    obj: dict[str, Any],
    relations: np.ndarray,
    pose: np.ndarray,
    intrinsic: CameraIntrinsic,
    top_k: int = 10,
) -> np.ndarray:
    classes, projected = projected_object_means(obj, pose, intrinsic)
    figure = plt.figure()
    axis = figure.add_subplot(111)
    axis.imshow(Image.fromarray(panel))

    shown = 0
    for subject, object_, predicate in ranked_3d_relations(relations, top_k):
        if (
            shown >= top_k
            or predicate not in valid_rel_class
            or subject == object_
            or int(classes[subject]) not in valid_class
            or int(classes[object_]) not in valid_class
            or (projected[subject] == -99999).any()
            or (projected[object_] == -99999).any()
        ):
            continue
        shown += 1
        curvature = curvature_sequence(1)[0]
        color = rel_colors[valid_rel_class.index(predicate)]
        sx, sy = projected[subject]
        ox, oy = projected[object_]
        axis.add_patch(
            FancyArrowPatch(
                (sx, sy),
                (ox, oy),
                connectionstyle=f"arc3,rad={curvature}",
                arrowstyle="-|>",
                mutation_scale=6,
                linewidth=0.5,
                color=color,
                shrinkA=7.5,
                shrinkB=7.5,
            )
        )
        lx, ly = arc_midpoint((sx, sy), (ox, oy), curvature)
        if not (20 < lx < intrinsic.width - 20 and 20 < ly < intrinsic.height - 20):
            continue
        axis.text(
            lx,
            ly,
            REL_CLASSES[predicate],
            ha="center",
            va="center",
            fontsize=2.5,
            color=_text_color(color),
            fontfamily="sans-serif",
            bbox=dict(
                boxstyle="round,pad=0.25,rounding_size=0.2",
                facecolor=color,
                edgecolor=color,
            ),
        )

    for index, class_id_value in enumerate(classes):
        mean_2d = projected[index]
        if (mean_2d == -99999).any():
            continue
        if not (40 < mean_2d[0] < intrinsic.width - 40 and 20 < mean_2d[1] < intrinsic.height - 20):
            continue
        class_id = int(class_id_value)
        color = obj_colors[valid_class.index(class_id)]
        axis.text(
            mean_2d[0],
            mean_2d[1],
            CLASSES[class_id],
            color=_text_color(color),
            fontsize=3,
            fontfamily="sans-serif",
            ha="center",
            va="center",
            bbox=dict(
                boxstyle="round,pad=0.25,rounding_size=0.2",
                facecolor=color,
                edgecolor=color,
                alpha=1,
            ),
        )

    axis.axis("off")
    figure.set_size_inches(intrinsic.width / 300, intrinsic.height / 300)
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    return _figure_to_rgb(figure, (intrinsic.width, intrinsic.height))


def save_rgb(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image.astype(np.uint8, copy=False), mode="RGB").save(path)
