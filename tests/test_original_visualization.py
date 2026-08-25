from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

import matplotlib
import numpy as np
from PIL import Image

from scripts.original_visualization import (
    PANEL_HEIGHT,
    PANEL_WIDTH,
    arc_midpoint,
    curvature_sequence,
    projected_object_means,
    ranked_3d_relations,
    render_2d_panel,
)
from scripts.replay_scene import (
    build_parser,
    combined_frame,
    ffmpeg_command,
    prepare_work_dir,
    resolve_frame_settings,
    valid_cached_panel,
    work_manifest,
)
from scripts.trace_io import CameraIntrinsic, select_frame_indices


class OriginalVisualizationTests(unittest.TestCase):
    def test_student_scene_cli_and_advanced_path_are_mutually_exclusive(self) -> None:
        parser = build_parser()
        self.assertEqual(parser.parse_args(["--scene", "apartment_1"]).scene, "apartment_1")
        self.assertIsNone(parser.parse_args([]).scene)
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            parser.parse_args(["--scene", "office_1", "--scene-dir", "data/office_1"])

    def test_frame_modes_have_safe_limits(self) -> None:
        parser = build_parser()

        self.assertEqual(parser.parse_args([]).fps, 12)
        stride, limit = resolve_frame_settings(parser.parse_args([]))
        self.assertEqual((stride, limit), (1, 360))
        default_indices = select_frame_indices(3598, 0, stride, limit)
        self.assertEqual((len(default_indices), default_indices[-1]), (360, 359))

        stride, limit = resolve_frame_settings(parser.parse_args(["--preview-only"]))
        self.assertEqual(select_frame_indices(3598, 20, stride, limit), [20])

        stride, limit = resolve_frame_settings(parser.parse_args(["--all-frames"]))
        self.assertEqual(len(select_frame_indices(3598, 0, stride, limit)), 3598)

        with self.assertRaises(ValueError):
            resolve_frame_settings(parser.parse_args(["--preview-only", "--all-frames"]))

    def test_prefix_compatible_cache_can_expand_to_all_frames(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            work_dir = Path(directory) / ".scene_frames"
            common = (Path("scene"), Path("trace"))
            first_thousand = work_manifest(*common, list(range(1000)), 15)
            all_frames = work_manifest(*common, list(range(3598)), 15)
            prepare_work_dir(work_dir, first_thousand, restart=False)
            prepare_work_dir(work_dir, all_frames, restart=False)
            stored = json.loads((work_dir / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(len(stored["source_indices"]), 3598)

    def test_stale_renderer_cache_requires_explicit_restart(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            work_dir = Path(directory) / ".scene_frames"
            desired = work_manifest(Path("scene"), Path("trace"), list(range(10)), 15)
            stale = {**desired, "manifest_version": desired["manifest_version"] - 1}
            prepare_work_dir(work_dir, stale, restart=False)
            with self.assertRaises(RuntimeError):
                prepare_work_dir(work_dir, desired, restart=False)
            prepare_work_dir(work_dir, desired, restart=True)
            stored = json.loads((work_dir / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(stored, desired)

    def test_fps_change_reuses_rendered_panel_cache(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            work_dir = Path(directory) / ".scene_frames"
            at_fifteen_fps = work_manifest(
                Path("scene"), Path("trace"), list(range(10)), 15
            )
            at_five_fps = {**at_fifteen_fps, "fps": 5}
            prepare_work_dir(work_dir, at_fifteen_fps, restart=False)
            prepare_work_dir(work_dir, at_five_fps, restart=False)
            stored = json.loads((work_dir / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(stored["fps"], 5)

    def test_curvature_and_midpoint_match_original(self) -> None:
        np.testing.assert_allclose(curvature_sequence(4), [-0.24, 0.24, -0.48, 0.48])
        np.testing.assert_allclose(arc_midpoint((0, 0), (10, 0), -0.24), [5, -1.2])

    def test_projection_uses_replica_camera_convention(self) -> None:
        classes = np.zeros((1, 150))
        classes[0, 8] = 1
        obj = {
            "classes": classes,
            "means": np.asarray([[0.0, 0.0, 2.0]]),
            "covs": np.asarray([np.eye(3) * 0.04]),
        }
        intrinsic = CameraIntrinsic(960, 540, 480, 480, 480, 270)
        labels, projected = projected_object_means(obj, np.eye(4), intrinsic)
        self.assertEqual(labels.tolist(), [8])
        np.testing.assert_allclose(projected[0], [480, 270])

    def test_dense_relation_ranking_preserves_subject_object(self) -> None:
        relations = np.zeros((2, 2, 50), dtype=np.int64)
        relations[0, 1, 30] = 7
        relations[1, 0, 28] = 5
        ranked = ranked_3d_relations(relations, top_k=10)
        self.assertEqual(ranked[:2], [(0, 1, 30), (1, 0, 28)])

    def test_compact_relations_rank_exactly_like_dense_relations(self) -> None:
        rng = np.random.default_rng(7)
        dense = rng.integers(0, 100, size=(8, 8, 50), dtype=np.int64)
        predicates = dense.argmax(-1).astype(np.uint8)
        compact = {
            "predicates": predicates,
            "scores": np.take_along_axis(dense, predicates[..., None], axis=-1)[..., 0].astype(
                np.uint32
            ),
        }
        self.assertEqual(
            ranked_3d_relations(compact, top_k=10),
            ranked_3d_relations(dense, top_k=10),
        )

    def test_sparse_unsigned_relation_scores_do_not_rank_zeros_first(self) -> None:
        dense = np.zeros((12, 12, 50), dtype=np.int64)
        for subject, object_, predicate, score in (
            (0, 1, 30, 11),
            (2, 3, 28, 9),
            (4, 5, 40, 7),
            (6, 7, 13, 5),
            (8, 9, 2, 3),
            (10, 11, 1, 1),
        ):
            dense[subject, object_, predicate] = score
        predicates = dense.argmax(-1).astype(np.uint8)
        compact = {
            "predicates": predicates,
            "scores": np.take_along_axis(dense, predicates[..., None], axis=-1)[
                ..., 0
            ].astype(np.uint32),
        }
        expected = ranked_3d_relations(dense, top_k=10)
        self.assertEqual(ranked_3d_relations(compact, top_k=10), expected)
        self.assertEqual(len(expected), 6)

    def test_2d_panel_has_original_dimensions(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            image_path = Path(directory) / "frame-000000.color.jpg"
            Image.new("RGB", (960, 540), "white").save(image_path)
            obj = {
                "classes": np.asarray([8]),
                "bboxes": np.asarray([[480, 270, 200, 100]], dtype=np.float32),
                "scores": np.asarray([0.9] + [0.0] * 299, dtype=np.float32),
            }
            rel = {
                "rels": np.empty((0, 2), dtype=np.int64),
                "rel_classes": np.empty((0,), dtype=np.int64),
            }
            panel = render_2d_panel(image_path, obj, rel)
        self.assertEqual(panel.shape, (PANEL_HEIGHT, PANEL_WIDTH, 3))

    def test_edge_label_cannot_expand_matplotlib_canvas(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            image_path = Path(directory) / "frame-000000.color.jpg"
            Image.new("RGB", (PANEL_WIDTH, PANEL_HEIGHT), "white").save(image_path)
            obj = {
                "classes": np.asarray([8]),
                # The label starts beyond the right edge, which used to make
                # bbox_inches="tight" produce a panel wider than 960 px.
                "bboxes": np.asarray([[959, 270, 2, 2]], dtype=np.float32),
                "scores": np.asarray([0.9] + [0.0] * 299, dtype=np.float32),
            }
            rel = {
                "rels": np.empty((0, 2), dtype=np.int64),
                "rel_classes": np.empty((0,), dtype=np.int64),
            }
            # Explicit figure bounds must also override a user's global
            # Matplotlib preference for tightly cropped saved figures.
            with matplotlib.rc_context({"savefig.bbox": "tight"}):
                panel = render_2d_panel(image_path, obj, rel)
        self.assertEqual(panel.shape, (PANEL_HEIGHT, PANEL_WIDTH, 3))

    def test_combined_frame_and_ffmpeg_layout(self) -> None:
        top = np.zeros((PANEL_HEIGHT, PANEL_WIDTH, 3), dtype=np.uint8)
        bottom = np.zeros_like(top)
        self.assertEqual(combined_frame(top, bottom).shape, (1080, 960, 3))
        command = ffmpeg_command(
            "ffmpeg", 15, Path("frames"), Path("result.mp4"), frame_count=1000
        )
        self.assertIn("vstack=inputs=2", " ".join(command))
        self.assertIn("yuv420p", command)
        self.assertIn("15", command)
        self.assertEqual(command[command.index("-frames:v") + 1], "1000")

    def test_corrupt_cached_panel_is_not_reused(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "frame.png"
            path.write_bytes(b"not a png")
            self.assertFalse(valid_cached_panel(path))
            Image.new("RGB", (970, PANEL_HEIGHT)).save(path)
            self.assertFalse(valid_cached_panel(path))
            Image.new("RGB", (960, 540)).save(path)
            self.assertTrue(valid_cached_panel(path))


if __name__ == "__main__":
    unittest.main()
