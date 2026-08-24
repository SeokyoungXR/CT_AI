from __future__ import annotations

import gzip
import hashlib
import json
import pickle
import tempfile
import unittest
from pathlib import Path

import numpy as np

from scripts.trace_io import (
    TraceBundle,
    ensure_trace_matches_scene,
    find_scene_inputs,
    load_trace,
    select_frame_indices,
)


def write_pickle(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wb") as stream:
        pickle.dump(value, stream, protocol=4)


def tiny_trace(directory: Path, frames: int = 3) -> None:
    objects = []
    relations = []
    objects_2d = []
    relations_2d = []
    for index in range(frames):
        count = index + 1
        objects.append(
            {
                "classes": np.zeros((count, 150)),
                "means": np.zeros((count, 3)),
                "covs": np.repeat(np.eye(3)[None], count, axis=0),
            }
        )
        relations.append(np.zeros((count, count, 50), dtype=np.int64))
        objects_2d.append(
            {
                "classes": np.zeros((0,), dtype=np.int64),
                "bboxes": np.zeros((0, 4), dtype=np.float32),
                "scores": np.zeros((300,), dtype=np.float32),
            }
        )
        relations_2d.append(
            {
                "rels": np.zeros((0, 2), dtype=np.int64),
                "rel_classes": np.zeros((0,), dtype=np.int64),
            }
        )
    for name, value in {
        "obj": objects,
        "rel": relations,
        "obj_2d": objects_2d,
        "rel_2d": relations_2d,
    }.items():
        write_pickle(directory / f"{name}.pkl.gz", value)


class TraceIOTests(unittest.TestCase):
    def test_selected_indices_retain_original_trace_positions(self) -> None:
        self.assertEqual(select_frame_indices(10, 0, 3, 0), [0, 3, 6, 9])
        self.assertEqual(select_frame_indices(10, 1, 3, 2), [1, 4])

    def test_numpy_only_trace_loads_without_torch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            tiny_trace(root)
            trace = load_trace(root, verify_checksums=False)
        self.assertEqual(trace.frame_count, 3)
        self.assertIsInstance(trace.obj_2d[0]["bboxes"], np.ndarray)

    def test_manifest_scene_mismatch_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            tiny_trace(root)
            (root / "manifest.json").write_text(
                json.dumps({"scene": "office_1"}), encoding="utf-8"
            )
            with self.assertRaises(ValueError):
                load_trace(root, verify_checksums=False, expected_scene="room_0")

    def test_scene_and_sequence_paths_are_both_accepted(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            scene = Path(directory) / "office_1"
            sequence = scene / "sequence"
            sequence.mkdir(parents=True)
            (scene / "mesh.ply").touch()
            (sequence / "_info.txt").touch()
            (sequence / "frame-000000.color.jpg").touch()
            np.savetxt(sequence / "frame-000000.pose.txt", np.eye(4))
            from_scene = find_scene_inputs(scene)
            from_sequence = find_scene_inputs(sequence)
        self.assertEqual(from_scene, from_sequence)
        self.assertEqual(from_scene.frame_count, 1)

    def test_trace_scene_count_mismatch_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            scene = Path(directory) / "office_1"
            sequence = scene / "sequence"
            sequence.mkdir(parents=True)
            (scene / "mesh.ply").touch()
            (sequence / "_info.txt").touch()
            (sequence / "frame-000000.color.jpg").touch()
            np.savetxt(sequence / "frame-000000.pose.txt", np.eye(4))
            inputs = find_scene_inputs(scene)
            bundle = TraceBundle([{}] * 2, [np.empty(0)] * 2, [{}] * 2, [{}] * 2)
            with self.assertRaises(ValueError):
                ensure_trace_matches_scene(bundle, inputs)

    def test_noncontiguous_replica_frame_names_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            scene = Path(directory) / "office_1"
            sequence = scene / "sequence"
            sequence.mkdir(parents=True)
            (scene / "mesh.ply").touch()
            (sequence / "_info.txt").touch()
            (sequence / "frame-000001.color.jpg").touch()
            np.savetxt(sequence / "frame-000001.pose.txt", np.eye(4))
            inputs = find_scene_inputs(scene)
            bundle = TraceBundle([{}], [np.empty(0)], [{}], [{}])
            with self.assertRaises(ValueError):
                ensure_trace_matches_scene(bundle, inputs)

    def test_packaged_trace_checksum_and_portability(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            tiny_trace(root)
            files = {}
            for path in root.glob("*.pkl.gz"):
                files[path.name] = {
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest()
                }
            (root / "manifest.json").write_text(
                json.dumps({"scene": "office_1", "files": files}), encoding="utf-8"
            )
            trace = load_trace(root, expected_scene="office_1")

        self.assertEqual(trace.frame_count, 3)
        self.assertTrue(
            all(
                isinstance(frame["bboxes"], np.ndarray)
                and isinstance(frame["scores"], np.ndarray)
                for frame in trace.obj_2d
            )
        )


if __name__ == "__main__":
    unittest.main()
