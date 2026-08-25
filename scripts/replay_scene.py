#!/usr/bin/env python3
"""Replay a saved ReplicaSSG temporal trace with the original renderer."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.original_visualization import (
    PANEL_HEIGHT,
    PANEL_WIDTH,
    render_2d_panel,
    render_3d_panel,
    render_3d_text_panel,
    save_rgb,
)
from scripts.trace_io import (
    clean_path,
    ensure_trace_matches_scene,
    find_scene_inputs,
    load_pose,
    load_trace,
    parse_intrinsics,
    select_frame_indices,
)


SUPPORTED_SCENES = ("office_1", "apartment_1")
WORK_OWNER = "CT_AI/scripts/replay_scene.py"
RENDER_CACHE_VERSION = 2
DEFAULT_FPS = 15
DEFAULT_MAX_FRAMES = 450


def path_argument(value: str) -> Path:
    try:
        return clean_path(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from error


def work_manifest(
    scene_dir: Path,
    trace_dir: Path,
    indices: list[int],
    fps: int,
) -> dict:
    return {
        "created_by": WORK_OWNER,
        "manifest_version": RENDER_CACHE_VERSION,
        "scene_dir": str(scene_dir),
        "trace_dir": str(trace_dir),
        "source_indices": indices,
        "fps": fps,
        "panel_size": [PANEL_WIDTH, PANEL_HEIGHT],
    }


def _load_work_manifest(path: Path) -> dict | None:
    if not path.is_file() or path.is_symlink():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if value.get("created_by") == WORK_OWNER else None


def _safe_remove_work_dir(work_dir: Path) -> None:
    manifest = _load_work_manifest(work_dir / "manifest.json")
    if manifest is None:
        raise RuntimeError(
            f"Refusing to remove unowned frame directory: {work_dir}. Remove it manually if appropriate."
        )
    shutil.rmtree(work_dir)


def _prefix_compatible_manifests(existing: dict, desired: dict) -> bool:
    """Allow a render to extend or shorten the same source-index prefix."""
    fixed_keys = (
        "created_by",
        "manifest_version",
        "scene_dir",
        "trace_dir",
        "panel_size",
    )
    if any(existing.get(key) != desired.get(key) for key in fixed_keys):
        return False
    existing_indices = existing.get("source_indices")
    desired_indices = desired.get("source_indices")
    if not isinstance(existing_indices, list) or not isinstance(desired_indices, list):
        return False
    common_length = min(len(existing_indices), len(desired_indices))
    return existing_indices[:common_length] == desired_indices[:common_length]


def prepare_work_dir(work_dir: Path, desired: dict, restart: bool) -> None:
    manifest_path = work_dir / "manifest.json"
    if work_dir.exists():
        existing = _load_work_manifest(manifest_path)
        if restart:
            _safe_remove_work_dir(work_dir)
        elif existing != desired and not _prefix_compatible_manifests(existing, desired):
            raise RuntimeError(
                f"{work_dir} belongs to a different replay configuration. "
                "Use --restart to replace only this generated frame cache."
            )
    work_dir.mkdir(parents=True, exist_ok=True)
    (work_dir / "2D").mkdir(exist_ok=True)
    (work_dir / "3D_text").mkdir(exist_ok=True)
    manifest_path.write_text(json.dumps(desired, indent=2), encoding="utf-8")


def ffmpeg_command(
    ffmpeg: str,
    fps: int,
    work_dir: Path,
    output: Path,
    frame_count: int,
) -> list[str]:
    return [
        ffmpeg,
        "-y",
        "-framerate",
        str(fps),
        "-i",
        str(work_dir / "2D" / "frame-%06d.color.png"),
        "-framerate",
        str(fps),
        "-i",
        str(work_dir / "3D_text" / "frame-%06d.color.png"),
        "-filter_complex",
        "[1:v][0:v]scale2ref=iw:iw*ih/iw[rb][ra];[ra][rb]vstack=inputs=2[v]",
        "-map",
        "[v]",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-crf",
        "24",
        "-preset",
        "veryfast",
        "-frames:v",
        str(frame_count),
        "-shortest",
        str(output),
    ]


def load_scene_mesh(path: Path):
    if sys.platform.startswith("linux") and not os.environ.get("DISPLAY"):
        os.environ.setdefault("VTK_DEFAULT_OPENGL_WINDOW", "vtkEGLRenderWindow")
    try:
        import pyvista as pv
    except ModuleNotFoundError as error:  # pragma: no cover - integration dependency
        raise RuntimeError(
            "PyVista/VTK is required. Run: conda env update -f environment.yml"
        ) from error
    return pv.read(path)


def combined_frame(top: np.ndarray, bottom: np.ndarray) -> np.ndarray:
    if top.shape != (PANEL_HEIGHT, PANEL_WIDTH, 3):
        raise ValueError(f"Unexpected 2D panel shape: {top.shape}")
    if bottom.shape != (PANEL_HEIGHT, PANEL_WIDTH, 3):
        raise ValueError(f"Unexpected 3D panel shape: {bottom.shape}")
    return np.concatenate((top, bottom), axis=0)


def valid_cached_panel(path: Path) -> bool:
    if not path.is_file() or path.is_symlink():
        return False
    try:
        with Image.open(path) as image:
            if image.size != (PANEL_WIDTH, PANEL_HEIGHT) or image.mode not in ("RGB", "RGBA"):
                return False
            image.verify()
    except (OSError, SyntaxError):
        return False
    return True


def resolve_frame_settings(args: argparse.Namespace) -> tuple[int, int]:
    """Return effective (stride, max frames), using zero only for all frames."""
    if args.preview_only:
        if args.max_frames is not None or args.all_frames:
            raise ValueError(
                "--preview-only cannot be combined with --max-frames or --all-frames"
            )
        return args.frame_stride, 1
    if args.all_frames:
        return args.frame_stride, 0
    max_frames = DEFAULT_MAX_FRAMES if args.max_frames is None else args.max_frames
    if max_frames < 1:
        raise ValueError("--max-frames must be at least 1; use --all-frames for no limit")
    return args.frame_stride, max_frames


def render_one(
    source_index: int,
    scene,
    trace,
    intrinsic,
    scene_mesh,
) -> tuple[np.ndarray, np.ndarray]:
    pose = load_pose(scene.pose_paths[source_index])
    top = render_2d_panel(
        scene.image_paths[source_index],
        trace.obj_2d[source_index],
        trace.rel_2d[source_index],
    )
    gaussian_panel = render_3d_panel(scene_mesh, pose, trace.obj[source_index])
    bottom = render_3d_text_panel(
        gaussian_panel,
        trace.obj[source_index],
        trace.rel[source_index],
        pose,
        intrinsic,
    )
    return top, bottom


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Replay an office_1 or apartment_1 temporal PKL trace using the original "
            "DeWorldSG 2D/3D visualization style."
        )
    )
    source = parser.add_mutually_exclusive_group()
    source.add_argument(
        "--scene",
        choices=SUPPORTED_SCENES,
        help="Scene unpacked under data/<scene> (default: office_1)",
    )
    source.add_argument(
        "--scene-dir",
        type=path_argument,
        help="Advanced: explicit ReplicaSSG scene or sequence directory",
    )
    parser.add_argument(
        "--trace-dir",
        type=path_argument,
        help="Advanced: explicit directory containing four .pkl/.pkl.gz files",
    )
    parser.add_argument(
        "--output",
        type=path_argument,
        help="Output MP4 path (defaults to outputs/<scene>.mp4)",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=DEFAULT_FPS,
        help=f"Output frames per second (default: {DEFAULT_FPS})",
    )
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--frame-stride", type=int, default=1)
    frame_limit = parser.add_mutually_exclusive_group()
    frame_limit.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help=f"Maximum output frames (default: {DEFAULT_MAX_FRAMES})",
    )
    frame_limit.add_argument(
        "--all-frames",
        action="store_true",
        help=(
            f"Remove the {DEFAULT_MAX_FRAMES}-frame cap; "
            "--start-frame/--frame-stride still apply"
        ),
    )
    parser.add_argument(
        "--preview-only",
        action="store_true",
        help="Render one 960x1080 PNG instead of a video",
    )
    parser.add_argument(
        "--keep-frames",
        action="store_true",
        help="Keep the generated 2D and 3D_text PNG cache after encoding",
    )
    parser.add_argument(
        "--restart",
        action="store_true",
        help="Discard this script's matching frame cache and start again",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.fps < 1:
        raise ValueError("--fps must be at least 1")
    args.frame_stride, args.max_frames = resolve_frame_settings(args)

    requested_scene = args.scene or "office_1"
    scene_path = (
        args.scene_dir
        if args.scene_dir is not None
        else PROJECT_ROOT / "data" / requested_scene
    )
    if args.scene_dir is None and not scene_path.is_dir():
        raise FileNotFoundError(
            f"Tutorial data is missing: {scene_path}\n"
            "Download the Google Drive asset ZIP and extract it inside the CT_AI folder."
        )
    scene = find_scene_inputs(scene_path)
    scene_name = scene.scene_dir.name
    if scene_name not in SUPPORTED_SCENES:
        raise ValueError(
            f"Unsupported scene '{scene_name}'. Choose one of: {', '.join(SUPPORTED_SCENES)}"
        )
    trace_dir = (
        args.trace_dir.expanduser().resolve()
        if args.trace_dir is not None
        else (PROJECT_ROOT / "assets" / "traces" / scene_name).resolve()
    )
    required_trace_names = ("obj", "rel", "obj_2d", "rel_2d")
    missing_trace = [
        name
        for name in required_trace_names
        if not (trace_dir / f"{name}.pkl.gz").is_file()
        and not (trace_dir / f"{name}.pkl").is_file()
    ]
    if missing_trace and args.trace_dir is None:
        raise FileNotFoundError(
            f"Tutorial trace is missing from {trace_dir}: {', '.join(missing_trace)}\n"
            "Download the Google Drive asset ZIP and extract it inside the CT_AI folder."
        )
    print("Loading and checking the four temporal trace files...")
    trace = load_trace(trace_dir, expected_scene=scene.scene_dir.name)
    ensure_trace_matches_scene(trace, scene)
    intrinsic = parse_intrinsics(scene.sequence_dir / "_info.txt")
    if (intrinsic.width, intrinsic.height) != (PANEL_WIDTH, PANEL_HEIGHT):
        raise ValueError(
            f"The original renderer expects 960x540 RGB, got {intrinsic.width}x{intrinsic.height}"
        )
    indices = select_frame_indices(
        trace.frame_count,
        args.start_frame,
        args.frame_stride,
        args.max_frames,
    )

    output = (
        args.output.expanduser().resolve()
        if args.output is not None
        else (PROJECT_ROOT / "outputs" / f"{scene_name}.mp4").resolve()
    )
    if output.suffix.lower() != ".mp4":
        raise ValueError("--output must end in .mp4")
    output.parent.mkdir(parents=True, exist_ok=True)
    preview_path = output.with_name(f"{output.stem}_preview.png")

    print(f"Scene: {scene.scene_dir}")
    print(f"Trace: {trace_dir}")
    print(f"Source frames: {trace.frame_count}")
    print(f"Selected frames: {len(indices)} (first={indices[0]}, last={indices[-1]})")
    print("Loading vertex-color mesh...")
    scene_mesh = load_scene_mesh(scene.mesh_path)

    if args.preview_only:
        top, bottom = render_one(indices[0], scene, trace, intrinsic, scene_mesh)
        save_rgb(preview_path, combined_frame(top, bottom))
        print(f"Preview complete: {preview_path}")
        return

    work_dir = output.with_name(f".{output.stem}_frames")
    desired_manifest = work_manifest(scene.scene_dir, trace_dir, indices, args.fps)
    prepare_work_dir(work_dir, desired_manifest, args.restart)
    print(f"Frame cache: {work_dir}")
    print("Existing cached frames are reused, so rerunning resumes an interrupted render.")

    for output_index, source_index in enumerate(indices):
        top_path = work_dir / "2D" / f"frame-{output_index:06d}.color.png"
        bottom_path = work_dir / "3D_text" / f"frame-{output_index:06d}.color.png"
        if not valid_cached_panel(top_path):
            top = render_2d_panel(
                scene.image_paths[source_index],
                trace.obj_2d[source_index],
                trace.rel_2d[source_index],
            )
            save_rgb(top_path, top)
        if not valid_cached_panel(bottom_path):
            pose = load_pose(scene.pose_paths[source_index])
            gaussian_panel = render_3d_panel(scene_mesh, pose, trace.obj[source_index])
            bottom = render_3d_text_panel(
                gaussian_panel,
                trace.obj[source_index],
                trace.rel[source_index],
                pose,
                intrinsic,
            )
            save_rgb(bottom_path, bottom)
        if output_index == 0:
            with Image.open(top_path) as top_image, Image.open(bottom_path) as bottom_image:
                preview = np.concatenate(
                    (np.asarray(top_image.convert("RGB")), np.asarray(bottom_image.convert("RGB"))),
                    axis=0,
                )
                save_rgb(preview_path, preview)
        if (output_index + 1) % 25 == 0 or output_index + 1 == len(indices):
            print(f"Rendered {output_index + 1}/{len(indices)} frames")

    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg was not found. Install it through environment.yml.")
    print(f"Encoding the original 960x1080 vertical layout at {args.fps} FPS...")
    process = subprocess.run(
        ffmpeg_command(ffmpeg, args.fps, work_dir, output, len(indices)),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    )
    if process.returncode != 0:
        raise RuntimeError(f"ffmpeg failed; cached frames were kept for retry:\n{process.stderr}")
    print(f"Video complete: {output}")
    print(f"Preview: {preview_path}")

    if not args.keep_frames:
        _safe_remove_work_dir(work_dir)
        print("Temporary rendered panels were removed (use --keep-frames to retain them).")


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, KeyError, TypeError, ValueError, RuntimeError) as error:
        print(f"Error: {error}", file=sys.stderr)
        raise SystemExit(1) from error
