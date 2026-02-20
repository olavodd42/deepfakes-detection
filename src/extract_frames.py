"""Frame extraction utilities for deepfake detection CNN training.

Provides efficient batch extraction of frames from video datasets using
hardware-accelerated decoding (Decord). Supports uniform temporal sampling,
optional resizing, and parallel processing of large video collections.
"""

import argparse
import cv2
import logging
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

import numpy as np
from decord import VideoReader
from decord import cpu, gpu

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mkv", ".mov", ".wmv", ".flv", ".webm"}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# Module-level cache so the GPU probe runs at most once.
_decord_ctx_cache: Optional[object] = None


def _get_decord_ctx(probe_video: Optional[str] = None):
    """Return a Decord context, preferring GPU with automatic CPU fallback.

    The GPU context is validated **once** by actually opening a video file
    with it.  Subsequent calls reuse the cached result.  If no
    ``probe_video`` is supplied on the first call the function defaults to
    CPU (safe path).

    Args:
        probe_video: Path to a small video used to test GPU decoding.
            Only needed on the first invocation.

    Returns:
        A ``gpu(0)`` or ``cpu(0)`` Decord context.
    """
    global _decord_ctx_cache
    if _decord_ctx_cache is not None:
        return _decord_ctx_cache

    if probe_video is not None:
        try:
            ctx = gpu(0)
            # Actually try to open a video — this is where CUDA errors surface
            _vr = VideoReader(probe_video, ctx=ctx)
            del _vr
            logger.info("Decord GPU decoding enabled.")
            _decord_ctx_cache = ctx
            return _decord_ctx_cache
        except Exception:
            logger.info("GPU decoding unavailable, falling back to CPU.")

    _decord_ctx_cache = cpu(0)
    return _decord_ctx_cache


def extract_frames(
    video_path: str | os.PathLike,
    frames_dir: str | os.PathLike,
    *,
    overwrite: bool = False,
    start: int = 0,
    end: int = -1,
    every: int = 1,
    num_frames: Optional[int] = None,
    resize: Optional[tuple[int, int]] = None,
    jpeg_quality: int = 95,
) -> int:
    """Extract frames from a single video and save them as JPEG images.

    Supports two sampling strategies:
      - **Stride-based** (``every``): extract every *n*-th frame in the
        ``[start, end)`` range.
      - **Uniform sampling** (``num_frames``): pick *n* frames uniformly
        spaced across the video duration. This is the recommended mode for
        CNN training, as it yields a representative temporal coverage
        regardless of the video length.

    When ``num_frames`` is supplied it overrides ``every``.

    Frames are saved under ``<frames_dir>/<video_stem>/<index>.jpg``, where
    ``<video_stem>`` is the filename without extension.

    Args:
        video_path: Path to the source video file.
        frames_dir: Root directory where extracted frames will be stored.
        overwrite: If ``True``, re-extract and overwrite existing frames.
        start: First frame index (inclusive, 0-based). Defaults to ``0``.
        end: Last frame index (exclusive). ``-1`` means up to the last frame.
        every: Extract one frame every *n* frames. Ignored when
            ``num_frames`` is set.
        num_frames: If given, uniformly sample this many frames from the
            video. Takes precedence over ``every``.
        resize: Optional ``(width, height)`` to resize frames before saving.
            Useful for normalising input size for CNN training.
        jpeg_quality: JPEG compression quality (0-100). Higher values
            preserve more detail at the cost of disk space. Defaults to 95.

    Returns:
        Number of frames actually written to disk.

    Raises:
        FileNotFoundError: If ``video_path`` does not exist.
        RuntimeError: If the video cannot be decoded.
    """
    video_path = Path(video_path).resolve()
    frames_dir = Path(frames_dir).resolve()

    if not video_path.is_file():
        raise FileNotFoundError(f"Video not found: {video_path}")

    try:
        ctx = _get_decord_ctx(probe_video=str(video_path))
        vr = VideoReader(str(video_path), ctx=ctx)
    except Exception:
        # If the cached context was GPU and this specific file fails,
        # retry once with CPU before giving up.
        try:
            vr = VideoReader(str(video_path), ctx=cpu(0))
        except Exception as exc:
            raise RuntimeError(
                f"Failed to open video {video_path}: {exc}"
            ) from exc

    total_frames = len(vr)
    if total_frames == 0:
        logger.warning("Video has 0 frames: %s", video_path)
        return 0

    # Resolve frame range
    start = max(0, start)
    end = min(total_frames, end) if end > 0 else total_frames

    # Build the list of frame indices to extract
    if num_frames is not None and num_frames > 0:
        # Uniform temporal sampling — best for CNN training diversity
        indices = np.linspace(start, end - 1, num=num_frames, dtype=int).tolist()
    else:
        indices = list(range(start, end, every))

    if not indices:
        return 0

    # Output directory: <frames_dir>/<video_stem>/
    video_stem = video_path.stem  # "001" from "001.mp4"
    out_dir = frames_dir / video_stem
    out_dir.mkdir(parents=True, exist_ok=True)

    encode_params = [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality]
    saved_count = 0

    # Batch decode when the number of frames is manageable (faster I/O)
    BATCH_THRESHOLD = 2048
    if len(indices) <= BATCH_THRESHOLD:
        frames = vr.get_batch(indices).asnumpy()  # (N, H, W, 3) RGB
        for idx, frame in zip(indices, frames):
            save_path = out_dir / f"{idx:010d}.jpg"
            if save_path.exists() and not overwrite:
                continue
            bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            if resize is not None:
                bgr = cv2.resize(bgr, resize, interpolation=cv2.INTER_AREA)
            cv2.imwrite(str(save_path), bgr, encode_params)
            saved_count += 1
    else:
        # Sequential access for very long videos to avoid OOM
        for idx in indices:
            save_path = out_dir / f"{idx:010d}.jpg"
            if save_path.exists() and not overwrite:
                continue
            frame = vr[idx].asnumpy()
            bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            if resize is not None:
                bgr = cv2.resize(bgr, resize, interpolation=cv2.INTER_AREA)
            cv2.imwrite(str(save_path), bgr, encode_params)
            saved_count += 1

    return saved_count


def video_to_frames(
    video_path: str | os.PathLike,
    frames_dir: str | os.PathLike,
    *,
    overwrite: bool = False,
    every: int = 1,
    num_frames: Optional[int] = None,
    resize: Optional[tuple[int, int]] = None,
    jpeg_quality: int = 95,
) -> Path:
    """High-level wrapper: extract frames from a single video with logging.

    Creates the output directory automatically and delegates to
    :func:`extract_frames`.

    Args:
        video_path: Path to the video file.
        frames_dir: Root output directory for extracted frames.
        overwrite: Re-extract frames even if they already exist.
        every: Frame stride (ignored when ``num_frames`` is set).
        num_frames: Number of uniformly sampled frames to extract.
        resize: Optional ``(width, height)`` for resizing.
        jpeg_quality: JPEG quality (0-100).

    Returns:
        Path to the directory containing the extracted frames.
    """
    video_path = Path(video_path).resolve()
    frames_dir = Path(frames_dir).resolve()
    video_stem = video_path.stem

    logger.info("Extracting frames from %s", video_path.name)
    saved = extract_frames(
        video_path,
        frames_dir,
        overwrite=overwrite,
        every=every,
        num_frames=num_frames,
        resize=resize,
        jpeg_quality=jpeg_quality,
    )
    logger.info("  -> saved %d frames to %s/", saved, video_stem)
    return frames_dir / video_stem


def batch_extract(
    video_dir: str | os.PathLike,
    frames_dir: str | os.PathLike,
    *,
    overwrite: bool = False,
    every: int = 1,
    num_frames: Optional[int] = None,
    resize: Optional[tuple[int, int]] = None,
    jpeg_quality: int = 95,
    max_workers: int = 1,
    limit: Optional[int] = None,
) -> dict[str, int]:
    """Extract frames from every video found in a directory tree.

    Walks ``video_dir`` recursively, processes each video file, and stores
    output under ``frames_dir`` preserving the relative sub-directory
    structure.  For example, a video at
    ``data/train_closed_set/real/001.mp4`` with ``video_dir=data`` and
    ``frames_dir=frames`` will produce
    ``frames/train_closed_set/real/001/<frame>.jpg``.

    Processing can be parallelised across multiple worker *processes*
    (``max_workers > 1``).  Each worker decodes on the Decord CPU backend
    to avoid GPU context conflicts across processes.

    Args:
        video_dir: Root directory to search for video files.
        frames_dir: Root output directory for all extracted frames.
        overwrite: Re-extract frames even if they already exist.
        every: Frame stride (ignored when ``num_frames`` is set).
        num_frames: Number of uniformly sampled frames per video.
        resize: Optional ``(width, height)`` for resizing.
        jpeg_quality: JPEG quality (0-100).
        max_workers: Number of parallel worker processes. Use ``1`` for
            sequential (and GPU-decoded) processing.
        limit: If set, only process the first *limit* videos found
            (useful for quick sanity checks).

    Returns:
        A dict mapping each video path (str) to the number of frames saved.
    """
    video_dir = Path(video_dir).resolve()
    frames_dir = Path(frames_dir).resolve()

    # Collect all video files
    video_paths = sorted(
        p for p in video_dir.rglob("*") if p.suffix.lower() in VIDEO_EXTENSIONS
    )
    if limit is not None:
        video_paths = video_paths[:limit]

    logger.info(
        "Found %d videos in %s (limit=%s)", len(video_paths), video_dir, limit
    )

    results: dict[str, int] = {}

    if max_workers <= 1:
        # Sequential — can use GPU decoding
        for i, vp in enumerate(video_paths, 1):
            rel = vp.relative_to(video_dir)
            out = frames_dir / rel.parent
            logger.info("[%d/%d] %s", i, len(video_paths), rel)
            try:
                count = extract_frames(
                    vp,
                    out,
                    overwrite=overwrite,
                    every=every,
                    num_frames=num_frames,
                    resize=resize,
                    jpeg_quality=jpeg_quality,
                )
                results[str(vp)] = count
            except Exception:
                logger.exception("Failed to process %s", vp)
                results[str(vp)] = 0
    else:
        # Parallel processing (CPU decoding per worker)
        with ProcessPoolExecutor(max_workers=max_workers) as pool:
            future_to_path = {}
            for vp in video_paths:
                rel = vp.relative_to(video_dir)
                out = frames_dir / rel.parent
                out.mkdir(parents=True, exist_ok=True)
                fut = pool.submit(
                    extract_frames,
                    vp,
                    out,
                    overwrite=overwrite,
                    every=every,
                    num_frames=num_frames,
                    resize=resize,
                    jpeg_quality=jpeg_quality,
                )
                future_to_path[fut] = vp

            done = 0
            for fut in as_completed(future_to_path):
                vp = future_to_path[fut]
                done += 1
                try:
                    count = fut.result()
                    results[str(vp)] = count
                    logger.info(
                        "[%d/%d] %s -> %d frames",
                        done,
                        len(video_paths),
                        vp.name,
                        count,
                    )
                except Exception:
                    logger.exception("Failed to process %s", vp)
                    results[str(vp)] = 0

    total_saved = sum(results.values())
    logger.info(
        "Done. Processed %d videos, saved %d total frames.",
        len(results),
        total_saved,
    )
    return results


def _parse_resize(value: str) -> tuple[int, int]:
    """Parse a 'WxH' string into a (width, height) tuple."""
    parts = value.lower().split("x")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError(
            f"resize must be in WxH format (e.g. 224x224), got '{value}'"
        )
    return int(parts[0]), int(parts[1])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract frames from videos for CNN training.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "input",
        help="Path to a single video file or a directory of videos.",
    )
    parser.add_argument(
        "output",
        help="Directory where extracted frames will be stored.",
    )
    parser.add_argument(
        "--every",
        type=int,
        default=1,
        help="Extract one frame every N frames (stride-based sampling).",
    )
    parser.add_argument(
        "--num-frames",
        type=int,
        default=None,
        help="Uniformly sample this many frames per video (overrides --every).",
    )
    parser.add_argument(
        "--resize",
        type=_parse_resize,
        default=None,
        metavar="WxH",
        help="Resize frames to WxH (e.g. 224x224) before saving.",
    )
    parser.add_argument(
        "--jpeg-quality",
        type=int,
        default=95,
        help="JPEG compression quality (0-100).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing frames.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of parallel workers (use >1 for CPU-decoded parallelism).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only process the first N videos (for quick testing).",
    )

    args = parser.parse_args()
    input_path = Path(args.input)

    if input_path.is_file():
        video_to_frames(
            input_path,
            args.output,
            overwrite=args.overwrite,
            every=args.every,
            num_frames=args.num_frames,
            resize=args.resize,
            jpeg_quality=args.jpeg_quality,
        )
    elif input_path.is_dir():
        batch_extract(
            input_path,
            args.output,
            overwrite=args.overwrite,
            every=args.every,
            num_frames=args.num_frames,
            resize=args.resize,
            jpeg_quality=args.jpeg_quality,
            max_workers=args.workers,
            limit=args.limit,
        )
    else:
        logger.error("Input path does not exist: %s", input_path)