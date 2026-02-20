import shutil
import os
from pathlib import Path

EXTENSIONS = {".jpg", ".jpeg", ".png"}


def flatten_to(src_dir: str, dst_dir: str, prefix: str = "") -> int:
    """Move all images from src_dir (recursively) into dst_dir,
    prefixing filenames with subfolder names to avoid collisions.

    Returns the number of files moved.
    """
    src = Path(src_dir)
    dst = Path(dst_dir)
    dst.mkdir(parents=True, exist_ok=True)

    count = 0
    for img_path in sorted(src.rglob("*")):
        if img_path.is_file() and img_path.suffix.lower() in EXTENSIONS:
            # Build a unique name: prefix + relative subfolder path + filename
            rel = img_path.relative_to(src)
            parts = list(rel.parts)
            if prefix:
                parts.insert(0, prefix)
            new_name = "_".join(parts)          # e.g. Deepfakes_000_003_0000000000.jpg
            dest_path = dst / new_name

            shutil.move(str(img_path), str(dest_path))
            count += 1
    return count


def build_dataset(
    frames_dir: str = "frames",
    test_dir: str = "data/test/unknown_attack/diffusion_images",
    output_dir: str = "dataset",
):
    """Reorganize frames/ and data/test/ into the standard layout:

    dataset/
    ├── train/
    │   ├── fake/   (Deepfakes + Face2Face + FaceSwap)
    │   └── real/
    └── test/
        ├── fake/   (ai-generated)
        └── real/   (nature)
    """
    frames = Path(frames_dir)
    test = Path(test_dir)
    out = Path(output_dir)

    # ── Train / fake ──────────────────────────────────────────
    train_fake_dst = out / "train" / "fake"
    for method_dir in sorted((frames / "fake").iterdir()):
        if method_dir.is_dir():
            n = flatten_to(str(method_dir), str(train_fake_dst), prefix=method_dir.name)
            print(f"  [train/fake] {method_dir.name}: {n} images")

    # ── Train / real ──────────────────────────────────────────
    train_real_dst = out / "train" / "real"
    n = flatten_to(str(frames / "real"), str(train_real_dst))
    print(f"  [train/real] {n} images")

    # ── Test / fake  (ai) ────────────────────────────────────
    test_fake_dst = out / "test" / "fake"
    n = flatten_to(str(test / "ai"), str(test_fake_dst))
    print(f"  [test/fake]  {n} images")

    # ── Test / real  (nature) ────────────────────────────────
    test_real_dst = out / "test" / "real"
    n = flatten_to(str(test / "nature"), str(test_real_dst))
    print(f"  [test/real]  {n} images")

    print(f"\nDataset pronto em: {out.resolve()}")


if __name__ == "__main__":
    build_dataset()