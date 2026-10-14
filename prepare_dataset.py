"""Build the training set in data/train from photos in the penguinoh.com repo.

data/train/metadata.jsonl is the manifest: one {"file_name", "text"} entry per
training image. To add a photo, add a line naming the website file (any
extension) with its caption, then re-run this script.
"""

import argparse
import json
from pathlib import Path

from PIL import Image, ImageOps

from main import SharedConfig

DEFAULT_SOURCE_DIR = "../penguinoh_website/src/assets/images"
DEFAULT_TRAIN_DIR = "./data/train"
MAX_SIZE = 1600  # longest side; training buckets downsample from here
SOURCE_EXTENSIONS = (".jpeg", ".jpg", ".JPG", ".JPEG")


def load_manifest(train_dir: Path) -> list[dict]:
    with open(train_dir / "metadata.jsonl") as f:
        return [json.loads(line) for line in f if line.strip()]


def find_source(source_dir: Path, file_name: str) -> Path:
    stem = Path(file_name).stem
    matches = [source_dir / f"{stem}{ext}" for ext in SOURCE_EXTENSIONS if (source_dir / f"{stem}{ext}").exists()]
    if not matches:
        raise FileNotFoundError(f"No source image for {file_name} in {source_dir}")
    return matches[0]


def process(source: Path, dest: Path):
    with Image.open(source) as im:
        im = ImageOps.exif_transpose(im).convert("RGB")
        im.thumbnail((MAX_SIZE, MAX_SIZE), Image.Resampling.LANCZOS)
        im.save(dest, "JPEG", quality=95)  # no exif= argument, so metadata is stripped


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source", default=DEFAULT_SOURCE_DIR, help="Website image directory")
    parser.add_argument("--train-dir", default=DEFAULT_TRAIN_DIR, help="Training set directory")
    parser.add_argument("--force", action="store_true", help="Re-process images that already exist")
    parser.add_argument("--prune", action="store_true", help="Delete images not listed in the manifest")
    args = parser.parse_args()

    source_dir, train_dir = Path(args.source), Path(args.train_dir)
    config = SharedConfig()
    trigger = f"{config.instance_name} the {config.class_name}"

    manifest = load_manifest(train_dir)
    names = [entry["file_name"] for entry in manifest]
    if len(names) != len(set(names)):
        raise ValueError("Duplicate file_name entries in metadata.jsonl")
    for entry in manifest:
        if trigger not in entry["text"]:
            raise ValueError(f'{entry["file_name"]}: caption must contain "{trigger}"')

    written = 0
    for entry in manifest:
        dest = train_dir / entry["file_name"]
        if dest.exists() and not args.force:
            continue
        process(find_source(source_dir, entry["file_name"]), dest)
        written += 1

    stale = sorted(p for p in train_dir.glob("*.jpg") if p.name not in set(names))
    for p in stale:
        if args.prune:
            p.unlink()
        else:
            print(f"warning: {p.name} is not in metadata.jsonl (use --prune to delete)")

    print(f"{len(manifest)} images in manifest, {written} written, {len(stale)} {'pruned' if args.prune else 'stale'}")


if __name__ == "__main__":
    main()
