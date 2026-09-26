"""Fetch immutable upstream files and resolve the exact base snapshot."""
import argparse
import hashlib
import json
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / "artifacts/vllm-audit"
BASE_REPO = "Qwen/Qwen3.5-4B-Base"
BASE_REV = "1001bb4d826a52d1f399e183466143f4da7b741b"
KEV_REV = "139fdd94f1b6a6ad80cc15e08fcb99cac885a101"
SOURCE_REV = "6d02f5d066cd34958dfd15ffa5d2f6f0f4c21a63"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--offline", action="store_true", help="Use prepared files and cached base only")
    args = parser.parse_args()
    from huggingface_hub import snapshot_download
    source = AUDIT / "kev-source"
    source.mkdir(parents=True, exist_ok=True)
    for filename in ("model.py", "checkpoint.py", "api.py", "serve.py"):
        dest = source / filename
        if dest.exists():
            continue
        if args.offline:
            raise FileNotFoundError(f"Missing pinned source: {dest}; rerun without --offline")
        url = f"https://raw.githubusercontent.com/jaredpalmer/kev/{SOURCE_REV}/kev/{filename}"
        with urllib.request.urlopen(url, timeout=60) as response:
            data = response.read()
        temp = dest.with_suffix(".download")
        temp.write_bytes(data)
        temp.replace(dest)
    checkpoint = AUDIT / "kev-checkpoint"
    required = ["adapter_config.json", "adapter_model.safetensors", "head.pt", "training_config.json"]
    if args.offline:
        for filename in required:
            if not (checkpoint / filename).is_file():
                raise FileNotFoundError(checkpoint / filename)
    else:
        snapshot_download("jaredpalmer/kev-4b", revision=KEV_REV, local_dir=checkpoint, allow_patterns=required)
    base = snapshot_download(BASE_REPO, revision=BASE_REV, local_files_only=args.offline,
                             allow_patterns=["config.json", "model.safetensors.index.json", "*.safetensors",
                                             "tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt"])
    base_path = Path(base)
    index = json.loads((base_path / "model.safetensors.index.json").read_text())
    for filename in set(index["weight_map"].values()) | {"config.json", "tokenizer.json", "tokenizer_config.json"}:
        if not (base_path / filename).is_file():
            raise FileNotFoundError(f"Incomplete base snapshot: {filename}")
    files = [*(source / f for f in ("model.py", "checkpoint.py", "api.py", "serve.py")), *(checkpoint / f for f in required)]
    hashes = {}
    for path in files:
        with path.open("rb") as file:
            hashes[str(path.relative_to(ROOT))] = hashlib.file_digest(file, "sha256").hexdigest()
    manifest_path = AUDIT / "inputs.json"
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text())
        if previous["sha256"] != hashes:
            raise ValueError("Pinned input hashes changed; inspect inputs before replacing the manifest")
    manifest = {"base_repo": BASE_REPO, "base_revision": BASE_REV, "base_path": base,
                "kev_revision": KEV_REV, "source_revision": SOURCE_REV, "sha256": hashes}
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"Pinned inputs ready; base snapshot: {base}")


if __name__ == "__main__":
    main()
