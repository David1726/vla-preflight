"""Prepare a curated source ZIP for the owner's manual GitHub publication."""

import hashlib
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def main():
    allowed_dirs = ("src", "tests", "scripts", "examples", "docs", "integrations", ".github")
    files = [
        ROOT / name
        for name in (
            "pyproject.toml",
            "MANIFEST.in",
            ".gitignore",
            ".gitattributes",
            "README.md",
            "README.zh-CN.md",
            "LICENSE",
            "THIRD_PARTY.md",
            "CONTRIBUTING.md",
        )
    ]
    for name in allowed_dirs:
        files += [
            p
            for p in (ROOT / name).rglob("*")
            if p.is_file()
            and "__pycache__" not in p.parts
            and p.suffix not in (".pyc", ".tar")
            and not any(part.endswith(".egg-info") for part in p.parts)
        ]
    destination = ROOT / "dist/vla-preflight-source-0.1.0rc1.zip"
    destination.parent.mkdir(exist_ok=True)
    with zipfile.ZipFile(destination, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(files):
            archive.write(path, "vla-preflight/" + path.relative_to(ROOT).as_posix())
    result = {
        "file": destination.name,
        "files": len(files),
        "bytes": destination.stat().st_size,
        "sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
        "excludes": [
            "environments",
            "Git internals",
            "workspace projects",
            "raw generated traces",
            "caches",
        ],
    }
    (destination.parent / "source-manifest.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
