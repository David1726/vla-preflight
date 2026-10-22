"""Run local release checks; never upload, create repositories or contact people."""

from datetime import datetime, timedelta, timezone
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import time
import zipfile

from evidence_utils import sanitized

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="release-check-output", help="New demo output directory")
    args = parser.parse_args()
    output = ROOT / "docs/evidence"
    output.mkdir(parents=True, exist_ok=True)
    checks = []

    def run(name, args):
        start = time.perf_counter()
        result = subprocess.run(
            [sys.executable, *args],
            cwd=ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        (output / f"{name}.log").write_text(sanitized(result.stdout, ROOT), encoding="utf-8")
        checks.append(
            {
                "name": name,
                "command": ["python", *args],
                "exit_code": result.returncode,
                "seconds": round(time.perf_counter() - start, 3),
                "log": f"{name}.log",
            }
        )
        print(f"{name}: exit {result.returncode}")
        if result.returncode:
            raise RuntimeError(f"Check failed: {name}; see {output / (name + '.log')}")

    paths = ["src", "tests", "scripts", "examples", "integrations/run_bridges.py"]
    run("lint", ["-m", "ruff", "check", *paths])
    run("format", ["-m", "ruff", "format", "--check", *paths])
    run("tests", ["-m", "pytest", "-q"])
    run("dependencies", ["-m", "pip", "check"])
    run("demo", ["-m", "vla_preflight", "demo", "--out", args.out])
    demo_root = ROOT / args.out
    shutil.copy2(demo_root / "summary.json", output / "demo-summary.json")
    for name in ("camera_mapping", "missing_reset", "fixed"):
        target = ROOT / "docs/reports" / name
        target.mkdir(parents=True, exist_ok=True)
        for file in ("index.html", "result.json"):
            shutil.copy2(demo_root / name / file, target / file)
    run(
        "example",
        ["examples/check_parity.py", "--out", ".cache/" + Path(args.out).name + "-example"],
    )
    run("build", ["-m", "build", "--no-isolation"])
    artifacts = sorted((ROOT / "dist").glob("vla_preflight-0.1.0rc1*"))
    if len(artifacts) != 2:
        raise RuntimeError("Expected exactly a wheel and sdist")
    run("metadata", ["-m", "twine", "check", *[str(p.relative_to(ROOT)) for p in artifacts]])
    wheel = next(p for p in artifacts if p.suffix == ".whl")
    with zipfile.ZipFile(wheel) as archive:
        names = archive.namelist()
        if any("vendor/" in n or ".venv" in n for n in names):
            raise RuntimeError("Unexpected vendored source or environment in runtime wheel")
        if not any(n.endswith("licenses/LICENSE") for n in names):
            raise RuntimeError("Wheel license missing")
    sdist = next(p for p in artifacts if p.name.endswith(".tar.gz"))
    with tarfile.open(sdist) as archive:
        names = archive.getnames()
        if not any(n.endswith("integrations/vendor/OPENPI_LICENSE") for n in names):
            raise RuntimeError("Source fixture license missing from sdist")
        if any(".venv/" in n or "/.git/" in n or "__pycache__" in n for n in names):
            raise RuntimeError("Environment, bytecode or Git internals included in sdist")
    run("freeze", ["-m", "pip", "freeze"])
    result = {
        "local_date": datetime.now(timezone(timedelta(hours=8))).isoformat(),
        "status": "PASS",
        "checks": checks,
        "artifacts": [
            {
                "file": p.name,
                "bytes": p.stat().st_size,
                "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
            }
            for p in artifacts
        ],
        "not_executed": ["remote CI", "trained model", "simulation", "robot", "browser visual QA"],
    }
    (output / "release-check.json").write_text(
        json.dumps(sanitized(result, ROOT), indent=2), encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
