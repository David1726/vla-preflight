"""Measure a genuinely isolated source install; preserve logs and cache conditions."""

from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import subprocess
import sys
import time

from evidence_utils import sanitized

ROOT = Path(__file__).resolve().parents[1]


def main():
    env = ROOT / ".venv-quickstart-measured"
    if env.exists() or (ROOT / "quickstart-output").exists():
        raise SystemExit("Use a new checkout for another measured first-run install")
    start = time.perf_counter()
    steps = []
    commands = [[sys.executable, "-m", "venv", str(env)]]
    python = env / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
    commands += [
        [
            str(python),
            "-m",
            "pip",
            "install",
            "--no-cache-dir",
            "--index-url",
            "https://pypi.org/simple",
            "--timeout",
            "15",
            "--retries",
            "1",
            "--disable-pip-version-check",
            ".",
        ],
        [str(python), "-m", "vla_preflight", "demo", "--out", "quickstart-output"],
    ]
    for cmd in commands:
        before = time.perf_counter()
        print("Starting: " + " ".join([Path(cmd[0]).name, *cmd[1:3]]), flush=True)
        run = subprocess.run(
            cmd, cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT
        )
        steps.append(
            {
                "command": [Path(cmd[0]).name, *cmd[1:]],
                "exit_code": run.returncode,
                "seconds": round(time.perf_counter() - before, 3),
                "output": run.stdout,
            }
        )
        if run.returncode:
            break
    evidence = {
        "local_date": datetime.now(timezone(timedelta(hours=8))).isoformat(),
        "cache_conditions": "New venv; pip cache disabled for install (--no-cache-dir); network available. Prior attempts may have warmed OS/network caches. Not a cold-network benchmark.",
        "prior_attempts": [
            "Default source install was interrupted before completion; no complete timing recorded.",
            "Verbose retry failed because the inherited global pip wheel cache was not writable. The documented recipe now disables that cache; no system permissions were changed.",
        ],
        "total_seconds": round(time.perf_counter() - start, 3),
        "steps": steps,
    }
    target = ROOT / "docs/evidence"
    target.mkdir(parents=True, exist_ok=True)
    (target / "quickstart.json").write_text(
        json.dumps(sanitized(evidence, ROOT), indent=2), encoding="utf-8"
    )
    print(f"Isolated source quickstart: {evidence['total_seconds']} seconds")
    return max(s["exit_code"] for s in steps)


if __name__ == "__main__":
    raise SystemExit(main())
