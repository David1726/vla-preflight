"""Install the built wheel in an isolated NumPy-only environment and run it."""

from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import subprocess
import sys
import time

from evidence_utils import sanitized

ROOT = Path(__file__).resolve().parents[1]


def main():
    env = ROOT / ".venv-wheel-smoke"
    if env.exists() or (ROOT / "wheel-smoke-output").exists():
        raise SystemExit("Use a fresh environment/output for wheel smoke verification")
    python = env / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
    wheel = ROOT / "dist/vla_preflight-0.1.0rc1-py3-none-any.whl"
    commands = [
        [sys.executable, "-m", "venv", str(env)],
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
            str(wheel),
        ],
        [
            str(python),
            "-c",
            "import importlib.util, numpy, vla_preflight; assert importlib.util.find_spec('torch') is None; print('No Torch installed; NumPy', numpy.__version__, 'Preflight', vla_preflight.__version__)",
        ],
        [str(python), "-m", "pip", "check"],
        [str(python), "-m", "vla_preflight", "demo", "--out", "wheel-smoke-output"],
    ]
    steps, start = [], time.perf_counter()
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
    result = {
        "local_date": datetime.now(timezone(timedelta(hours=8))).isoformat(),
        "total_seconds": round(time.perf_counter() - start, 3),
        "steps": steps,
        "status": "PASS" if all(s["exit_code"] == 0 for s in steps) else "FAIL",
    }
    (ROOT / "docs/evidence/wheel-smoke.json").write_text(
        json.dumps(sanitized(result, ROOT), indent=2), encoding="utf-8"
    )
    print(result["status"])
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
