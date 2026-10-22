import argparse
import json
from pathlib import Path
import sys

from .compare import compare
from .demo import run_demo
from .report import write_report


def main(argv=None):
    parser = argparse.ArgumentParser(description="Replayable policy I/O parity checks")
    parser.add_argument("--version", action="version", version="vla-preflight 0.1.0rc1")
    sub = parser.add_subparsers(dest="command", required=True)
    demo = sub.add_parser("demo", help="Run healthy, seeded-fault, fixed and UNKNOWN controls")
    demo.add_argument("--out", default="demo-output", help="New output directory")
    diff = sub.add_parser("compare", help="Compare captured reference/candidate traces")
    diff.add_argument("reference")
    diff.add_argument("candidate")
    diff.add_argument("--out", required=True)
    diff.add_argument("--atol", type=float, default=1e-6)
    diff.add_argument("--rtol", type=float, default=1e-5)
    diff.add_argument("--config", help="JSON with field_map and per-field tolerances")
    args = parser.parse_args(argv)
    try:
        if args.command == "demo":
            result = run_demo(args.out)
            print(
                f"Demo verified: {len(result['scenarios'])} scenarios; {result['elapsed_seconds']} s"
            )
            print((Path(args.out) / "index.html").resolve())
            return 0
        config = json.loads(Path(args.config).read_text(encoding="utf-8")) if args.config else {}
        if set(config) - {"field_map", "tolerances"}:
            raise ValueError("Config accepts field_map and tolerances only")
        result = compare(args.reference, args.candidate, atol=args.atol, rtol=args.rtol, **config)
        write_report(result, args.reference, args.candidate, args.out)
        print(f"{result['status']}: {result['counts']}")
        print((Path(args.out) / "index.html").resolve())
        return result["exit_code"]
    except (ValueError, OSError, KeyError, TypeError) as error:
        print(f"Invalid evidence or command: {error}", file=sys.stderr)
        return 3


if __name__ == "__main__":
    sys.exit(main())
