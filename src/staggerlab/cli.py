"""Command-line reproduction of the fixed validation study."""

import argparse
import json

from .study import run_study


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Run StaggerLab's analytical validation study")
    parser.add_argument(
        "--output", default="out", help="directory for CSV, JSON and PNG artifacts"
    )
    args = parser.parse_args(argv)
    report = run_study(args.output)
    print(
        json.dumps({"all_passed": report["all_passed"], "checks": report["checks"]}, indent=2)
    )
    return 0 if report["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
