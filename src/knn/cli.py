"""Command line interface: python -m knn.cli build config/mkw4.yaml"""

import argparse
import sys

from .build import build
from .config import ConfigError, load_config


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="knn", description="Environment measures for galaxy catalogues.")
    sub = parser.add_subparsers(dest="command", required=True)

    p_build = sub.add_parser("build", help="build the enriched catalogue")
    p_build.add_argument("config", help="path to the YAML config file")
    p_build.add_argument("--output", default=None,
                         help="output parquet path (default: 'output' from the config)")

    args = parser.parse_args(argv)

    if args.command == "build":
        try:
            cfg = load_config(args.config)
        except ConfigError as err:
            print(f"error: {err}", file=sys.stderr)
            return 2
        df = build(cfg, output=args.output)
        print(f"Wrote {len(df)} rows")
        print(df["z_type"].value_counts(dropna=False).to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
