"""Command line interface: python -m knn.cli build config/mkw4.yaml"""

import argparse
import sys

from .build import build
from .config import ConfigError, load_config
from .validation import format_summary, validate_photoz


def main(argv=None):
    """Run the knn command line interface.

    Subcommands: 'build CONFIG [--output PATH]' writes the enriched
    catalogue, 'validate-photoz CONFIG [--output-dir DIR]' measures how well
    photo-z environments recover the spec-z environment and writes
    photoz_validation.json. argv defaults to sys.argv[1:]. Returns the exit
    status: 0 on success, 2 for an invalid configuration.
    """
    parser = argparse.ArgumentParser(
        prog="knn", description="Environment measures for galaxy catalogues.")
    sub = parser.add_subparsers(dest="command", required=True)

    p_build = sub.add_parser("build", help="build the enriched catalogue")
    p_build.add_argument("config", help="path to the YAML config file")
    p_build.add_argument("--output", default=None,
                         help="output parquet path (default: 'output' from the config)")

    p_val = sub.add_parser("validate-photoz",
                           help="validate photo-z environments against spec-z")
    p_val.add_argument("config", help="path to the YAML config file")
    p_val.add_argument("--output-dir", default=None,
                       help="directory for photoz_validation.json (default: data/derived)")

    args = parser.parse_args(argv)

    try:
        cfg = load_config(args.config)
    except ConfigError as err:
        print(f"error: {err}", file=sys.stderr)
        return 2

    if args.command == "build":
        df = build(cfg, output=args.output)
        print(f"Wrote {len(df)} rows")
        print(df["z_type"].value_counts(dropna=False).to_string())
    elif args.command == "validate-photoz":
        results = validate_photoz(cfg, output_dir=args.output_dir)
        print(format_summary(results))
        print(f"\nWrote {results['output_path']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
