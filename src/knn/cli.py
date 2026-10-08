"""Command line interface: python -m knn.cli build config/mkw4.yaml"""

import argparse
import sys

from .build import build
from .config import ConfigError, load_config
from .validation import format_summary, validate_photoz


def serve(cfg, host="127.0.0.1", port=8765, debug=False):
    """Load the explorer for cfg and serve the Dash app (blocks until interrupted)."""
    from .explorer import Explorer

    try:
        from .app import create_app
    except ImportError as err:
        raise SystemExit(f"error: the web app needs the 'app' extra (pip install -e .[app]): "
                         f"{err}") from None
    explorer = Explorer.from_config(cfg)
    if explorer.source == "build":
        print("note: the enriched catalogue or its tracer table is missing or older than the "
              "config; it was built in memory (run `knn build` to cache it)", file=sys.stderr)
    app = create_app(explorer)
    print(f"Serving on http://{host}:{port}/")
    app.run(host=host, port=port, debug=debug, use_reloader=False)


def main(argv=None):
    """Run the knn command line interface.

    Subcommands: 'build CONFIG [--output PATH]' writes the enriched
    catalogue, 'validate-photoz CONFIG [--output-dir DIR]' measures how well
    photo-z environments recover the spec-z environment and writes
    photoz_validation.json, 'serve CONFIG [--host H] [--port P] [--debug]'
    runs the interactive explorer (needs the 'app' extra). argv defaults to
    sys.argv[1:]. Returns the exit status: 0 on success, 2 for an invalid
    configuration.
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

    p_serve = sub.add_parser("serve", help="run the interactive explorer (Dash app)")
    p_serve.add_argument("config", help="path to the YAML config file")
    p_serve.add_argument("--host", default="127.0.0.1", help="address to bind (default 127.0.0.1)")
    p_serve.add_argument("--port", type=int, default=8765, help="port (default 8765)")
    p_serve.add_argument("--debug", action="store_true",
                         help="Dash debug mode (error display in the browser; no reloader)")

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
    elif args.command == "serve":
        serve(cfg, host=args.host, port=args.port, debug=args.debug)
    elif args.command == "validate-photoz":
        results = validate_photoz(cfg, output_dir=args.output_dir)
        print(format_summary(results))
        print(f"\nWrote {results['output_path']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
