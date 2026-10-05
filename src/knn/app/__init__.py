"""Dash web app of the environment explorer: create_app(explorer)."""

from pathlib import Path

ASSETS = Path(__file__).parent / "assets"


def create_app(explorer):
    """Build the Dash app for an Explorer (knn.explorer); nothing is served."""
    from dash import Dash

    from .callbacks import register_callbacks
    from .layout import build_layout

    app = Dash("knn", assets_folder=str(ASSETS), title="knn explorer",
               suppress_callback_exceptions=False)
    app.layout = build_layout(explorer)
    register_callbacks(app, explorer)
    return app
