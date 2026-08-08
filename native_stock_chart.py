from pathlib import Path

import streamlit.components.v1 as components


_COMPONENT = components.declare_component(
    "native_stock_chart",
    path=str(Path(__file__).parent / "components" / "native_stock_chart" / "dist"),
)


def render_native_stock_chart(charts, *, key: str):
    """Render the View Stock charts with native Lightweight Charts primitives."""
    return _COMPONENT(charts=charts, key=key, default=None)


def browser_bookmark_storage(*, storage_key: str, symbols=None, initial_symbols=None, key: str):
    """Read or update the browser-local ordered bookmark ticker list."""
    return _COMPONENT(
        mode="bookmark_storage",
        storageKey=storage_key,
        symbols=symbols,
        initialSymbols=initial_symbols or [],
        key=key,
        default=None,
    )
