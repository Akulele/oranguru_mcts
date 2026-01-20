#!/usr/bin/env python3
"""
Server configuration helper for local Pokemon Showdown.

Allows overriding host/port or websocket URL via environment variables:
- ORANGURU_SHOWDOWN_WS / ORANGURU_SHOWDOWN_URL
- ORANGURU_SHOWDOWN_HOST
- ORANGURU_SHOWDOWN_PORT
"""

from __future__ import annotations

import os

from poke_env import ServerConfiguration


def _derive_server_address(ws_url: str) -> str:
    trimmed = ws_url.replace("ws://", "").replace("wss://", "")
    return trimmed.split("/")[0]


def get_server_configuration(
    default_host: str = "localhost",
    default_port: int = 8000,
) -> ServerConfiguration:
    """Return a ServerConfiguration honoring env overrides."""
    ws_url = os.getenv("ORANGURU_SHOWDOWN_WS") or os.getenv("ORANGURU_SHOWDOWN_URL")
    server_address = os.getenv("ORANGURU_SHOWDOWN_SERVER")

    if ws_url:
        if not server_address:
            server_address = _derive_server_address(ws_url)
        return ServerConfiguration(ws_url, server_address)

    host = os.getenv("ORANGURU_SHOWDOWN_HOST", default_host)
    port_raw = os.getenv("ORANGURU_SHOWDOWN_PORT", str(default_port))
    try:
        port = int(port_raw)
    except ValueError:
        port = default_port

    ws_url = f"ws://{host}:{port}/showdown/websocket"
    return ServerConfiguration(ws_url, f"{host}:{port}")
