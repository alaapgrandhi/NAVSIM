#!/usr/bin/env bash
# Run pixi install with a longer HTTP timeout to avoid download/extract timeouts.
# Use this if "pixi install" fails with "network timeout" (UV_HTTP_TIMEOUT current value: 30s).
export UV_HTTP_TIMEOUT=120
exec pixi install "$@"
