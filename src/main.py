"""Main entry point for HR Policy Assistant (local development).

Render deployments use ``granian`` (see ``render.yaml``). For local
development we keep the same ASGI server so behaviour matches what
runs in production.
"""

import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.api.main import app  # noqa: E402
import granian  # noqa: E402


def main():
    """Run the HR Policy Assistant application locally via granian."""
    port = int(os.getenv("PORT", "8000"))
    host = os.getenv("APP_HOST", "0.0.0.0")

    print(f"Starting HR Policy Assistant on {host}:{port}")
    print(f"Open http://localhost:{port} in your browser")

    # ``granian.Granian`` accepts the same ASGI app and binds to host/port
    # the same way uvicorn does. Mirrors ``render.yaml``'s startCommand.
    granian.Granian(
        target=app,
        interface="asgi",
        address=host,
        port=port,
        workers=1,
    ).serve()


if __name__ == "__main__":
    main()
