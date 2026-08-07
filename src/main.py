"""Main entry point for HR Policy Assistant."""

import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.api.main import app
import uvicorn


def main():
    """Run the HR Policy Assistant application."""
    port = int(os.getenv("PORT", "10000"))
    host = os.getenv("APP_HOST", "0.0.0.0")

    print(f"Starting HR Policy Assistant on {host}:{port}")
    print("Open http://localhost:10000 in your browser")

    uvicorn.run(
        app,
        host=host,
        port=port,
        reload=False
    )


if __name__ == "__main__":
    main()
