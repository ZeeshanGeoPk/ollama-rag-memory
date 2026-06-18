from __future__ import annotations

import shutil
import sys

import uvicorn

from .config import Settings


def main() -> None:
    """Validate local prerequisites and launch the FastAPI application."""
    settings = Settings.load()
    if settings.ollama_bootstrap and shutil.which("ollama") is None:
        print(
            "Ollama is not installed or is not available on PATH.\n\n"
            "Install Ollama first:\n"
            "  curl -fsSL https://ollama.com/install.sh | sh\n\n"
            "Then run this project again:\n"
            "  .venv/bin/python main.py\n\n"
            "For API-only development without starting Ollama automatically:\n"
            "  OLLAMA_BOOTSTRAP=false .venv/bin/python main.py",
            file=sys.stderr,
        )
        raise SystemExit(1)

    uvicorn.run(
        "ollama_middle_layer.app:app",
        host=settings.middleware_host,
        port=settings.middleware_port,
        reload=False,
    )


if __name__ == "__main__":
    main()
