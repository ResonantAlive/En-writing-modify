"""Launcher: python -m app   (single worker, always)."""
import sys

import uvicorn

from .config import load_config
from .errors import StartupError
from .main import create_app


def main() -> int:
    cfg = load_config()
    try:
        app = create_app(cfg=cfg)
    except StartupError as e:
        print(f"REFUSING TO START: {e}", file=sys.stderr)
        return 2
    n = cfg.net
    uvicorn.run(app, host=n["host"], port=n["port"], workers=1,
                ssl_certfile=n.get("ssl_certfile"), ssl_keyfile=n.get("ssl_keyfile"),
                proxy_headers=False, log_level="info")
    return 0


if __name__ == "__main__":
    sys.exit(main())
