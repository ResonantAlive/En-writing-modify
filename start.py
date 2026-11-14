"""One-click launcher:  python start.py

Run it with the Python of the environment you want to use (e.g. an activated virtualenv). It never switches
interpreters by itself.

What it does, in order:
 1. checks that the required packages and (for the local OCR route) the model folders are present --
    it only REPORTS what is missing and prints the command; it never installs anything;
 2. starts the app (single worker, 127.0.0.1 unless settings say otherwise);
 3. warms up the local OCR engine in the background (first run exports the OpenVINO model, ~30-45 s);
 4. opens the browser once the server answers;
 5. on first start, tells you where the invite-code file is (the codes themselves are never printed).

Options:  --no-browser   do not open the browser
          --no-warmup    do not pre-load the local OCR engine
          --check        only run the checks, do not start the server
"""
from __future__ import annotations

import os
import sys
import threading
import time
import urllib.request
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REQUIRED = ["fastapi", "uvicorn", "multipart", "argon2", "filelock", "httpx", "PIL", "json_repair"]
# local OCR (PP-OCRv6): openvino at run time; torch + transformers only for the one-time model export
OPTIONAL_OCR = ["openvino", "torch", "transformers"]
MODEL_DIRS = ["PP-OCRv6_medium_det_safetensors", "PP-OCRv6_medium_rec_safetensors"]


def _missing(mods: list[str]) -> list[str]:
    import importlib.util
    return [m for m in mods if importlib.util.find_spec(m) is None]


def _wait_and_open(url: str, health: str) -> None:
    for _ in range(120):
        try:
            with urllib.request.urlopen(health, timeout=1):
                break
        except Exception:
            time.sleep(0.5)
    else:
        return
    try:
        webbrowser.open(url)
    except Exception:
        pass


def main(argv: list[str]) -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    os.chdir(ROOT)
    sys.path.insert(0, str(ROOT))

    miss = _missing(REQUIRED)
    if miss:
        print(f"[start] 缺少必需的 Python 包: {', '.join(miss)}")
        print(f"[start] 请手动安装:  \"{sys.executable}\" -m pip install -r requirements.txt")
        return 1
    exported = all((ROOT / "data" / "ov_models" / f).is_file() for f in ("det.xml", "rec.xml"))
    ocr_miss = _missing(OPTIONAL_OCR if not exported else ["openvino"])   # torch only needed to export once
    no_models = [d for d in MODEL_DIRS if not (ROOT / d).is_dir()]
    if ocr_miss or no_models:
        print("[start] 提示: 本地识别路线不可用 (缺少: "
              + ", ".join([*ocr_miss, *no_models]) + ")。视觉大模型路线不受影响。")
    if "--check" in argv:
        print("[start] 检查完成")
        return 0

    from app.config import load_config
    from app.errors import StartupError
    from app.main import create_app

    cfg = load_config()
    try:
        app = create_app(cfg=cfg)
    except StartupError as e:
        print(f"[start] 无法启动: {e}", file=sys.stderr)
        return 2
    n = cfg.net
    host = n["host"]
    scheme = "https" if n.get("ssl_certfile") else "http"
    shown = "127.0.0.1" if host in ("0.0.0.0", "::") else host
    url = f"{scheme}://{shown}:{n['port']}/"
    invite = Path(cfg.data_dir) / "app" / "invite_codes.json"
    print(f"[start] 地址: {url}")
    print(f"[start] 邀请码文件(仅服务器上可见): {invite}")
    if "--no-warmup" not in argv:
        from app.ocr import service as ocr_service
        ocr_service.warmup_async(cfg)
    if "--no-browser" not in argv:
        threading.Thread(target=_wait_and_open, args=(url, url), daemon=True).start()
    import uvicorn
    uvicorn.run(app, host=host, port=n["port"], workers=1, ssl_certfile=n.get("ssl_certfile"),
                ssl_keyfile=n.get("ssl_keyfile"), proxy_headers=False, log_level="info")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
