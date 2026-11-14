"""Image helpers: magic-byte sniffing (extensions are never trusted), thumbnails, EXIF, hashing."""
from __future__ import annotations

import hashlib
from datetime import datetime
from pathlib import Path

from PIL import Image, ImageOps

try:  # HEIC/HEIF support
    import pillow_heif
    pillow_heif.register_heif_opener()
except Exception:  # pragma: no cover - dependency is installed in this project
    pillow_heif = None

ALLOWED_MIMES = ["image/jpeg", "image/png", "image/webp", "image/heic", "image/heif"]

_HEIC_BRANDS = {b"heic", b"heix", b"hevc", b"hevx", b"heim", b"heis", b"hevm", b"hevs"}
_HEIF_BRANDS = {b"mif1", b"msf1", b"heif"}


def sniff_type(head: bytes) -> tuple[str, str] | None:
    """Detect (mime, ext) from the first bytes of a file; None if not an allowed image type."""
    if head[:3] == b"\xff\xd8\xff":
        return "image/jpeg", "jpg"
    if head[:8] == b"\x89PNG\r\n\x1a\n":
        return "image/png", "png"
    if len(head) >= 12 and head[:4] == b"RIFF" and head[8:12] == b"WEBP":
        return "image/webp", "webp"
    if len(head) >= 12 and head[4:8] == b"ftyp":
        brand = head[8:12]
        if brand in _HEIC_BRANDS:
            return "image/heic", "heic"
        if brand in _HEIF_BRANDS:
            return "image/heif", "heif"
    return None


def sha256_file(path: Path | str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _to_rgb(img: Image.Image) -> Image.Image:
    if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
        rgba = img.convert("RGBA")
        bg = Image.new("RGB", rgba.size, (255, 255, 255))
        bg.paste(rgba, mask=rgba.split()[-1])
        return bg
    return img.convert("RGB")


def load_oriented(src: Path | str) -> Image.Image:
    """Fully decode and apply EXIF orientation. Raises on corrupt data."""
    with Image.open(src) as im:
        im.load()
        out = ImageOps.exif_transpose(im)
        return out if out is not im else im.copy()


def make_thumbnail(src: Path | str, dst: Path | str, size: int = 320) -> tuple[int, int]:
    """Write an EXIF-oriented JPEG thumbnail (long edge <= size). Returns the ORIENTED full-size (width, height)."""
    img = load_oriented(src)
    w, h = img.size
    img = _to_rgb(img)
    img.thumbnail((size, size), Image.LANCZOS)
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    img.save(dst, "JPEG", quality=85)
    return w, h


def _parse_exif_dt(v) -> str | None:
    if not isinstance(v, str):
        return None
    v = v.strip().strip("\x00")
    for fmt in ("%Y:%m:%d %H:%M:%S", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(v, fmt).isoformat(timespec="seconds")
        except ValueError:
            pass
    return None


def exif_datetime(path: Path | str) -> str | None:
    """Capture time from EXIF (DateTimeOriginal > DateTimeDigitized > DateTime) as naive ISO string."""
    try:
        with Image.open(path) as im:
            ex = im.getexif()
            sub = ex.get_ifd(0x8769) if hasattr(ex, "get_ifd") else {}
            for v in (sub.get(36867), sub.get(36868), ex.get(36867), ex.get(306)):
                iso = _parse_exif_dt(v)
                if iso:
                    return iso
    except Exception:
        return None
    return None


def preprocess_for_vlm(src: Path | str, dst: Path | str, max_edge: int, quality: int) -> dict:
    """EXIF-fix, convert to RGB JPEG, shrink so the long edge <= max_edge. Returns a params dict."""
    img = load_oriented(src)
    ow, oh = img.size
    img = _to_rgb(img)
    resized = False
    if max(ow, oh) > max_edge:
        img.thumbnail((max_edge, max_edge), Image.LANCZOS)
        resized = True
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    img.save(dst, "JPEG", quality=int(quality))
    return {"orig_width": ow, "orig_height": oh, "width": img.size[0], "height": img.size[1],
            "max_edge": max_edge, "quality": int(quality), "resized": resized,
            "bytes": dst.stat().st_size}
