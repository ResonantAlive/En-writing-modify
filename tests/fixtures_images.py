"""Tiny real image bytes (JPEG/PNG/WebP/HEIC) for tests, with optional EXIF DateTimeOriginal / Orientation."""
from __future__ import annotations

import io

from PIL import Image

try:
    import pillow_heif
    pillow_heif.register_heif_opener()
except Exception:  # pragma: no cover
    pillow_heif = None


def _exif(dt: str | None, orientation: int | None) -> Image.Exif:
    ex = Image.Exif()
    if orientation:
        ex[0x0112] = orientation
    if dt:  # "YYYY:MM:DD HH:MM:SS"
        ex[0x0132] = dt
        ex.get_ifd(0x8769)[36867] = dt
    return ex


def make_image(fmt: str = "jpeg", size: tuple[int, int] = (40, 20), color=(200, 30, 30),
               exif_time: str | None = None, orientation: int | None = None) -> bytes:
    """fmt in jpeg|png|webp|heic. exif_time like '2026:11:13 10:45:12'."""
    img = Image.new("RGB", size, color)
    # a marker pixel block so orientation changes are detectable (top-left is blue)
    k = max(4, min(size) // 4)
    for x in range(min(k, size[0])):
        for y in range(min(k, size[1])):
            img.putpixel((x, y), (0, 0, 255))
    buf = io.BytesIO()
    fmt = fmt.lower()
    ex = _exif(exif_time, orientation) if (exif_time or orientation) else None
    if fmt in ("jpeg", "jpg"):
        img.save(buf, "JPEG", exif=ex) if ex else img.save(buf, "JPEG")
    elif fmt == "png":
        img.save(buf, "PNG", exif=ex) if ex else img.save(buf, "PNG")
    elif fmt == "webp":
        img.save(buf, "WEBP", exif=ex) if ex else img.save(buf, "WEBP")
    elif fmt in ("heic", "heif"):
        if pillow_heif is None:
            raise RuntimeError("pillow-heif not available")
        img.save(buf, "HEIF", exif=ex) if ex else img.save(buf, "HEIF")
    else:
        raise ValueError(fmt)
    return buf.getvalue()
