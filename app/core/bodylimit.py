"""Request body size limits (pure ASGI middleware).

- teacher review (POST .../subtasks/<id>/review): settings review.max_body_mb (default 8 MB) -- 200 items x
  10000 characters of Chinese text is about 6 MB in UTF-8;
- multipart uploads: not limited here (the upload code streams every file against max_file_mb and caps the
  number of files);
- every other request with a body: settings max_json_mb (default 2 MB).
A declared Content-Length over the limit is refused before reading; a body without a length (chunked) is
counted while it is read. Answer: 413 {"code": "body_too_large", ...}.
"""
from __future__ import annotations

import json
import re

from starlette.exceptions import HTTPException

REVIEW_PATH = re.compile(r"^/api/v1/batches/[^/]+/subtasks/[^/]+/review/?$")


class _TooLarge(HTTPException):
    """Raised while reading an over-long body; FastAPI passes HTTPException through its body parsing."""
    def __init__(self, cap: int):
        super().__init__(413, f"request body exceeds {cap // (1024 * 1024)} MB")


class BodyLimit:
    def __init__(self, app, cfg):
        self.app, self.cfg = app, cfg

    def _cap(self, scope) -> int | None:
        if REVIEW_PATH.match(scope.get("path", "")):
            return int(float(self.cfg["review"]["max_body_mb"]) * 1024 * 1024)
        ctype = b""
        for k, v in scope.get("headers") or []:
            if k == b"content-type":
                ctype = v.lower()
        if ctype.startswith(b"multipart/form-data"):
            return None
        return int(float(self.cfg["max_json_mb"]) * 1024 * 1024)

    @staticmethod
    async def _reject(send, cap: int) -> None:
        body = json.dumps({"code": "body_too_large", "message": f"request body exceeds {cap // (1024 * 1024)} MB",
                           "detail": {"max_bytes": cap}}).encode()
        await send({"type": "http.response.start", "status": 413,
                    "headers": [(b"content-type", b"application/json"), (b"content-length", str(len(body)).encode())]})
        await send({"type": "http.response.body", "body": body})

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or scope.get("method") in ("GET", "HEAD", "OPTIONS"):
            return await self.app(scope, receive, send)
        cap = self._cap(scope)
        if cap is None:
            return await self.app(scope, receive, send)
        for k, v in scope.get("headers") or []:
            if k == b"content-length":
                try:
                    if int(v) > cap:
                        return await self._reject(send, cap)
                except ValueError:
                    return await self._reject(send, cap)
        seen = 0
        started = False

        async def counted_receive():
            nonlocal seen
            msg = await receive()
            if msg.get("type") == "http.request":
                seen += len(msg.get("body") or b"")
                if seen > cap:
                    raise _TooLarge(cap)
            return msg

        async def tracked_send(msg):
            nonlocal started
            if msg.get("type") == "http.response.start":
                started = True
            await send(msg)

        try:
            await self.app(scope, counted_receive, tracked_send)
        except _TooLarge:
            if not started:
                await self._reject(send, cap)
