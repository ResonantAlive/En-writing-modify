"""Local mock of an OpenAI-compatible chat-completions server (real HTTP on 127.0.0.1, no internet).

Usage (fixture `mock_llm`):
    mock_llm.script(["fenced_json", "ok_json"])        # behaviours consumed in call order
    mock_llm.script(lambda req, n: "ok_json")           # or a callable(request_record, call_index) -> behaviour
    mock_llm.default = "ok_json"                         # used when the script is exhausted
    mock_llm.base_url                                    # "http://127.0.0.1:PORT/v1"
    mock_llm.requests                                    # every received request: {path, headers, body, raw}

Behaviours (str, or dict with "kind"):
  reasoning_bad  ok_json fenced_json trailing_comma text_around missing_field score_out_of_range empty refusal not_json
  regex_only reasoning_content think_tags connection_drop
  "http:500" / {"kind": "http", "status": 429}         plain HTTP error with a small JSON body
  "delay:0.3" / {"kind": "delay", "seconds": 0.3, "then": "ok_json"}   sleep, then answer with `then`
  {"kind": "content", "content": "...", "finish_reason": "stop"}        arbitrary assistant content
"""
from __future__ import annotations

import json
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

SCORE = 7
CRITERIA = [{"name": "内容", "body": "切题，论点清楚"}, {"name": "语言", "body": "有少量语法错误"}]


def good_obj(score=SCORE) -> dict:
    return {"判分": score, "判据": CRITERIA, "评语": "整体不错，注意时态一致。"}


class MockLLM:
    def __init__(self):
        self.requests: list[dict] = []
        self.default: object = "ok_json"
        self._script: object = []
        self._lock = threading.Lock()
        self._stop = threading.Event()
        mock = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *a):  # silence
                pass

            def do_POST(self):
                n = int(self.headers.get("Content-Length") or 0)
                raw = self.rfile.read(n) if n else b""
                try:
                    body = json.loads(raw.decode("utf-8")) if raw else None
                except Exception:
                    body = None
                rec = {"path": self.path, "headers": {k: v for k, v in self.headers.items()}, "body": body,
                       "raw": raw}
                with mock._lock:
                    idx = len(mock.requests)
                    mock.requests.append(rec)
                if not self.path.rstrip("/").endswith("/chat/completions"):
                    return self._send(404, {"error": "not found"})
                beh = mock._next(rec, idx)
                mock._respond(self, beh)

            def _send(self, status, obj, extra=None):
                data = json.dumps(obj, ensure_ascii=False).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(data)))
                for k, v in (extra or {}).items():
                    self.send_header(k, v)
                self.end_headers()
                self.wfile.write(data)

        self._handler = Handler
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.server.daemon_threads = True
        self.port = self.server.server_address[1]
        self.base_url = f"http://127.0.0.1:{self.port}/v1"
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    # ---- scripting
    def script(self, behaviours) -> None:
        with self._lock:
            self._script = list(behaviours) if not callable(behaviours) else behaviours
            self.requests.clear()
            self._consumed = 0

    _consumed = 0

    def _next(self, rec, idx):
        with self._lock:
            sc = self._script
            if callable(sc):
                pass
            else:
                if self._consumed < len(sc):
                    b = sc[self._consumed]
                    self._consumed += 1
                    return b
                return self.default
        return sc(rec, idx)

    def chat_requests(self) -> list[dict]:
        return [r for r in self.requests if r["path"].rstrip("/").endswith("/chat/completions")]

    # ---- responses
    @staticmethod
    def _completion(content, *, finish="stop", extra_msg=None, usage=True):
        msg = {"role": "assistant", "content": content}
        msg.update(extra_msg or {})
        out = {"id": "chatcmpl-mock", "object": "chat.completion", "model": "mock-model",
               "choices": [{"index": 0, "message": msg, "finish_reason": finish}]}
        if usage:
            out["usage"] = {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120}
        return out

    def _respond(self, h, beh):
        if isinstance(beh, str):
            if ":" in beh:
                kind, _, arg = beh.partition(":")
                beh = {"kind": kind, **({"status": int(arg)} if kind == "http" else {"seconds": float(arg)})}
            else:
                beh = {"kind": beh}
        kind = beh["kind"]
        if kind == "delay":
            self._stop.wait(float(beh.get("seconds", 1)))
            return self._respond(h, beh.get("then", "ok_json"))
        if kind == "http":
            return h._send(int(beh["status"]), {"error": {"message": f"mock http {beh['status']}"}},
                           beh.get("headers"))
        if kind == "connection_drop":
            try:
                h.connection.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            h.close_connection = True
            return
        j = lambda o: json.dumps(o, ensure_ascii=False)  # noqa: E731
        good = good_obj()
        content, finish, extra = None, "stop", None
        if kind == "content":
            content, finish = beh["content"], beh.get("finish_reason", "stop")
        elif kind == "ok_json":
            content = j(good)
        elif kind == "fenced_json":
            content = "```json\n" + j(good) + "\n```"
        elif kind == "trailing_comma":
            content = '{"判分": 7, "判据": [{"name": "内容", "body": "切题"},], "评语": "好",}'
        elif kind == "text_around":
            content = "好的，结果如下：\n" + j(good) + "\n希望有帮助。"
        elif kind == "missing_field":
            content = j({"判分": 7})
        elif kind == "score_out_of_range":
            content = j({**good, "判分": 999})
        elif kind == "empty":
            content = ""
        elif kind == "refusal":
            content = "I'm sorry, but I can't help with that request."
        elif kind == "not_json":
            content = "这篇作文写得不错，但是我不打算给出JSON。"
        elif kind == "regex_only":
            content = "判分: 7"
        elif kind == "reasoning_content":
            content, extra = j(good), {"reasoning_content": "SECRET-THINKING-STEPS reasoning_content"}
        elif kind == "reasoning_bad":        # thinking present, but the answer itself is unusable
            content, extra = "这不是JSON", {"reasoning_content": "SECRET-THINKING-STEPS reasoning_bad"}
        elif kind == "think_tags":
            content = "<think>SECRET-THINKING-STEPS think_tags</think>\n" + j(good)
        else:
            raise ValueError(f"unknown mock behaviour {kind!r}")
        h._send(200, self._completion(content, finish=finish, extra_msg=extra))

    def close(self):
        self._stop.set()
        self.server.shutdown()
        self.server.server_close()


@pytest.fixture
def mock_llm():
    m = MockLLM()
    try:
        yield m
    finally:
        m.close()
