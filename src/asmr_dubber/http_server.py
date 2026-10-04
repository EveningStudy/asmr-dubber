"""Static assets, JSON transport and capability based media streaming."""

import base64
import ipaddress
import json
import logging
import mimetypes
import os
import secrets
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib.resources import files
from urllib.parse import unquote, urlsplit

from pydantic import ValidationError

from .api import API
from .app_logging import redact_sensitive
from .services.application import Application


def remote_auth(host):
    try:
        loopback = (
            host.casefold() == "localhost" or ipaddress.ip_address(host.strip("[]")).is_loopback
        )
    except ValueError:
        loopback = False
    if loopback:
        return None
    username = os.getenv("ASMR_DUBBER_UI_USERNAME", "asmr").strip() or "asmr"
    password = os.getenv("ASMR_DUBBER_UI_PASSWORD", "").strip() or secrets.token_urlsafe(18)
    print(f"ASMR Dubber login: {username}; password: {password}", flush=True)
    return username, password


class Server(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address, application=None):
        self.application = application or Application()
        self.api = API(self.application)
        self.token = secrets.token_urlsafe(32)
        self.auth = remote_auth(address[0])
        super().__init__(address, Handler)

    def server_close(self):
        self.application.tasks.close()
        super().server_close()


class Handler(BaseHTTPRequestHandler):
    server: Server
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):
        logging.getLogger(__name__).debug(fmt, *args)

    def handle(self):
        try:
            super().handle()
        except (ConnectionResetError, BrokenPipeError):
            self.close_connection = True

    def _allowed(self, write=False):
        if self.server.auth:
            expected = "Basic " + base64.b64encode(":".join(self.server.auth).encode()).decode()
            if not secrets.compare_digest(self.headers.get("Authorization", ""), expected):
                self._discard_body()
                self.close_connection = True
                self.send_response(401)
                self.send_header("Connection", "close")
                self.send_header("WWW-Authenticate", 'Basic realm="ASMR Dubber"')
                self.send_header("Content-Length", "0")
                self.end_headers()
                return False
        host = self.headers.get("Host", "")
        valid_hosts = {
            f"127.0.0.1:{self.server.server_port}",
            f"localhost:{self.server.server_port}",
            f"{self.server.server_address[0]}:{self.server.server_port}",
        }
        if host not in valid_hosts:
            self._discard_body()
            self.close_connection = True
            self._json({"error": "Invalid Host"}, 403)
            return False
        if write and not secrets.compare_digest(
            self.headers.get("X-ASMR-Token", ""), self.server.token
        ):
            self._discard_body()
            self.close_connection = True
            self._json({"error": "Invalid session token"}, 403)
            return False
        origin = self.headers.get("Origin")
        if origin and origin != "http://" + host:
            self._discard_body()
            self.close_connection = True
            self._json({"error": "Invalid Origin"}, 403)
            return False
        return True

    def _discard_body(self):
        if self.command != "POST":
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            return
        if 0 < length <= 16 * 1024 * 1024:
            self.rfile.read(length)

    def _headers(self, status, content_type, length):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(length))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        if self.close_connection:
            self.send_header("Connection", "close")

    def _json(self, value, status=200):
        data = json.dumps(value, ensure_ascii=False).encode("utf-8")
        self._headers(status, "application/json; charset=utf-8", len(data))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        path = urlsplit(self.path).path
        if not self._allowed(write=path.startswith("/api/")):
            return
        try:
            if path.startswith("/api/"):
                self._json(self.server.api.call(path.removeprefix("/api/"), {}))
            elif path.startswith("/media/"):
                self._media(path.removeprefix("/media/"))
            else:
                self._static(path)
        except (KeyError, FileNotFoundError):
            self._json({"error": "Not found"}, 404)
        except Exception as exc:
            self._json({"error": redact_sensitive(exc)}, 400)

    def do_POST(self):
        if not self._allowed(write=True):
            self.close_connection = True
            return
        path = urlsplit(self.path).path
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length < 0:
                raise ValueError("Invalid content length")
            if path == "/api/uploads":
                self._json(
                    self.server.application.media.upload(
                        self.rfile, length, unquote(self.headers.get("X-Filename", ""))
                    )
                )
                return
            if length > 16 * 1024 * 1024:
                raise ValueError("JSON request is too large")
            payload = json.loads(self.rfile.read(length))
            if not isinstance(payload, dict):
                raise ValueError("JSON object required")
            self._json(self.server.api.call(path.removeprefix("/api/"), payload))
        except ValidationError as exc:
            self._json({"error": "Invalid parameters", "detail": redact_sensitive(str(exc))}, 422)
        except (KeyError, FileNotFoundError):
            self._json({"error": "Not found"}, 404)
        except Exception as exc:
            self._json({"error": redact_sensitive(exc)}, 400)

    def _static(self, path):
        name = "index.html" if path == "/" else path.removeprefix("/")
        if any(part in {"..", ".", ""} for part in name.split("/")) or "\\" in name:
            raise FileNotFoundError(name)
        resource = files("asmr_dubber").joinpath("frontend", name)
        if not resource.is_file():
            raise FileNotFoundError(name)
        data = resource.read_bytes()
        if name == "index.html":
            data = data.replace(b"__SESSION_TOKEN__", self.server.token.encode())
        self._headers(
            200,
            (mimetypes.guess_type(name)[0] or "application/octet-stream") + "; charset=utf-8",
            len(data),
        )
        self.end_headers()
        self.wfile.write(data)

    def _media(self, token):
        file, content_type = self.server.application.media.resolve(token)
        size = file.stat().st_size
        start, end = 0, size - 1
        range_header = self.headers.get("Range")
        if range_header:
            try:
                unit, limits = range_header.split("=", 1)
                first, last = limits.split("-", 1)
                if unit != "bytes" or "," in limits:
                    raise ValueError()
                if first:
                    start, end = int(first), int(last) if last else size - 1
                else:
                    start = max(0, size - int(last))
                end = min(end, size - 1)
                if not 0 <= start <= end < size:
                    raise ValueError()
            except ValueError:
                self.send_response(416)
                self.send_header("Content-Range", f"bytes */{size}")
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
        self._headers(206 if range_header else 200, content_type, end - start + 1)
        self.send_header("Accept-Ranges", "bytes")
        if range_header:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.end_headers()
        with file.open("rb") as stream:
            stream.seek(start)
            remaining = end - start + 1
            while remaining:
                block = stream.read(min(1024 * 1024, remaining))
                if not block:
                    break
                try:
                    self.wfile.write(block)
                except (BrokenPipeError, ConnectionResetError):
                    break
                remaining -= len(block)
