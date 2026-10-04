"""Start the native application with the same public host and port entry point."""

import logging
from contextlib import suppress

from .app_logging import configure_logging
from .http_server import Server
from .platforms import require_supported_platform


def launch(host: str = "127.0.0.1", port: int = 7860) -> None:
    configure_logging()
    require_supported_platform()
    logging.getLogger(__name__).info("启动界面：host=%s port=%s", host, port)
    with Server((host, port)) as server, suppress(KeyboardInterrupt):
        server.serve_forever()
