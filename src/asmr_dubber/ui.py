"""Start the native application with the same public host and port entry point."""

import argparse
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


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Start the ASMR Dubber web UI.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args(argv)
    launch(host=args.host, port=args.port)


if __name__ == "__main__":
    main()
