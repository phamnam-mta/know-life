import json
import asyncio
import logging
import os
import warnings
from decimal import Decimal
from pathlib import Path
from typing import (
    Any,
    Dict,
    Optional,
    Set,
    Text,
    Tuple,
    Union,
)
from asyncio import AbstractEventLoop
import numpy as np

from src.utils.constants import (
    DEFAULT_ENCODING, 
    DEFAULT_SANIC_WORKERS, 
    ENV_SANIC_WORKERS,
    TCP_PROTOCOL,
    ENV_LOG_LEVEL_LIBRARIES,
    DEFAULT_LOG_LEVEL_LIBRARIES
)

from sanic import Sanic
from sanic.views import CompositionView
from socket import SOCK_DGRAM, SOCK_STREAM


logger = logging.getLogger(__name__)

def configure_file_logging(
    logger_obj: logging.Logger,
    log_file: Optional[Text],
    use_syslog: Optional[bool],
    syslog_address: Optional[Text] = None,
    syslog_port: Optional[int] = None,
    syslog_protocol: Optional[Text] = None,
) -> None:
    """Configure logging to a file.
    Args:
        logger_obj: Logger object to configure.
        log_file: Path of log file to write to.
        use_syslog: Add syslog as a logger.
        syslog_address: Adress of the syslog server.
        syslog_port: Port of the syslog server.
        syslog_protocol: Protocol with the syslog server
    """
    if use_syslog:
        formatter = logging.Formatter(
            "%(asctime)s [%(levelname)-5.5s] [%(process)d]" " %(message)s"
        )
        socktype = SOCK_STREAM if syslog_protocol == TCP_PROTOCOL else SOCK_DGRAM
        syslog_handler = logging.handlers.SysLogHandler(
            address=(syslog_address, syslog_port), socktype=socktype,
        )
        syslog_handler.setLevel(logger_obj.level)
        syslog_handler.setFormatter(formatter)
        logger_obj.addHandler(syslog_handler)
    if log_file:
        formatter = logging.Formatter("%(asctime)s [%(levelname)-5.5s]  %(message)s")
        file_handler = logging.FileHandler(
            log_file, encoding=DEFAULT_ENCODING
        )
        file_handler.setLevel(logger_obj.level)
        file_handler.setFormatter(formatter)
        logger_obj.addHandler(file_handler)


def number_of_sanic_workers() -> int:
    """Get the number of Sanic workers to use in `app.run()`.
    If the environment variable constants.ENV_SANIC_WORKERS is set and is not equal to
    1, that value will only be permitted if the used lock store is not the
    `InMemoryLockStore`.
    """

    def _log_and_get_default_number_of_workers() -> int:
        logger.debug(
            f"Using the default number of Sanic workers ({DEFAULT_SANIC_WORKERS})."
        )
        return DEFAULT_SANIC_WORKERS

    try:
        env_value = int(os.environ.get(ENV_SANIC_WORKERS, DEFAULT_SANIC_WORKERS))
    except ValueError:
        logger.error(
            f"Cannot convert environment variable `{ENV_SANIC_WORKERS}` "
            f"to int ('{os.environ[ENV_SANIC_WORKERS]}')."
        )
        return _log_and_get_default_number_of_workers()

    if env_value == DEFAULT_SANIC_WORKERS:
        return _log_and_get_default_number_of_workers()

    if env_value < 1:
        logger.debug(
            f"Cannot set number of Sanic workers to the desired value "
            f"({env_value}). The number of workers must be at least 1."
        )
        return _log_and_get_default_number_of_workers()

    logger.debug(
        f"Unable to assign desired number of Sanic workers ({env_value}) as "
        f"no `RedisLockStore` or custom `LockStore` endpoint "
        f"configuration has been found."
    )
    return _log_and_get_default_number_of_workers()

def list_routes(app: Sanic) -> Dict[Text, Text]:
    """List all the routes of a sanic application. Mainly used for debugging."""
    from urllib.parse import unquote

    output = {}

    def find_route(suffix: Text, path: Text) -> Optional[Text]:
        for name, (uri, _) in app.router.routes_names.items():
            if name.split(".")[-1] == suffix and uri == path:
                return name
        return None

    for route in app.router.routes:
        endpoint = route.parts
        if endpoint[:-1] in app.router.routes_all and endpoint[-1] == "/":
            continue

        options = {}
        for arg in route._params:
            options[arg] = f"[{arg}]"

        if not isinstance(route.handler, CompositionView):
            handlers = [
                (list(route.methods)[0], route.name.replace("rasa.server.", ""))
            ]
        else:
            handlers = [
                (method, find_route(v.__name__, endpoint) or v.__name__)
                for method, v in route.handler.handlers.items()
            ]

        for method, name in handlers:
            full_endpoint = "/" + "/".join(endpoint)
            line = unquote(f"{full_endpoint:50s} {method:30s} {name}")
            output[name] = line

    url_table = "\n".join(output[url] for url in sorted(output))
    logger.debug(f"Available web server routes: \n{url_table}")

    return output

def enable_async_loop_debugging(
    event_loop: AbstractEventLoop, slow_callback_duration: float = 0.1
) -> AbstractEventLoop:
    """Enables debugging on an event loop.
    Args:
        event_loop: The event loop to enable debugging on
        slow_callback_duration: The threshold at which a callback should be
                                alerted as slow.
    """
    logging.info(
        "Enabling coroutine debugging. Loop id {}.".format(id(asyncio.get_event_loop()))
    )

    # Enable debugging
    event_loop.set_debug(True)

    # Make the threshold for "slow" tasks very very small for
    # illustration. The default is 0.1 (= 100 milliseconds).
    event_loop.slow_callback_duration = slow_callback_duration

    # Report all mistakes managing asynchronous resources.
    warnings.simplefilter("always", ResourceWarning)
    return event_loop

def update_sanic_log_level(
    log_file: Optional[Text] = None,
    use_syslog: Optional[bool] = False,
    syslog_address: Optional[Text] = None,
    syslog_port: Optional[int] = None,
    syslog_protocol: Optional[Text] = None,
) -> None:
    """Set the log level to 'LOG_LEVEL_LIBRARIES' environment variable ."""
    from sanic.log import logger, error_logger, access_logger

    log_level = os.environ.get(ENV_LOG_LEVEL_LIBRARIES, DEFAULT_LOG_LEVEL_LIBRARIES)

    logger.setLevel(log_level)
    error_logger.setLevel(log_level)
    access_logger.setLevel(log_level)

    logger.propagate = False
    error_logger.propagate = False
    access_logger.propagate = False

    if log_file is not None:
        formatter = logging.Formatter("%(asctime)s [%(levelname)-5.5s]  %(message)s")
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)

        logger.addHandler(file_handler)
        error_logger.addHandler(file_handler)
        access_logger.addHandler(file_handler)
    if use_syslog:
        formatter = logging.Formatter(
            "%(asctime)s [%(levelname)-5.5s] [%(process)d]" " %(message)s"
        )
        socktype = SOCK_STREAM if syslog_protocol == TCP_PROTOCOL else SOCK_DGRAM
        syslog_handler = logging.handlers.SysLogHandler(
            address=(syslog_address, syslog_port), socktype=socktype,
        )
        syslog_handler.setFormatter(formatter)
        logger.addHandler(syslog_handler)
        error_logger.addHandler(syslog_handler)
        access_logger.addHandler(syslog_handler)