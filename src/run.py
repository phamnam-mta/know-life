import asyncio
import logging
import uuid
import os
from functools import partial
from typing import Any, List, Optional, Text, Union, Dict


from sanic import Sanic
from asyncio import AbstractEventLoop

from src import server
from src.agent import Agent
from src.utils import sanic_utils
from src.utils.constants import (
    KB_DEFAULT_MODEL_DIR,
    KB_DEFAULT_DATA_DIR,
    KB_DATABASE_PATH,
    KB_RELATION_PATH,
    DEFAULT_RESPONSE_TIMEOUT,
    DEFAULT_SERVER_PORT,
    DEFAULT_SERVER_INTERFACE,
    ENV_SANIC_BACKLOG
)
from src import ELASTICSEARCH_URL
from src.utils.constants import (
    QA_INDEX,
    QA_MODEL_DIR,
)

logger = logging.getLogger()

def configure_app(
    cors: Optional[Union[Text, List[Text], None]] = None,
    auth_token: Optional[Text] = None,
    response_timeout: int = DEFAULT_RESPONSE_TIMEOUT,
    jwt_secret: Optional[Text] = None,
    jwt_method: Optional[Text] = None
) -> Sanic:
    """Run the agent."""

    app = server.create_app(
        cors_origins=cors,
        auth_token=auth_token,
        response_timeout=response_timeout,
        jwt_secret=jwt_secret,
        jwt_method=jwt_method,
    )

    if logger.isEnabledFor(logging.DEBUG):
        sanic_utils.list_routes(app)

    async def configure_async_logging() -> None:
        if logger.isEnabledFor(logging.DEBUG):
            sanic_utils.enable_async_loop_debugging(asyncio.get_event_loop())

    app.add_task(configure_async_logging)

    return app


def serve_application(
    database_path: Text = KB_DATABASE_PATH,
    relation_path: Text = KB_RELATION_PATH,
    kb_model_dir: Text = KB_DEFAULT_MODEL_DIR,
    kb_data_dir: Text = KB_DEFAULT_DATA_DIR,
    qa_model_dir: Text = QA_MODEL_DIR,
    es_url: Text = ELASTICSEARCH_URL,
    index: Text = QA_INDEX,
    interface: Optional[Text] = DEFAULT_SERVER_INTERFACE,
    port: int = DEFAULT_SERVER_PORT,
    cors: Optional[Union[Text, List[Text]]] = None,
    auth_token: Optional[Text] = None,
    response_timeout: int = DEFAULT_RESPONSE_TIMEOUT,
    jwt_secret: Optional[Text] = None,
    jwt_method: Optional[Text] = None,
    log_file: Optional[Text] = None,
    ssl_certificate: Optional[Text] = None,
    ssl_keyfile: Optional[Text] = None,
    ssl_ca_file: Optional[Text] = None,
    ssl_password: Optional[Text] = None,
    use_syslog: Optional[bool] = False,
    syslog_address: Optional[Text] = None,
    syslog_port: Optional[int] = None,
    syslog_protocol: Optional[Text] = None,
) -> None:
    """Run the API entrypoint."""

    app = configure_app(
        cors,
        auth_token,
        response_timeout,
        jwt_secret,
        jwt_method,
    )

    ssl_context = server.create_ssl_context(
        ssl_certificate, ssl_keyfile, ssl_ca_file, ssl_password
    )
    protocol = "https" if ssl_context else "http"

    logger.info(f"Starting Knowlife server on {protocol}://{interface}:{port}")

    app.register_listener(
        partial(load_agent_on_start,
                database_path,
                relation_path,
                kb_model_dir,
                kb_data_dir,
                qa_model_dir,
                es_url,
                index),
        "before_server_start",
    )
    # app.register_listener(close_resources, "after_server_stop")

    number_of_workers = sanic_utils.number_of_sanic_workers()

    sanic_utils.update_sanic_log_level(
        log_file, use_syslog, syslog_address, syslog_port, syslog_protocol,
    )

    app.run(
        host=interface,
        port=port,
        ssl=ssl_context,
        backlog=int(os.environ.get(ENV_SANIC_BACKLOG, "100")),
        workers=number_of_workers,
    )



def load_agent_on_start(
    database_path: Text, 
    relation_path: Text, 
    kb_model_dir: Text,
    kb_data_dir: Text,
    qa_model_dir: Text, 
    es_url: Text, 
    index: Text,
    app: Sanic,
    loop: Text,
    ) -> Agent:
    """Load an agent.
    Used to be scheduled on server start
    (hence the `app` and `loop` arguments)."""
    app.agent = Agent.load_agent(
                database_path=database_path, 
                relation_path=relation_path, 
                kb_model_dir=kb_model_dir,
                kb_data_dir=kb_data_dir,
                qa_model_dir=qa_model_dir, 
                es_url=es_url, 
                index=index
            )
    if not app.agent:
        logger.warning(
            "Agent could not be loaded with the provided configuration. "
            "Load default agent without any model."
        )
        app.agent = Agent.load_agent(
            database_path = KB_DATABASE_PATH,
            relation_path = KB_RELATION_PATH,
            kb_model_dir = KB_DEFAULT_MODEL_DIR,
            kb_data_dir = KB_DEFAULT_DATA_DIR,
            qa_model_dir = QA_MODEL_DIR,
            es_url = ELASTICSEARCH_URL,
            index = QA_INDEX
        )

    return app.agent

if __name__ == '__main__':
    serve_application()