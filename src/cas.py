import mimetypes
import os
import time
import warnings
from typing import Generator, List, Optional, Union
from urllib.parse import urlparse

import numpy as np
import supervisely as sly
from clip_client import Client
from docarray import Document, DocumentArray
from supervisely.sly_logger import logger

import src.globals as g
from src.utils import send_request, timeit, with_retries

CLIP_SLUG = "supervisely-ecosystem/deploy-clip-as-service"
client = None


class SlyCasClient(Client):
    def __init__(self, server: str, credential: dict = {}, **kwargs):
        """Create a Clip client object that connects to the Clip server.
        Server scheme is in the format of ``scheme://netloc:port``, where
            - scheme: one of grpc, websocket, http, grpcs, websockets, https
            - netloc: the server ip address or hostname
            - port: the public port of the server
        :param server: the server URI
        :param credential: the credential for authentication ``{'Authentication': '<token>'}``
        """
        try:
            r = urlparse(server)
            _port = r.port
            self._scheme = r.scheme
            if self._scheme == "https":
                _new_port = 443
            elif self._scheme == "http":
                _new_port = 80
        except:
            raise ValueError(f"{server} is not a valid scheme")

        _tls = False
        if self._scheme in ("grpcs", "https", "wss"):
            self._scheme = self._scheme[:-1]
            _tls = True

        if self._scheme == "ws":
            self._scheme = "websocket"  # temp fix for the core
            if credential:
                warnings.warn("Credential is not supported for websocket, please use grpc or http")

        if self._scheme in ("grpc", "http", "websocket"):
            if self._scheme == "http":
                if r.path.startswith("/net/") and _port is None:
                    _kwargs = dict(host=r.hostname, port=_port, protocol=self._scheme, tls=_tls)
                elif r.path.startswith("/net/") and _port:
                    _new_port = _port
                    _kwargs = dict(host=r.hostname, port=_port, protocol=self._scheme, tls=_tls)
            else:
                _kwargs = dict(host=r.hostname, port=_port, protocol=self._scheme, tls=_tls)

            from jina import Client

            self._client = Client(**_kwargs)
            self._async_client = Client(**_kwargs, asyncio=True)
            self._client.args.port = str(_new_port) + r.path
            self._async_client.args.port = str(_new_port) + r.path
        else:
            raise ValueError(f"{server} is not a valid scheme")

        self._authorization = credential.get("Authorization", os.environ.get("CLIP_AUTH_TOKEN"))

    def _iter_doc(
        self, content, results: Optional["DocumentArray"] = None
    ) -> Generator["Document", None, None]:
        """Differs from the base class method by handling Supervisely remote URLs."""

        for c in content:
            if isinstance(c, str):
                _mime = mimetypes.guess_type(c)[0]
                # Check if URL contains image-related paths or parameters
                is_image_url = _mime and _mime.startswith("image") or "/remote/" in c

                if is_image_url:
                    logger.trace("[Clip Client] Processing string content as image URL")
                    d = Document(
                        uri=c,
                    ).load_uri_to_blob()
                else:
                    logger.trace("[Clip Client] Processing string content as text")
                    d = Document(text=c)
            elif isinstance(c, Document):
                if c.content_type in ("text", "blob"):
                    logger.trace(
                        "[Clip Client] Processing Document content of type: %s", c.content_type
                    )
                    d = c
                elif not c.blob and c.uri:
                    logger.trace("[Clip Client] Processing Document content as URI")
                    c.load_uri_to_blob()
                    d = c
                elif c.tensor is not None:
                    logger.trace("[Clip Client] Processing Document content as tensor")
                    d = c
                else:
                    raise TypeError(f"unsupported input type {c!r} {c.content_type}")
            else:
                raise TypeError(f"unsupported input type {c!r}")

            if results is not None:
                results.append(d)
            yield d

    @staticmethod
    def _gather_result(response, results: "DocumentArray", attribute: Optional[str] = None):
        r = response.docs
        if attribute:
            results[r[:, "id"]][:, attribute] = r[:, attribute]


class CasClient:
    async def get_vectors(self, queries: List[str]) -> List[List[float]]:
        raise NotImplementedError


class CasTaskClient(CasClient):
    # TODO This class is not used and must be refactored before to be used
    def __init__(self, api: sly.Api, task_id: int):
        self.api = api
        self.task_id = task_id

    async def get_vectors(self, queries: List[str]) -> List[List[float]]:
        return await send_request(
            self.api,
            self.task_id,
            "get_vectors",
            data={},
            context={"queries": queries},
            retries=3,
            timeout=60 * 5,
        )


class CasUrlClient(CasClient):
    STARTUP_TIMEOUT = 60 * 5  # 5 minutes

    def __init__(self, url: str):
        self.url = url
        self.client = SlyCasClient(url)
        self.__wait_for_start()

    def __wait_for_start(self):
        logger.info("Connecting to CLIP at %s...", self.url)
        t = time.monotonic()
        delay = 1
        last_exception = None
        while time.monotonic() - t < self.STARTUP_TIMEOUT:
            try:
                self.client.profile()
                logger.info("Connected to CLIP at %s!", self.url)
                return
            except Exception as e:
                last_exception = e
                logger.debug(
                    "Failed to connect to CLIP at %s. Retrying after %d seconds...",
                    self.url,
                    delay,
                    exc_info=True,
                )
                time.sleep(delay)
                if delay < 4:
                    delay *= 2
        raise RuntimeError(f"Failed to connect to CLIP at {self.url}") from last_exception

    @with_retries(retries=5, sleep_time=2)
    @timeit
    async def get_vectors(self, queries: List[str]) -> List[np.ndarray]:
        """Use CLIP to get vectors from the list of queries.
        List of queries is a list of URLs for images or text prompts.

        :param queries: List of queries (URLs for images or text prompts).
        :type queries: List[str]
        :return: List of vectors.
        :rtype: List[np.ndarray]
        """
        vectors = await self.client.aencode(queries)
        return vectors.tolist()


def _init_client() -> Union[CasUrlClient, CasClient]:
    # Always fetch fresh host information on each initialization
    processed_clip_host = g.clip_host

    if processed_clip_host is None or processed_clip_host == "":
        from src.utils import get_app_host

        sly.logger.info("CLIP host not set in environment, fetching from app host...")
        processed_clip_host = get_app_host(g.api, CLIP_SLUG)
        sly.logger.info("Fetched CLIP host from app: %s", processed_clip_host)

    if not processed_clip_host:
        raise ValueError("CLIP_HOST is not set and cannot be determined automatically")

    try:
        # Try to parse as task ID
        task_id = int(processed_clip_host)
    except ValueError:
        # Not a task ID, treat as URL
        if processed_clip_host[:4] not in ["http", "ws:/", "grpc"]:
            processed_clip_host = "grpc://" + processed_clip_host

        sly.logger.info("Using CLIP host as URL: %s", processed_clip_host)
        return CasUrlClient(processed_clip_host)

    # If we reach here, processed_clip_host is a task ID
    sly.logger.debug("CLIP host appears to be a task ID: %s, fetching task info...", task_id)
    task_info = g.api.task.get_info_by_id(task_id)

    try:
        processed_clip_host = (
            g.api.server_address + task_info["settings"]["message"]["appInfo"]["baseUrl"]
        )
        sly.logger.debug("Resolved CLIP URL from task settings: %s", processed_clip_host)
    except Exception as e:
        sly.logger.warning("Cannot get CLIP URL from task settings", exc_info=True)
        raise RuntimeError("Cannot connect to CLIP Service")

    # Use the resolved URL to create CasUrlClient instead of CasTaskClient
    sly.logger.info("Resolved CLIP host from Task ID and using it as URL: %s", processed_clip_host)
    return CasUrlClient(processed_clip_host)


@timeit
async def _ensure_client_ready():
    """Ensure that the CLIP client is initialized and ready to handle requests."""
    global client

    # Check if client is None or not properly initialized
    if client is None:
        try:
            sly.logger.info("CLIP client is None, attempting to initialize...")
            client = _init_client()
            sly.logger.info("CLIP client successfully initialized")
        except Exception as e:
            error_msg = f"Failed to initialize CLIP client: {str(e)}"
            sly.logger.error(error_msg, exc_info=True)
            raise RuntimeError(error_msg) from e

    # ALWAYS check for CasUrlClient readiness, even if client exists
    if isinstance(client, CasUrlClient):
        try:
            # Test if client is ready to handle requests
            sly.logger.debug("Ensuring CLIP client is ready for requests...")
            result = await client.client._async_client.is_flow_ready()
            if not result:
                sly.logger.warning("CLIP client flow is not ready, invalidating client")
                # IMPORTANT: Set client to None immediately when it's not working
                client = None
            else:
                sly.logger.debug("CLIP client is ready for requests")
        except Exception as e:
            sly.logger.warning("CLIP client flow is not ready, invalidating client: %s", str(e))
            # IMPORTANT: Set client to None immediately when it's not working
            client = None
            sly.logger.debug("CLIP client invalidated due to flow not ready")

            # Try to reinitialize, but don't fail if it doesn't work
            try:
                sly.logger.info("Attempting to reinitialize CLIP client...")
                client = _init_client()
                sly.logger.info("CLIP client successfully reinitialized")

                # Verify the new client is ready
                if isinstance(client, CasUrlClient):
                    result = await client.client._async_client.is_flow_ready()
                    if not result:
                        sly.logger.warning("CLIP client flow is not ready, invalidating client")
                        # IMPORTANT: Set client to None immediately when it's not working
                        client = None
                    else:
                        sly.logger.debug("CLIP client is ready for requests")

            except Exception as init_e:
                sly.logger.warning(
                    "Failed to reinitialize CLIP client, will retry on next request: %s",
                    str(init_e),
                )
                # Don't raise error here - client is already None, next call will try again
                client = None


@timeit
async def get_vectors(queries: List[str]) -> List[List[float]]:
    global client

    # Light check
    if client is None:
        await _ensure_client_ready()

    if client is None:
        raise RuntimeError("CLIP client is not available")

    try:
        return await client.get_vectors(queries)
    except Exception as e:
        # Try to reinitialize the client
        sly.logger.warning(
            "Error during get_vectors, attempting to reinitialize client: %s", str(e)
        )
        try:
            await _ensure_client_ready()
            if client is not None:
                return await client.get_vectors(queries)
        except Exception as reinit_e:
            sly.logger.error("Failed to reinitialize client: %s", str(reinit_e))

        # If reinitialization fails - raise the original error
        error_msg = f"Failed to get vectors from CLIP service: {str(e)}"
        sly.logger.error(error_msg, exc_info=True)
        raise RuntimeError(error_msg) from e


@timeit
async def is_flow_ready():
    global client

    try:
        await _ensure_client_ready()
    except Exception as e:
        sly.logger.warning("Failed to ensure CLIP client ready in health check: %s", str(e))
        return False

    return client is not None
