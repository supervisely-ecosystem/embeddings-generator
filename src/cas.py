import time
from typing import List

import numpy as np
import supervisely as sly
from clip_client import Client
from supervisely.sly_logger import logger

import src.globals as g
from src.utils import send_request, timeit, with_retries


class CasClient:
    async def get_vectors(self, queries: List[str]) -> List[List[float]]:
        raise NotImplementedError


class CasTaskClient(CasClient):
    def __init__(self, api: sly.Api, task_id: int):
        self.api = api
        self.task_id = task_id

    async def get_vectors(self, queries: List[str]) -> List[List[float]]:
        return await send_request(
            self.api, self.task_id, "get_vectors", data={}, context={"queries": queries}
        )


class CasUrlClient(CasClient):
    STARTUP_TIMEOUT = 60 * 5  # 5 minutes

    def __init__(self, url: str):
        self.url = url
        self.client = Client(url)
        self.__wait_for_start()

    def __wait_for_start(self):
        logger.info(f"Connecting to CAS at {self.url}...")
        t = time.monotonic()
        delay = 1
        while time.monotonic() - t < self.STARTUP_TIMEOUT:
            try:
                self.client.profile()
                logger.info(f"Connected to CAS at {self.url}!")
                return
            except Exception:
                logger.debug(
                    f"Failed to connect to CAS at {self.url}. Retrying after {delay} seconds...",
                    exc_info=True,
                )
                time.sleep(delay)
                if delay < 4:
                    delay *= 2
        raise RuntimeError(f"Failed to connect to CAS at {self.url}")

    @with_retries(retries=5, sleep_time=2)
    @timeit
    async def get_vectors(self, queries: List[str]) -> List[np.ndarray]:
        """Use CAS to get vectors from the list of queries.
        List of queries is a list of URLs for images or text prompts.

        :param queries: List of queries (URLs for images or text prompts).
        :type queries: List[str]
        :return: List of vectors.
        :rtype: List[np.ndarray]
        """
        vectors = await self.client.aencode(queries)
        return vectors.tolist()


def _init_client() -> CasClient:
    if isinstance(g.cas_host, int):
        return CasTaskClient(g.api, g.cas_host)
    else:
        return CasUrlClient(g.cas_host)


client = _init_client()


@timeit
async def get_vectors(queries: List[str]) -> List[List[float]]:
    return await client.get_vectors(queries)
