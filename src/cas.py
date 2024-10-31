from typing import List

import numpy as np
import supervisely as sly
from clip_client import Client

import src.globals as g
from src.utils import with_retries

client = Client(f"grpc://{g.cas_host}")

try:
    sly.logger.info(f"Connecting to CAS at {g.cas_host}...")
    client.profile()
    sly.logger.info(f"Connected to CAS at {g.cas_host}")
except Exception as e:
    sly.logger.error(f"Failed to connect to CAS at {g.cas_host}: {e}")


@with_retries(retries=5, sleep_time=2)
@sly.timeit
async def get_vectors(queries: List[str]) -> List[np.ndarray]:
    """Use CAS to get vectors from the list of queries.
    List of queries is a list of URLs for images or text prompts.

    :param queries: List of queries (URLs for images or text prompts).
    :type queries: List[str]
    :return: List of vectors.
    :rtype: List[np.ndarray]
    """
    vectors = await client.aencode(queries)
    return [vector.tolist() for vector in vectors]
