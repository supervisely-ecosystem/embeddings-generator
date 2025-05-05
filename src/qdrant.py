from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import numpy as np
import supervisely as sly
from qdrant_client import AsyncQdrantClient, QdrantClient
from qdrant_client.http.exceptions import UnexpectedResponse
from qdrant_client.models import Batch, CollectionInfo, Distance, VectorParams

import src.globals as g
from src.utils import ImageInfoLite, TupleFields, timeit, with_retries

client = AsyncQdrantClient(g.qdrant_host)

try:
    sly.logger.info("Connecting to Qdrant at %s...", g.qdrant_host)
    QdrantClient(g.qdrant_host).get_collections()
    sly.logger.info("Connected to Qdrant at %s", g.qdrant_host)
except Exception as e:
    sly.logger.error("Failed to connect to Qdrant at %s with error: %s", g.qdrant_host, e)


IMAGES_COLLECTION = "images"


class SearchResultField:
    ITEMS = "items"
    VECTORS = "vectors"
    SCORES = "scores"


import base64
import uuid

from qdrant_client.models import PointStruct


def hash_to_uuid(image_hash: str) -> uuid.UUID:
    """Converts a base64-encoded image hash to a UUID."""
    raw_bytes = base64.b64decode(image_hash)
    if len(raw_bytes) != 32:
        raise ValueError("Expected 32-byte hash input")

    selected_bytes = raw_bytes[:8] + raw_bytes[-8:]
    return uuid.UUID(bytes=selected_bytes)


def uuid_to_bytes(hash_uuid: uuid.UUID) -> bytes:
    """Returns 16-byte UUID as bytes. This is a first 8 bytes and last 8 bytes of the image hash."""
    return hash_uuid.bytes


@with_retries()
async def delete_collection_items(
    item_ids: List[str], collection_name: str = IMAGES_COLLECTION
) -> None:
    """Delete a collection items with the specified IDs.
    For IMAGES_COLLECTION IDs must be strings, that are image hashes.

    :param item_ids: A list of item IDs to delete.
    :type item_ids: List[str]
    :param collection_name: The name of the collection to delete items from, defaults to IMAGES_COLLECTION.
    :type collection_name: str
    """
    sly.logger.debug("Deleting items from collection %s...", item_ids)
    try:
        await client.delete(collection_name, item_ids)
        sly.logger.debug("Items %s deleted.", item_ids)
    except UnexpectedResponse:
        sly.logger.debug("Something went wrong, while deleting %s .", item_ids)


@with_retries()
@timeit
async def get_or_create_collection(
    collection_name: str, size: int = 512, distance: Distance = Distance.COSINE
) -> CollectionInfo:
    """Get or create a collection with the specified name.

    :param collection_name: The name of the collection to get or create.
    :type collection_name: str
    :param size: The size of the vectors in the collection, defaults to 512.
    :type size: int, optional
    :param distance: The distance metric to use for the collection, defaults to Distance.COSINE.
    :type distance: Distance, optional
    :return: The CollectionInfo object.
    :rtype: CollectionInfo
    """
    if isinstance(collection_name, int):
        collection_name = str(collection_name)
    try:
        collection = await client.get_collection(collection_name)
        sly.logger.debug("Collection %s already exists.", collection_name)
    except UnexpectedResponse:
        await client.create_collection(
            collection_name,
            vectors_config=VectorParams(size=size, distance=distance),
        )
        sly.logger.debug("Collection %s created.", collection_name)
        collection = await client.get_collection(collection_name)
    return collection


async def collection_exists(collection_name: str) -> bool:
    """Check if a collection with the specified name exists.

    :param collection_name: The name of the collection to check.
    :type collection_name: str
    :return: True if the collection exists, False otherwise.
    :rtype: bool
    """
    if isinstance(collection_name, int):
        collection_name = str(collection_name)
    try:
        await client.get_collection(collection_name)
        return True
    except UnexpectedResponse:
        return False


@with_retries()
@timeit
async def get_items_by_hashes(
    collection_name: str, image_hashes: List[str], with_vectors: bool = False
) -> Union[List[ImageInfoLite], Tuple[List[ImageInfoLite], List[np.ndarray]]]:
    """Get vectors from the collection based on the specified image hashes.

    :param collection_name: The name of the collection to get vectors from.
    :type collection_name: str
    :param image_hashes: A list of image hashes to get vectors for.
    :type image_hashes: List[int]
    :param with_vectors: Whether to return vectors along with ImageInfoLite objects, defaults to False.
    :type with_vectors: bool, optional
    :return: A list of vectors.
    :rtype: List[np.ndarray]
    """
    if isinstance(collection_name, int):
        collection_name = str(collection_name)

    point_ids = [hash_to_uuid(image_hash).hex for image_hash in image_hashes]
    points = await client.retrieve(
        collection_name, point_ids, with_payload=True, with_vectors=with_vectors
    )
    image_infos = [ImageInfoLite(id=None, **point.payload) for point in points]
    if with_vectors:
        vectors = [point.vector for point in points]
        return image_infos, vectors
    return image_infos


@with_retries(retries=5, sleep_time=2)
@timeit
async def upsert(
    collection_name: str,
    vectors: List[np.ndarray],
    image_infos: List[ImageInfoLite],
) -> None:
    """Upsert vectors and payloads to the collection.

    :param collection_name: The name of the collection to upsert to.
    :type collection_name: str
    :param vectors: A list of vectors to upsert.
    :type vectors: List[np.ndarray]
    :param image_infos: A list of ImageInfoLite objects.
    :type image_infos: List[ImageInfoLite]
    """
    if isinstance(collection_name, int):
        collection_name = str(collection_name)
    payloads = get_payloads(image_infos)
    sly.logger.debug("Upserting %d vectors to collection %s.", len(vectors), collection_name)
    sly.logger.debug("Upserting %d payloads to collection %s.", len(payloads), collection_name)
    await client.upsert(
        collection_name,
        Batch(
            vectors=vectors,
            ids=[hash_to_uuid(image_info.hash).hex for image_info in image_infos],
            payloads=payloads,
        ),
    )

    if sly.is_development():
        # By default qdrant should overwrite vectors with the same ids
        # so this line is only needed to check if vectors were upserted correctly.
        # Do not use this in production since it will slow down the process.
        collecton_info = await client.get_collection(collection_name)
        sly.logger.debug(
            "Collection %s has %d vectors.", collection_name, collecton_info.points_count
        )


@with_retries()
async def get_diff(collection_name: str, image_infos: List[ImageInfoLite]) -> List[ImageInfoLite]:
    """Get the difference between ImageInfoLite objects and points from the collection.
    Returns ImageInfoLite objects that need to be updated.

    :param collection_name: The name of the collection to get items from.
    :type collection_name: str
    :param image_infos: A list of ImageInfoLite objects.
    :type image_infos: List[ImageInfoLite]
    :return: A list of ImageInfoLite objects that need to be updated.
    :rtype: List[ImageInfoLite]
    """
    # Get specified ids from collection, compare updated_at and return ids that need to be updated.
    if isinstance(collection_name, int):
        collection_name = str(collection_name)
    points = await client.retrieve(
        collection_name,
        [hash_to_uuid(image_info.hash).hex for image_info in image_infos],
        with_payload=True,
    )
    sly.logger.debug("Retrieved %d points from collection %s", len(points), collection_name)

    diff = _diff(image_infos, points)

    sly.logger.debug("Found %d points that need to be updated.", len(diff))
    if sly.is_development():
        # To avoid unnecessary computations in production,
        # only log the percentage of points that need to be updated in development.
        percent = round(len(diff) / len(image_infos) * 100, 2)
        sly.logger.debug(
            "From the total of %d points, %d points need to be updated. (%.2f%%)",
            len(image_infos),
            len(diff),
            percent,
        )

    return diff


@timeit
def _diff(image_infos: List[ImageInfoLite], points: List[Dict[str, Any]]) -> List[ImageInfoLite]:
    """Compare ImageInfoLite objects with points from the collection and return the difference.
    Uses updated_at field to compare points.

    :param image_infos: A list of ImageInfoLite objects.
    :type image_infos: List[ImageInfoLite]
    :param points: A list of dictionaries with points from the collection.
    :type points: List[Dict[str, Any]]
    :return: A list of ImageInfoLite objects that need to be updated.
    :rtype: List[ImageInfoLite]
    """
    # If the point with the same id doesn't exist in the collection, it will be added to the diff.
    # If the point with the same id exsts - check if updated_at is exactly the same, or add to the diff.
    # Image infos and points have different length, so we need to iterate over image infos.
    diff = []
    points_dict = {point.payload.get(TupleFields.HASH): point for point in points}

    for image_info in image_infos:
        point = points_dict.get(image_info.hash)
        if point is None or point.payload.get(TupleFields.UPDATED_AT) != image_info.updated_at:
            diff.append(image_info)

    return diff


@timeit
def get_payloads(image_infos: List[ImageInfoLite]) -> List[Dict[str, Any]]:
    """Get payloads from ImageInfoLite objects.
    Converts named tuples to dictionaries and removes the id field.

    :param image_infos: A list of ImageInfoLite objects.
    :type image_infos: List[ImageInfoLite]
    :return: A list of payloads.
    :rtype: List[Dict[str, Any]]
    """
    ignore_fields = [TupleFields.ID]
    payloads = [
        {k: v for k, v in image_info.to_json().items() if k not in ignore_fields}
        for image_info in image_infos
    ]
    return payloads


@with_retries()
@timeit
async def search(
    collection_name: str,
    query_vector: np.ndarray,
    limit: int,
    return_vectors: bool = False,
    return_scores: bool = True,
) -> Dict[str, Union[List[ImageInfoLite], List[np.ndarray]]]:
    """Search for similar items in the collection based on the query vector.
    If return_vectors is True, returns vectors along with ImageInfoLite objects.
    NOTE: Do not set return_vectors to True unless necessary, since it will slow down the process
    and increase the memory usage.

    :param collection_name: The name of the collection to search in.
    :type collection_name: str
    :param query_vector: The vector to use for searching.
    :type query_vector: np.ndarray
    :param limit: The number of items to return.
    :type limit: int
    :param return_vectors: Whether to return vectors along with ImageInfoLite objects, defaults to False.
    :type return_vectors: bool, optional
    :param return_scores: Whether to return scores along with ImageInfoLite objects, defaults to True.
    :type return_scores: bool, optional
    :return: A dictionary with keys "items", "vectors" and "scores".
    :rtype: Dict[str, Union[List[ImageInfoLite], List[np.ndarray]]]
    """
    if isinstance(collection_name, int):
        collection_name = str(collection_name)
    points = await client.search(
        collection_name,
        query_vector,
        limit=limit,
        with_payload=True,
        with_vectors=return_vectors,
    )
    result = {"items": [ImageInfoLite(point.id, **point.payload) for point in points]}

    if return_vectors:
        result["vectors"] = [point.vector for point in points]

    if return_scores:
        result["scores"] = [point.score for point in points]

    return result


@with_retries()
@timeit
async def get_items(
    collection_name: str, limit: int = None
) -> Tuple[List[ImageInfoLite], List[np.ndarray]]:
    """Returns specified number of items from the collection. If limit is not specified, returns all items.

    :param collection_name: The name of the collection to get items from.
    :type collection_name: str
    :param limit: The number of items to return, defaults to None.
    :type limit: int, optional
    :return: A tuple of ImageInfoLite objects and vectors.
    :rtype: Tuple[List[ImageInfoLite], List[np.ndarray]]
    """
    if isinstance(collection_name, int):
        collection_name = str(collection_name)
    if not limit:
        collection = await client.get_collection(collection_name)
        limit = collection.points_count
    points, _ = await client.scroll(
        collection_name, limit=limit, with_payload=True, with_vectors=True
    )
    sly.logger.debug("Retrieved %d points from collection %s", len(points), collection_name)
    return [ImageInfoLite(id=None, **point.payload) for point in points], [
        point.vector for point in points
    ]


@timeit
async def get_single_point(
    collection_name: str,
    vector: np.ndarray,
    limit: int,
    option: Literal["farthest", "closest", "random"] = "farthest",
) -> Tuple[ImageInfoLite, np.ndarray]:
    """Get a single point from the collection based on the specified option.
    Options available: "farthest", "closest", "random".
    NOTE: This function is too slow, need to find an option to get farthest point from Qdrant.

    :param collection_name: The name of the collection to get items from.
    :type collection_name: str
    :param vector: The vector to use for searching.
    :type vector: np.ndarray
    :param limit: The number of items to search.
    :type limit: int
    :param option: The option to use for choosing the point, defaults to "farthest".
    :type option: Literal["farthest", "closest", "random"], optional
    :return: The ImageInfoLite object and the vector of the chosen point.
    :rtype: Tuple[ImageInfoLite, np.ndarray]
    """
    raise NotImplementedError(
        "This function is too slow, need to find an option to get farthest point from Qdrant."
    )
    # image_infos, vectors = await search(collection_name, vector, limit, return_vectors=True)
