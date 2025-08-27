from typing import Any, Dict, List, Literal, Optional, Tuple, Union
from urllib.parse import urlparse

import numpy as np
import supervisely as sly
from fastapi.responses import JSONResponse
from qdrant_client import AsyncQdrantClient, QdrantClient
from qdrant_client.conversions import common_types as types
from qdrant_client.http import models
from qdrant_client.http.exceptions import UnexpectedResponse
from qdrant_client.http.models import PointStruct, UpdateStatus
from qdrant_client.models import (
    Batch,
    CollectionInfo,
    Distance,
    OverwritePayloadOperation,
    SetPayload,
    VectorParams,
)

import src.globals as g
from src.utils import (
    ImageInfoLite,
    ObjectInfoLite,
    QdrantFields,
    ResponseFields,
    TupleFields,
    parse_timestamp,
    timeit,
    with_retries,
)


def create_client_from_url(url: str) -> AsyncQdrantClient:
    """Create a Qdrant client instance from URL.

    Args:
        url: The Qdrant service URL in format http(s)://<host>[:port]

    Returns:
        AsyncQdrantClient: Configured client instance
    """
    parsed_host = urlparse(url)

    # Validate URL format
    if parsed_host.scheme not in ["http", "https"]:
        raise ValueError(f"Qdrant host should be in format http(s)://<host>[:port], got {url}")

    # Create client with appropriate settings based on URL
    return AsyncQdrantClient(
        url=parsed_host.hostname,
        https=parsed_host.scheme == "https",
        port=parsed_host.port or 6333,  # Default Qdrant port
    )


client = create_client_from_url(g.qdrant_host)


try:
    sly.logger.info(f"Connecting to Qdrant at {g.qdrant_host}...")
    QdrantClient(g.qdrant_host).get_collections()
    sly.logger.info(f"Connected to Qdrant successfully.")
except Exception as e:
    sly.logger.error(f"Failed to connect to Qdrant at {g.qdrant_host}: {e}")


class SearchResultField:
    ITEMS = "items"
    VECTORS = "vectors"
    SCORES = "scores"


class ImageReferences:
    IMAGE_IDS = "image_ids"
    PROJECT_IDS = "project_ids"
    DATASET_IDS = "dataset_ids"

    def __init__(self, payload: Dict[str, Any]):
        self.image_ids = payload.get(self.IMAGE_IDS, [])
        self.project_ids = payload.get(self.PROJECT_IDS, [])
        self.dataset_ids = payload.get(self.DATASET_IDS, [])

    def update(
        self,
        image_ids: Optional[List[int]] = None,
        project_ids: Optional[List[int]] = None,
        dataset_ids: Optional[List[int]] = None,
    ):
        """Update the references with new values.

        :param image_id: Image IDs to add.
        :type image_id: Optional[List[int]], optional
        :param project_id: Project IDs to add.
        :type project_id: Optional[List[int]], optional
        :param dataset_id: Dataset IDs to add.
        :type dataset_id: Optional[List[int]], optional
        """
        if image_ids is None:
            image_ids = []
        if project_ids is None:
            project_ids = []
        if dataset_ids is None:
            dataset_ids = []
        self.image_ids.extend(image_ids)
        self.project_ids.extend(project_ids)
        self.dataset_ids.extend(dataset_ids)
        self.image_ids = list(set(self.image_ids))
        self.project_ids = list(set(self.project_ids))
        self.dataset_ids = list(set(self.dataset_ids))

    @classmethod
    def clear_payload(cls, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Clear the payload from references.

        :param payload: Payload to clear.
        :type payload: Dict[str, Any]
        :return: Cleared payload.
        :rtype: Dict[str, Any]
        """
        for field in [cls.IMAGE_IDS, cls.PROJECT_IDS, cls.DATASET_IDS]:
            if field in payload:
                del payload[field]
        return payload


def get_search_filter(
    dataset_id: Optional[int] = None,
    image_ids: Optional[List[int]] = None,
    object_image_ids: Optional[List[int]] = None,
):
    """Get search filter for Qdrant collection.

    :param dataset_id: Dataset ID to filter by.
    :type dataset_id: Optional[int], optional
    :param image_ids: List of image IDs to filter by.
    :type image_ids: Optional[List[int]], optional
    :param object_image_ids: List of object image IDs to filter by.
    :type object_image_ids: Optional[List[int]], optional
    :return: Filter for Qdrant collection.
    :rtype: dict
    """

    filter = None
    if image_ids:
        filter = models.Filter(
            must=[
                models.HasIdCondition(has_id=image_ids),
            ],
        )
    elif dataset_id:
        filter = models.Filter(
            must=[
                models.FieldCondition(
                    key=QdrantFields.DATASET_ID,
                    match=models.MatchAny(any=[dataset_id]),
                ),
            ],
        )
    elif object_image_ids:
        filter = models.Filter(
            must=[
                models.FieldCondition(
                    key=QdrantFields.IMAGE_ID,
                    match=models.MatchAny(any=object_image_ids),
                ),
            ],
        )
    return filter


@with_retries()
async def delete_collection_items(
    collection_name: str,
    items_info: List[Union[sly.ImageInfo, ImageInfoLite]],
    objects: bool = False,
) -> Dict[str, Any]:
    """Delete a collection items with the specified IDs.

    :param collection_name: The name of the collection to delete items from
    :type collection_name: str
    :param items_info: A list of ImageInfo or ImageInfoLite objects to delete.
    :type items_info: List[Union[sly.ImageInfo, ImageInfoLite]]
    :param objects: If True, delete object embeddings instead of image embeddings.
    :type objects: bool, optional
    :return: The payloads of the deleted items.
    :rtype: Dict[str, Any]
    """
    objects_filter = None
    ids = [info.id for info in items_info]
    if objects:
        objects_filter = get_search_filter(object_image_ids=ids)

    sly.logger.debug(f"[Collection: {collection_name}] Deleting items from collection %s...", ids)
    try:
        await client.delete(
            collection_name=collection_name,
            points_selector=objects_filter
            or ids,  # Use filter if objects, otherwise use ids directly
            wait=False,
        )
    except UnexpectedResponse:
        sly.logger.debug(
            f"[Collection: {collection_name}] Something went wrong, while deleting {ids}."
        )


@with_retries()
@timeit
async def get_or_create_collection(
    collection_name: str,
    size: int = 512,
    distance: Distance = Distance.COSINE,
    objects: bool = False,
) -> CollectionInfo:
    """Get or create a collection with the specified name.

    :param collection_name: The name of the collection to get or create.
    :type collection_name: str
    :param size: The size of the vectors in the collection, defaults to 512.
    :type size: int, optional
    :param distance: The distance metric to use for the collection, defaults to Distance.COSINE.
    :type distance: Distance, optional
    :param objects: If True, create a collection for object embeddings instead of image embeddings.
    :type objects: bool, optional
    :return: The CollectionInfo object.
    :rtype: CollectionInfo
    """
    msg_prefix = f"[Project: {collection_name}]"
    try:
        collection = await client.get_collection(collection_name)
        sly.logger.debug(f"{msg_prefix} Qdrant collection already exists.")
    except UnexpectedResponse:
        await client.create_collection(
            collection_name,
            vectors_config=VectorParams(size=size, distance=distance),
        )
        sly.logger.debug(f"{msg_prefix} Qdrant collection created.")

        # Create necessary indexes for efficient filtering

        await client.create_payload_index(
            collection_name=collection_name,
            field_name=f"{QdrantFields.DATASET_ID}",
            field_schema="keyword",
        )

        msg_indexed_fields = (
            f"{msg_prefix} Fields indexed for collection: {QdrantFields.DATASET_ID}"
        )

        if objects:
            await client.create_payload_index(
                collection_name=collection_name,
                field_name=f"{QdrantFields.IMAGE_ID}",
                field_schema="keyword",
            )

            await client.create_payload_index(
                collection_name=collection_name,
                field_name=f"{QdrantFields.CLASS_ID}",
                field_schema="keyword",
            )
            msg_indexed_fields += f", {QdrantFields.IMAGE_ID}, {QdrantFields.CLASS_ID}"

        sly.logger.debug(msg_indexed_fields)

        collection = await client.get_collection(collection_name)
    return collection


async def collection_exists(collection_name: str) -> bool:
    """Check if a collection with the specified name exists.

    :param collection_name: The name of the collection to check.
    :type collection_name: str
    :return: True if the collection exists, False otherwise.
    :rtype: bool
    """

    try:
        await client.get_collection(collection_name)
        return True
    except UnexpectedResponse:
        return False


@with_retries(retries=5, sleep_time=2)
@timeit
async def upsert(
    collection_name: str,
    vectors: List[np.ndarray],
    items_info: List[Union[ImageInfoLite, ObjectInfoLite]],
) -> None:
    """Upsert vectors and payloads to the collection.

    :param collection_name: The name of the collection to upsert to.
    :type collection_name: str
    :param vectors: A list of vectors to upsert.
    :type vectors: List[np.ndarray]
    :param image_infos: A list of ImageInfoLite objects.
    :type image_infos: List[ImageInfoLite]
    """

    ids = [item_info.id for item_info in items_info]
    payloads = create_payloads(items_info)
    msg_prefix = f"[Project: {collection_name}]"
    sly.logger.debug(f"{msg_prefix} Upserting {len(vectors)} vectors to Qdrant collection.")
    await client.upsert(collection_name, Batch(vectors=vectors, ids=ids, payloads=payloads))

    if sly.is_development():
        # By default qdrant should overwrite vectors with the same ids
        # so this line is only needed to check if vectors were upserted correctly.
        # Do not use this in production since it will slow down the process.
        collection_info = await client.get_collection(collection_name)
        sly.logger.debug(
            f"{msg_prefix} Qdrant Collection has {collection_info.points_count} vectors."
        )


@timeit
def create_payloads(items_info: List[Union[ImageInfoLite, ObjectInfoLite]]) -> List[Dict[str, Any]]:
    """
    Prepare payloads for ImageInfoLite or ObjectInfoLite objects before upserting to Qdrant.
    Converts named tuples to dictionaries and removes fields:
       - ID
       - SCORE

    :param items_info: A list of ImageInfoLite or ObjectInfoLite objects.
    :type items_info: List[Union[ImageInfoLite, ObjectInfoLite]]
    :return: A list of payloads.
    :rtype: List[Dict[str, Any]]
    """
    ignore_fields = [TupleFields.ID, TupleFields.SCORE]
    payloads = [
        {k: v for k, v in item_info.to_json().items() if k not in ignore_fields}
        for item_info in items_info
    ]
    return payloads


@with_retries()
@timeit
async def search(
    collection_name: str,
    query_vector: np.ndarray,
    limit: int,
    query_filter: Optional[types.Filter] = None,
    return_vectors: bool = False,
    return_scores: bool = True,
    score_threshold: Optional[float] = None,
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
    :param query_filter: The filter to use for searching, defaults to None.
    :type query_filter: Optional[types.Filter], optional
    :param return_vectors: Whether to return vectors along with ImageInfoLite objects, defaults to False.
    :type return_vectors: bool, optional
    :param return_scores: Whether to return scores along with ImageInfoLite objects, defaults to True.
    :type return_scores: bool, optional
    :param score_threshold: The threshold for scores, defaults to None.
    :type score_threshold: Optional[float], optional
    :return: A dictionary with keys "items", "vectors" and "scores".
    :rtype: Dict[str, Union[List[ImageInfoLite], List[np.ndarray]]]
    """

    response = await client.query_points(
        collection_name=collection_name,
        query=query_vector,
        query_filter=query_filter,
        limit=limit,
        with_payload=True,
        with_vectors=return_vectors,
        score_threshold=score_threshold,
    )
    result = {}

    result[SearchResultField.ITEMS] = [
        ImageInfoLite(id=point.id, **point.payload) for point in response.points
    ]

    if return_vectors:
        result[SearchResultField.VECTORS] = [point.vector for point in response.points]

    if return_scores:
        result[SearchResultField.SCORES] = [point.score for point in response.points]

    return result


@with_retries()
@timeit
async def get_items(
    collection_name: str,
    limit: int = None,
    batch_size: int = 10000,
    with_vectors: bool = False,
    objects: bool = False,
) -> Tuple[List[Union[ImageInfoLite, ObjectInfoLite]], List[np.ndarray]]:
    """Returns specified number of items from the collection. If limit is not specified, returns all items.

    :param collection_name: The name of the collection to get items from.
    :type collection_name: str
    :param limit: The number of items to return, defaults to None.
    :type limit: int, optional
    :param batch_size: The number of items to retrieve in each batch to efficiently get all items, defaults to 10000.
    :type batch_size: int, optional
    :param with_vectors: Whether to return vectors along with ImageInfoLite objects, defaults to False.
    :type with_vectors: bool, optional
    :param objects: If True, return object embeddings instead of image embeddings.
    :type objects: bool, optional
    :return: A tuple of two lists: list of ImageInfoLite or ObjectInfoLite objects and list of vectors.
    :rtype: Tuple[List[Union[ImageInfoLite, ObjectInfoLite]], List[np.ndarray]]    
    """
    all_points: List[types.Record] = []
    next_offset = None
    total = 0

    if not limit:
        collection = await client.get_collection(collection_name)
        limit = collection.points_count

    while total < limit:
        current_batch_size = min(batch_size, limit - total)
        points, next_offset = await client.scroll(
            collection_name=collection_name,
            limit=current_batch_size,
            with_payload=True,
            with_vectors=with_vectors,
            offset=next_offset,
        )
        if not points:
            break
        all_points.extend(points)
        total += len(points)
        if len(points) < current_batch_size:
            break  # No more points to fetch

    all_points = all_points[:limit]

    if objects:
        items_info = [ObjectInfoLite(id=point.id, **point.payload) for point in all_points]
    else:
        items_info = [ImageInfoLite(id=point.id, **point.payload) for point in all_points]

    sly.logger.debug(
        f"[Project: {collection_name}] Retrieved {len(all_points)} points from Qdrant collection."
    )
    if with_vectors:
        vectors = [point.vector for point in all_points]
    else:
        vectors = []
    return items_info, vectors


@with_retries()
@timeit
async def get_items_by_id(
    collection_name: str,
    item_ids: List[int],
    with_vectors: bool = False,
    objects: bool = False,
) -> Tuple[List[Union[ImageInfoLite, ObjectInfoLite]], List[np.ndarray]]:
    """Get vectors from the collection based on the image IDs.

    :param collection_name: The name of the collection to get vectors from.
    :type collection_name: str
    :param image_ids: A list image IDs to retrieve from the collection.
    :type image_ids: List[int]
    :param with_vectors: Whether to return vectors along with ImageInfoLite objects, defaults to False.
    :type with_vectors: bool, optional
    :return: A tuple of ImageInfoLite or ObjectInfoLite objects and vectors.
    :rtype: Tuple[List[Union[ImageInfoLite, ObjectInfoLite]], List[np.ndarray]]
    """

    points = await client.retrieve(
        collection_name=collection_name,
        ids=item_ids,
        with_payload=True,
        with_vectors=with_vectors,
    )
    if objects:
        item_infos = [ObjectInfoLite(id=point.id, **point.payload) for point in points]
    else:
        item_infos = [ImageInfoLite(id=point.id, **point.payload) for point in points]

    if with_vectors:
        vectors = [point.vector for point in points]
    else:
        vectors = []
    return item_infos, vectors


@with_retries()
async def get_item_payloads(collection_name: str, ids=List[str]) -> Dict[str, Any]:
    """
    Get payloads of items from the collection based on the specified IDs.
    IDs that don't exist in the collection will have None as their payload value.

    :param collection_name: The name of the collection to get items from.
    :type collection_name: str
    :param ids: The IDs of the items to return.
    :type ids: List[Union[int, str]]
    :return: A dictionary with ID keys and payload values.
    :rtype: Dict[str, Any]
    """

    points = await client.retrieve(
        collection_name,
        ids,
        with_payload=True,
        with_vectors=False,
    )
    sly.logger.debug(f"[Project: {collection_name}] Retrieved {len(points)} points from Qdrant collection.")

    # Create initial dictionary with all IDs set to None
    result = {str(id_): None for id_ in ids}

    # Update with actual payloads for IDs that exist
    for point in points:
        result[str(point.id)] = point.payload

    return result


@with_retries()
async def update_payloads(
    collection_name: str, id_to_payload: Dict[str, Dict], wait: bool = False
) -> None:
    """Update payloads of items in the collection based on the specified IDs.

    :param collection_name: The name of the collection to update items in.
    :type collection_name: str
    :param id_to_payload: A dictionary with ID keys and payload values.
    :type id_to_payload: Dict[str, Dict]
    :param wait: Whether to wait for the operation to complete, defaults to False.
    :type wait: bool, optional
    """
    if len(id_to_payload) == 0:
        return

    await client.batch_update_points(
        collection_name,
        update_operations=[
            OverwritePayloadOperation(
                overwrite_payload=SetPayload(
                    payload=payload,
                    points=[point_id],
                )
            )
            for point_id, payload in id_to_payload.items()
        ],
        wait=wait,
    )


@with_retries()
async def delete_collection(collection_name: str) -> None:
    """Delete a collection with the specified name.

    :param collection_name: The name of the collection to delete.
    :type collection_name: str
    """
    sly.logger.debug(f"Deleting collection {collection_name}...")
    try:
        await client.delete_collection(collection_name)
        sly.logger.debug(f"Collection {collection_name} deleted.")
    except UnexpectedResponse:
        sly.logger.debug(f"Collection {collection_name} wasn't found while deleting.")
