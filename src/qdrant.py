from typing import Any, Dict, List, Literal, Optional, Tuple, Union
from urllib.parse import urlparse

import numpy as np
import supervisely as sly
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
    QdrantFields,
    TupleFields,
    parse_timestamp,
    prepare_ids,
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
    sly.logger.info(f"Connected to Qdrant at {g.qdrant_host}")
except Exception as e:
    sly.logger.error(f"Failed to connect to Qdrant at {g.qdrant_host}: {e}")


IMAGES_COLLECTION = "images"


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
    project_id: Optional[int] = None,
    dataset_id: Optional[int] = None,
    image_ids: Optional[List[int]] = None,
):
    """Get search filter for Qdrant collection.

    :param project_id: Project ID to filter by.
    :type project_id: Optional[int], optional
    :param dataset_id: Dataset ID to filter by.
    :type dataset_id: Optional[int], optional
    :param image_ids: List of image IDs to filter by.
    :type image_ids: Optional[List[int]], optional
    :return: Filter for Qdrant collection.
    :rtype: dict
    """

    filter = None
    if image_ids:
        filter = models.Filter(
            must=[
                models.FieldCondition(
                    key=QdrantFields.IMAGE_IDS,
                    match=models.MatchAny(any=image_ids),
                ),
            ],
        )
    elif dataset_id:
        filter = models.Filter(
            must=[
                models.FieldCondition(
                    key=QdrantFields.DATASET_IDS,
                    match=models.MatchAny(any=[dataset_id]),
                ),
            ],
        )
    elif project_id:
        filter = models.Filter(
            must=[
                models.FieldCondition(
                    key=QdrantFields.PROJECT_IDS,
                    match=models.MatchAny(any=[project_id]),
                ),
            ],
        )

    return filter


@with_retries()
async def delete_collection_items(
    image_infos: List[sly.ImageInfo], collection_name: str = IMAGES_COLLECTION
) -> Dict[str, Any]:
    """Delete a collection items with the specified IDs.
    For IMAGES_COLLECTION IDs must be strings, that are UUIDs of the images.
    Returns the payloads of the deleted items.

    :param image_infos: A list of ImageInfo objects to delete.
    :type image_infos: List[ImageInfo]
    :param collection_name: The name of the collection to delete items from, defaults to IMAGES_COLLECTION.
    :type collection_name: str
    :return: The payloads of the deleted items.
    :rtype: Dict[str, Any]
    """
    ids = await prepare_ids(image_infos)

    sly.logger.debug("Deleting items from collection %s...", ids)
    try:
        payloads = await get_item_payloads(collection_name, ids)
        await client.delete(collection_name, ids)
        sly.logger.debug("Items %s deleted.", ids)
        return payloads
    except UnexpectedResponse:
        sly.logger.debug("Something went wrong, while deleting %s .", ids)


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

        # Create necessary indexes for efficient filtering
        await client.create_payload_index(
            collection_name=collection_name,
            field_name=f"{QdrantFields.PROJECT_IDS}",
            field_schema="keyword",
        )

        await client.create_payload_index(
            collection_name=collection_name,
            field_name=f"{QdrantFields.DATASET_IDS}",
            field_schema="keyword",
        )

        await client.create_payload_index(
            collection_name=collection_name,
            field_name=f"{QdrantFields.IMAGE_IDS}",
            field_schema="keyword",
        )

        sly.logger.debug("Created payload indexes for collection %s", collection_name)

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
async def get_items_by_info(
    collection_name: str,
    image_infos: List[Union[sly.ImageInfo, ImageInfoLite]],
    with_vectors: bool = False,
) -> Union[List[ImageInfoLite], Tuple[List[ImageInfoLite], List[np.ndarray]]]:
    """Get vectors from the collection based on the specified image hashes.

    :param collection_name: The name of the collection to get vectors from.
    :type collection_name: str
    :param image_infos: A list of ImageInfo or ImageInfoLite objects.
                        If ImageInfoLite, the hash or link field must be set.
    :type image_infos: List[Union[sly.ImageInfo, ImageInfoLite]]
    :param with_vectors: Whether to return vectors along with ImageInfoLite objects, defaults to False.
    :type with_vectors: bool, optional
    :return: A list of vectors.
    :rtype: List[np.ndarray]
    """
    if isinstance(collection_name, int):
        collection_name = str(collection_name)

    point_ids = await prepare_ids(image_infos)

    points = await client.retrieve(
        collection_name=collection_name,
        ids=point_ids,
        with_payload=True,
        with_vectors=with_vectors,
    )

    for point in points:
        point.payload = ImageReferences.clear_payload(point.payload)

    image_infos = [
        ImageInfoLite(id=None, dataset_id=None, project_id=None, **point.payload)
        for point in points
    ]

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
    reference_ids: List[Optional[Dict[str, int]]],
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
    payloads = prepare_payloads(image_infos, reference_ids)
    ids = await prepare_ids(image_infos)
    sly.logger.debug("Upserting %d vectors to collection %s.", len(vectors), collection_name)
    sly.logger.debug("Upserting %d payloads to collection %s.", len(payloads), collection_name)
    await client.upsert(collection_name, Batch(vectors=vectors, ids=ids, payloads=payloads))

    if sly.is_development():
        # By default qdrant should overwrite vectors with the same ids
        # so this line is only needed to check if vectors were upserted correctly.
        # Do not use this in production since it will slow down the process.
        collecton_info = await client.get_collection(collection_name)
        sly.logger.debug(
            "Collection %s has %d vectors.", collection_name, collecton_info.points_count
        )


@with_retries()
@timeit
async def get_diff(
    collection_name: str,
    image_infos: List[ImageInfoLite],
    payloads: Optional[Dict[str, Dict]] = None,
    ids: Optional[List[str]] = None,
) -> Dict[str, Dict[str, Any]]:
    """Get the difference between ImageInfoLite objects and points from the collection.
    Returns a dictionary with IDs as keys and dictionaries with ImageInfoLite objects and ImageReferences objects as values.

    :param collection_name: The name of the collection to get items from.
    :type collection_name: str
    :param image_infos: A list of ImageInfoLite objects.
    :type image_infos: List[ImageInfoLite]
    :param payloads: A dictionary with payloads to update the points in the collection.
    :type payloads: Optional[Dict[str, Dict]], optional
    :param ids: A list of IDs to get from the collection.
    :type ids: Optional[List[str]], optional
    :return: A dictionary with IDs as keys and dictionaries with ImageInfoLite objects and ImageReferences objects as values.
    :rtype: Dict[str, Dict[str, Any]]
    """
    # Get specified ids from collection, compare updated_at and return ids that need to be updated.
    if isinstance(collection_name, int):
        collection_name = str(collection_name)

    if ids is None:
        ids = await prepare_ids(image_infos)

    points = await client.retrieve(collection_name, ids, with_payload=True)
    sly.logger.debug("Retrieved %d points from collection %s", len(points), collection_name)

    diff = _diff(ids, image_infos, points, payloads)

    sly.logger.debug("Found %d points that need to be updated.", len(diff))
    if sly.is_development():
        # To avoid unnecessary computations in production,
        # only log the percentage of points that need to be updated in development.
        percent = round(len(diff) / len(image_infos) * 100, 2) if len(image_infos) > 0 else 0
        sly.logger.debug(
            "From the total of %d points, %d points need to be updated. (%.2f%%)",
            len(image_infos),
            len(diff),
            percent,
        )

    return diff


@timeit
def _diff(
    ids: List[str],
    image_infos: List[ImageInfoLite],
    points: List[Dict[str, Any]],
    payloads: Optional[Dict[str, Dict]] = None,
) -> Tuple[List[ImageInfoLite], List[ImageReferences]]:
    """Get the difference between ImageInfoLite objects and points from the collection.
    Uses payloads to check if need to update the points or only their payload in the collection.
    Returns dict with IDs as keys and dicts with ImageInfoLite objects and ImageReferences objects as values.

    :param ids: A list of IDs to get from the collection.
    :type ids: Optional[List[str]], optional
    :param image_infos: A list of ImageInfoLite objects.
    :type image_infos: List[ImageInfoLite]
    :param points: A list of dictionaries with points from the collection.
    :type points: List[Dict[str, Any]]
    :param payloads: A dictionary with payloads to update the points in the collection.
    :type payloads: Optional[Dict[str, Dict]], optional

    :return: Dictionary with IDs as keys and dicts with ImageInfoLite objects and ImageReferences objects as values.
    :rtype: Dict[str, Dict[str, Any]]
    """
    if len(image_infos) != len(ids):
        raise ValueError(
            "Length of image_infos and ids must be the same. "
            f"Got {len(image_infos)} and {len(ids)}."
        )
    # If the point with the same id doesn't exist in the collection, it will be added to the diff.
    # If the point with the same id exsts - check if IDs are in the payload, if not - add them to the diff.
    diff = {point_id: {} for point_id in ids}
    points_dict = {}
    for point in points:
        if point.payload.get(TupleFields.HASH) is not None:
            points_dict[point.payload.get(TupleFields.HASH)] = point
        elif point.payload.get(TupleFields.LINK) is not None:
            points_dict[point.payload.get(TupleFields.LINK)] = point

    for point_id, image_info in zip(ids, image_infos):
        if image_info.hash is not None:
            point = points_dict.get(image_info.hash)
        elif image_info.link is not None:
            point = points_dict.get(image_info.link)
        if point is None:
            diff[point_id]["info"] = image_info
            diff[point_id]["payload"] = None
            continue
        payload = payloads.get(point_id) if payloads else None
        if payload is None:
            continue
        update_payload = False
        if image_info.id not in point.payload.get(QdrantFields.IMAGE_IDS, []):
            point.payload[QdrantFields.IMAGE_IDS].append(image_info.id)
            update_payload = True
        if image_info.dataset_id not in point.payload.get(QdrantFields.DATASET_IDS, []):
            point.payload[QdrantFields.DATASET_IDS].append(image_info.dataset_id)
            update_payload = True
        if image_info.project_id not in point.payload.get(QdrantFields.PROJECT_IDS, []):
            point.payload[QdrantFields.PROJECT_IDS].append(image_info.project_id)
            update_payload = True
        if update_payload:
            diff[point_id]["info"] = None
            diff[point_id]["payload"] = point.payload
    diff = {k: v for k, v in diff.items() if v}
    return diff


@timeit
def prepare_payloads(
    image_infos: List[ImageInfoLite], references: List[Optional[ImageReferences]]
) -> List[Dict[str, Any]]:
    """
    Prepare payloads for ImageInfoLite objects before upserting to Qdrant.
    Converts named tuples to dictionaries and removes fields:
       - ID
       - PROJECT_ID
       - DATASET_ID
       - SCORE

    Adds reference IDs to the payloads.

    :param image_infos: A list of ImageInfoLite objects.
    :type image_infos: List[ImageInfoLite]
    :param reference_ids: A dictionary with reference IDs.
    :type reference_ids: List[Optional[Dict[str, int]]]
    :return: A list of payloads.
    :rtype: List[Dict[str, Any]]
    """
    ignore_fields = [
        TupleFields.ID,
        TupleFields.PROJECT_ID,
        TupleFields.DATASET_ID,
        TupleFields.SCORE,
    ]
    payloads = []
    for image_info, reference in zip(image_infos, references):
        payload = {k: v for k, v in image_info.to_json().items() if k not in ignore_fields}
        if reference is not None:
            reference.update(
                image_ids=[image_info.id],
                project_ids=[image_info.project_id],
                dataset_ids=[image_info.dataset_id],
            )
            payload[QdrantFields.IMAGE_IDS] = reference.image_ids
            payload[QdrantFields.PROJECT_IDS] = reference.project_ids
            payload[QdrantFields.DATASET_IDS] = reference.dataset_ids
        else:
            payload[QdrantFields.IMAGE_IDS] = [image_info.id]
            payload[QdrantFields.PROJECT_IDS] = [image_info.project_id]
            payload[QdrantFields.DATASET_IDS] = [image_info.dataset_id]
        payloads.append(payload)
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
    if isinstance(collection_name, int):
        collection_name = str(collection_name)

    response = await client.query_points(
        collection_name,
        query_vector,
        query_filter=query_filter,
        limit=limit,
        with_payload=True,
        with_vectors=return_vectors,
        score_threshold=score_threshold,
    )
    result = {}
    items = []
    for point in response.points:
        payload = ImageReferences.clear_payload(point.payload)
        items.append(ImageInfoLite(id=None, dataset_id=None, project_id=None, **payload))
    result[SearchResultField.ITEMS] = items
    if return_vectors:
        result[SearchResultField.VECTORS] = [point.vector for point in response.points]

    if return_scores:
        result[SearchResultField.SCORES] = [point.score for point in response.points]

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
        collection_name,
        limit=limit,
        with_payload=True,
        with_vectors=True,
    )
    sly.logger.debug("Retrieved %d points from collection %s", len(points), collection_name)
    return [
        ImageInfoLite(id=None, project_id=None, dataset_id=None, **point.payload)
        for point in points
    ], [point.vector for point in points]


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
    if isinstance(collection_name, int):
        collection_name = str(collection_name)

    points = await client.retrieve(
        collection_name,
        ids,
        with_payload=True,
        with_vectors=False,
    )
    sly.logger.debug("Retrieved %d points from collection %s", len(points), collection_name)

    # Create initial dictionary with all IDs set to None
    result = {str(id_): None for id_ in ids}

    # Update with actual payloads for IDs that exist
    for point in points:
        result[str(point.id)] = point.payload

    return result


@with_retries()
async def update_payloads(collection_name: str, id_to_payload: Dict[str, Dict]) -> None:
    """Update payloads of items in the collection based on the specified IDs.

    :param collection_name: The name of the collection to update items in.
    :type collection_name: str
    :param id_to_payload: A dictionary with ID keys and payload values.
    :type id_to_payload: Dict[str, Dict]
    """
    if len(id_to_payload) == 0:
        return

    if isinstance(collection_name, int):
        collection_name = str(collection_name)
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
        wait=False,
    )


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


@timeit
async def is_project_in_qdrant(project_id: int) -> bool:
    """Check if any point of Qdrant collection has the specified project ID in its payload.

    :param project_id: The ID of the project to check.
    :type project_id: int
    :return: True if the project is in Qdrant, False otherwise.
    :rtype: bool
    """
    filter = get_search_filter(project_id=project_id)
    try:
        points, _ = await client.scroll(
            collection_name=IMAGES_COLLECTION,
            limit=1,
            scroll_filter=filter,
            with_payload=True,
            with_vectors=False,
        )
        return len(points) > 0
    except UnexpectedResponse:
        return None
