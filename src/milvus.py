import asyncio
from typing import Any, Dict, List, Optional, Tuple, Union
from urllib.parse import urlparse

import numpy as np
import supervisely as sly
from pymilvus import AsyncMilvusClient, CollectionSchema, DataType, FieldSchema

import src.globals as g
from src.utils import (
    ImageInfoLite,
    MilvusFields,
    MilvusParams,
    ObjectInfoLite,
    TupleFields,
    memoryit,
    timeit,
    with_retries,
)


def create_client_from_url(url: str) -> AsyncMilvusClient:
    """Create a Milvus client instance from URL.

    Args:
        url: The Milvus service URL in format http(s)://<host>[:port]

    Returns:
        MilvusClient: Configured client instance
    """
    parsed_host = urlparse(url)

    # Validate URL format
    if parsed_host.scheme not in ["http", "https"]:
        raise ValueError(f"Milvus host should be in format http(s)://<host>[:port], got {url}")

    # Create client with appropriate settings based on URL
    return AsyncMilvusClient(
        uri=g.milvus_host,
        pool_size=50,
        timeout=30,
    )


client = create_client_from_url(g.milvus_host)

# Global flag to track connection status
_connection_verified = False


def ensure_connection(func):
    """Decorator to ensure Milvus connection is established before function execution.
    Connection check is performed only once on first call.
    """

    @wraps(func)
    async def wrapper(*args, **kwargs):
        global _connection_verified

        if not _connection_verified:
            try:
                await client.get_server_version()
                sly.logger.debug("Milvus client connection verified.")
                _connection_verified = True
            except Exception as e:
                sly.logger.error(f"Failed to connect to Milvus at {g.milvus_host}: {e}")
                raise

        return await func(*args, **kwargs)

    return wrapper


# Import functools.wraps for the decorator
from functools import wraps

# Remove the immediate connection check from module import
# try:
#     sly.logger.info(f"Connecting to Milvus at {g.milvus_host}...")
#     check_connection()
#     sly.logger.info(f"Milvus client configured successfully.")
# except Exception as e:
#     sly.logger.error(f"Failed to configure Milvus client for {g.milvus_host}: {e}")


class SearchResultField:
    ITEMS = "items"
    VECTORS = "vectors"
    SCORES = "scores"


def prepare_name(project_id: int) -> str:
    """Prepare a valid collection name from the project ID.

    :param project_id: The project ID to convert.
    :type project_id: int
    :return: A valid collection name.
    :rtype: str
    """
    # Milvus collection names must start with a letter or underscore and contain only letters, numbers, and underscores.
    name = str(project_id).replace("-", "_").replace(" ", "_")
    if not name.startswith("_"):
        name = f"_{name}"
    return name


def get_search_filter(
    dataset_id: Optional[int] = None,
    image_ids: Optional[List[int]] = None,
    object_ids: Optional[List[int]] = None,
) -> Tuple[Optional[str], Dict[str, Any]]:
    """Get search filter template and parameters for Milvus collection.

    :param dataset_id: Dataset ID to filter by.
    :type dataset_id: Optional[int], optional
    :param image_ids: List of image IDs to filter by.
    :type image_ids: Optional[List[int]], optional
    :param object_ids: List of object image IDs to filter by.
    :type object_ids: Optional[List[int]], optional
    :return: Tuple of filter template and filter parameters.
    :rtype: Tuple[Optional[str], Dict[str, Any]]
    """
    filter_template = None
    filter_params = {}

    if image_ids:
        filter_template = f"{MilvusFields.ID} in {image_ids}"
        filter_params = {"image_ids": image_ids}
    elif dataset_id:
        filter_template = f"{MilvusFields.DATASET_ID} == {dataset_id}"
        filter_params = {"dataset_id": dataset_id}
    elif object_ids:
        filter_template = f"{MilvusFields.IMAGE_ID} in {object_ids}"
        filter_params = {"object_ids": object_ids}

    return filter_template, filter_params


@ensure_connection
@with_retries()
async def delete_collection_items(
    collection_name: str,
    items_info: List[Union[sly.ImageInfo, ImageInfoLite, ObjectInfoLite]],
):
    """Delete a collection items with the specified IDs.

    :param collection_name: The name of the collection to delete items from
    :type collection_name: str
    :param items_info: A list of ImageInfo, ImageInfoLite or ObjectInfoLite objects to delete.
    :type items_info: List[Union[sly.ImageInfo, ImageInfoLite, ObjectInfoLite]]
    """
    ids = [info.id for info in items_info]

    if not ids or len(ids) == 0:
        sly.logger.debug(f"[Project: {collection_name}] No items to delete from Milvus collection.")
        return []

    sly.logger.debug(
        f"[Project: {collection_name}] Deleting items from Milvus collection %s...", ids
    )
    try:
        partition_name = None
        if isinstance(items_info[0], ObjectInfoLite):
            partition_name = MilvusParams.OBJECTS

        return await client.delete(
            prepare_name(collection_name),
            ids=ids,
            partition_name=partition_name,
        )
    except Exception as e:
        sly.logger.debug(
            f"[Project: {collection_name}] Something went wrong, while deleting {len(ids)} item(s) from Milvus collection: {e}"
        )


@ensure_connection
@with_retries()
@timeit
async def get_or_create_collection(
    collection_name: str,
    size: int = 512,
    distance: MilvusParams = MilvusParams.COSINE,
    index_type: MilvusParams = MilvusParams.IVF_SQ8,
    params: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:  # Changed return type from CollectionInfo
    """Get or create a collection with the specified name.

    :param collection_name: The name of the collection to get or create.
    :type collection_name: str
    :param size: The size of the vectors in the collection, defaults to 512.
    :type size: int, optional
    :param distance: The distance metric to use for the collection, defaults to "COSINE".
    :type distance: str, optional
    :param index_type: The index type to use for the collection, defaults to "HNSW".
    :type index_type: str, optional
    :param params: The index parameters to use for the collection, defaults to None.
                    If None, uses {"M": 16, "efConstruction": 200} for HNSW and {"nlist": 1024} for other index types.
    :type params: Optional[Dict[str, Any]], optional
    :return: Collection information.
    :rtype: Dict[str, Any]
    """
    msg_prefix = f"[Project: {collection_name}]"
    prepared_name = prepare_name(collection_name)
    try:
        exists = await client.has_collection(prepared_name)
        if exists:
            try:
                info = await client.describe_collection(prepared_name)
                return info
            except Exception as e:
                sly.logger.warning(f"{msg_prefix} Failed to load existing collection: {e}")
                # Try to drop and recreate if loading fails
                await client.drop_collection(prepared_name)
                sly.logger.debug(f"{msg_prefix} Dropped corrupted collection, will recreate")

        sly.logger.debug(f"{msg_prefix} Creating new Milvus collection...")
        # Define schema based on whether it's for objects or images
        fields = [
            FieldSchema(name=MilvusFields.ID, dtype=DataType.INT64, is_primary=True, auto_id=False),
            FieldSchema(name=MilvusFields.VECTOR, dtype=DataType.FLOAT_VECTOR, dim=size),
            FieldSchema(name=MilvusFields.DATASET_ID, dtype=DataType.INT64),
            FieldSchema(name=MilvusFields.FULL_URL, dtype=DataType.VARCHAR, max_length=1000),
            FieldSchema(name=MilvusFields.CAS_URL, dtype=DataType.VARCHAR, max_length=1000),
            FieldSchema(name=MilvusFields.IMAGE_ID, dtype=DataType.INT64, nullable=True),
            FieldSchema(name=MilvusFields.CLASS_ID, dtype=DataType.INT64, nullable=True),
        ]

        schema = CollectionSchema(
            fields=fields, description=f"Collection for project ID: {collection_name}"
        )

        if params is None:
            if index_type == MilvusParams.HNSW:
                params = {"M": 16, "efConstruction": 200}
            else:
                params = {MilvusParams.NLIST: 4096, MilvusParams.NPROBE: 64}

        sly.logger.debug(
            f"{msg_prefix} Creating collection with index_type: {index_type}, distance: {distance}"
        )

        # Create index parameters
        index_params = client.prepare_index_params()
        index_params.add_index(
            field_name=MilvusFields.VECTOR,
            index_type=index_type,
            metric_type=distance,
            params=params,
        )
        # Add indexes for scalar fields to improve filtering performance
        index_params.add_index(field_name=MilvusFields.DATASET_ID, index_type="AUTOINDEX")
        index_params.add_index(field_name=MilvusFields.IMAGE_ID, index_type="AUTOINDEX")
        index_params.add_index(field_name=MilvusFields.CLASS_ID, index_type="AUTOINDEX")

        # Create collection
        await client.create_collection(
            collection_name=prepared_name,
            schema=schema,
            index_params=index_params,
        )

        try:
            await client.create_partition(
                collection_name=prepared_name,
                partition_name=MilvusParams.OBJECTS,
            )
            sly.logger.debug(f"{msg_prefix} Created objects partition")
        except Exception as e:
            sly.logger.warning(f"{msg_prefix} Failed to create objects partition: {e}")

        # Load the collection
        await client.load_collection(prepared_name)
        sly.logger.debug(f"{msg_prefix} Milvus collection created and loaded with indexed fields.")

        collection_info = await client.describe_collection(prepared_name)
        return collection_info
    except Exception as e:
        sly.logger.error(f"{msg_prefix} Error creating/getting collection: {e}")
        raise


@ensure_connection
async def collection_exists(collection_name: str) -> bool:
    """Check if a collection with the specified name exists.

    :param collection_name: The name of the collection to check.
    :type collection_name: str
    :return: True if the collection exists, False otherwise.
    :rtype: bool
    """
    try:
        exists = await client.has_collection(prepare_name(collection_name))
        return exists
    except Exception:
        return False


@ensure_connection
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
    :param items_info: A list of ImageInfoLite or ObjectInfoLite objects.
    :type items_info: List[Union[ImageInfoLite, ObjectInfoLite]]
    """
    msg_prefix = f"[Project: {collection_name}]"
    sly.logger.debug(f"{msg_prefix} Upserting {len(vectors)} vectors to Milvus collection.")

    # Prepare data for insertion
    data = []
    for vector, item_info in zip(vectors, items_info):
        vector_info = item_info.to_json()
        # Remove score from vector_info as they are handled separately
        vector_info.pop(TupleFields.SCORE, None)

        record = {MilvusFields.VECTOR: vector, **vector_info}
        data.append(record)

    # Insert data into collection
    await client.upsert(collection_name=prepare_name(collection_name), data=data)

    if sly.is_development():
        # Check collection stats
        stats = await client.get_collection_stats(prepare_name(collection_name))
        sly.logger.debug(f"{msg_prefix} Milvus Collection has {stats['row_count']} vectors.")


@ensure_connection
@with_retries()
@timeit
@memoryit
async def search(
    collection_name: str,
    query_vector: np.ndarray,
    limit: int,
    query_filter: Optional[Tuple[str, dict]] = None,
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
    :param query_filter: A tuple of filter template and filter parameters to apply during search, defaults to None.
    :type query_filter: Optional[Tuple[str, dict]], optional
    :param return_vectors: Whether to return vectors along with ImageInfoLite objects, defaults to False.
    :type return_vectors: bool, optional
    :param return_scores: Whether to return scores along with ImageInfoLite objects, defaults to True.
    :type return_scores: bool, optional
    :param score_threshold: The threshold for scores, defaults to None.
    :type score_threshold: Optional[float], optional
    :return: A dictionary with keys "items", "vectors" and "scores".
    :rtype: Dict[str, Union[List[ImageInfoLite], List[np.ndarray]]]
    """
    prepared_name = prepare_name(collection_name)

    search_params = {
        MilvusFields.METRIC_TYPE: MilvusParams.COSINE,
        MilvusFields.PARAMS: {MilvusParams.NPROBE: 64, MilvusParams.RANGE_FILTER: 1.0},
    }

    output_fields = [
        MilvusFields.ID,
        MilvusFields.DATASET_ID,
        MilvusFields.FULL_URL,
        MilvusFields.CAS_URL,
    ]
    if return_vectors:
        output_fields.append(MilvusFields.VECTOR)

    await client.load_collection(prepared_name)

    response = await client.search(
        collection_name=prepared_name,
        data=[query_vector],
        anns_field=MilvusFields.VECTOR,
        search_params=search_params,
        limit=limit,
        filter=query_filter[0] if query_filter else None,
        filter_params=query_filter[1] if query_filter else {},
        output_fields=output_fields,
    )
    await client.release_collection(prepared_name)

    response = response[0] if response else []
    result = {}

    # Convert results to ImageInfoLite objects
    items = []
    for hit in response:
        # Milvus with COSINE returns similarity score directly (higher is better)
        # For other metrics it would return distance (lower is better)
        similarity_score = hit.get(MilvusFields.DISTANCE, 0.0)

        # Handle score threshold (for COSINE, score is similarity directly)
        if score_threshold is not None:
            if similarity_score < score_threshold:
                continue

        # Extract entity data - Milvus structure is different
        entity_data = hit.get(MilvusFields.ENTITY, hit)  # Some versions return data directly in hit

        item = ImageInfoLite(
            id=entity_data.get(MilvusFields.ID),
            dataset_id=entity_data.get(MilvusFields.DATASET_ID),
            full_url=entity_data.get(MilvusFields.FULL_URL),
            cas_url=entity_data.get(MilvusFields.CAS_URL),
            score=similarity_score,
        )
        items.append(item)

    result[SearchResultField.ITEMS] = items

    if return_vectors:
        result[SearchResultField.VECTORS] = [
            hit.get(MilvusFields.ENTITY, hit).get(MilvusFields.VECTOR, []) for hit in response
        ]

    if return_scores:
        # Return similarity scores directly for COSINE metric
        result[SearchResultField.SCORES] = [hit.get(MilvusFields.DISTANCE, 0.0) for hit in response]

    return result


@ensure_connection
@with_retries()
@timeit
@memoryit
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
    prepared_name = prepare_name(collection_name)
    # Get collection stats to determine total count
    stats = await client.get_collection_stats(prepared_name)
    total_count = stats.get("row_count", 0)

    if not limit:
        actual_limit = total_count
    else:
        actual_limit = min(limit, total_count)

    output_fields = [
        MilvusFields.ID,
        MilvusFields.DATASET_ID,
        MilvusFields.FULL_URL,
        MilvusFields.CAS_URL,
    ]
    if with_vectors:
        output_fields.append(MilvusFields.VECTOR)
    if objects:
        output_fields.extend([MilvusFields.IMAGE_ID, MilvusFields.CLASS_ID])

    # Milvus has a limitation on the number of entities returned per query (16384)
    # We need to retrieve data in batches
    MAX_QUERY_LIMIT = 16384  # Milvus limit for single query
    effective_batch_size = min(batch_size, MAX_QUERY_LIMIT)

    items_info = []
    vectors = []
    retrieved_count = 0

    # Process data in batches with offset pagination
    offset = 0

    await client.load_collection(prepared_name)

    while retrieved_count < actual_limit:
        # Calculate the limit for this batch
        remaining_items = actual_limit - retrieved_count
        current_batch_size = min(effective_batch_size, remaining_items)

        try:
            # Query current batch
            batch_results = await client.query(
                collection_name=prepared_name,
                expr="",
                output_fields=output_fields,
                offset=offset,
                limit=current_batch_size,
            )

            # If no results returned, we've reached the end
            if not batch_results:
                break

            # Process batch results
            for result in batch_results:
                distance = result.get(MilvusFields.DISTANCE, 0)
                similarity_score = 1.0 - distance if distance is not None else 0.0
                if objects:
                    item = ObjectInfoLite(
                        id=result.get(MilvusFields.ID),
                        image_id=result.get(MilvusFields.IMAGE_ID),
                        dataset_id=result.get(MilvusFields.DATASET_ID),
                        class_id=result.get(MilvusFields.CLASS_ID),
                        full_url=result.get(MilvusFields.FULL_URL),
                        cas_url=result.get(MilvusFields.CAS_URL),
                        score=similarity_score,
                    )
                else:
                    item = ImageInfoLite(
                        id=result.get(MilvusFields.ID),
                        dataset_id=result.get(MilvusFields.DATASET_ID),
                        full_url=result.get(MilvusFields.FULL_URL),
                        cas_url=result.get(MilvusFields.CAS_URL),
                        score=similarity_score,
                    )
                items_info.append(item)

                if with_vectors:
                    vectors.append(result.get("vector", []))

            retrieved_count += len(batch_results)
            offset += len(batch_results)

            sly.logger.debug(
                f"[Project: {collection_name}] Retrieved batch of {len(batch_results)} items. "
                f"Total: {retrieved_count}/{actual_limit}"
            )

            # If we got fewer items than requested, we've reached the end
            if len(batch_results) < current_batch_size:
                break

        except Exception as e:
            sly.logger.error(
                f"[Project: {collection_name}] Error retrieving batch at offset {offset}: {e}"
            )
            await client.release_collection(prepared_name)
            break

    await client.release_collection(prepared_name)
    sly.logger.debug(
        f"[Project: {collection_name}] Retrieved {len(items_info)} points from Milvus collection."
    )

    return items_info, vectors


@ensure_connection
@with_retries()
@timeit
@memoryit
async def get_items_by_id(
    collection_name: str,
    item_ids: List[int],
    with_vectors: bool = False,
    objects: bool = False,
) -> Tuple[List[Union[ImageInfoLite, ObjectInfoLite]], List[np.ndarray]]:
    """Get vectors from the collection based on the item IDs.

    :param collection_name: The name of the collection to get vectors from.
    :type collection_name: str
    :param item_ids: A list of item IDs to retrieve from the collection.
    :type item_ids: List[int]
    :param with_vectors: Whether to return vectors along with ImageInfoLite objects, defaults to False.
    :type with_vectors: bool, optional
    :param objects: If True, return object embeddings instead of image embeddings.
    :type objects: bool, optional
    :return: A tuple of ImageInfoLite or ObjectInfoLite objects and vectors.
    :rtype: Tuple[List[Union[ImageInfoLite, ObjectInfoLite]], List[np.ndarray]]
    """
    prepared_name = prepare_name(collection_name)

    partition_names = None
    if objects:
        partition_names = [MilvusParams.OBJECTS]

    output_fields = [
        MilvusFields.ID,
        MilvusFields.DATASET_ID,
        MilvusFields.FULL_URL,
        MilvusFields.CAS_URL,
    ]
    if with_vectors:
        output_fields.append(MilvusFields.VECTOR)
    if objects:
        output_fields.extend([MilvusFields.IMAGE_ID, MilvusFields.CLASS_ID])

    await client.load_collection(prepared_name)

    results = await client.get(
        collection_name=prepared_name,
        ids=item_ids,
        output_fields=output_fields,
        partition_names=partition_names,
    )

    await client.release_collection(prepared_name)

    item_infos = []
    vectors = []

    for result in results:
        distance = result.get(MilvusFields.DISTANCE, 0)
        similarity_score = 1.0 - distance if distance is not None else 0.0
        if objects:
            item = ObjectInfoLite(
                id=result.get(MilvusFields.ID),
                image_id=result.get(MilvusFields.IMAGE_ID),
                dataset_id=result.get(MilvusFields.DATASET_ID),
                class_id=result.get(MilvusFields.CLASS_ID),
                full_url=result.get(MilvusFields.FULL_URL),
                cas_url=result.get(MilvusFields.CAS_URL),
                score=similarity_score,
            )
        else:
            item = ImageInfoLite(
                id=result.get(MilvusFields.ID),
                dataset_id=result.get(MilvusFields.DATASET_ID),
                full_url=result.get(MilvusFields.FULL_URL),
                cas_url=result.get(MilvusFields.CAS_URL),
                score=similarity_score,
            )
        item_infos.append(item)

        if with_vectors:
            vectors.append(result.get(MilvusFields.VECTOR, []))

    return item_infos, vectors


@ensure_connection
@with_retries()
async def delete_collection(collection_name: str) -> None:
    """Delete a collection with the specified name.

    :param collection_name: The name of the collection to delete.
    :type collection_name: str
    """
    sly.logger.debug(f"[Project: {collection_name}] Deleting Milvus collection...")

    try:
        await client.drop_collection(prepare_name(collection_name))
    except Exception as e:
        sly.logger.debug(f"[Project: {collection_name}] Unable to delete Milvus collection: {e}")
