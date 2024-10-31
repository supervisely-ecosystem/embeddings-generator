from math import sqrt
from random import choice
from typing import Any, Dict, List, Literal, Tuple, Union

import numpy as np
import supervisely as sly
from pympler import asizeof
from qdrant_client import AsyncQdrantClient, QdrantClient
from qdrant_client.http.exceptions import UnexpectedResponse
from qdrant_client.models import Batch, CollectionInfo, Distance, VectorParams
from sklearn.cluster import KMeans

import src.globals as g
from src.utils import ImageInfoLite, QdrantFields, TupleFields, timeit, with_retries

client = AsyncQdrantClient(g.qdrant_host)

try:
    sly.logger.info(f"Connecting to Qdrant at {g.qdrant_host}...")
    QdrantClient(g.qdrant_host).get_collections()
    sly.logger.info(f"Connected to Qdrant at {g.qdrant_host}")
except Exception as e:
    sly.logger.error(f"Failed to connect to Qdrant at {g.qdrant_host} with error: {e}")


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
    try:
        collection = await client.get_collection(collection_name)
        sly.logger.debug(f"Collection {collection_name} already exists.")
    except UnexpectedResponse:
        await client.create_collection(
            collection_name,
            vectors_config=VectorParams(size=size, distance=distance),
        )
        sly.logger.debug(f"Collection {collection_name} created.")
        collection = await client.get_collection(collection_name)
    return collection


@with_retries()
@timeit
async def get_items_by_ids(
    collection_name: str, image_ids: List[int], with_vectors: bool = False
) -> Union[List[ImageInfoLite], Tuple[List[ImageInfoLite], List[np.ndarray]]]:
    """Get vectors from the collection based on the specified image IDs.

    :param collection_name: The name of the collection to get vectors from.
    :type collection_name: str
    :param image_ids: A list of image IDs to get vectors for.
    :type image_ids: List[int]
    :param with_vectors: Whether to return vectors along with ImageInfoLite objects, defaults to False.
    :type with_vectors: bool, optional
    :return: A list of vectors.
    :rtype: List[np.ndarray]
    """
    points = await client.retrieve(
        collection_name, image_ids, with_payload=True, with_vectors=with_vectors
    )
    image_infos = [ImageInfoLite(point.id, **point.payload) for point in points]
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
    await client.upsert(
        collection_name,
        Batch(
            vectors=vectors,
            ids=[image_info.id for image_info in image_infos],
            payloads=get_payloads(image_infos),
        ),
    )

    if sly.is_development():
        # By default qdrant should overwrite vectors with the same ids
        # so this line is only needed to check if vectors were upserted correctly.
        # Do not use this in production since it will slow down the process.
        collecton_info = await client.get_collection(collection_name)
        sly.logger.debug(
            f"Collection {collection_name} has {collecton_info.points_count} vectors."
        )


@with_retries()
async def get_diff(
    collection_name: str, image_infos: List[ImageInfoLite]
) -> List[ImageInfoLite]:
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
    points = await client.retrieve(
        collection_name,
        [image_info.id for image_info in image_infos],
        with_payload=True,
    )
    sly.logger.debug(
        f"Retrieved {len(points)} points from collection {collection_name}"
    )

    diff = _diff(image_infos, points)

    sly.logger.debug(f"Found {len(diff)} points that need to be updated.")
    if sly.is_development():
        # To avoid unnecessary computations in production,
        # only log the percentage of points that need to be updated in development.
        percent = round(len(diff) / len(image_infos) * 100, 2)
        sly.logger.debug(
            f"From the total of {len(image_infos)} points, {len(diff)} points need to be updated. ({percent}%)"
        )

    return diff


@timeit
def _diff(
    image_infos: List[ImageInfoLite], points: List[Dict[str, Any]]
) -> List[ImageInfoLite]:
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
    points_dict = {point.id: point for point in points}

    for image_info in image_infos:
        point = points_dict.get(image_info.id)
        if (
            point is None
            or point.payload.get(TupleFields.UPDATED_AT) != image_info.updated_at
        ):
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
        {k: v for k, v in image_info._asdict().items() if k not in ignore_fields}
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
) -> Union[List[ImageInfoLite], Tuple[List[ImageInfoLite], List[np.ndarray]]]:
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
    :return: A list of ImageInfoLite objects and vectors if return_vectors is True.
    :rtype: Union[List[ImageInfoLite], Tuple[List[ImageInfoLite], List[np.ndarray]]]
    """
    points = await client.search(
        collection_name,
        query_vector,
        limit=limit,
        with_payload=True,
        with_vectors=return_vectors,
    )
    image_infos = [ImageInfoLite(point.id, **point.payload) for point in points]
    if return_vectors:
        vectors = [point.vector for point in points]
        return image_infos, vectors
    return image_infos


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
    if not limit:
        collection = await client.get_collection(collection_name)
        limit = collection.points_count
    points, _ = await client.scroll(
        collection_name, limit=limit, with_payload=True, with_vectors=True
    )
    sly.logger.debug(
        f"Retrieved {len(points)} points from collection {collection_name}."
    )
    return [ImageInfoLite(point.id, **point.payload) for point in points], [
        point.vector for point in points
    ]


@timeit
async def diverse(
    collection_name: str,
    num_images: int,
    method: str,
    option: str,
) -> List[ImageInfoLite]:
    """Generate a diverse population of images using the specified method.

    :param collection_name: The name of the collection to get items from.
    :type collection_name: str
    :param num_images: The number of diverse images to generate.
    :type num_images: int
    :param method: The method to use for generating diverse images.
    :type method: str
    :param option: Option is an additional parameter for the method.
    :type option: str
    :raises ValueError: If the method is not supported.
    :return: A list of diverse images as ImageInfoLite objects.
    :rtype: List[ImageInfoLite]
    """
    if method == QdrantFields.KMEANS:
        return await diverse_kmeans(collection_name, num_images, option)
    else:
        raise ValueError(f"Method {method} is not supported.")


@timeit
async def diverse_kmeans(
    collection_name: str,
    num_images: int,
    option: Literal["random", "centroids"] = None,
    num_clusters: int = None,
) -> List[ImageInfoLite]:
    """Generate a diverse population of images using KMeans clustering.
    Two options are available: "random" and "centroids".
    The "random" option chooses a random image from each cluster.
    The "centroids" option chooses the image closest to the centroid of each cluster.

    :param collection_name: The name of the collection to get items from.
    :type collection_name: str
    :param num_images: The number of diverse images to generate.
    :type num_images: int
    :param option: The option to use for choosing images from clusters, defaults to None.
    :type option: Literal["random", "centroids"], optional
    :param num_clusters: The number of clusters to use in KMeans clustering.
    :type num_clusters: int
    :return: A list of diverse images as ImageInfoLite objects.
    :rtype: List[ImageInfoLite]
    """
    image_infos, vectors = await get_items(collection_name)
    if sly.is_development():
        vectors_size = asizeof.asizeof(vectors) / 1024 / 1024
        sly.logger.debug(f"Vectors size: {vectors_size:.2f} MB.")
    if not num_clusters:
        num_clusters = int(sqrt(len(image_infos) / 2))
        sly.logger.debug(f"Number of clusters is set to {num_clusters}.")
    if not option:
        option = QdrantFields.RANDOM
        sly.logger.debug(f"Option is set to {option}.")
    kmeans = KMeans(n_clusters=num_clusters).fit(vectors)
    labels = kmeans.labels_
    sly.logger.debug(
        f"KMeans clustering with {num_clusters} and {option} option is done."
    )

    diverse_images = []
    while len(diverse_images) < num_images:
        for cluster_id in set(labels):
            cluster_image_infos = [
                image_info
                for image_info, label in zip(image_infos, labels)
                if label == cluster_id
            ]
            if not cluster_image_infos:
                continue

            if option == QdrantFields.RANDOM:
                # Randomly choose an image from the cluster.
                image_info = choice(cluster_image_infos)
            elif option == QdrantFields.CENTROIDS:
                # Choose the image closest to the centroid of the cluster.
                cluster_vectors = [
                    vector
                    for vector, label in zip(vectors, labels)
                    if label == cluster_id
                ]
                centroid = np.mean(cluster_vectors, axis=0)
                distances = [
                    np.linalg.norm(vector - centroid) for vector in cluster_vectors
                ]
                image_info = cluster_image_infos[distances.index(min(distances))]
            diverse_images.append(image_info)
            image_infos.remove(image_info)
            if len(diverse_images) == num_images:
                break

    return diverse_images


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
    image_infos, vectors = await search(
        collection_name, vector, limit, return_vectors=True
    )
