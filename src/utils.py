import asyncio
import base64
import datetime
import hashlib
import json
import urllib.parse
import uuid
from dataclasses import dataclass
from functools import partial, wraps
from time import perf_counter
from typing import Callable, Dict, List, Literal, Optional, Union

import aiohttp
import supervisely as sly
from supervisely._utils import batched
from supervisely.api.app_api import SessionInfo
from supervisely.api.entities_collection_api import CollectionItem, CollectionType
from supervisely.api.module_api import ApiField

PROJECTIONS_SLUG = "supervisely-ecosystem/projections_service"
projections_task_map = {}


class TupleFields:
    """Fields of the named tuples used in the project."""

    ID = "id"
    HASH = "hash"
    LINK = "link"
    DATASET_ID = "dataset_id"
    PROJECT_ID = "project_id"
    FULL_URL = "full_url"
    CAS_URL = "cas_url"
    HDF5_URL = "hdf5_url"
    UPDATED_AT = "updated_at"
    UNIT_SIZE = "unitSize"
    URL = "url"
    THUMBNAIL = "thumbnail"
    ATLAS_ID = "atlasId"
    ATLAS_INDEX = "atlasIndex"
    VECTOR = "vector"
    IMAGES = "images"
    SCORE = "score"


class QdrantFields:
    """Fields for the queries to the Qdrant API."""

    KMEANS = "kmeans"
    NUM_CLUSTERS = "num_clusters"
    OPTION = "option"
    RANDOM = "random"
    CENTROIDS = "centroids"

    # Payload Fields
    DATASET_ID = "dataset_id"
    IMAGE_ID = "image_id"
    ID = "id"


class EventFields:
    """Fields of the event in request objects."""

    PROJECT_ID = "project_id"
    DATASET_ID = "dataset_id"
    TEAM_ID = "team_id"
    IMAGE_IDS = "image_ids"
    FORCE = "force"
    PROMPT = "prompt"
    LIMIT = "limit"
    METHOD = "method"
    REDUCTION_DIMENSIONS = "reduction_dimensions"
    SAMPLING_METHOD = "sampling_method"
    SAMPLE_SIZE = "sample_size"
    CLUSTERING_METHOD = "clustering_method"
    NUM_CLUSTERS = "num_clusters"
    SAVE = "save"
    RETURN_VECTORS = "return_vectors"
    THRESHOLD = "threshold"

    ATLAS = "atlas"
    POINTCLOUD = "pointcloud"

    # Search by fields
    BY_PROJECT_ID = "by_project_id"
    BY_DATASET_ID = "by_dataset_id"
    BY_IMAGE_IDS = "by_image_ids"

    # Event types
    SEARCH = "search"
    DIVERSE = "diverse"
    CLUSTERING = "clustering"
    EMBEDDINGS = "embeddings"


class SamplingMethods:
    """Sampling methods for the images."""

    RANDOM = "random"
    CENTROIDS = "centroids"


class ClusteringMethods:
    """Clustering methods for the images."""

    KMEANS = "kmeans"
    DBSCAN = "dbscan"


class ResponseFields:
    """Fields of the response file."""

    COLLECTION_ID = "collection_id"
    MESSAGE = "message"
    STATUS = "status"
    VECTORS = "vectors"
    IMAGE_IDS = "image_ids"
    BACKGROUND_TASK_ID = "background_task_id"
    RESULT = "result"
    IS_RUNNING = "is_running"
    PROGRESS = "progress"


class ResponseStatus:
    """Status of the response."""

    SUCCESS = "success"
    COMPLETED = "completed"
    ERROR = "error"
    IN_PROGRESS = "in_progress"
    NOT_FOUND = "not_found"
    NO_TASK = "no_task"
    CANCELLED = "cancelled"
    FAILED = "failed"
    RUNNING = "running"


class CustomDataFields:
    """Fields of the custom data."""

    EMBEDDINGS_UPDATE_STARTED_AT = "embeddings_update_started_at"


@dataclass
class ImageInfoLite:
    id: int
    dataset_id: int
    full_url: str
    cas_url: str
    updated_at: str  # or datetime.datetime if you parse it
    score: float = None

    def to_json(self):
        return {
            TupleFields.ID: self.id,
            TupleFields.DATASET_ID: self.dataset_id,
            TupleFields.FULL_URL: self.full_url,
            TupleFields.CAS_URL: self.cas_url,
            TupleFields.UPDATED_AT: self.updated_at,
            TupleFields.SCORE: self.score,
        }
        # Alternative: return asdict(self)  # if field names match keys

    @classmethod
    def from_json(cls, data: dict):
        return cls(
            id=data[TupleFields.ID],
            dataset_id=data[TupleFields.DATASET_ID],
            full_url=data[TupleFields.FULL_URL],
            cas_url=data[TupleFields.CAS_URL],
            updated_at=data[TupleFields.UPDATED_AT],
            score=data.get(TupleFields.SCORE, None),
        )


def timeit(func: Callable) -> Callable:
    """Decorator to measure the execution time of the function.
    Works with both async and sync functions.

    :param func: Function to measure the execution time of.
    :type func: Callable
    :return: Decorated function.
    :rtype: Callable
    """

    if asyncio.iscoroutinefunction(func):

        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            start_time = perf_counter()
            result = await func(*args, **kwargs)
            end_time = perf_counter()
            execution_time = end_time - start_time
            _log_execution_time(func.__name__, execution_time)
            return result

        return async_wrapper
    else:

        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            start_time = perf_counter()
            result = func(*args, **kwargs)
            end_time = perf_counter()
            execution_time = end_time - start_time
            _log_execution_time(func.__name__, execution_time)
            return result

        return sync_wrapper


def _log_execution_time(function_name: str, execution_time: float) -> None:
    """Log the execution time of the function.

    :param function_name: Name of the function.
    :type function_name: str
    :param execution_time: Execution time of the function.
    :type execution_time: float
    """
    sly.logger.debug("%.4f sec | %s", execution_time, function_name)


def to_thread(func: Callable) -> Callable:
    """Decorator to run the function in a separate thread.
    Can be used for slow synchronous functions inside of the asynchronous code
    to avoid blocking the event loop.

    :param func: Function to run in a separate thread.
    :type func: Callable
    :return: Decorated function.
    :rtype: Callable
    """

    # For Python 3.9+.
    # @wraps(func)
    # def wrapper(*args, **kwargs):
    #     return asyncio.to_thread(func, *args, **kwargs)

    # For Python 3.7+.
    @wraps(func)
    def wrapper(*args, **kwargs):
        loop = asyncio.get_event_loop()
        func_with_args = partial(func, *args, **kwargs)
        return loop.run_in_executor(None, func_with_args)

    return wrapper


def with_retries(retries: int = 3, sleep_time: int = 1, on_failure: Callable = None) -> Callable:
    """Decorator to retry the function in case of an exception.
    Works only with async functions. Custom function can be executed on failure.
    NOTE: The on_failure function should be idempotent and synchronous.

    :param retries: Number of retries.
    :type retries: int
    :param sleep_time: Time to sleep between retries.
    :type sleep_time: int
    :param on_failure: Function to execute on failure, if None, raise an exception.
    :type on_failure: Callable, optional
    :raises Exception: If the function fails after all retries.
    :return: Decorator.
    :rtype: Callable
    """

    def retry_decorator(func):
        @wraps(func)
        async def async_function_with_retries(*args, **kwargs):
            for _ in range(retries):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    sly.logger.debug(
                        "Failed to execute %s, retrying. Error: %s", func.__name__, str(e)
                    )
                    await asyncio.sleep(sleep_time)
            if on_failure is not None:
                return on_failure()
            else:
                raise RuntimeError(f"Failed to execute {func.__name__} after {retries} retries.")

        return async_function_with_retries

    return retry_decorator


@to_thread
@timeit
def get_datasets(api: sly.Api, project_id: int, recursive: bool = False) -> List[sly.DatasetInfo]:
    """Returns list of datasets from the project.

    :param api: Instance of supervisely API.
    :type api: sly.Api
    :param project_id: ID of the project to get datasets from.
    :type project_id: int
    :return: List of datasets.
    :rtype: List[sly.DatasetInfo]
    """
    return api.dataset.get_list(project_id, recursive=recursive)


@to_thread
@timeit
def get_project_info(api: sly.Api, project_id: int) -> sly.ProjectInfo:
    """Returns project info by ID.

    :param api: Instance of supervisely API.
    :type api: sly.Api
    :param project_id: ID of the project to get info.
    :type project_id: int
    :return: Project info.
    :rtype: sly.ProjectInfo
    """
    return api.project.get_info_by_id(project_id)


@to_thread
@timeit
def get_team_info(api: sly.Api, team_id: int) -> sly.TeamInfo:
    """Returns team info by ID.

    :param api: Instance of supervisely API.
    :type api: sly.Api
    :param team_id: ID of the team to get info.
    :type team_id: int
    :return: Team info.
    :rtype: sly.TeamInfo
    """
    return api.team.get_info_by_id(team_id)


def _get_project_info_by_name(
    api: sly.Api, workspace_id: int, project_name: str
) -> sly.ProjectInfo:
    return api.project.get_info_by_name(workspace_id, project_name)


@to_thread
@timeit
def get_project_info_by_name(api: sly.Api, workspace_id: int, project_name: str) -> sly.ProjectInfo:
    """Returns project info by name.

    :param api: Instance of supervisely API.
    :type api: sly.Api
    :param workspace_id: ID of the workspace to get project from.
    :type workspace_id: int
    :param project_name: Name of the project to get info.
    :type project_name: str
    :return: Project info.
    :rtype: sly.ProjectInfo
    """
    return _get_project_info_by_name(api, workspace_id, project_name)


def _get_or_create_project(
    api: sly.Api, workspace_id: int, project_name: str, project_type: str
) -> sly.ProjectInfo:
    project_info = _get_project_info_by_name(api, workspace_id, project_name)
    if project_info is None:
        project_info = api.project.create(workspace_id, project_name, project_type)
    return project_info


@to_thread
@timeit
def get_or_create_project(
    api: sly.Api, workspace_id: int, project_name: str, project_type: str
) -> sly.ProjectInfo:
    return _get_or_create_project(api, workspace_id, project_name, project_type)


@to_thread
@timeit
def get_dataset_by_name(api: sly.Api, project_id: int, dataset_name: str) -> sly.DatasetInfo:
    return api.dataset.get_info_by_name(project_id, name=dataset_name)


@to_thread
@timeit
def get_or_create_dataset(api: sly.Api, project_id: int, dataset_name: str) -> sly.DatasetInfo:
    dataset_info = api.dataset.get_info_by_name(project_id, name=dataset_name)
    if dataset_info is None:
        dataset_info = api.dataset.create(project_id, dataset_name)
    return dataset_info


@to_thread
@timeit
def get_pcd_by_name(
    api: sly.Api, dataset_id: int, pcd_name: str
) -> sly.api.pointcloud_api.PointcloudInfo:
    return api.pointcloud.get_info_by_name(dataset_id, pcd_name)


@to_thread
@timeit
def set_image_embeddings_updated_at(
    api: sly.Api,
    image_infos: List[Union[sly.ImageInfo, ImageInfoLite]],
    timestamps: Optional[List[str]] = None,
):
    """Sets the embeddings updated at timestamp for the images."""
    ids = [image_info.id for image_info in image_infos]
    ids = list(set(ids))
    api.image.set_embeddings_updated_at(ids, timestamps)


@to_thread
@timeit
def set_project_embeddings_updated_at(
    api: sly.Api,
    project_id: int,
    timestamp: str = None,
):
    """Sets the embeddings updated at timestamp for the project."""
    api.project.set_embeddings_updated_at(project_id, timestamp)


@to_thread
@timeit
def get_project_embeddings_updated_at(api: sly.Api, project_id: int) -> Optional[str]:
    """Gets the embeddings updated at timestamp for the project."""
    project_info = api.project.get_info_by_id(project_id)
    return project_info.embeddings_updated_at


@to_thread
@timeit
def set_embeddings_in_progress(
    api: sly.Api, project_id: int, in_progress: bool, error_message: Optional[str] = None
):
    """Sets the embeddings in progress flag for the project."""
    api.project.set_embeddings_in_progress(
        id=project_id, in_progress=in_progress, error_message=error_message
    )


@to_thread
@timeit
def get_team_file_info(api: sly.Api, team_id: int, path: str):
    return api.file.get_info_by_path(team_id, path)


def resize_image_url(
    full_storage_url: str,
    imgproxy_address: Optional[str] = None,
    ext: Literal["jpeg", "png"] = "jpeg",
    method: Literal["fit", "fill", "fill-down", "force", "auto"] = "auto",
    width: int = 0,
    height: int = 0,
    quality: int = 70,
) -> str:
    """Returns a URL to a resized image with given parameters.
    Default sizes are 0, which means that the image will not be resized,
    just compressed if the extension is jpeg to the given quality.
    Learn more about resize parameters `here <https://docs.imgproxy.net/usage/processing#resize>`_.

    :param full_storage_url: Full Image storage URL, can be obtained from ImageInfo.
    :type full_storage_url: str
    :param ext: Image extension, jpeg or png.
    :type ext: Literal["jpeg", "png"], optional
    :param method: Resize type, fit, fill, fill-down, force, auto.
    :type method: Literal["fit", "fill", "fill-down", "force", "auto"], optional
    :param width: Width of the resized image.
    :type width: int, optional
    :param height: Height of the resized image.
    :type height: int, optional
    :param quality: Quality of the resized image.
    :type quality: int, optional
    :return: Full URL to a resized image.
    :rtype: str

    :Usage example:

    .. code-block:: python

        import supervisely as sly
        from supervisely_utils import resize_image_url

        api = sly.Api(server_address, token)

        image_id = 376729
        img_info = api.image.get_info_by_id(image_id)

        img_resized_url = resize_image_url(
            img_info.full_storage_url, ext="jpeg", method="fill", width=512, height=256)
        print(img_resized_url)
        # Output: https://app.supervisely.com/previews/q/ext:jpeg/resize:fill:512:256:0/q:70/plain/h5un6l2bnaz1vj8a9qgms4-public/images/original/2/X/Re/<image_name>.jpg
    """
    # original url example: https://app.supervisely.com/h5un6l2bnaz1vj8a9qgms4-public/images/original/2/X/Re/<image_name>.jpg
    # resized url example:  https://app.supervisely.com/previews/q/ext:jpeg/resize:fill:300:0:0/q:70/plain/h5un6l2bnaz1vj8a9qgms4-public/images/original/2/X/Re/<image_name>.jpg
    # to add: previews/q/ext:jpeg/resize:fill:300:0:0/q:70/plain/
    try:
        parsed_url = urllib.parse.urlparse(full_storage_url)
        server_address = f"{parsed_url.scheme}://{parsed_url.netloc}"

        resize_string = f"q/ext:{ext}/resize:{method}:{width}:{height}:0/q:{quality}/plain"
        # Determine base address for imgproxy
        if imgproxy_address:
            # Remove trailing slash if present
            imgproxy_address = imgproxy_address.rstrip("/")
            base_address = imgproxy_address
        else:
            base_address = f"{server_address}/previews"

        url = full_storage_url.replace(server_address, f"{base_address}/{resize_string}")
        return url
    except Exception as e:
        sly.logger.debug(f"Failed to resize image with url: {full_storage_url}: {repr(e)}")
        return full_storage_url


@timeit
async def create_lite_image_infos(
    cas_size: int,
    image_infos: List[sly.ImageInfo],
    imgproxy_address: Optional[str] = None,
) -> List[ImageInfoLite]:
    """Returns lite version of image infos to cut off unnecessary data.

    :param cas_size: Size of the image for CLIP, it will be added to URL.
    :type cas_size: int
    :param image_infos: List of image infos to get lite version from.
    :type image_infos: List[sly.ImageInfo]
    :param imgproxy_address: Imgproxy address to use for resizing images, if None, will use the full storage URL.
    :type imgproxy_address: Optional[str], optional
    :return: List of lite version of image infos.
    :rtype: List[ImageInfoLite]
    """
    if imgproxy_address is not None:
        sly.logger.debug(
            "Imgproxy address is set to %s while creating lite image infos", imgproxy_address
        )
    if not image_infos or len(image_infos) == 0:
        return []
    if isinstance(image_infos[0], ImageInfoLite):
        return image_infos
    images_list = []
    for image_info in image_infos:
        cas_url = resize_image_url(
            image_info.full_storage_url,
            imgproxy_address=imgproxy_address,
            method="fit",
            width=cas_size,
            height=cas_size,
        )
        images_list.append(
            ImageInfoLite(
                id=image_info.id,
                dataset_id=image_info.dataset_id,
                full_url=image_info.full_storage_url,
                cas_url=cas_url,
                updated_at=image_info.updated_at,
            )
        )
    return images_list


@timeit
async def get_lite_image_infos(
    api: sly.Api,
    cas_size: int,
    project_id: int,
    dataset_id: int = None,
    image_ids: List[int] = None,
    image_infos: List[sly.ImageInfo] = None,
    imgproxy_address: Optional[str] = None,
) -> List[ImageInfoLite]:
    """Returns lite version of image infos to cut off unnecessary data.
    Uses either dataset_id or image_ids to get image infos.
    If dataset_id is provided, it will be used to get all images from the dataset.
    If image_ids are provided, they will be used to get image infos.

    :param api: Instance of supervisely API.
    :type api: sly.Api
    :param cas_size: Size of the image for CLIP, it will be added to URL.
    :type cas_size: int
    :param project_id: ID of the project to get images from.
    :type project_id: int
    :param dataset_id: ID of the dataset to get images from.
    :type dataset_id: int, optional
    :param image_ids: List of image IDs to get image infos.
    :type image_ids: List[int], optional
    :param image_infos: List of image infos to get lite version from.
    :type image_infos: List[sly.ImageInfo], optional
    :param imgproxy_address: Imgproxy address to use for resizing images, if None, will use the full storage URL.
    :type imgproxy_address: Optional[str], optional
    :return: List of lite version of image infos.
    :rtype: List[ImageInfoLite]
    """
    if not image_infos or len(image_infos) == 0:
        image_infos = await image_get_list_async(api, project_id, dataset_id, image_ids)

    if len(image_infos) == 0:
        return []
    image_infos = await create_lite_image_infos(
        cas_size, image_infos, imgproxy_address=imgproxy_address
    )
    return image_infos


def parse_timestamp(
    timestamp: str, timestamp_format: str = "%Y-%m-%dT%H:%M:%S.%fZ"
) -> datetime.datetime:
    """
    Parse timestamp string to datetime object.
    Timestamp format: "2021-01-22T19:37:50.158Z".
    """
    return datetime.datetime.strptime(timestamp, timestamp_format)


async def send_request(
    api: sly.Api,
    task_id: int,
    method: str,
    data: Dict,
    context: Optional[Dict] = None,
    skip_response: bool = False,
    timeout: Optional[int] = 60,
    outside_request: bool = True,
    retries: int = 10,
    raise_error: bool = False,
):
    """send_request"""
    if type(data) is not dict:
        raise TypeError("data argument has to be a dict")
    if context is None:
        context = {}
    context["outside_request"] = outside_request
    resp = await api.post_async(
        "tasks.request.direct",
        {
            ApiField.TASK_ID: task_id,
            ApiField.COMMAND: method,
            ApiField.CONTEXT: context,
            ApiField.STATE: data,
            "skipResponse": skip_response,
            "timeout": timeout,
        },
        retries=retries,
        raise_error=raise_error,
    )
    return resp.json()


async def base64_from_url(api: sly.Api, url):
    api._set_async_client()
    r = await api.async_httpx_client.get(url)
    b = bytes()
    async for chunk in r.aiter_bytes():
        b += chunk
    img_base64 = base64.b64encode(b)
    data_url = f"data:image/png;base64,{str(img_base64, 'utf-8')}"
    return data_url


@timeit
def fix_vectors(vectors_batch):
    for i, vector in enumerate(vectors_batch):
        for j, value in enumerate(vector):
            if not isinstance(value, float):
                sly.logger.debug(
                    "Value %s is not of type float: %s. Converting to float.", value, type(value)
                )
                vectors_batch[i][j] = float(value)
    return vectors_batch


async def run_safe(func, *args, **kwargs):
    try:
        return await func(*args, **kwargs)
    except Exception as e:
        sly.logger.error("Error in function %s: %s", func.__name__, e, exc_info=True)
        return None


def get_filter_images_wo_embeddings() -> Dict:
    """Create filter to get images that dont have embeddings.

    :return: Dictionary representing the filter.
    :rtype: Dict
    """
    return {
        ApiField.FIELD: ApiField.EMBEDDINGS_UPDATED_AT,
        ApiField.OPERATOR: "eq",
        ApiField.VALUE: None,
    }


def get_filter_deleted_after(timestamp: str) -> Dict:
    """Create filter to get images deleted after a specific date.

    :return: Dictionary representing the filter.
    :rtype: Dict
    """
    return {
        ApiField.FIELD: ApiField.UPDATED_AT,
        ApiField.OPERATOR: "gt",
        ApiField.VALUE: timestamp,
    }


@timeit
async def image_get_list_async(
    api: sly.Api,
    project_id: int,
    dataset_id: int = None,
    image_ids: List[int] = None,
    per_page: int = 1000,
    wo_embeddings: Optional[bool] = False,
    deleted_after: Optional[str] = None,
) -> List[sly.ImageInfo]:
    """
    Get list of images from the project or dataset.
     - If `image_ids` is provided, it will return only those images.
     - If `dataset_id` is provided, it will return images from that dataset.
     - If neither `dataset_id` nor `image_ids` is provided, it will return all images from the project.
     - If `wo_embeddings` is True, it will return only images without embeddings.
     - If `deleted_after` is provided, it will return only images that were updated after that date.

    :param api: Instance of supervisely API.
    :type api: sly.Api
    :param project_id: ID of the project to get images from.
    :type project_id: int
    :param dataset_id: ID of the dataset to get images from. If None, will get images from the whole project.
    :type dataset_id: int, optional
    :param image_ids: List of image IDs to get images from. If None, will get all images.
    :type image_ids: List[int], optional
    :param per_page: Number of images to return per page. Default is 1000.
    :type per_page: int
    :param wo_embeddings: If True, will return only images without embeddings. Default is False.
    :type wo_embeddings: bool, optional
    :param deleted_after: If provided, will return only images that were updated after this date.
    :type deleted_after: str, optional
    :return: List of images from the project or dataset.
    :rtype: List[sly.ImageInfo]
    :raises ValueError: If both `wo_embeddings` and `deleted_after` are set to True.
    """
    method = "images.list"
    base_data = {
        ApiField.PROJECT_ID: project_id,
        ApiField.FORCE_METADATA_FOR_LINKS: False,
        ApiField.PER_PAGE: per_page,
    }

    if dataset_id is not None:
        base_data[ApiField.DATASET_ID] = dataset_id

    if wo_embeddings and deleted_after:
        raise ValueError("Both created_after and deleted_after cannot be set at the same time.")
    if wo_embeddings:
        base_data[ApiField.FILTER] = [get_filter_images_wo_embeddings()]
    if deleted_after is not None:
        if ApiField.FILTER not in base_data:
            base_data[ApiField.FILTER] = []
        base_data[ApiField.FILTER].append(get_filter_deleted_after(deleted_after))
        base_data[ApiField.SHOW_DISABLED] = True

    semaphore = api.get_default_semaphore()
    all_items = []
    tasks = []

    async def _get_all_pages(ids_filter: List[Dict]):
        page_data = base_data.copy()
        if ids_filter:
            if ApiField.FILTER not in page_data:
                page_data[ApiField.FILTER] = []
            page_data[ApiField.FILTER].extend(ids_filter)

        page_data[ApiField.PAGE] = 1
        first_response = await api.post_async(method, page_data)
        first_response_json = first_response.json()

        total_pages = first_response_json.get("pagesCount", 1)
        batch_items = []

        entities = first_response_json.get("entities", [])
        for item in entities:
            image_info = api.image._convert_json_info(item)
            batch_items.append(image_info)

        if total_pages > 1:

            async def fetch_page(page_num):
                page_data_copy = page_data.copy()
                page_data_copy[ApiField.PAGE] = page_num

                async with semaphore:
                    response = await api.post_async(method, page_data_copy)
                    response_json = response.json()

                    page_items = []
                    entities = response_json.get("entities", [])
                    for item in entities:
                        image_info = api.image._convert_json_info(item)
                        page_items.append(image_info)

                    return page_items

            # Create tasks for all remaining pages
            tasks = []
            for page_num in range(2, total_pages + 1):
                tasks.append(asyncio.create_task(fetch_page(page_num)))

            page_results = await asyncio.gather(*tasks)

            for page_items in page_results:
                batch_items.extend(page_items)

        return batch_items

    if image_ids is None:
        # If no image IDs specified, get all images
        tasks.append(asyncio.create_task(_get_all_pages([])))
    else:
        # Process image IDs in batches of 50
        for batch in batched(image_ids):
            ids_filter = [
                {ApiField.FIELD: ApiField.ID, ApiField.OPERATOR: "in", ApiField.VALUE: batch}
            ]
            tasks.append(asyncio.create_task(_get_all_pages(ids_filter)))
            await asyncio.sleep(0.02)  # Small delay to avoid overwhelming the server

    # Wait for all tasks to complete
    batch_results = await asyncio.gather(*tasks)

    # Combine results from all batches
    for batch_items in batch_results:
        all_items.extend(batch_items)

    return all_items


async def embeddings_up_to_date(
    api: sly.Api, project_id: int, project_info: Optional[sly.ProjectInfo] = None
):
    if project_info is None:
        project_info = await get_project_info(api, project_id)
    if project_info is None:
        return None
    if project_info.embeddings_updated_at is None:
        return False
    images_to_create = await image_get_list_async(api, project_id, wo_embeddings=True)
    if len(images_to_create) > 0:
        return False
    return True


async def get_list_all_pages_async(
    api: sly.Api,
    method,
    data,
    convert_json_info_cb,
    progress_cb=None,
    limit: int = None,
    return_first_response: bool = False,
    semaphore: Optional[List[asyncio.Semaphore]] = None,
):
    """
    Get list of all or limited quantity entities from the Supervisely server.
    """
    from copy import deepcopy

    def _add_sort_param(data):
        """_add_sort_param"""
        results = deepcopy(data)
        results[ApiField.SORT] = ApiField.ID
        results[ApiField.SORT_ORDER] = "asc"  # @TODO: move to enum
        return results

    convert_func = convert_json_info_cb

    if ApiField.SORT not in data:
        data = _add_sort_param(data)
    if semaphore is None:
        semaphore = api.get_default_semaphore()

    # Get first page to determine pagination details
    first_page_data = {**data, "page": 1}
    first_response = await api.post_async(method, first_page_data)
    first_response_json = first_response.json()

    total = first_response_json["total"]
    per_page = first_response_json["perPage"]
    pages_count = first_response_json["pagesCount"]

    results = first_response_json["entities"]
    if progress_cb is not None:
        progress_cb(len(results))

    # If only one page or limit is already exceeded with first page
    if (pages_count == 1 and len(results) == total) or (
        limit is not None and len(results) >= limit
    ):
        if limit is not None:
            results = results[:limit]
        if return_first_response:
            return [convert_func(item) for item in results], first_response_json
        return [convert_func(item) for item in results]

    # Process remaining pages concurrently
    async def fetch_page(page_num):
        async with semaphore:
            page_data = {**data, "page": page_num, "per_page": per_page}
            response = await api.post_async(method, page_data)
            response_json = response.json()
            page_items = response_json.get("entities", [])
            if progress_cb is not None:
                progress_cb(len(page_items))
            return page_items

    # Create tasks for all remaining pages
    tasks = []
    for page_num in range(2, pages_count + 1):
        tasks.append(asyncio.create_task(fetch_page(page_num)))

    # Wait for all tasks to complete
    for task in asyncio.as_completed(tasks):
        page_items = await task
        results.extend(page_items)
        if limit is not None and len(results) >= limit:
            break

    if len(results) != total and limit is None:
        raise RuntimeError(f"Method {method!r}: error during pagination, some items are missed")

    if limit is not None:
        results = results[:limit]

    return [convert_func(item) for item in results]


@timeit
async def get_all_projects(
    api: sly.Api,
    project_ids: Optional[List[int]] = None,
) -> List[sly.ProjectInfo]:
    """
    Get all projects from the Supervisely server that have a flag for automatic embeddings update.

    Fields that will be returned:
        - id
        - name
        - updated_at
        - embeddings_enabled
        - embeddings_in_progress
        - embeddings_updated_at
        - team_id
        - workspace_id
        - items_count

    """
    method = "projects.list.all"
    convert_json_info = api.project._convert_json_info
    fields = [
        ApiField.EMBEDDINGS_ENABLED,
        ApiField.EMBEDDINGS_IN_PROGRESS,
        ApiField.EMBEDDINGS_UPDATED_AT,
    ]
    data = {
        ApiField.SKIP_EXPORTED: True,
        ApiField.EXTRA_FIELDS: fields,
        ApiField.FILTER: [
            {
                ApiField.FIELD: ApiField.EMBEDDINGS_ENABLED,
                ApiField.OPERATOR: "=",
                ApiField.VALUE: True,
            }
        ],
    }
    tasks = []
    if project_ids is not None:
        for batch in batched(project_ids):
            data[ApiField.FILTER] = [
                {
                    ApiField.FIELD: ApiField.ID,
                    ApiField.OPERATOR: "in",
                    ApiField.VALUE: batch,
                },
                {
                    ApiField.FIELD: ApiField.EMBEDDINGS_ENABLED,
                    ApiField.OPERATOR: "=",
                    ApiField.VALUE: True,
                },
            ]
            tasks.append(
                get_list_all_pages_async(
                    api,
                    method,
                    data=data,
                    convert_json_info_cb=convert_json_info,
                    progress_cb=None,
                    limit=None,
                    return_first_response=False,
                )
            )
    else:
        tasks.append(
            get_list_all_pages_async(
                api,
                method,
                data=data,
                convert_json_info_cb=convert_json_info,
                progress_cb=None,
                limit=None,
                return_first_response=False,
            )
        )
    results = []
    for task in asyncio.as_completed(tasks):
        results.extend(await task)
    return results


def get_key(prompt: str, project_id: str, settings: Dict) -> str:
    """
    Generate a unique hash key for a search request.

    :param prompt: Prompt for the search request.
    :type prompt: str
    :param project_id: ID of the project.
    :type project_id: str
    :param settings: Settings for the search request.
    :type settings: Dict
    :return: Unique hash key for the search request.
    :rtype: str
    """
    cache_data = {"prompt": prompt, "project_id": project_id, "settings": settings}
    serialized = json.dumps(cache_data, sort_keys=True)
    return hashlib.md5(serialized.encode()).hexdigest()


def generate_key(project_id: int, user_token: str) -> str:
    """
    Generate a unique hash key based on the provided parameters.

    Parameters that must be used to generate the key:
     - project_id
     - user_token
    """
    params = {"project_id": project_id, "user_token": user_token}
    serialized = json.dumps(params, sort_keys=True)
    return hashlib.md5(serialized.encode()).hexdigest()


@to_thread
def create_collection_and_populate(
    api: sly.Api,
    project_id: int,
    name: str,
    image_ids: List[int],
    event: EventFields,
    collection_type: str = CollectionType.AI_SEARCH,
    ai_search_key: str = None,
) -> int:
    """Create Entities Collection for project.

    **NOTE**: For events CLUSTERING, DIVERSE, and EMBEDDINGS, collection will be recreated if it already exists.

    :param api: Instance of supervisely API.
    :type api: sly.Api
    :param project_id: ID of the project to create collection in.
    :type project_id: int
    :param name: Name of the collection.
    :type name: str
    :param image_ids: List of image IDs to populate the collection.
    :type image_ids: List[int]
    :param event: Event type to determine default AI Search key.
                For diverse and clustering search, AI Search key will be the same and automatically generated inside this function.
                For prompt search, must be generated and passed to the function.
    :type event: EventFields
    :param collection_type: Type of the collection.
    :type collection_type: str
    :param ai_search_key: AI search key for the collection.
    :type ai_search_key: str, optional
    :return: ID of the created collection.
    :rtype: int
    """
    if event in [EventFields.DIVERSE, EventFields.CLUSTERING, EventFields.EMBEDDINGS]:
        # Generate AI search key for diverse and clustering search which will be the same
        ai_search_key = get_key(event, project_id, {"event": event})

        # Remove existing collection with the same AI search key
        while True:
            collection_info = api.entities_collection.get_info_by_ai_search_key(
                project_id, ai_search_key
            )
            if collection_info is None:
                break
            api.entities_collection.remove(collection_info.id)

    collection_id = api.entities_collection.create(
        project_id=project_id,
        name=name,
        type=collection_type,
        ai_search_key=ai_search_key,
    ).id
    items = [
        CollectionItem(entity_id=image_id, meta=CollectionItem.Meta(score=1))
        for image_id in image_ids
    ]
    api.entities_collection.add_items(collection_id, items)
    return collection_id


def hash_to_uuid(image_hash: str) -> uuid.UUID:
    """Converts a base64-encoded image hash to a UUID."""
    raw_bytes = base64.b64decode(image_hash)
    if len(raw_bytes) != 32:
        raise ValueError("Expected 32-byte hash input")

    selected_bytes = raw_bytes[:8] + raw_bytes[-8:]
    return uuid.UUID(bytes=selected_bytes)


def link_to_uuid(image_link: str) -> uuid.UUID:
    """Converts a string represented image link to a UUID."""
    # Create a deterministic UUID based on the image link using uuid5 with a namespace
    # Using URL namespace for links
    return uuid.uuid5(uuid.NAMESPACE_URL, image_link)


def _start_projections_service(
    api: sly.Api,
    module_id: int,
    workspace_id: int,
) -> SessionInfo:
    """Starts the projections service app."""
    session = api.app.start(
        agent_id=None,
        module_id=module_id,
        workspace_id=workspace_id,
    )
    api.app.wait(session.task_id, target_status=sly.task.Status.STARTED)
    return session


@with_retries()
@to_thread
def start_projections_service(api: sly.Api, project_id: int):
    msg_prefix = f"[Project: {project_id}]"
    try:
        e_msg = ""
        module_info = api.app.get_ecosystem_module_info(slug=PROJECTIONS_SLUG)
    except Exception as e:
        e_msg = f"Error: {str(e)}"
        module_info = None
    if module_info is None:
        raise RuntimeError(
            f"{msg_prefix} Projections service module not found in ecosystem. {e_msg}"
        )
    project = api.project.get_info_by_id(project_id)
    team_id = project.team_id
    workspace_id = project.workspace_id

    # Check if the task is already in projections_task_map
    if team_id in projections_task_map:
        task_ids = projections_task_map[team_id]
        for task_id in task_ids:
            task_status = api.task.get_status(task_id)
            if task_status == api.task.Status.STARTED:
                return task_id
            elif task_status == api.task.Status.QUEUED:
                task_info = api.task.get_info_by_id(task_id)

                if task_info.meta.get("retries", 0) == 0:
                    created_at = parse_timestamp(task_info.created_at)
                    elapsed_time = (
                        datetime.datetime.now(datetime.timezone.utc) - created_at
                    ).total_seconds()

                    if elapsed_time > 120:  # Task created more than 2 minutes ago
                        projections_task_map[team_id].remove(task_id)
                    else:  # Task created less than 2 minutes ago
                        # Wait 10 seconds for the task to start
                        try:
                            api.app.wait(
                                task_id,
                                target_status=api.task.Status.STARTED,
                                attempts=1,
                                attempt_delay_sec=20,
                            )
                            if api.app.wait_until_ready_for_api_calls(task_id):
                                return task_id
                        except Exception as e:
                            sly.logger.debug(
                                f"{msg_prefix} Error waiting for task {task_id} to start: {str(e)}"
                            )
                            # If the task is still not started, remove it from the map
                            projections_task_map[team_id].remove(task_id)
                            continue
                        else:
                            # If the task started, return its ID
                            return task_id

    # If no task is found, start a new projections service
    sly.logger.debug(f"{msg_prefix} Starting Projections service for team ID {team_id}.")

    sessions = api.app.get_sessions(team_id, module_info.id, statuses=[api.task.Status.STARTED])
    me = api.user.get_my_info()
    sessions = [s for s in sessions if s.user_id == me.id]
    if len(sessions) == 0:
        session = _start_projections_service(api, module_info.id, workspace_id)
    else:
        session = sessions[0]

    ready = api.app.wait_until_ready_for_api_calls(session.task_id)
    if not ready:
        sly.logger.debug(f"{msg_prefix} Restarting Projections service...")
        session = _start_projections_service(api, module_info.id, workspace_id)
        ready = api.app.wait_until_ready_for_api_calls(session.task_id)
        if not ready:
            raise RuntimeError(
                f"{msg_prefix} Projections service is not ready for API calls after restart"
            )

    # Add the task to projections_task_map
    if projections_task_map.get(team_id, None) is None:
        projections_task_map[team_id] = []
    projections_task_map[team_id].append(session.task_id)

    return session.task_id


@with_retries()
@to_thread
def stop_projections_service(api: sly.Api, task_id: int):
    status = api.task.stop(task_id)
    return status


@to_thread
@timeit
def is_team_plan_sufficient(api: sly.Api, team_id: int) -> bool:
    """Check if the team has a usage plan that allows for embeddings.

    :param api: Instance of supervisely API.
    :type api: sly.Api
    :param team_id: ID of the team to check.
    :type team_id: int
    :return: True if the team has a usage plan that allows for embeddings, False otherwise.
    :rtype: bool
    """
    team_info = api.team.get_info_by_id(team_id)

    # If usage is None or plan is None, allow embeddings
    if team_info.usage is None or team_info.usage.plan is None:
        return True

    return team_info.usage.plan != "free"


def get_app_host(api: sly.Api, slug: str, net_server_address: str = None) -> str:
    """Get the app host URL from the Supervisely API.

    :param api: Instance of supervisely API.
    :type api: sly.Api
    :param slug: Slug of the app to get the host URL for.
    :type slug: str
    :param net_server_address: Optional server address for the app. If not provided, uses
        the server address from the API instance.
    :type net_server_address: str, optional
    :return: The app host URL.
    :rtype: str
    """

    server_address = net_server_address or api.server_address
    net_appendix = "/" if net_server_address else "/net/"
    session_token = api.app.get_session_token(slug)
    sly.logger.debug("Session token for CLIP slug %s: %s", slug, session_token)
    host = server_address.rstrip("/") + net_appendix + session_token
    sly.logger.debug("App host URL for CLIP: %s", host)
    return host


@to_thread
def clean_image_embeddings_updated_at(api: sly.Api, project_id: int):
    """Set embeddings updated at timestamp to None for all images in the project."""
    msg_prefix = f"[Project: {project_id}]"
    try:
        sly.logger.debug(f"{msg_prefix} Starting to set embeddings updated at to None for images.")
        datasets = api.dataset.get_list(project_id=project_id, recursive=True)
        if len(datasets) == 0:
            return
        dataset_ids = []
        for dataset in datasets:
            if dataset.images_count != 0 or dataset.items_count != 0:
                dataset_ids.append(dataset.id)
                continue
        if len(dataset_ids) == 0:
            return
        try:
            image_ids = []
            for dataset_id in dataset_ids:
                sly.logger.debug(f"{msg_prefix} Getting images for dataset ID {dataset_id}")
                image_ids.extend([image.id for image in api.image.get_list(dataset_id=dataset_id)])
            timestamps = [None] * len(image_ids)
        except Exception as e:
            sly.logger.warning(
                f"{msg_prefix} Failed to get images for dataset ID {dataset_id}: {e}"
            )
            return
        api.image.set_embeddings_updated_at(ids=image_ids, timestamps=timestamps)
        sly.logger.debug(f"{msg_prefix} Set embeddings updated at to None for images successfully.")
    except Exception as e:
        sly.logger.error(
            f"{msg_prefix} Failed to set embeddings updated at to None for images: {e}",
            exc_info=True,
        )


@to_thread
@timeit
def set_update_flag(api: sly.Api, project_id: int, timestamp: Optional[str] = None):
    custom_data = api.project.get_custom_data(project_id)
    custom_data[CustomDataFields.EMBEDDINGS_UPDATE_STARTED_AT] = timestamp or datetime.datetime.now(
        datetime.timezone.utc
    ).strftime("%Y-%m-%dT%H:%M:%S.%fZ")
    api.project.update_custom_data(project_id, custom_data, silent=True)


@to_thread
@timeit
def clear_update_flag(api: sly.Api, project_id: int):
    custom_data = api.project.get_custom_data(project_id)
    if custom_data is None or custom_data == {}:
        return
    if CustomDataFields.EMBEDDINGS_UPDATE_STARTED_AT in custom_data:
        del custom_data[CustomDataFields.EMBEDDINGS_UPDATE_STARTED_AT]
        api.project.update_custom_data(project_id, custom_data, silent=True)


async def cleanup_task_and_flags(
    api: sly.Api, project_id: int, error_message: Optional[str] = None
):
    """
    Helper function to clean up task resources and reset project flags.
    Used across multiple endpoints to avoid code duplication.
    """
    from src.globals import background_tasks

    await set_embeddings_in_progress(api, project_id, False, error_message)
    await clear_update_flag(api, project_id)
    task_id = int(project_id)
    if task_id in background_tasks:
        del background_tasks[task_id]


async def validate_project_for_ai_features(
    api: sly.Api, project_info: sly.ProjectInfo, msg_prefix: str
):
    """
    Common validation for AI features (embeddings, search, diverse).
    Returns JSONResponse with error if validation fails, None if validation passes.
    """
    from fastapi.responses import JSONResponse

    # Check team subscription plan
    if not await is_team_plan_sufficient(api, project_info.team_id):
        message = f"Team {project_info.team_id} with 'free' plan cannot use AI features."
        sly.logger.warning(message)
        return JSONResponse({ResponseFields.MESSAGE: message}, status_code=403)

    # Check if embeddings are enabled for the project
    if project_info.embeddings_enabled is not None and project_info.embeddings_enabled is False:
        message = f"{msg_prefix} AI Search is disabled. Skipping."
        sly.logger.info(message)
        return JSONResponse({ResponseFields.MESSAGE: message}, status_code=200)

    return None  # Validation passed


@to_thread
def create_current_timestamp() -> str:
    """Create a timestamp in the format 'YYYY-MM-DDTHH:MM:SS.ssssssZ'."""
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


@to_thread
def disable_embeddings(api: sly.Api, project_id: int):
    """Disable embeddings for the project."""
    api.project.disable_embeddings(project_id)
    sly.logger.debug(f"[Project: {project_id}] Embeddings disabled.")


@to_thread
def set_processing_progress(
    project_id: int, total: int, current: int = 0, status: str = "processing"
):
    """Set processing progress for a project.

    :param project_id: Project ID
    :param total: Total number of items to process
    :param current: Current number of processed items
    :param status: Status of processing (processing, completed, error)
    """
    import src.globals as g

    g.image_processing_progress[project_id] = {"total": total, "current": current, "status": status}


@to_thread
def update_processing_progress(project_id: int, current: int, status: str = "processing"):
    """Update current progress for a project.

    :param project_id: Project ID
    :param current: Current number of processed items
    :param status: Status of processing (processing, completed, error)
    """
    import src.globals as g

    if project_id in g.image_processing_progress:
        g.image_processing_progress[project_id]["current"] = current
        g.image_processing_progress[project_id]["status"] = status


@to_thread
def get_processing_progress(project_id: int) -> Optional[Dict]:
    """Get processing progress for a project.

    :param project_id: Project ID
    :return: Dictionary with progress info or None if not found
    """
    import src.globals as g

    return g.image_processing_progress.get(project_id, None)


@to_thread
def clear_processing_progress(project_id: int):
    """Clear processing progress for a project.

    :param project_id: Project ID
    """
    import src.globals as g

    if project_id in g.image_processing_progress:
        del g.image_processing_progress[project_id]


@to_thread
def get_all_processing_progress() -> Dict:
    """Get processing progress for all projects.

    :return: Dictionary with all projects progress
    """
    import src.globals as g

    return g.image_processing_progress.copy()


@timeit
async def download_resized_images(image_urls: List[str]) -> List[bytes]:
    """Download resized images in parallel with concurrency limit.

    :param image_urls: List of image URLs to download
    :type image_urls: List[str]
    :return: List of image bytes in the same order as input URLs
    :rtype: List[bytes]
    """
    import src.globals as g

    concurrency = int(g.imgproxy_concurrency)

    # Create semaphore to limit concurrent downloads
    semaphore = asyncio.Semaphore(concurrency)

    @with_retries()
    async def download_single_image(session: aiohttp.ClientSession, url: str) -> bytes:
        """Download a single image with semaphore protection."""
        async with semaphore:
            async with session.get(url) as response:
                response.raise_for_status()
                return await response.read()

    # Create HTTP session
    async with aiohttp.ClientSession() as session:
        # Create tasks for all downloads, preserving order
        tasks = [download_single_image(session, url) for url in image_urls]

        # Wait for all downloads to complete
        image_bytes_list = await asyncio.gather(*tasks)

        return image_bytes_list
