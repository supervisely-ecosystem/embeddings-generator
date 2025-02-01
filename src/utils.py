import asyncio
import base64
import datetime
from collections import namedtuple
from functools import partial, wraps
from time import perf_counter
from typing import Callable, Dict, List, Optional

import supervisely as sly
from supervisely._utils import resize_image_url
from supervisely.api.module_api import ApiField


class TupleFields:
    """Fields of the named tuples used in the project."""

    ID = "id"
    DATASET_ID = "dataset_id"
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


class QdrantFields:
    """Fields for the queries to the Qdrant API."""

    KMEANS = "kmeans"
    NUM_CLUSTERS = "num_clusters"
    OPTION = "option"
    RANDOM = "random"
    CENTROIDS = "centroids"


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

    ATLAS = "atlas"
    POINTCLOUD = "pointcloud"


_ImageInfoLite = namedtuple(
    "_ImageInfoLite",
    [
        TupleFields.ID,
        TupleFields.DATASET_ID,
        TupleFields.FULL_URL,
        TupleFields.CAS_URL,
        TupleFields.UPDATED_AT,
    ],
)


class ImageInfoLite(_ImageInfoLite):
    def to_json(self):
        return {
            TupleFields.ID: self.id,
            TupleFields.DATASET_ID: self.dataset_id,
            TupleFields.FULL_URL: self.full_url,
            TupleFields.CAS_URL: self.cas_url,
            TupleFields.UPDATED_AT: self.updated_at,
        }

    @classmethod
    def from_json(cls, data):
        return cls(
            id=data[TupleFields.ID],
            dataset_id=data[TupleFields.DATASET_ID],
            full_url=data[TupleFields.FULL_URL],
            cas_url=data[TupleFields.CAS_URL],
            updated_at=data[TupleFields.UPDATED_AT],
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
def update_custom_data(api: sly.Api, project_id: int, custom_data: Dict):
    return api.project.update_custom_data(project_id, custom_data)


@to_thread
@timeit
def get_all_projects(api: sly.Api) -> List[sly.ProjectInfo]:
    return api.project.get_list_all()["entities"]


@to_thread
@timeit
def get_file_info(api: sly.Api, team_id: int, path: str):
    return api.file.get_info_by_path(team_id, path)


@timeit
async def get_image_infos(
    api: sly.Api,
    cas_size: int,
    project_id: int,
    dataset_id: int = None,
    image_ids: List[int] = None,
) -> List[ImageInfoLite]:
    """Returns lite version of image infos to cut off unnecessary data.
    Uses either dataset_id or image_ids to get image infos.
    If dataset_id is provided, it will be used to get all images from the dataset.
    If image_ids are provided, they will be used to get image infos.

    :param api: Instance of supervisely API.
    :type api: sly.Api
    :param cas_size: Size of the image for CAS, it will be added to URL.
    :type cas_size: int
    :param dataset_id: ID of the dataset to get images from.
    :type dataset_id: int, optional
    :param image_ids: List of image IDs to get image infos.
    :type image_ids: List[int], optional
    :return: List of lite version of image infos.
    :rtype: List[ImageInfoLite]
    """

    image_infos = await image_get_list_async(api, project_id, dataset_id, image_ids)

    return [
        ImageInfoLite(
            id=image_info.id,
            dataset_id=image_info.dataset_id,
            full_url=image_info.full_storage_url,
            cas_url=resize_image_url(
                image_info.full_storage_url,
                method="fit",
                width=cas_size,
                height=cas_size,
            ),
            updated_at=image_info.updated_at,
        )
        for image_info in image_infos
    ]


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


@timeit
async def image_get_list_async(
    api: sly.Api,
    project_id: int,
    dataset_id: int = None,
    images_ids: List[int] = None,
    per_page: int = 500,
):
    method = "images.list"
    data = {
        ApiField.PROJECT_ID: project_id,
        ApiField.FORCE_METADATA_FOR_LINKS: False,
        ApiField.PER_PAGE: per_page,
    }
    if dataset_id is not None:
        data[ApiField.DATASET_ID] = dataset_id
    if images_ids is not None:
        data[ApiField.FILTERS] = [{"field": ApiField.ID, "operator": "in", "value": images_ids}]
    semaphore = api.get_default_semaphore()
    pages_count = None
    tasks: List[asyncio.Task] = []

    async def _r(data_):
        nonlocal pages_count
        async with semaphore:
            response = await api.post_async(method, data_)
            response_json = response.json()
            items = response_json.get("entities", [])
            pages_count = response_json["pagesCount"]
        return [api.image._convert_json_info(item) for item in items]

    data[ApiField.PAGE] = 1
    items = await _r(data)

    for page_n in range(2, pages_count + 1):
        data[ApiField.PAGE] = page_n
        tasks.append(asyncio.create_task(_r(data.copy())))
    for task in tasks:
        items.extend(await task)
    return items
