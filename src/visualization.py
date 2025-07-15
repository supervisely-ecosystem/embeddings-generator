import json
import tempfile
from pathlib import Path
from typing import List, Optional, Tuple

import supervisely as sly
from fastapi.responses import JSONResponse
from supervisely.api.file_api import FileInfo

import src.globals as g
import src.qdrant as qdrant
from src.pointcloud import download as download_pcd
from src.pointcloud import upload as upload_pcd
from src.utils import (
    ImageInfoLite,
    ResponseFields,
    get_dataset_by_name,
    get_lite_image_infos,
    get_or_create_dataset,
    get_or_create_project,
    get_pcd_by_name,
    get_project_info,
    get_project_info_by_name,
    get_team_file_info,
    parse_timestamp,
    send_request,
    start_projections_service,
    timeit,
)


def projections_path(project_id):
    return f"/embeddings/visualizations/{project_id}/projections.json"


def projections_project_name():
    return "Embeddings Projections"


def projections_dataset_name(project_id):
    return str(project_id)


def get_projections_pcd_name():
    return "pcd_2_dim.pcd"


@timeit
async def create_projections(
    api: sly.Api, project_id: int, dataset_id: int = None, image_ids: List[int] = None
) -> Tuple[List[ImageInfoLite], List[List[float]]]:

    msg_prefix = f"[Project: {project_id}]"

    if image_ids is None:
        image_infos = await get_lite_image_infos(
            api, cas_size=g.IMAGE_SIZE_FOR_CLIP, project_id=project_id, dataset_id=dataset_id
        )
    else:
        image_infos = await get_lite_image_infos(
            api, cas_size=g.IMAGE_SIZE_FOR_CLIP, project_id=project_id, image_ids=image_ids
        )
    # image_hashes = [info.hash for info in image_infos]

    image_infos_result, vectors = await qdrant.get_items_by_id(
        qdrant.IMAGES_COLLECTION, image_infos, with_vectors=True
    )

    try:
        projections_service_task_id = await start_projections_service(api, project_id)
    except Exception as e:
        message = f"{msg_prefix} Failed to start projections service: {str(e)}"
        sly.logger.error(message, exc_info=True)
        return JSONResponse({ResponseFields.MESSAGE: message}, status_code=500)

    projections = await send_request(
        api,
        projections_service_task_id,
        "projections",
        data={"vectors": vectors, "method": "umap"},
        timeout=60 * 5,
        retries=3,
        raise_error=True,
    )
    return image_infos_result, projections


@timeit
async def save_projections(
    api: sly.Api,
    project_id: int,
    image_infos: List[ImageInfoLite],
    projections: List[List[float]],
    project_info: Optional[sly.ProjectInfo] = None,
    cluster_labels: List[int] = None,
):
    """Saves projections to a PCD file and uploads it to the point cloud project."""
    if project_info is None:
        project_info = await get_project_info(api, project_id)
    pcd_dataset_info = await get_or_create_projections_dataset(
        api, project_info.id, image_project_info=project_info
    )

    pcd_info = await upload_pcd(
        api,
        projections,
        [info.id for info in image_infos],
        get_projections_pcd_name(),
        pcd_dataset_info.id,
        cluster_labels,
    )
    return pcd_info


@timeit
async def get_pcd_info(
    api: sly.Api, project_id: int, project_info: Optional[sly.ProjectInfo] = None
) -> sly.api.pointcloud_api.PointcloudInfo:
    """
    Retrieves point cloud information for projections associated with a specific project.
    Validates the existence of project, dataset, and point cloud for projections.
    """
    if project_info is None:
        project_info = await get_project_info(api, project_id)
    pcd_project_info = await get_project_info_by_name(
        api, project_info.workspace_id, projections_project_name()
    )
    if pcd_project_info is None:
        raise ValueError(f"Project with projections not found: {projections_project_name()}")

    pcd_dataset_info = await get_dataset_by_name(api, pcd_project_info.id, str(project_info.id))
    if pcd_dataset_info is None:
        raise ValueError(
            f"Dataset with projections not found: {projections_dataset_name(project_id)}"
        )

    pcd_item_info: sly.api.pointcloud_api.PointcloudInfo = await get_pcd_by_name(
        api, pcd_dataset_info.id, get_projections_pcd_name()
    )
    if pcd_item_info is None:
        raise ValueError("PCD with projections not found: pcd_2_dim.pcd")
    return pcd_item_info


@timeit
async def get_projections(
    api: sly.Api,
    project_id: int,
    project_info: Optional[sly.ProjectInfo] = None,
    pcd_info: Optional[sly.api.pointcloud_api.PointcloudInfo] = None,
) -> Tuple[List[ImageInfoLite], List[List[float]]]:
    """
    Retrieves the 2D projections of image embeddings from a point cloud file.
    This function downloads a point cloud file associated with a project, extracts the 2D projection
    vectors (first two dimensions of the point cloud points) and returns them along with the corresponding
    image information.
    """
    if pcd_info is None:
        if project_info is None:
            project_info = await get_project_info(api, project_id)
        pcd_info = await get_pcd_info(api, project_id, project_info)

    pcd = await download_pcd(api, pcd_info.id)
    vectors = pcd.points[:, :2]
    image_ids = pcd.image_ids
    image_infos = await get_lite_image_infos(
        api, cas_size=g.IMAGE_SIZE_FOR_CLIP, project_id=project_id, image_ids=image_ids
    )
    return image_infos, vectors.tolist()


async def get_or_create_projections_dataset(
    api: sly.Api, image_project_id: int, image_project_info: sly.ProjectInfo = None
) -> sly.DatasetInfo:
    """
    Gets or creates a dataset for projections based on an image project.

    This function checks if a dataset exists for storing projections associated with the specified image project.
    If it exists, it returns the dataset info; if not, it creates a new dataset and returns its info.
    The dataset is created within a point clouds project that is either retrieved or created if not exists.
    """
    if image_project_info is None:
        image_project_info = await get_project_info(api, image_project_id)
    workspace_id = image_project_info.workspace_id
    project_info = await get_or_create_project(
        api, workspace_id, projections_project_name(), project_type=sly.ProjectType.POINT_CLOUDS
    )
    dataset_info = await get_or_create_dataset(api, project_info.id, str(image_project_id))
    return dataset_info


async def is_projections_up_to_date(
    api: sly.Api,
    project_id: int,
    project_info: Optional[sly.ProjectInfo] = None,
) -> bool:
    """
    Checks if the projections (embeddings visualization) are up to date with the current project state.
    Compares the timestamp of the last update of the point cloud data with the timestamp of the last
    project update to determine if the projections need to be recalculated.
    """
    try:
        pcd_info = await get_pcd_info(api, project_id)
    except ValueError:
        return False
    if project_info is None:
        project_info = await get_project_info(api, project_id)
    return parse_timestamp(pcd_info.updated_at) >= parse_timestamp(project_info.updated_at)


async def get_or_create_projections(api: sly.Api, project_id, project_info):
    """
    Retrieves existing projections for a project or creates new ones if needed.

    This function checks if projections exist for the given project and are up to date.
    If projections don't exist or are outdated (project was updated after projections
    were created), new projections will be generated.
    """
    if project_info is None:
        project_info = api.project.get_info_by_id(project_id)
    try:
        pcd_info: sly.api.pointcloud_api.PointcloudInfo = await get_pcd_info(
            api, project_id, project_info=project_info
        )
    except ValueError as e:
        sly.logger.debug("Projections not found. Creating new projections.")
    else:
        if parse_timestamp(pcd_info.updated_at) < parse_timestamp(project_info.updated_at):
            sly.logger.debug("Projections are not up to date. Creating new projections.")
            pcd_info = None

    if pcd_info is None:
        # create new projections
        image_infos, projections = await create_projections(
            api,
            project_id,
            # image_ids=image_ids, #! fix before enabling projections endpoints
        )
        # save projections
        await save_projections(
            api,
            project_id=project_id,
            image_infos=image_infos,
            projections=projections,
            project_info=project_info,
        )
    else:
        image_infos, projections = await get_projections(
            api, project_id, project_info=project_info, pcd_info=pcd_info
        )
