import json
import tempfile
from pathlib import Path
from typing import List, Optional, Tuple

import supervisely as sly
from supervisely.api.file_api import FileInfo

import src.globals as g
import src.qdrant as qdrant
from src.pointcloud import download as download_pcd
from src.utils import (
    ImageInfoLite,
    get_dataset_by_name,
    get_file_info,
    get_image_infos,
    get_or_create_dataset,
    get_or_create_project,
    get_pcd_by_name,
    get_project_info,
    get_project_info_by_name,
    parse_timestamp,
    send_request,
    timeit,
)


def projections_path(project_id):
    return f"/embeddings/visualizations/{project_id}/projections.json"


def projections_project_name():
    return "Embeddings projections"


def projections_dataset_name(project_id):
    return str(project_id)


def get_projections_pcd_name():
    return "pcd_2_dim.pcd"


@timeit
async def create_projections(
    api: sly.Api, project_id: int, dataset_id: int = None, image_ids: List[int] = None
) -> Tuple[List[ImageInfoLite], List[List[float]]]:
    if image_ids is None:
        image_infos = await get_image_infos(
            api, cas_size=g.IMAGE_SIZE_FOR_CAS, project_id=project_id, dataset_id=dataset_id
        )
        image_ids = [info.id for info in image_infos]

    image_infos, vectors = await qdrant.get_items_by_ids(project_id, image_ids, with_vectors=True)
    projections = await send_request(
        api,
        g.projections_service_task_id,
        "projections",
        data={"vectors": vectors, "method": "umap"},
        timeout=60 * 5,
        retries=3,
        raise_error=True,
    )
    return image_infos, projections


@timeit
async def save_projections(
    api: sly.Api,
    project_id: int,
    image_infos: List[ImageInfoLite],
    projections: List[List[float]],
    project_info: Optional[sly.ProjectInfo] = None,
):
    if project_info is None:
        project_info = await get_project_info(api, project_id)
    data = []
    for info, proj in zip(image_infos, projections):
        data.append({"image_info": info.to_json(), "projection": proj})
    with open("tmp.json", "w", encoding="utf-8") as f:
        json.dump(data, f)
    await api.file.upload_async(project_info.team_id, "tmp.json", projections_path(project_id))
    Path("tmp.json").unlink()


@timeit
async def get_pcd_info(
    api: sly.Api, project_id: int, project_info: Optional[sly.ProjectInfo] = None
) -> sly.api.pointcloud_api.PointcloudInfo:
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
    if pcd_info is None:
        if project_info is None:
            project_info = await get_project_info(api, project_id)
        pcd_info = await get_pcd_info(api, project_id, project_info)

    pcd = await download_pcd(api, pcd_info.id)
    vectors = pcd.points[:, :2]
    image_ids = pcd.image_ids
    image_infos = await get_image_infos(
        api, cas_size=g.IMAGE_SIZE_FOR_CAS, project_id=project_id, image_ids=image_ids
    )
    return image_infos, vectors.tolist()


async def get_or_create_projections_dataset(
    api: sly.Api, image_project_id: int, image_project_info: sly.ProjectInfo = None
) -> sly.DatasetInfo:
    if image_project_info is None:
        image_project_info = await get_project_info(api, image_project_id)
    workspace_id = image_project_info.workspace_id
    project_info = await get_or_create_project(
        api, workspace_id, projections_project_name(), project_type=sly.ProjectType.POINT_CLOUDS
    )
    dataset_info = await get_or_create_dataset(api, project_info.id, str(image_project_id))
    return dataset_info


async def projections_up_to_date(
    api: sly.Api,
    project_id: int,
    project_info: Optional[sly.ProjectInfo] = None,
    file_info: Optional[FileInfo] = None,
) -> bool:
    if project_info is None:
        project_info = await get_project_info(api, project_id)

    if file_info is None:
        file_info = await get_file_info(api, project_info.team_id, projections_path(project_id))

    sly.logger.debug(
        "project_info: %s", "not found" if project_info is None else project_info._asdict()
    )
    sly.logger.debug("file_info: %s", "not found" if file_info is None else file_info._asdict())
    if file_info is None:
        return False

    return parse_timestamp(file_info.updated_at) > parse_timestamp(project_info.updated_at)
