import json
import tempfile
from pathlib import Path
from typing import List, Optional, Tuple

import supervisely as sly
from supervisely.api.file_api import FileInfo

import src.globals as g
import src.qdrant as qdrant
from src.utils import (
    ImageInfoLite,
    get_file_info,
    get_image_infos,
    get_project_info,
    parse_timestamp,
    send_request,
    timeit,
)


def projections_path(project_id):
    return f"/embeddings/visualizations/{project_id}/projections.json"


@timeit
async def create_projections(
    api: sly.Api, project_id: int, dataset_id: int = None, image_ids: List[int] = None
) -> Tuple[List[ImageInfoLite], List[List[float]]]:
    if image_ids is None:
        image_infos = get_image_infos(
            api, cas_size=g.IMAGE_SIZE_FOR_CAS, project_id=project_id, dataset_id=dataset_id
        )
        image_ids = [info.id for info in image_infos]

    image_infos, vectors = await qdrant.get_items_by_ids(project_id, image_ids, with_vectors=True)
    projections = await send_request(
        api,
        g.projections_service_task_id,
        "create_projections",
        data={"vectors": vectors, "method": "UMAP"},
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
async def get_projections(
    api: sly.Api, project_id: int, project_info: Optional[sly.ProjectInfo] = None
) -> Tuple[List[ImageInfoLite], List[List[float]]]:
    if project_info is None:
        project_info = await get_project_info(api, project_id)
    file_info = await get_file_info(api, project_info.team_id, projections_path(project_id))
    if file_info is None:
        sly.logger.warning(
            "File with projections not found",
            extra={"project_id": project_id, "path": projections_path(project_id)},
        )
        return [], []
    with tempfile.NamedTemporaryFile("r") as f:
        await api.file.download_async(project_info.team_id, projections_path(project_id), f.name)
        data = json.load(f)
    image_infos = [ImageInfoLite.from_json(item["image_info"]) for item in data]
    projections = [item["projection"] for item in data]
    return image_infos, projections


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
    if file_info is None:
        return False

    return parse_timestamp(file_info.updated_at) > parse_timestamp(project_info.updated_at)
