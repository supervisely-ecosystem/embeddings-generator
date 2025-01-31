import json
import tempfile
from pathlib import Path
from typing import List, Tuple

import supervisely as sly

import src.globals as g
import src.qdrant as qdrant
from src.utils import ImageInfoLite, parse_timestamp, timeit


def projections_path(project_id):
    return f"/embeddings/visualizations/{project_id}/projections.json"


@timeit
async def create_projections(
    api: sly.Api, project_id: int, dataset_id: int = None, image_ids: List[int] = None
) -> Tuple[List[ImageInfoLite], List[List[float]]]:
    if image_ids is None:
        if dataset_id is None:
            image_ids = []
            datasets = api.dataset.get_list(project_id)
            for dataset in datasets:
                image_ids.extend([info.id for info in api.image.get_list(dataset.id)])
        else:
            image_ids = [info.id for info in api.image.get_list(dataset_id)]

    image_infos, vectors = await qdrant.get_items_by_ids(
        project_id, image_ids, with_vectors=True
    )
    projections = g.api.task.send_request(g.projections_service_task_id, "create_projections", data={"vectors": vectors, "method": "UMAP"})
    return image_infos, projections


@timeit
async def save_projections(
    api: sly.Api,
    project_id: int,
    image_infos: List[ImageInfoLite],
    projections: List[List[float]],
):
    project_info = api.project.get_info_by_id(project_id)
    data = []
    for info, proj in zip(image_infos, projections):
        data.append({"image_info": info.to_json(), "projection": proj})
    with open("tmp.json", "w") as f:
        json.dump(data, f)

    api.file.upload(project_info.team_id, "tmp.json", projections_path(project_id))
    Path("tmp.json").unlink()

@timeit
async def get_projections(api: sly.Api, project_id: int) -> Tuple[List[ImageInfoLite], List[List[float]]]:
    project_info = api.project.get_info_by_id(project_id)
    file_info = api.file.get_info_by_path(
        project_info.team_id, projections_path(project_id)
    )
    if file_info is None:
        sly.logger.debug("file not found")
        return [], []
    with tempfile.NamedTemporaryFile("r") as f:
        api.file.download(project_info.team_id, projections_path(project_id), f.name)
        data = json.load(f)
    image_infos = [ImageInfoLite.from_json(item["image_info"]) for item in data]
    projections = [item["projection"] for item in data]
    return image_infos, projections


def projections_up_to_date(api: sly.Api, project_id):
    project_info = api.project.get_info_by_id(project_id)

    file_info = api.file.get_info_by_path(
        project_info.team_id, projections_path(project_id)
    )
    if file_info is None:
        return False

    return parse_timestamp(file_info.updated_at) > parse_timestamp(
        project_info.updated_at
    )
