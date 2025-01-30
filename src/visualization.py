import json
import tempfile
from pathlib import Path
from typing import List, Tuple

import sklearn.decomposition
import supervisely as sly
import umap

import src.qdrant as qdrant
from src.utils import ImageInfoLite, parse_timestamp


def projections_path(project_id):
    return f"/embeddings/visualizations/{project_id}/projections.json"


def calculate_projections(
    embeddings, all_info_list, projection_method, metric="euclidean", umap_min_dist=0.05
) -> List[List[float]]:
    try:
        if projection_method == "PCA":
            decomp = sklearn.decomposition.PCA(2)
            projections = decomp.fit_transform(embeddings)
        elif projection_method == "UMAP":
            decomp = umap.UMAP(min_dist=umap_min_dist, metric=metric)
            projections = decomp.fit_transform(embeddings)
        elif projection_method == "PCA-UMAP":
            decomp = sklearn.decomposition.PCA(64)
            projections = decomp.fit_transform(embeddings)
            decomp = umap.UMAP(min_dist=umap_min_dist, metric=metric)
            projections = decomp.fit_transform(projections)
        # elif projection_method == "t-SNE":
        #     decomp = sklearn.manifold.TSNE(2, perplexity=min(30, len(all_info_list) - 1), metric=metric, n_jobs=-1)
        #     projections = decomp.fit_transform(embeddings)
        # elif projection_method == "PCA-t-SNE":
        #     decomp = sklearn.decomposition.PCA(64)
        #     projections = decomp.fit_transform(embeddings)
        #     decomp = sklearn.manifold.TSNE(2, perplexity=min(30, len(all_info_list) - 1), metric=metric, n_jobs=-1)
        #     projections = decomp.fit_transform(projections)
        else:
            raise ValueError(f"unexpexted projection_method {projection_method}")
    except Exception as e:
        print(e)
        raise RuntimeError(
            f"count of embeddings = {len(embeddings)}, not enough to calculate projections."
        )
    return projections.tolist()


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
    projections = calculate_projections(vectors, image_infos, "UMAP")
    return image_infos, projections


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

async def get_projections(api: sly.Api, project_id: int) -> Tuple[List[ImageInfoLite], List[List[float]]]:
    project_info = api.project.get_info_by_id(project_id)
    file_info = api.file.get_info_by_path(
        project_info.team_id, projections_path(project_id)
    )
    if file_info is None:
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
