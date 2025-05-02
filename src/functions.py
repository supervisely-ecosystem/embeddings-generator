import asyncio
import time
from typing import List, Optional

import supervisely as sly
from supervisely.sly_logger import logger

import src.cas as cas
import src.globals as g
import src.qdrant as qdrant
from src.utils import (
    fix_vectors,
    get_all_projects,
    get_datasets,
    get_lite_image_infos,
    get_project_info,
    get_team_info,
    parse_timestamp,
    timeit,
    update_embeddings_data,
)


@timeit
async def process_images(
    api: sly.Api,
    project_id: int,
    dataset_id: int = None,
    image_ids: List[int] = None,
    image_infos: List[sly.ImageInfo] = None,
) -> None:
    """Process images from the specified project. Download images, save them to HDF5,
    get vectors from the images and upsert them to Qdrant.
    Either dataset_id or image_ids should be provided. If the dataset_id is provided,
    all images from the dataset will be processed. If the image_ids are provided,
    only images with the specified IDs will be processed.

    Note: this function works only with Qdrant image collection "images".

    :param api: Supervisely API object.
    :type api: sly.Api
    :param project_id: Project ID to process images from.
    :type project_id: int
    :param dataset_id: Dataset ID to process images from.
    :type dataset_id: int, optional
    :param image_ids: List of image IDs to process.
    :type image_ids: List[int], optional
    :param image_infos: List of image infos to process.
    :type image_infos: List[sly.ImageInfo], optional
    """
    # Get image infos from the project.
    image_infos = await get_lite_image_infos(
        api,
        cas_size=g.IMAGE_SIZE_FOR_CAS,
        project_id=project_id,
        dataset_id=dataset_id,
        image_ids=image_ids,
        image_infos=image_infos,
    )

    await qdrant.get_or_create_collection(qdrant.IMAGES_COLLECTION)
    if await qdrant.collection_exists(qdrant.IMAGES_COLLECTION):
        # Get diff of image infos, check if they are already
        # in the Qdrant collection and have the same updated_at field.
        image_infos = await qdrant.get_diff(qdrant.IMAGES_COLLECTION, image_infos)

    if len(image_infos) == 0:
        logger.debug("All images are up-to-date.")
        return image_infos

    current_progress = 0
    total_progress = len(image_infos)
    logger.debug("Images to be processed: %d.", total_progress)
    for image_batch in sly.batched(image_infos):
        # Get vectors from images.
        # base64_data = [await base64_from_url(image_info.cas_url) for image_info in image_batch]
        # vectors_batch = await cas.get_vectors(
        #     base64_data
        # )
        vectors_batch = await cas.get_vectors([image_info.cas_url for image_info in image_batch])
        vectors_batch = fix_vectors(vectors_batch)
        logger.debug("Got %d vectors from images.", len(vectors_batch))

        # Upsert vectors to Qdrant.
        await qdrant.upsert(
            qdrant.IMAGES_COLLECTION,
            vectors_batch,
            image_batch,
        )
        current_progress += len(image_batch)
        logger.debug(
            "Upserted %d vectors to Qdrant. [%d/%d]",
            len(vectors_batch),
            current_progress,
            total_progress,
        )
    logger.debug("All %d images have been processed.", total_progress)
    return image_infos


@timeit
async def update_embeddings(
    api: sly.Api,
    project_id: int,
    force: bool = False,
    project_info: Optional[sly.ProjectInfo] = None,
):
    if project_info is None:
        project_info = await get_project_info(api, project_id)
    # Check if embeddings are up-to-date
    if project_info.embeddings_updated_at is None or force:
        image_infos = await process_images(api, project_id)
    elif parse_timestamp(project_info.embeddings_updated_at) < parse_timestamp(
        project_info.updated_at
    ):
        # Get all datasets that were updated after the embeddings were updated.
        dataset_infos = await get_datasets(api, project_id, recursive=True)
        dataset_infos = [
            dataset_info
            for dataset_info in dataset_infos
            if parse_timestamp(dataset_info.updated_at)
            > parse_timestamp(project_info.embeddings_updated_at)
        ]
        # Get image IDs for all datasets.
        tasks = []
        for dataset_info in dataset_infos:
            tasks.append(
                asyncio.create_task(
                    get_lite_image_infos(
                        api,
                        cas_size=g.IMAGE_SIZE_FOR_CAS,
                        project_id=project_id,
                        dataset_id=dataset_info.id,
                    )
                )
            )
        image_infos = [info for result in asyncio.gather(*tasks) for info in result]
        if len(image_infos) > 0:
            # image_ids = [info.id for info in image_infos]
            image_infos = await process_images(api, project_id, image_infos=image_infos)
    else:
        logger.debug("Embeddings for project %d are up-to-date.", project_info.id)
        return
    if len(image_infos) > 0:
        await update_embeddings_data(api, project_id, project_info.updated_at)


@timeit
async def auto_update_embeddings(
    api: sly.Api, project_id: int, project_info: Optional[sly.ProjectInfo] = None
):
    """
    Update embeddings for the specified project if needed.
    """
    if project_info is None:
        project_info = await get_project_info(api, project_id)

    team_info: sly.TeamInfo = await get_team_info(project_info.team_id)
    if team_info.usage.plan == "free":
        logger.debug(
            "Auto update embeddings is not available on free plan.",
            extra={
                "project_id": project_id,
                "project_name": project_info.name,
                "team_id": team_info.id,
                "team_name": team_info.name,
            },
        )
        api.project.disable_embeddings(project_id)
        logger.info(
            "Embeddings are disabled for project %s (id: %d) due to free plan.",
            project_info.name,
            project_id,
        )
        return

    # Check if embeddings activated for the project
    log_extra = {
        "team_id": project_info.team_id,
        "workspace_id": project_info.workspace_id,
        "project_name": project_info.name,
        "project_id": project_id,
        "items_count": project_info.items_count,
        "embeddings_updated_at": project_info.embeddings_updated_at,
        "updated_at": project_info.updated_at,
        "embeddings_enabled": project_info.embeddings_enabled,
    }
    if not project_info.embeddings_enabled:
        logger.debug("Embeddings are not activated for project %d.", project_id, extra=log_extra)
        return
    logger.debug(
        "Auto update embeddings for project %s (id: %d).",
        project_info.name,
        project_id,
        extra=log_extra,
    )
    t = time.monotonic()
    await update_embeddings(api, project_id, force=False, project_info=project_info)
    t = time.monotonic() - t
    logger.debug(
        "Auto update embeddings for project %s (id: %d) finished. Time: %.3f sec.",
        project_info.name,
        project_id,
        t,
        extra={**log_extra, "time": t},
    )


@timeit
async def auto_update_all_embeddings():
    """Update embeddings for all available projects"""
    logger.info("Auto update all embeddings task started.")
    # collection_names = await qdrant.get_collection_names()
    # project_ids = [int(name) for name in collection_names]
    project_infos: List[sly.ProjectInfo] = await get_all_projects(g.api)
    for project_info in project_infos:
        await auto_update_embeddings(
            g.api,
            project_info.id,
            project_info=project_info,
        )
    logger.info("Auto update all embeddings task finished.")
