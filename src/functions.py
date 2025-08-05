from asyncio import sleep as asyncio_sleep
from typing import List, Optional, Tuple

import supervisely as sly
from docarray import Document
from supervisely.sly_logger import logger

import src.cas as cas
import src.globals as g
import src.qdrant as qdrant
from src.utils import (
    clear_processing_progress,
    create_lite_image_infos,
    download_resized_images,
    fix_vectors,
    get_project_info,
    image_get_list_async,
    parse_timestamp,
    set_image_embeddings_updated_at,
    set_processing_progress,
    set_project_embeddings_updated_at,
    timeit,
    update_processing_progress,
)


@timeit
async def process_images(
    api: sly.Api,
    project_id: int,
    to_create: List[sly.ImageInfo],
    to_delete: List[sly.ImageInfo],
    return_vectors: bool = False,
    check_collection_exists: bool = True,
) -> Tuple[List[sly.ImageInfo], List[List[float]]]:
    """Process images from the specified project. Download images, save them to HDF5,
    get vectors from the images and upsert them to Qdrant.

    :param api: Supervisely API object.
    :type api: sly.Api
    :param project_id: Project ID to process images from.
    :type project_id: int
    :param to_create: List of image infos to create in Qdrant.
    :type to_create: List[sly.ImageInfo]
    :param to_delete: List of image infos to delete from Qdrant.
    :type to_delete: List[sly.ImageInfo]
    :param return_vectors: If True, return vectors of the created images.
    :type return_vectors: bool
    :param check_collection_exists: If True, check if the Qdrant collection exists.
    :type check_collection_exists: bool
    :return: Tuple of two lists: list of created image infos and list of vectors.
    :rtype: Tuple[List[sly.ImageInfo], List[List[float]]]
    """

    msg_prefix = f"[Project: {project_id}]"
    vectors = []
    current_progress = 0

    if len(to_create) == 0 and len(to_delete) == 0:
        logger.debug(f"{msg_prefix} Nothing to update.")
        return to_create, vectors

    try:
        to_create = await create_lite_image_infos(
            cas_size=g.IMAGE_SIZE_FOR_CLIP,
            image_infos=to_create,
            imgproxy_address=g.imgproxy_address,
        )

        # if await qdrant.collection_exists(project_id):
        # Get diff of image infos, check if they are already in the Qdrant collection

        if check_collection_exists:
            await qdrant.get_or_create_collection(project_id)

        current_progress = 0
        total_progress = len(to_create)

        # Initialize progress tracking
        if total_progress > 0:
            await set_processing_progress(project_id, total_progress, 0, "processing")

        if len(to_create) > 0:
            logger.debug(f"{msg_prefix} Images to be vectorized: {total_progress}.")
            for image_batch in sly.batched(to_create):
                # Download images as bytes and create Document objects
                image_urls = [image_info.cas_url for image_info in image_batch]
                image_bytes_list = await download_resized_images(image_urls)
                # Create Document objects with blob data
                queries = [Document(blob=image_bytes) for image_bytes in image_bytes_list]

                # Get vectors from images using Document objects.
                vectors_batch = await cas.get_vectors(queries)
                vectors_batch = fix_vectors(vectors_batch)
                logger.debug(f"{msg_prefix} Got {len(vectors_batch)} vectors for images.")

                # Upsert vectors to Qdrant.
                await qdrant.upsert(project_id, vectors_batch, image_batch)
                current_progress += len(image_batch)

                # Update progress
                await update_processing_progress(project_id, current_progress, "processing")

                logger.debug(
                    f"{msg_prefix} Upserted {len(vectors_batch)} vectors to Qdrant. [{current_progress}/{total_progress}]",
                )
                await set_image_embeddings_updated_at(api, image_batch)

                if return_vectors:
                    vectors.extend(vectors_batch)

            logger.debug(f"{msg_prefix} All {total_progress} images have been vectorized.")
            # Mark as completed
            await update_processing_progress(project_id, current_progress, "completed")

        if len(to_delete) > 0:
            logger.debug(f"{msg_prefix} Vectors for images to be deleted: {len(to_delete)}.")
            for image_batch in sly.batched(to_delete):
                # Delete images from the Qdrant.
                await qdrant.delete_collection_items(
                    collection_name=project_id, image_infos=image_batch
                )
                await set_image_embeddings_updated_at(api, image_batch, [None] * len(image_batch))
                logger.debug(f"{msg_prefix} Deleted {len(image_batch)} images from Qdrant.")

        logger.info(
            f"{msg_prefix} Embeddings Created: {len(to_create)}, Deleted: {len(to_delete)}."
        )

        await asyncio_sleep(1)  # Brief delay to allow final status to be read
        await clear_processing_progress(project_id)

        return to_create, vectors

    except Exception as e:
        # Mark as error and log
        await update_processing_progress(project_id, current_progress, "error")
        logger.error(f"{msg_prefix} Error during image processing: {str(e)}")
        raise


@timeit
async def update_embeddings(
    api: sly.Api,
    project_id: int,
    force: bool = False,
    project_info: Optional[sly.ProjectInfo] = None,
):
    msg_prefix = f"[Project: {project_id}] "

    if project_info is None:
        project_info = await get_project_info(api, project_id)

    if force:
        logger.info(f"{msg_prefix} Force enabled, recreating embeddings for all images.")
        await qdrant.delete_collection(project_id)
        # do not need to create collection here, it will be created in process_images
        images_to_create = await image_get_list_async(api, project_id)
        images_to_delete = []
    elif project_info.embeddings_updated_at is None:
        # do not need to check or create collection here, it will be created in process_images
        logger.info(
            f"{msg_prefix} Embeddings are not updated yet, creating embeddings for all images."
        )
        images_to_create = await image_get_list_async(api, project_id)
        images_to_delete = []
    elif parse_timestamp(project_info.embeddings_updated_at) < parse_timestamp(
        project_info.updated_at
    ):
        logger.info(
            f"{msg_prefix} Embeddings are outdated, will check for images that need to be updated."
        )
        images_to_create = await image_get_list_async(api, project_id, wo_embeddings=True)
        if project_info.embeddings_updated_at is not None:
            images_to_delete = await image_get_list_async(
                api, project_id, deleted_after=project_info.embeddings_updated_at
            )
        else:
            images_to_delete = []

    else:
        logger.debug("Embeddings for project %d are up-to-date.", project_info.id)
        return
    image_infos = await process_images(api, project_id, images_to_create, images_to_delete)
    if len(image_infos) > 0:
        await set_image_embeddings_updated_at(api, image_infos)
        await set_project_embeddings_updated_at(api, project_id)
