from asyncio import sleep as asyncio_sleep
from typing import AsyncGenerator, List, Optional, Tuple

import supervisely as sly
from docarray import Document
from supervisely.sly_logger import logger

import src.cas as cas
import src.globals as g
import src.qdrant as qdrant
from src.utils import (
    ImageInfoLite,
    ObjectInfoLite,
    clear_processing_progress,
    create_lite_image_infos,
    download_resized_images,
    fix_vectors,
    get_lite_object_infos,
    get_project_info,
    image_get_list_async,
    parse_timestamp,
    set_image_embeddings_updated_at,
    set_processing_progress,
    set_project_embeddings_updated_at,
    timeit,
    update_processing_progress,
)


async def process_items_generator(
    api: sly.Api,
    project_id: int,
    to_create: List[sly.ImageInfo],
    objects: bool = False,
    batch_size: int = 1000,
) -> AsyncGenerator[List[sly.ImageInfo], None]:
    """Generator that yields batches of processed lite image/object infos.

    :param api: Supervisely API object.
    :type api: sly.Api
    :param project_id: Project ID to process images from.
    :type project_id: int
    :param to_create: List of image infos to process.
    :type to_create: List[sly.ImageInfo]
    :param objects: If True, process as objects instead of images.
    :type objects: bool
    :param batch_size: Size of each batch. If None, uses sly.batched default.
    :type batch_size: int
    :return: AsyncGenerator yielding batches of processed lite infos.
    :rtype: AsyncGenerator[List[sly.ImageInfo], None]
    """

    for batch in sly.batched(to_create, batch_size):
        if objects:
            processed_batch = await get_lite_object_infos(
                api,
                cas_size=g.IMAGE_SIZE_FOR_CLIP,
                project_id=project_id,
                image_infos=batch,
                imgproxy_address=g.imgproxy_address,
            )
        else:
            processed_batch = await create_lite_image_infos(
                cas_size=g.IMAGE_SIZE_FOR_CLIP,
                image_infos=batch,
                imgproxy_address=g.imgproxy_address,
            )
        yield processed_batch


@timeit
async def process_images(
    api: sly.Api,
    project_id: int,
    to_create: List[sly.ImageInfo],
    to_delete: List[sly.ImageInfo],
    return_vectors: bool = False,
    check_collection_exists: bool = True,
    objects: bool = False,
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
    item_name = "objects" if objects else "images"
    vectors = []
    current_progress = 0

    if len(to_create) == 0 and len(to_delete) == 0:
        logger.debug(f"{msg_prefix} Nothing to update.")
        return to_create, vectors

    try:
        # Calculate total progress beforehand without processing all items
        total_progress = len(to_create)

        if check_collection_exists:
            await qdrant.get_or_create_collection(project_id, objects=objects)

        current_progress = 0

        # Initialize progress tracking
        if total_progress > 0:
            await set_processing_progress(project_id, total_progress, 0, "processing")

        if len(to_create) > 0:
            logger.debug(
                f"{msg_prefix} {item_name} to be vectorized {'for images in a number of' if not objects else ''}: {total_progress}."
            )

            # Use generator to process items in batches without loading all in memory
            async for items_batch in process_items_generator(
                api=api, project_id=project_id, to_create=to_create, objects=objects
            ):
                if len(items_batch) == 0:
                    continue
                # Download images (or cropped images for objects) as bytes and create Document objects
                item_urls = [item_info.cas_url for item_info in items_batch]
                image_bytes_list = await download_resized_images(item_urls)
                # Create Document objects with blob data
                queries = [Document(blob=image_bytes) for image_bytes in image_bytes_list]

                # Get vectors from images using Document objects.
                vectors_batch = await cas.get_vectors(queries)
                vectors_batch = fix_vectors(vectors_batch)
                logger.debug(f"{msg_prefix} Got {len(vectors_batch)} vectors for {item_name}.")

                # Upsert vectors to Qdrant.
                await qdrant.upsert(project_id, vectors_batch, items_batch)

                if isinstance(items_batch[0], ImageInfoLite):
                    processed_items_num = len(items_batch)
                else:  # isinstance(items_batch[0], ObjectInfoLite):
                    processed_items_num = [item.image_id for item in items_batch]
                    processed_items_num = len(list(set(processed_items_num)))
                current_progress += processed_items_num

                # Update progress
                await update_processing_progress(project_id, current_progress, "processing")

                logger.debug(
                    f"{msg_prefix} Upserted {len(vectors_batch)} vectors to Qdrant. [{current_progress}/{total_progress}]",
                )
                await set_image_embeddings_updated_at(api, items_batch)

                if return_vectors:
                    vectors.extend(vectors_batch)

            logger.debug(f"{msg_prefix} All {total_progress} {item_name} have been vectorized.")
            # Mark as completed
            await update_processing_progress(project_id, current_progress, "completed")

        if len(to_delete) > 0:
            logger.debug(f"{msg_prefix} Vectors for {item_name} to be deleted: {len(to_delete)}.")
            for items_batch in sly.batched(to_delete):
                # Delete embeddings from the Qdrant.
                await qdrant.delete_collection_items(
                    collection_name=project_id, items_info=items_batch, objects=objects
                )
                await set_image_embeddings_updated_at(api, items_batch, [None] * len(items_batch))
                logger.debug(f"{msg_prefix} Deleted {len(items_batch)} {item_name} from Qdrant.")

        logger.info(
            f"{msg_prefix} Embeddings Created: {len(to_create)}, Deleted: {len(to_delete)}."
        )

        await asyncio_sleep(1)  # Brief delay to allow final status to be read
        await clear_processing_progress(project_id)

        # Return the original to_create list, as we process it in batches without storing processed items
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
    objects: bool = False,
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
    items_info = await process_images(
        api=api,
        project_id=project_id,
        to_create=images_to_create,
        to_delete=images_to_delete,
        objects=objects,
    )
    if len(items_info) > 0:
        await set_image_embeddings_updated_at(api, items_info)
        await set_project_embeddings_updated_at(api, project_id)
