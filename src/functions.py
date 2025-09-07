from asyncio import sleep as asyncio_sleep
from typing import AsyncGenerator, List, Literal, Optional, Tuple

import supervisely as sly
from docarray import Document
from supervisely.sly_logger import logger

import src.cas as cas
import src.globals as g
import src.milvus as milvus
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
    items: Literal["all", "images", "objects"] = "all",
    batch_size: int = 1000,
) -> AsyncGenerator[Tuple[List[ImageInfoLite], List[ObjectInfoLite]], None]:
    """Generator that yields batches of processed lite image and object infos.

    :param api: Supervisely API object.
    :type api: sly.Api
    :param project_id: Project ID to process images from.
    :type project_id: int
    :param to_create: List of image infos to process.
    :type to_create: List[sly.ImageInfo]
    :param items: Type of items to process - "all", "images", or "objects".
    :type items: Literal["all", "images", "objects"]
    :param batch_size: Size of each batch. If None, uses sly.batched default.
    :type batch_size: int
    :return: AsyncGenerator yielding tuples of (image_infos, object_infos) batches.
    :rtype: AsyncGenerator[Tuple[List[ImageInfoLite], List[ObjectInfoLite]], None]
    """

    for batch in sly.batched(to_create, batch_size):
        image_batch = []
        object_batch = []

        if items in ["all", "images"]:
            image_batch = await create_lite_image_infos(
                cas_size=g.IMAGE_SIZE_FOR_CLIP,
                image_infos=batch,
                imgproxy_address=g.imgproxy_address,
            )

        if items in ["all", "objects"]:
            object_batch = await get_lite_object_infos(
                api,
                cas_size=g.IMAGE_SIZE_FOR_CLIP,
                project_id=project_id,
                image_infos=batch,
                imgproxy_address=g.imgproxy_address,
            )

        yield (image_batch, object_batch)


@timeit
async def process_images(
    api: sly.Api,
    project_id: int,
    to_create: List[sly.ImageInfo],
    to_delete: List[sly.ImageInfo],
    return_vectors: bool = False,  #! make separate function for returning vectors
    check_collection_exists: bool = True,
    items: Literal["all", "images", "objects"] = "all",
) -> Tuple[List[sly.ImageInfo], List[List[float]]]:
    """Process images from the specified project. Download images, save them to HDF5,
    get vectors from the images and upsert them to vector DB.

    :param api: Supervisely API object.
    :type api: sly.Api
    :param project_id: Project ID to process images from.
    :type project_id: int
    :param to_create: List of image infos to create in vector DB.
    :type to_create: List[sly.ImageInfo]
    :param to_delete: List of image infos to delete from vector DB.
    :type to_delete: List[sly.ImageInfo]
    :param return_vectors: If True, return vectors of the created images.
    :type return_vectors: bool
    :param check_collection_exists: If True, check if the vector DB collection exists.
    :type check_collection_exists: bool
    :param items: Type of items to process - "all", "images", or "objects".
    :type items: Literal["all", "images", "objects"]
    :return: Tuple of two lists: list of created image infos and list of vectors.
    :rtype: Tuple[List[sly.ImageInfo], List[List[float]]]
    """

    msg_prefix = f"[Project: {project_id}]"
    images_vectors = []
    objects_vectors = []
    current_progress = 0

    if len(to_create) == 0 and len(to_delete) == 0:
        logger.debug(f"{msg_prefix} Nothing to update.")
        return to_create, (images_vectors, objects_vectors)

    try:
        # Calculate total progress beforehand without processing all items
        total_progress = len(to_create)

        if check_collection_exists:
            await milvus.get_or_create_collection(project_id)

        current_progress = 0

        # Initialize progress tracking
        if total_progress > 0:
            await set_processing_progress(project_id, total_progress, 0, "processing")

        # Determine what to process based on items parameter
        process_images_flag = items in ["all", "images"]
        process_objects_flag = items in ["all", "objects"]

        if len(to_create) > 0:
            logger.debug(f"{msg_prefix} Processing {items} embeddings for {total_progress} images.")

            # Use generator to process items in batches without loading all in memory
            async for image_batch, object_batch in process_items_generator(
                api=api, project_id=project_id, to_create=to_create, items=items
            ):
                processed_images_count = 0
                batch_images_to_update = []

                # Process images if needed
                if process_images_flag and len(image_batch) > 0:
                    # Download images as bytes and create Document objects
                    item_urls = [item_info.cas_url for item_info in image_batch]
                    image_bytes_list = await download_resized_images(item_urls)
                    # Create Document objects with blob data
                    queries = [Document(blob=image_bytes) for image_bytes in image_bytes_list]

                    # Get vectors from images using Document objects.
                    vectors_batch = await cas.get_vectors(queries)
                    vectors_batch = fix_vectors(vectors_batch)
                    logger.debug(f"{msg_prefix} Got {len(vectors_batch)} vectors for images.")

                    # Upsert vectors to vector DB.
                    await milvus.upsert(project_id, vectors_batch, image_batch)

                    processed_images_count = len(image_batch)
                    batch_images_to_update.extend([item.id for item in image_batch])

                    if return_vectors:
                        images_vectors.extend(vectors_batch)

                # Process objects if needed
                if process_objects_flag and len(object_batch) > 0:
                    # Download cropped images for objects as bytes and create Document objects
                    item_urls = [item_info.cas_url for item_info in object_batch]
                    image_bytes_list = await download_resized_images(item_urls)
                    # Create Document objects with blob data
                    queries = [Document(blob=image_bytes) for image_bytes in image_bytes_list]

                    # Get vectors from objects using Document objects.
                    vectors_batch = await cas.get_vectors(queries)
                    vectors_batch = fix_vectors(vectors_batch)
                    logger.debug(f"{msg_prefix} Got {len(vectors_batch)} vectors for objects.")

                    # Upsert vectors to vector DB.
                    await milvus.upsert(project_id, vectors_batch, object_batch)

                    # For objects, count unique image_ids
                    object_image_ids = [item.image_id for item in object_batch]
                    batch_images_to_update.extend(object_image_ids)
                    if not process_images_flag:  # Only count if images weren't processed
                        processed_images_count = len(list(set(object_image_ids)))

                    if return_vectors:
                        objects_vectors.extend(vectors_batch)

                # Update progress
                current_progress += processed_images_count
                await update_processing_progress(project_id, current_progress, "processing")

                logger.debug(
                    f"{msg_prefix} Processed batch: {processed_images_count} images. [{current_progress}/{total_progress}]",
                )

                # Update embeddings timestamp for this batch if conditions are met
                if items == "all":
                    # For "all" mode, update timestamp only when both images and objects are processed for this batch
                    if len(image_batch) > 0 and len(object_batch) > 0:
                        # Get unique image IDs that have both image and object embeddings created
                        image_ids_with_images = set([item.id for item in image_batch])
                        image_ids_with_objects = set([item.image_id for item in object_batch])
                        # Only update timestamp for images that have both types of embeddings
                        complete_image_ids = image_ids_with_images.intersection(
                            image_ids_with_objects
                        )
                        if complete_image_ids:
                            batch_image_infos = [
                                img_info
                                for img_info in to_create
                                if img_info.id in complete_image_ids
                            ]
                            await set_image_embeddings_updated_at(api, batch_image_infos)
                elif items == "images" and len(image_batch) > 0:
                    # For images-only mode, update when images are processed
                    # Convert ImageInfoLite back to original ImageInfo objects
                    image_ids = [item.id for item in image_batch]
                    batch_image_infos = [
                        img_info for img_info in to_create if img_info.id in image_ids
                    ]
                    await set_image_embeddings_updated_at(api, batch_image_infos)
                elif items == "objects" and len(object_batch) > 0:
                    # For objects-only mode, update when objects are processed
                    # Get the original ImageInfo objects for the processed object images
                    object_image_ids = set([item.image_id for item in object_batch])
                    batch_image_infos = [
                        img_info for img_info in to_create if img_info.id in object_image_ids
                    ]
                    await set_image_embeddings_updated_at(api, batch_image_infos)

            logger.debug(f"{msg_prefix} All {total_progress} items have been vectorized.")
            # Mark as completed
            await update_processing_progress(project_id, current_progress, "completed")

        if len(to_delete) > 0:
            # Determine what to delete based on items parameter
            delete_item_name = (
                "images" if items == "images" else "objects" if items == "objects" else "items"
            )
            logger.debug(
                f"{msg_prefix} Vectors for {delete_item_name} to be deleted: {len(to_delete)}."
            )
            for items_batch in sly.batched(to_delete):
                # Delete embeddings from the vector DB.
                await milvus.delete_collection_items(
                    collection_name=project_id, items_info=items_batch
                )
                await set_image_embeddings_updated_at(api, items_batch, [None] * len(items_batch))
                logger.debug(
                    f"{msg_prefix} Deleted {len(items_batch)} {delete_item_name} from vector DB."
                )

        logger.info(
            f"{msg_prefix} Embeddings Created: {len(to_create)}, Deleted: {len(to_delete)}."
        )

        await asyncio_sleep(1)  # Brief delay to allow final status to be read
        await clear_processing_progress(project_id)

        # Return the original to_create list, as we process it in batches without storing processed items
        return to_create, (images_vectors, objects_vectors)

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
    items: Literal["all", "images", "objects"] = "all",
):
    """Update embeddings for a project.

    :param api: Supervisely API object.
    :type api: sly.Api
    :param project_id: Project ID to update embeddings for.
    :type project_id: int
    :param force: If True, recreate all embeddings regardless of timestamps.
    :type force: bool
    :param project_info: Optional project info to avoid fetching it again.
    :type project_info: Optional[sly.ProjectInfo]
    :param items: Type of items to process - "all", "images", or "objects".
    :type items: Literal["all", "images", "objects"]
    """
    msg_prefix = f"[Project: {project_id}] "

    if project_info is None:
        project_info = await get_project_info(api, project_id)

    if force:
        logger.info(f"{msg_prefix} Force enabled, recreating embeddings for all images.")
        await milvus.delete_collection(project_id)
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
        items=items,
    )
    if len(items_info[0]) > 0:  # items_info is a tuple (to_create, vectors)
        await set_project_embeddings_updated_at(api, project_id)
