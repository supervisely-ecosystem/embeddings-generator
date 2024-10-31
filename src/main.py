from typing import List

import supervisely as sly

import src.cas as cas
import src.globals as g
import src.qdrant as qdrant
from src.events import Event
from src.utils import ImageInfoLite, get_datasets, get_image_infos

app = sly.Application()


@app.event(Event.Embeddings, use_state=True)
@sly.timeit
async def create_embeddings(api: sly.Api, event: Event.Embeddings) -> None:
    sly.logger.info(
        f"Started creating embeddings for project {event.project_id}. "
        f"Force: {event.force}, Image IDs: {event.image_ids}."
    )

    if event.force:
        # Step 1: If force is True, delete the collection and recreate it.
        sly.logger.debug(f"Force enabled, deleting collection {event.project_id}.")
        await qdrant.delete_collection(event.project_id)
        sly.logger.debug(f"Deleting HDF5 file for project {event.project_id}.")

    # Step 2: Ensure collection exists in Qdrant.
    await qdrant.get_or_create_collection(event.project_id)

    # Step 3: Process images.
    if not event.image_ids:
        # Step 3A: If image_ids are not provided, get all datasets from the project.
        # Then iterate over datasets and process images from each dataset.
        datasets = await get_datasets(api, event.project_id)
        for dataset in datasets:
            await process_images(api, event.project_id, dataset_id=dataset.id)
    else:
        # Step 3B: If image_ids are provided, process images with specific IDs.
        await process_images(api, event.project_id, image_ids=event.image_ids)

    sly.logger.debug(f"Embeddings for project {event.project_id} have been created.")


@app.event(Event.Search, use_state=True)
@sly.timeit
async def search(api: sly.Api, event: Event.Search) -> List[ImageInfoLite]:
    sly.logger.info(
        f"Searching for similar images in project {event.project_id}. "
        f"Query: {event.query}, Limit: {event.limit}."
    )

    # ? Add support for image IDs in the query.
    # * In this case, we'll need to get the resized image URLs from the API
    # * and then get vectors from these URLs.

    # Vectorize the query data (can be a text prompt or an image URL).
    query_vectors = await cas.get_vectors([event.query])

    image_infos = await qdrant.search(event.project_id, query_vectors[0], event.limit)
    sly.logger.debug(f"Found {len(image_infos)} similar images.")

    return image_infos


@app.event(Event.Diverse, use_state=True)
@sly.timeit
async def diverse(api: sly.Api, event: Event.Diverse) -> List[ImageInfoLite]:
    sly.logger.info(
        f"Generating diverse population for project {event.project_id}. "
        f"Method: {event.query}, Limit: {event.limit}, Option: {event.option}."
    )

    image_infos = await qdrant.diverse(
        event.project_id,
        event.limit,
        event.method,
        event.option,
    )
    sly.logger.debug(f"Generated {len(image_infos)} diverse images.")

    return image_infos


@sly.timeit
async def process_images(
    api: sly.Api, project_id: int, dataset_id: int = None, image_ids: List[int] = None
) -> None:
    """Process images from the specified project. Download images, save them to HDF5,
    get vectors from the images and upsert them to Qdrant.
    Either dataset_id or image_ids should be provided. If the dataset_id is provided,
    all images from the dataset will be processed. If the image_ids are provided,
    only images with the specified IDs will be processed.

    :param api: Supervisely API object.
    :type api: sly.Api
    :param project_id: Project ID to process images from.
    :type project_id: int
    :param dataset_id: Dataset ID to process images from.
    :type dataset_id: int, optional
    :param image_ids: List of image IDs to process.
    :type image_ids: List[int], optional
    """
    # Get image infos from the project.
    image_infos = await get_image_infos(
        api,
        g.IMAGE_SIZE_FOR_CAS,
        g.IMAGE_SIZE_FOR_ATLAS,
        dataset_id=dataset_id,
        image_ids=image_ids,
    )

    # Get diff of image infos, check if they are already
    # in the Qdrant collection and have the same updated_at field.
    qdrant_diff = await qdrant.get_diff(project_id, image_infos)

    await image_infos_to_db(project_id, qdrant_diff)


@sly.timeit
async def image_infos_to_db(project_id: int, image_infos: List[ImageInfoLite]) -> None:
    """Save image infos to the database.

    :param image_infos: List of image infos to save.
    :type image_infos: List[ImageInfoLite]
    """
    sly.logger.debug(f"Upserting {len(image_infos)} vectors to Qdrant.")
    for image_batch in sly.batched(image_infos):
        # Get vectors from images.
        vectors_batch = await cas.get_vectors(
            [image_info.cas_url for image_info in image_batch]
        )

        sly.logger.debug(f"Received {len(vectors_batch)} vectors.")

        # Upsert vectors to Qdrant.
        await qdrant.upsert(
            project_id,
            vectors_batch,
            image_batch,
        )
    sly.logger.debug(f"Upserted {len(image_infos)} vectors to Qdrant.")