from typing import List

import supervisely as sly

import src.cas as cas
import src.globals as g
import src.qdrant as qdrant
from src.events import Event
from src.utils import ImageInfoLite, get_datasets, get_image_infos, timeit

app = sly.Application()

# This will enable Advanced Debugging mode only in development mode.
# Do not need to remove it in production.
sly.app.development.enable_advanced_debug()


@app.event(Event.Embeddings, use_state=True)
@timeit
async def create_embeddings(api: sly.Api, event: Event.Embeddings) -> None:
    # Examples of requests:
    # 1. Calculate embeddings for all images in the project.
    # data = {"project_id": <your-project-id>, "team_id": <your-team-id>}
    # 2. Calculate embeddings for specific images.
    # data = {"image_ids": [<image-id-1>, <image-id-2>, ...], "team_id": <your-team-id>}
    # api.task.send_request(task_id, "embeddings", data, skip_response=True)

    sly.logger.info(
        f"Started creating embeddings for project {event.project_id}. "
        f"Force: {event.force}, Image IDs: {event.image_ids}."
    )

    if event.force:
        # Step 1: If force is True, delete the collection and recreate it.
        sly.logger.debug(f"Force enabled, deleting collection {event.project_id}.")
        await qdrant.delete_collection(event.project_id)

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
@timeit
async def search(api: sly.Api, event: Event.Search) -> List[ImageInfoLite]:
    sly.logger.info(
        f"Searching for similar images in project {event.project_id}, limit: {event.limit}. "
        f"Text prompt: {event.prompt}, Image IDs: {event.image_ids}."
    )

    # If request contains text prompt, prepare a list for query.
    text_prompts = [event.prompt] if event.prompt else []

    # If request contains image IDs, get image URLs to add to the query.
    image_infos = []
    if event.image_ids:
        image_infos = get_image_infos(
            api, g.IMAGE_SIZE_FOR_CAS, image_ids=event.image_ids
        )
        sly.logger.debug(
            f"Request contains image IDs, obtained {len(image_infos)} image infos. "
            "Will use their URLs for the query."
        )
    image_urls = [image_info.cas_url for image_info in image_infos]

    # Combine text prompts and image URLs to create a query.
    query = text_prompts + image_urls
    sly.logger.info(f"Final query: {query}")

    # Vectorize the query data (can be a text prompt or an image URL).
    query_vectors = await cas.get_vectors([query])

    image_infos = await qdrant.search(event.project_id, query_vectors[0], event.limit)
    sly.logger.debug(f"Found {len(image_infos)} similar images.")

    return image_infos


@app.event(Event.Diverse, use_state=True)
@timeit
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


@timeit
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
        dataset_id=dataset_id,
        image_ids=image_ids,
    )

    # Get diff of image infos, check if they are already
    # in the Qdrant collection and have the same updated_at field.
    qdrant_diff = await qdrant.get_diff(project_id, image_infos)

    await image_infos_to_db(project_id, qdrant_diff)


@timeit
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
