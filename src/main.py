import base64
from typing import List

import supervisely as sly
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from supervisely.api.module_api import ApiField

import src.cas as cas
import src.globals as g
import src.qdrant as qdrant
from src.events import Event
from src.utils import (ImageInfoLite, get_datasets, get_image_infos,
                       parse_timestamp, timeit)
from src.visualization import (create_projections, get_projections,
                               projections_up_to_date, save_projections)

app = sly.Application()
server = app.get_server()

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
    # Examples of requests:
    # 1. Search for similar images using text prompt.
    # data = {"project_id": <your-project-id>, "limit": <limit>, "prompt": <text-prompt>}
    # 2. Search for similar images using image IDs.
    # data = {"project_id": <your-project-id>, "limit": <limit>, "image_ids": [<image-id-1>, <image-id-2>, ...]}
    # 3. Both text prompt and image IDs can be provided at the same time.
    # response =api.task.send_request(task_id, "search", data) # Do not skip response.

    sly.logger.info(
        f"Searching for similar images in project {event.project_id}, limit: {event.limit}. "
        f"Text prompt: {event.prompt}, Image IDs: {event.image_ids}."
    )

    text_prompts = []
    if event.prompt:
        if isinstance(event.prompt, list):
            text_prompts = event.prompt
        else:
            # If prompt is a comma-separated string, split it into a list.
            text_prompts = event.prompt.split(",")

    sly.logger.debug(f"Formatted text prompts: {text_prompts}")

    # If request contains image IDs, get image URLs to add to the query.
    image_infos = []
    if event.image_ids:
        image_infos = await get_image_infos(
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
    query_vectors = await cas.get_vectors(query)
    sly.logger.debug(
        f"The query has been vectorized and will be used for search. Number of vectors: {len(query_vectors)}."
    )

    image_infos = await qdrant.search(event.project_id, query_vectors[0], event.limit)
    sly.logger.debug(f"Found {len(image_infos)} similar images.")

    return image_infos


@app.event(Event.Diverse, use_state=True)
@timeit
async def diverse(api: sly.Api, event: Event.Diverse) -> List[ImageInfoLite]:
    # Examples of requests:
    # 1. Generate diverse population using KMeans method.
    # data = {"project_id": <your-project-id>, "limit": <limit>, "method": "kmeans"}
    # response = api.task.send_request(task_id, "diverse", data) # Do not skip response.
    sly.logger.info(
        f"Generating diverse population for project {event.project_id}. "
        f"Method: {event.method}, Limit: {event.limit}."
    )

    image_infos = await qdrant.diverse(
        event.project_id,
        event.limit,
        event.method,
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
    if dataset_id is None and image_ids is None:
        datasets = api.dataset.get_list(project_id)
        image_ids = [info.id for dataset in datasets for info in api.image.get_list(dataset_id=dataset.id)]

    # Get image infos from the project.
    image_infos = await get_image_infos(
        api,
        g.IMAGE_SIZE_FOR_CAS,
        dataset_id=dataset_id,
        image_ids=image_ids,
    )

    if qdrant.collection_exists(project_id):
        # Get diff of image infos, check if they are already
        # in the Qdrant collection and have the same updated_at field.
        image_infos = await qdrant.get_diff(project_id, image_infos)

    await image_infos_to_db(project_id, image_infos)


async def base64_from_url(url):
    g.api._set_async_client()
    r = await g.api.async_httpx_client.get(url)
    b = bytes()
    async for chunk in r.aiter_bytes():
        b += chunk
    img_base64 = base64.b64encode(b)
    data_url = "data:image/png;base64,{}".format(str(img_base64, "utf-8"))
    return data_url


@timeit
async def image_infos_to_db(project_id: int, image_infos: List[ImageInfoLite]) -> None:
    """Save image infos to the database.

    :param image_infos: List of image infos to save.
    :type image_infos: List[ImageInfoLite]
    """
    sly.logger.debug(f"Upserting {len(image_infos)} vectors to Qdrant.")
    for image_batch in sly.batched(image_infos):
        # Get vectors from images.
        base64_data = [await base64_from_url(image_info.cas_url) for image_info in image_batch]
        vectors_batch = await cas.get_vectors(
            base64_data
        )
        sly.logger.debug(f"Received {len(vectors_batch)} vectors: {vectors_batch[0]}")
        sly.logger.debug(f"Received {len(image_batch)} images: {image_batch[0]}")
        # Upsert vectors to Qdrant.
        await qdrant.upsert(
            project_id,
            vectors_batch,
            image_batch,
        )
    sly.logger.debug(f"Upserted {len(image_infos)} vectors to Qdrant.")


@app.event(Event.Projections, use_state=True)
@timeit
async def projections_event_endpoint(api: sly.Api, event: Event.Projections):
    if projections_up_to_date(api, event.project_id):
        image_infos, projections = await get_projections(api, event.project_id)
    image_infos, projections = await create_projections(api, event.project_id)
    await save_projections(api, event.project_id, image_infos, projections)
    return [[info.to_json() for info in image_infos], projections]


@timeit
async def _update_embeddings(api: sly.Api, project_id: int):
    project_info = api.project.get_info_by_id(project_id)
    custom_data = project_info.custom_data
    if custom_data is None:
        custom_data = {}
    if custom_data.get("embeddings", False):
        return
    emb_updated_at = custom_data.get("embeddings_updated_at", None)
    if emb_updated_at is None:
        await process_images(api, project_id)
    elif parse_timestamp(emb_updated_at) < project_info.updated_at:
        datasets = api.dataset.get_list(project_id)
        datasets = [dataset for dataset in datasets if parse_timestamp(dataset.updated_at) > parse_timestamp(emb_updated_at)]
        image_ids = [info.id for dataset in datasets for info in api.image.get_list(dataset_id=dataset.id)]
        await process_images(api, project_id, image_ids)
    custom_data["embeddings_updated_at"] = project_info.updated_at
    api.project.update_custom_data(project_id)


@app.event(Event.UpdateEmbeddings, use_state=True)
async def update_embeddings_event_endpoint(api: sly.Api, event: Event.UpdateEmbeddings):
    return await _update_embeddings(api, event.project_id)


async def update_all_embeddings():
    projects = g.api.project.get_list_all()["entities"]
    for project in projects:
        if project.custom_data is None:
            continue
        if project.custom_data.get("embeddings", False):
            await _update_embeddings(g.api, project.id)


async def run_safe(func, *args, **kwargs):
    try:
        return await func(*args, **kwargs)
    except Exception as e:
        sly.logger.error(f"Error in function {func.__name__}: {e}")
        return None


scheduler = AsyncIOScheduler()
scheduler.add_job(run_safe, args=[update_all_embeddings], trigger="interval", minutes=g.UPDATE_EMBEDDINGS_INTERVAL)


@server.on_event("startup")
def on_startup():
    scheduler.start()

app.call_before_shutdown(scheduler.shutdown)
