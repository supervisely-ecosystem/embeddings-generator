import asyncio
from typing import Dict, List

import supervisely as sly
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from fastapi import Request

import src.cas as cas
import src.globals as g
import src.qdrant as qdrant
from src.events import Event
from src.functions import auto_update_all_embeddings, process_images
from src.utils import (
    ImageInfoLite,
    embeddings_up_to_date,
    get_image_infos,
    get_project_info,
    run_safe,
    timeit,
    update_custom_data,
)
from src.visualization import (
    create_projections,
    get_projections,
    projections_up_to_date,
    save_projections,
)

app = sly.Application()
server = app.get_server()

if sly.is_development():
    # This will enable Advanced Debugging mode only in development mode.
    # Do not need to remove it in production.
    sly.app.development.enable_advanced_debug()


@app.event(Event.Embeddings, use_state=True)
@timeit
async def create_embeddings(api: sly.Api, event: Event.Embeddings) -> None:
    """
    Examples of requests:
    1. Calculate embeddings for all images in the project.
    data = {"project_id": <your-project-id>, "team_id": <your-team-id>}
    2. Calculate embeddings for specific images.
    data = {"image_ids": [<image-id-1>, <image-id-2>, ...], "team_id": <your-team-id>}
    api.task.send_request(task_id, "embeddings", data, skip_response=True)
    """
    sly.logger.info(
        "Started creating embeddings for project %s. Force: %s, Image IDs: %s.",
        event.project_id,
        event.force,
        event.image_ids,
    )

    if event.force:
        # Step 1: If force is True, delete the collection and recreate it.
        sly.logger.debug("Force enabled, deleting collection %s.", event.project_id)
        await qdrant.delete_collection(event.project_id)

    # Step 2: Ensure collection exists in Qdrant.
    await qdrant.get_or_create_collection(event.project_id)

    # Step 3: Process images.
    await process_images(api, event.project_id, image_ids=event.image_ids)

    if event.image_ids is None:
        # Step 4: Update custom data.
        project_info = await get_project_info(api, event.project_id)
        custom_data = project_info.custom_data or {}
        custom_data["embeddings_updated_at"] = project_info.updated_at
        await update_custom_data(api, event.project_id, custom_data)

    sly.logger.debug("Embeddings for project %s have been created.", event.project_id)


@app.event(Event.Search, use_state=True)
@timeit
async def search(api: sly.Api, event: Event.Search) -> List[List[Dict]]:
    # Examples of requests:
    # 1. Search for similar images using text prompt.
    # data = {"project_id": <your-project-id>, "limit": <limit>, "prompt": <text-prompt>}
    # 2. Search for similar images using image IDs.
    # data = {"project_id": <your-project-id>, "limit": <limit>, "image_ids": [<image-id-1>, <image-id-2>, ...]}
    # 3. Both text prompt and image IDs can be provided at the same time.
    # response =api.task.send_request(task_id, "search", data) # Do not skip response.
    # returns a list of ImageInfoLite objects for each query.
    # response: List[List[Dict]]

    sly.logger.info(
        "Searching for similar images in project %s, limit: %s. Text prompt: %s, Image IDs: %s.",
        event.project_id,
        event.limit,
        event.prompt,
        event.image_ids,
    )

    text_prompts = []
    if event.prompt:
        if isinstance(event.prompt, list):
            text_prompts = event.prompt
        else:
            # If prompt is a comma-separated string, split it into a list.
            text_prompts = event.prompt.split(",")

    sly.logger.debug("Formatted text prompts: %s", text_prompts)

    # If request contains image IDs, get image URLs to add to the query.
    image_infos = []
    if event.image_ids:
        image_infos = await get_image_infos(
            api,
            cas_size=g.IMAGE_SIZE_FOR_CAS,
            project_id=event.project_id,
            image_ids=event.image_ids,
        )
        sly.logger.debug(
            "Request contains image IDs, obtained %d image infos. Will use their URLs for the query.",
            len(image_infos),
        )
    image_urls = [image_info.cas_url for image_info in image_infos]

    # Combine text prompts and image URLs to create a query.
    queries = text_prompts + image_urls
    sly.logger.info("Final query: %s", queries)

    # Vectorize the query data (can be a text prompt or an image URL).
    query_vectors = await cas.get_vectors(queries)
    sly.logger.debug(
        "The query has been vectorized and will be used for search. Number of vectors: %d.",
        len(query_vectors),
    )

    result = []
    tasks = []

    async def _search_task(project_id, vector, limit, query):
        return (await qdrant.search(project_id, vector, limit), query)

    for vector, query in zip(query_vectors, queries):
        tasks.append(
            asyncio.create_task(_search_task(event.project_id, vector, event.limit, query))
        )

    for task in asyncio.as_completed(tasks):
        infos, query = await task
        result.append([info.to_json() for info in infos])
        sly.logger.debug("Found %d similar images for a query %s", len(infos), query)

    return result


@app.event(Event.Diverse, use_state=True)
@timeit
async def diverse(api: sly.Api, event: Event.Diverse) -> List[ImageInfoLite]:
    # Examples of requests:
    # 1. Generate diverse population using KMeans method.
    # data = {"project_id": <your-project-id>, "limit": <limit>, "method": "kmeans"}
    sly.logger.info(
        "Generating diverse population for project %s. Method: %s, Limit: %s.",
        event.project_id,
        event.method,
        event.limit,
    )

    image_infos = await qdrant.diverse(
        event.project_id,
        event.limit,
        event.method,
    )
    sly.logger.debug("Generated %d diverse images.", len(image_infos))

    return image_infos


@app.event(Event.Projections, use_state=True)
@timeit
async def projections_event_endpoint(api: sly.Api, event: Event.Projections):
    project_info = await get_project_info(api, event.project_id)
    if await projections_up_to_date(api, event.project_id, project_info=project_info):
        sly.logger.debug("Projections are up to date. Loading from file.")
        image_infos, projections = await get_projections(
            api, event.project_id, project_info=project_info
        )
    else:
        sly.logger.debug("Projections are not up to date. Creating new projections.")
        image_infos, projections = await create_projections(api, event.project_id)
        await save_projections(
            api, event.project_id, image_infos, projections, project_info=project_info
        )
    indexes = []
    for i, info in enumerate(image_infos):
        if event.image_ids is None or info.id in event.image_ids:
            indexes.append(i)
    sly.logger.debug("Returning %d projections.", len(indexes))
    return [[image_infos[i].to_json() for i in indexes], [projections[i] for i in indexes]]


@server.post("/embeddings_up_to_date")
async def embeddings_up_to_date_endpoint(request: Request):
    state = request.state.state
    project_id = state["project_id"]
    return await embeddings_up_to_date(g.api, project_id)


@server.post("/projections_up_to_date")
async def projections_up_to_date_endpoint(request: Request):
    state = request.state.state
    project_id = state["project_id"]
    return await projections_up_to_date(g.api, project_id)


scheduler = AsyncIOScheduler()
scheduler.add_job(
    run_safe,
    args=[auto_update_all_embeddings],
    trigger="interval",
    minutes=g.UPDATE_EMBEDDINGS_INTERVAL,
)


@server.on_event("startup")
def on_startup():
    scheduler.start()


app.call_before_shutdown(scheduler.shutdown)
