import asyncio
import datetime
from typing import Dict, List

import numpy as np
import supervisely as sly
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from fastapi import Request
from supervisely.imaging.color import get_predefined_colors

import src.cas as cas
import src.globals as g
import src.qdrant as qdrant
from src.events import Event
from src.functions import auto_update_all_embeddings, process_images, update_embeddings
from src.pointcloud import download as download_pcd
from src.pointcloud import upload as upload_pcd
from src.search_cache import CollectionItem, SearchCache
from src.utils import (
    ClusteringMethods,
    EventFields,
    ImageInfoLite,
    ResponseFields,
    SamplingMethods,
    create_collection_and_populate,
    embeddings_up_to_date,
    get_lite_image_infos,
    get_project_info,
    image_get_list_async,
    parse_timestamp,
    run_safe,
    send_request,
    timeit,
    update_embeddings_data,
    update_id_by_hash,
)
from src.visualization import (
    create_projections,
    get_or_create_projections_dataset,
    get_pcd_info,
    get_projections,
    get_projections_pcd_name,
    is_projections_up_to_date,
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
        "Started creating embeddings for project %s. Force: %s, Image IDs: %s. Return vectors: %s.",
        event.project_id,
        event.force,
        event.image_ids,
        event.return_vectors,
    )
    project_info: sly.ProjectInfo = await get_project_info(api, event.project_id)

    if project_info.embeddings_in_progress:
        message = (
            f"Embeddings creation is already in progress for project {event.project_id}. Skipping."
        )
        sly.logger.info(message)
        return {ResponseFields.MESSAGE: message}
    if not event.force:
        if project_info.embeddings_updated_at is not None and parse_timestamp(
            project_info.embeddings_updated_at
        ) >= parse_timestamp(project_info.updated_at):
            message = f"Embeddings are up to date for project {event.project_id}. Skipping."
            sly.logger.info(message)
            return {ResponseFields.MESSAGE: message}

    try:
        api.project.set_embeddings_in_progress(event.project_id, True)
        # Step 1: Ensure collection exists in Qdrant.
        await qdrant.get_or_create_collection(qdrant.IMAGES_COLLECTION)

        if event.image_ids:
            image_infos = await image_get_list_async(
                api, event.project_id, images_ids=event.image_ids
            )
        else:
            image_infos = await image_get_list_async(api, event.project_id)

        # Step 2: If force is True, delete existing collection items.
        if event.force:
            image_hashes = [info.hash for info in image_infos]
            sly.logger.debug(
                "Force enabled for project %s, will recreate items in collection %s.",
                event.project_id,
                qdrant.IMAGES_COLLECTION,
            )
            await qdrant.delete_collection_items(image_hashes)

        # Step 3: Process images.
        image_infos = await process_images(api, event.project_id, image_infos=image_infos)
        image_hashes = [info.hash for info in image_infos]

        # if event.image_ids is None and len(image_infos) > 0:
        if len(image_infos) > 0:
            # Step 4: Update embeddings data.
            # project_info = await get_project_info(api, event.project_id)
            await update_embeddings_data(api, event.project_id)

        sly.logger.debug("Embeddings for project %s have been created.", event.project_id)

        if event.return_vectors:
            _, vectors = await qdrant.get_items_by_hashes(
                qdrant.IMAGES_COLLECTION, image_hashes, with_vectors=True
            )
            # image_infos_result = update_id_by_hash(image_infos, image_infos_result)

            sly.logger.debug("Got %d image infos and %d vectors.", len(image_infos), len(vectors))
            api.project.set_embeddings_in_progress(event.project_id, False)
            sly.logger.info(
                "Embeddings creation for project %s has been completed. %d images processed.",
                event.project_id,
                len(image_infos),
            )
            # collection_id = await create_collection_and_populate(
            #     api=api,
            #     project_id=event.project_id,
            #     name=f"Embeddings for {event.project_id}",
            #     image_ids=[info.id for info in image_infos],
            #     event=EventFields.EMBEDDINGS,
            # )
            api.project.set_embeddings_in_progress(event.project_id, False)
            return {
                # ResponseFields.COLLECTION_ID: collection_id,
                ResponseFields.IMAGE_IDS: [info.id for info in image_infos],
                ResponseFields.VECTORS: vectors,
            }
        else:
            return {
                ResponseFields.MESSAGE: f"Embeddings creation for project {event.project_id} has been completed. "
            }
    except Exception as e:
        sly.logger.error(
            "Error while creating embeddings for project %s: %s", event.project_id, str(e)
        )
        api.project.set_embeddings_in_progress(event.project_id, False)
        raise e


@app.event(Event.Search, use_state=True)
@timeit
async def search(api: sly.Api, event: Event.Search) -> List[List[Dict]]:
    # Examples of requests:
    # 1. Search for similar images using text prompt.
    # data = {"project_id": <your-project-id>, "limit": <limit>, "prompt": <text-prompt>}
    # 2. Search for similar images using image IDs.
    # data = {"project_id": <your-project-id>, "limit": <limit>, "by_image_ids": [<image-id-1>, <image-id-2>, ...]}
    # 3. Both text prompt and image IDs can be provided at the same time.
    # response =api.task.send_request(task_id, "search", data) # Do not skip response.
    # returns a list of ImageInfoLite objects for each query.
    # response: List[List[Dict]]

    sly.logger.info(
        "Searching for similar images in project %s, limit: %s. Prompts: text - %s, Image IDs - %s. "
        "Search will be performed by image IDs: %s, dataset ID: %s.",
        event.project_id,
        event.limit,
        event.prompt,
        event.by_image_ids,
        event.image_ids,
        event.dataset_id,
    )

    settings = {
        EventFields.LIMIT: event.limit,
        EventFields.BY_IMAGE_IDS: event.by_image_ids if event.by_image_ids else None,
        EventFields.IMAGE_IDS: event.image_ids,
        EventFields.DATASET_ID: event.dataset_id,
    }

    # Try to get cached results first
    prompt_str = (
        ",".join(event.prompt) if isinstance(event.prompt, list) else str(event.prompt or "")
    )
    cache = SearchCache(api, event.project_id, prompt_str, settings)

    if cache.collection_id is not None and cache.updated_at is not None:
        if cache.project_info.embeddings_updated_at is None:
            # If the project has no embeddings, invalidate the cache.
            sly.logger.info("Project %s has no embeddings. Invalidating cache.", event.project_id)
            cache.clear()
        elif parse_timestamp(cache.updated_at) < parse_timestamp(
            cache.project_info.embeddings_updated_at
        ):
            # If the cache is older than the project update timestamp, invalidate it.
            sly.logger.info(
                "Cached search results are older than the project update. Invalidating cache."
            )
            cache.clear()
        else:
            sly.logger.info("Using cached search results")
            return {ResponseFields.COLLECTION_ID: cache.collection_id}

    # Collect project images info to use later for mapping.
    image_infos = await image_get_list_async(api, event.project_id)
    if image_infos is None or len(image_infos) == 0:
        return {ResponseFields.MESSAGE: "Project is empty."}

    text_prompts = []
    if event.prompt:
        if isinstance(event.prompt, list):
            text_prompts = event.prompt
        else:
            # If prompt is a comma-separated string, split it into a list.
            text_prompts = event.prompt.split(",")

    sly.logger.debug("Formatted text prompts: %s", text_prompts)

    image_urls = []
    # If request contains image IDs, get image URLs to add to the query.
    if event.by_image_ids:
        filtered_image_infos = [info for info in image_infos if info.id in event.by_image_ids]
        lite_image_infos = await get_lite_image_infos(
            api,
            cas_size=g.IMAGE_SIZE_FOR_CAS,
            project_id=event.project_id,
            image_infos=filtered_image_infos,
        )
        sly.logger.debug(
            "Request contains image IDs, obtained %d image infos. Will use their URLs for the query.",
            len(lite_image_infos),
        )
        image_urls = [image_info.cas_url for image_info in lite_image_infos]
        #! check if we need to separate requests

    # Combine text prompts and image URLs to create a query.
    queries = text_prompts + image_urls
    sly.logger.info("Final query: %s", queries)

    # Vectorize the query data (can be a text prompt or an image URL).
    query_vectors = await cas.get_vectors(queries)
    sly.logger.debug(
        "The query has been vectorized and will be used for search. Number of vectors: %d.",
        len(query_vectors),
    )

    results = []
    tasks = []

    async def _search_task(collection: int, vector: np.ndarray, limit: int, query_filter, query):
        return (
            await qdrant.search(
                collection_name=collection,
                query_vector=vector,
                limit=limit,
                query_filter=query_filter,
            ),
            query,
        )

    search_filter = None

    if event.image_ids:
        # If image_ids are provided, create a filter for the search.
        search_filter = qdrant.get_search_filter(image_ids=event.image_ids)
    elif event.dataset_id:
        # If dataset_id is provided, create a filter for the search.
        search_filter = qdrant.get_search_filter(dataset_id=event.dataset_id)
    else:
        # If project_id is provided, create a filter for the search.
        search_filter = qdrant.get_search_filter(project_id=event.project_id)

    for vector, query in zip(query_vectors, queries):
        tasks.append(
            asyncio.create_task(
                _search_task(
                    collection=qdrant.IMAGES_COLLECTION,
                    vector=vector,
                    limit=event.limit,
                    query_filter=search_filter,
                    query=query,
                )
            )
        )

    for task in asyncio.as_completed(tasks):
        search_results, query = await task
        items = search_results[qdrant.SearchResultField.ITEMS]
        items = update_id_by_hash(image_infos, items)
        # items = [info.to_json() for info in items]
        if search_results.get(qdrant.SearchResultField.SCORES, None) is not None:
            for i, score in enumerate(search_results[qdrant.SearchResultField.SCORES]):
                items[i].score = score
        else:
            for i in range(len(items)):
                items[i].score = None
        results.extend(items)
        sly.logger.debug("Found %d similar images for a query %s", len(items), query)

    # Cache the results before returning them
    collection_id = cache.save(results)
    return {ResponseFields.COLLECTION_ID: collection_id}


@app.event(Event.Diverse, use_state=True)
@timeit
async def diverse(api: sly.Api, event: Event.Diverse) -> List[ImageInfoLite]:
    """
    Generates a representative subset of images from a project by leveraging CLIP embeddings and clustering techniques.
    It works by retrieving image vectors from Qdrant, sending them to a projections service that applies clustering algorithms (like KMeans),
    and then returning a selection of images that maximally represent the visual diversity of the entire collection.
    This approach is particularly valuable for creating balanced training datasets, getting a quick overview of content variety,
    and identifying outliers without having to manually review the entire image collection.

    # Examples of requests:
    # 1. Generate diverse population using KMeans method.
    # data = {"project_id": <your-project-id>, "limit": <limit>, "method": "kmeans"}

    """

    sly.logger.info(
        f"Generating diverse population for project {event.project_id}.",
        extra={
            "sampling_method": event.sampling_method,
            "sample_size": event.sample_size,
            "clustering_method": event.clustering_method,
            "num_clusters": event.num_clusters,
            "dataset_id": event.dataset_id,
            "image_ids": event.image_ids,
        },
    )

    if event.image_ids:
        image_infos = await image_get_list_async(api, event.project_id, images_ids=event.image_ids)
    elif event.dataset_id:
        image_infos = await image_get_list_async(api, event.project_id, dataset_id=event.dataset_id)
    else:
        image_infos = await image_get_list_async(api, event.project_id)
    image_hashes = [info.hash for info in image_infos]
    image_infos_result, vectors = await qdrant.get_items_by_hashes(
        qdrant.IMAGES_COLLECTION, image_hashes, with_vectors=True
    )

    image_infos_result = update_id_by_hash(image_infos, image_infos_result)

    data = {
        "vectors": vectors,
        "sample_size": event.sample_size,
        "sampling_method": event.sampling_method,
    }
    if event.clustering_method == ClusteringMethods.KMEANS:
        data["settings"] = {
            "num_clusters": event.num_clusters,
            "clustering_method": event.clustering_method,
        }
    elif event.clustering_method == ClusteringMethods.DBSCAN:
        data["settings"] = {"clustering_method": event.clustering_method}

    samples = await send_request(
        api, task_id=g.projections_service_task_id, method="diverse", data=data
    )
    result = []
    sly.logger.debug("Generated diverse samples: %s", samples)
    for label, sample in samples.items():
        result.extend([image_infos_result[i].id for i in sample])

    sly.logger.debug("Generated %d diverse images.", len(result))

    if len(result) == 0:
        return {ResponseFields.MESSAGE: "No diverse images found."}

    # Delete existing collection items if any and create a new collection for the diverse population.
    collection_id = await create_collection_and_populate(
        api=api,
        project_id=event.project_id,
        name=f"Diverse Population for {event.project_id}",
        event=EventFields.DIVERSE,
        image_ids=result,
    )
    return {ResponseFields.COLLECTION_ID: collection_id}


@app.event(Event.Projections, use_state=True)
@timeit
async def projections_event_endpoint(api: sly.Api, event: Event.Projections):
    project_info = await get_project_info(api, event.project_id)
    pcd_info = None
    try:
        pcd_info: sly.api.pointcloud_api.PointcloudInfo = await get_pcd_info(
            api, event.project_id, project_info=project_info
        )
    except ValueError as e:
        sly.logger.debug("Projections not found. Creating new projections.")
    else:
        if parse_timestamp(pcd_info.updated_at) < parse_timestamp(project_info.updated_at):
            sly.logger.debug("Projections are not up to date. Creating new projections.")
            pcd_info = None

    if pcd_info is None:
        # update embeddings
        await update_embeddings(
            api,
            event.project_id,
            force=False,
            project_info=project_info,
        )

        # create new projections
        image_infos, projections = await create_projections(
            api, event.project_id, image_ids=event.image_ids
        )
        # save projections
        await save_projections(
            api,
            project_id=event.project_id,
            image_infos=image_infos,
            projections=projections,
            project_info=project_info,
        )
    else:
        image_infos, projections = await get_projections(
            api, event.project_id, project_info=project_info, pcd_info=pcd_info
        )
    indexes = []
    for i, info in enumerate(image_infos):
        if event.image_ids is None or info.id in event.image_ids:
            indexes.append(i)
    sly.logger.debug("Returning %d projections.", len(indexes))
    return [[image_infos[i].to_json() for i in indexes], [projections[i] for i in indexes]]


@app.event(Event.Clusters, use_state=True)
async def clusters_event_endpoint(api: sly.Api, event: Event.Clusters):
    if event.image_ids:
        image_infos = await image_get_list_async(api, event.project_id, images_ids=event.image_ids)
    else:
        image_infos = await image_get_list_async(api, event.project_id)
    image_hashes = [info.hash for info in image_infos]
    image_infos_result, vectors = await qdrant.get_items_by_hashes(
        qdrant.IMAGES_COLLECTION, image_hashes, with_vectors=True
    )
    image_infos_result = update_id_by_hash(image_infos, image_infos_result)
    data = {"vectors": vectors, "reduce": True}
    if event.reduction_dimensions:
        data["reduction_dimensions"] = event.reduction_dimensions
    labels = await send_request(
        api,
        g.projections_service_task_id,
        "clusters",
        data=data,
        timeout=60 * 5,
        retries=3,
        raise_error=True,
    )
    if event.save:
        project_info = await get_project_info(api, event.project_id)
        image_infos_result, vectors = await get_projections(
            api, event.project_id, project_info=project_info
        )
        image_infos_result, vectors = zip(
            *[
                (image_info, vector)
                for image_info, vector in zip(image_infos, vectors)
                if event.image_ids is None or image_info.id in event.image_ids
            ]
        )
        dataset = await get_or_create_projections_dataset(
            api, event.project_id, image_project_info=project_info
        )

        palette = get_predefined_colors((len(labels)))
        colors = [palette[(label + 1) % len(palette)] for label in labels]
        await upload_pcd(
            api,
            vectors,
            [info.id for info in image_infos_result],
            f"clusters_{datetime.datetime.now()}.pcd",
            dataset.id,
            cluster_ids=labels,
            colors=colors,
        )

    return [info.to_json() for info in image_infos_result], labels


@server.post("/embeddings_up_to_date")
async def embeddings_up_to_date_endpoint(request: Request):
    state = request.state.state
    project_id = state["project_id"]
    return await embeddings_up_to_date(g.api, project_id)


@server.post("/projections_up_to_date")
async def projections_up_to_date_endpoint(request: Request):
    state = request.state.state
    project_id = state["project_id"]
    return await is_projections_up_to_date(g.api, project_id)


scheduler = AsyncIOScheduler(
    job_defaults={
        "misfire_grace_time": 120,  # Allow jobs to be 2 minutes late
        "coalesce": True,  # Combine missed runs into a single run
    }
)
scheduler.add_job(
    run_safe,
    args=[auto_update_all_embeddings],
    trigger="interval",
    minutes=g.UPDATE_EMBEDDINGS_INTERVAL,
    max_instances=1,  # Prevent overlapping job instances
)


@server.on_event("startup")
def on_startup():
    scheduler.start()


app.call_before_shutdown(scheduler.shutdown)
