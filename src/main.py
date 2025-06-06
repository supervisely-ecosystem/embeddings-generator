import asyncio
import datetime
from typing import Dict, List

import numpy as np
import supervisely as sly
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from fastapi import Request
from fastapi.responses import JSONResponse
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
    ApiField,
    ClusteringMethods,
    EventFields,
    ImageInfoLite,
    ResponseFields,
    ResponseStatus,
    SamplingMethods,
    create_collection_and_populate,
    embeddings_up_to_date,
    get_lite_image_infos,
    get_project_info,
    image_get_list_async,
    parse_timestamp,
    run_safe,
    send_request,
    set_embeddings_in_progress,
    set_image_embeddings_updated_at,
    set_project_embeddings_updated_at,
    timeit,
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
        f"Started creating embeddings for project {event.project_id}",
        extra={
            "force": event.force,
            "return_vectors": event.return_vectors,
            "image_ids": event.image_ids,
        },
    )

    collection_msg = f"[Collection: {event.project_id}] "

    project_info: sly.ProjectInfo = await get_project_info(api, event.project_id)

    # Step 0: Check if embeddings are already in progress.
    if project_info.embeddings_in_progress:
        message = f"{collection_msg} Embeddings creation is already in progress. Skipping."
        sly.logger.info(message)
        return JSONResponse({ResponseFields.MESSAGE: message}, status_code=409)

    # Step 1: Check if embeddings are already up to date.
    if event.force:
        images_to_create = await image_get_list_async(api, event.project_id)
        images_to_delete = []
    elif event.image_ids:
        images_to_create = await image_get_list_async(
            api,
            event.project_id,
            images_ids=event.image_ids,
            wo_embeddings=True,
        )
        if project_info.embeddings_updated_at is not None:
            images_to_delete = await image_get_list_async(
                api,
                event.project_id,
                images_ids=event.image_ids,
                deleted_after=project_info.embeddings_updated_at,
            )
        else:
            images_to_delete = []
    else:
        images_to_create = await image_get_list_async(api, event.project_id, wo_embeddings=True)
        if project_info.embeddings_updated_at is not None:
            images_to_delete = await image_get_list_async(
                api, event.project_id, deleted_after=project_info.embeddings_updated_at
            )
        else:
            images_to_delete = []

    if len(images_to_create) == 0 and len(images_to_delete) == 0:
        message = f"{collection_msg} Embeddings are up to date. Skipping."
        sly.logger.info(message)
        return JSONResponse({ResponseFields.MESSAGE: message}, status_code=409)

    async def execute():
        try:
            await set_embeddings_in_progress(api, event.project_id, True)

            # Step 2: If force is True, delete the collection and recreate it.
            if event.force and not project_info.embeddings_in_progress:
                sly.logger.debug(f"{collection_msg} Force enabled, deleting collection.")
                await qdrant.delete_collection(event.project_id)

            await qdrant.get_or_create_collection(event.project_id)

            # Step 3: Process images.
            image_infos, vectors = await process_images(
                api,
                event.project_id,
                to_create=images_to_create,
                to_delete=images_to_delete,
                return_vectors=event.return_vectors,
                check_collection_exists=False,
            )

            # if len(image_infos) > 0:
            #     # Step 4: Delete all AI Search Entities Collections in the project as they are outdated.

            #     await SearchCache.drop_cache(api, event.project_id)
            await set_project_embeddings_updated_at(api, event.project_id)
            if event.return_vectors:
                # _, vectors = await qdrant.get_items_by_id(
                #     qdrant.IMAGES_COLLECTION, image_infos, with_vectors=True
                # )

                sly.logger.info(
                    f"{collection_msg} Embeddings creation has been completed. "
                    f"{len(image_infos)} images vectorized. {len(images_to_delete)} images deleted. {len(vectors)} vectors returned.",
                )
                await set_embeddings_in_progress(api, event.project_id, False)
                return JSONResponse(
                    {
                        ResponseFields.IMAGE_IDS: [info.id for info in image_infos],
                        ResponseFields.VECTORS: vectors,
                    }
                )
            else:
                await set_embeddings_in_progress(api, event.project_id, False)
                message = f"{collection_msg} Embeddings creation has been completed."
                sly.logger.info(message)
                return JSONResponse({ResponseFields.MESSAGE: message})
        except Exception as e:
            message = f"{collection_msg} Error while creating embeddings: {str(e)}"
            sly.logger.error(message, exc_info=True)
            await set_embeddings_in_progress(api, event.project_id, False)
            return JSONResponse({ResponseFields.MESSAGE: message}, status_code=500)

    task_id = f"embeddings_{event.project_id}_{datetime.datetime.now().timestamp()}"
    task = asyncio.create_task(execute())
    g.background_tasks[task_id] = task
    return JSONResponse(
        {
            ResponseFields.MESSAGE: "Embeddings creation started.",
            ResponseFields.BACKGROUND_TASK_ID: task_id,
        }
    )


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
    try:
        sly.logger.info(
            f"Searching for similar images in project {event.project_id}",
            extra={
                "prompt": event.prompt,
                "by_image_ids": event.by_image_ids,
                "limit": event.limit,
                "image_ids": event.image_ids,
                "dataset_id": event.dataset_id,
                "threshold": event.threshold,
            },
        )

        settings = {
            EventFields.LIMIT: event.limit,
            EventFields.BY_IMAGE_IDS: event.by_image_ids if event.by_image_ids else None,
            EventFields.IMAGE_IDS: event.image_ids,
            EventFields.DATASET_ID: event.dataset_id,
            EventFields.THRESHOLD: event.threshold,
        }

        # ------------------------ Step 1: Get Cache For The Search If Available. ------------------------ #
        prompt_str = (
            ",".join(event.prompt) if isinstance(event.prompt, list) else str(event.prompt or "")
        )
        cache = SearchCache(api, event.project_id, prompt_str, settings)

        if cache.collection_id is not None and cache.updated_at is not None:
            if cache.project_info.is_embeddings_updated is False:
                # If the project has outdated embeddings, invalidate the cache.
                sly.logger.info(
                    "Project %s has outdated embeddings. Invalidating cache.", event.project_id
                )
                await cache.clear()
            else:
                sly.logger.info("Using cached search results")
                return JSONResponse({ResponseFields.COLLECTION_ID: cache.collection_id})

        # Collect project images info to use later for mapping.
        image_infos = await image_get_list_async(api, event.project_id)
        if image_infos is None or len(image_infos) == 0:
            return JSONResponse({ResponseFields.MESSAGE: "Project is empty."})

        # ------------------------- Step 2: Update Payloads In Qdrant If Needed. ------------------------- #
        populate_response = await qdrant.populate_payloads(event.project_id, image_infos)
        if isinstance(populate_response, List):
            sly.logger.debug(
                f"Project {event.project_id} does not have {len(populate_response)} images in Qdrant. Will update embeddings."
            )
            image_infos_to_update = [image_infos[i] for i in populate_response]
            if len(image_infos_to_update) > 0:
                await process_images(api, event.project_id, image_infos=image_infos_to_update)
        elif isinstance(populate_response, JSONResponse):
            return populate_response
        #! remove deleted items from payloads
        # -------------------- Step 3: Prepare Text Prompts And Image URLs For Search -------------------- #
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

        if len(queries) == 0:
            return JSONResponse({ResponseFields.MESSAGE: "No queries provided."})

        # ----------- Step 4: Vectorize The Query Data (can Be A Text Prompt Or An Image URL). ----------- #
        query_vectors = await cas.get_vectors(queries)
        sly.logger.debug(
            "The query has been vectorized and will be used for search. Number of vectors: %d.",
            len(query_vectors),
        )

        # -------------------------- Step 5: Search For Similar Images In Qdrant ------------------------- #
        tasks = []

        async def _search_task(
            collection: int, vector: np.ndarray, limit: int, query_filter, query
        ):
            return (
                await qdrant.search(
                    collection_name=collection,
                    query_vector=vector,
                    limit=limit,
                    query_filter=query_filter,
                    score_threshold=event.threshold,
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

        results = []
        for task in asyncio.as_completed(tasks):
            search_results, query = await task
            items = search_results[qdrant.SearchResultField.ITEMS]
            if len(items) == 0:
                sly.logger.debug("No similar images found for query %s", query)
                continue
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
        if len(results) == 0:
            return JSONResponse({ResponseFields.MESSAGE: "No similar images found."})
        # Cache the results before returning them
        collection_id = await cache.save(results)
        return JSONResponse({ResponseFields.COLLECTION_ID: collection_id})
    except Exception as e:
        sly.logger.error(f"Error during search: {str(e)}", exc_info=True)
        return JSONResponse({ResponseFields.MESSAGE: f"Search failed: {str(e)}"})


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
            "num_clusters": event.sample_size,
            "dataset_id": event.dataset_id,
            "image_ids": event.image_ids,
        },
    )

    # ------------------------------------ Step 1: Get Image Info ------------------------------------ #
    if event.image_ids:
        image_infos = await image_get_list_async(api, event.project_id, images_ids=event.image_ids)
    elif event.dataset_id:
        image_infos = await image_get_list_async(api, event.project_id, dataset_id=event.dataset_id)
    else:
        image_infos = await image_get_list_async(api, event.project_id)
        # image_hashes = [info.hash for info in image_infos]

    # ------------------------- Step 2: Update Payloads In Qdrant If Needed. ------------------------- #
    populate_response = await qdrant.populate_payloads(event.project_id, image_infos)
    if isinstance(populate_response, List):
        sly.logger.debug(
            f"Project {event.project_id} does not have {len(populate_response)} images in Qdrant. Will update embeddings."
        )
        image_infos_to_update = [image_infos[i] for i in populate_response]
        await process_images(api, event.project_id, image_infos=image_infos_to_update)
    elif isinstance(populate_response, JSONResponse):
        return populate_response
    #! remove deleted items from payloads

    # ----------------------------- Step 3: Get Image Vectors From Qdrant ---------------------------- #
    image_infos_result, vectors = await qdrant.get_items_by_id(
        qdrant.IMAGES_COLLECTION, image_infos, with_vectors=True
    )

    if len(vectors) == 0:
        return JSONResponse({ResponseFields.MESSAGE: "No vectors found."})

    image_infos_result = update_id_by_hash(image_infos, image_infos_result)

    # ------------------------- Step 4: Prepare Data For Projections Service ------------------------- #
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

    # --------------- Step 5: Send Request To Projections Service And Return The Result -------------- #
    samples = await send_request(
        api, task_id=g.projections_service_task_id, method="diverse", data=data
    )
    result = []
    sly.logger.debug("Generated diverse samples: %s", samples)
    for label, sample in samples.items():
        result.extend([image_infos_result[i].id for i in sample])

    sly.logger.debug("Generated %d diverse images.", len(result))

    if len(result) == 0:
        return JSONResponse({ResponseFields.MESSAGE: "No diverse images found."})

    # Delete existing collection items if any and create a new collection for the diverse population.
    collection_id = await create_collection_and_populate(
        api=api,
        project_id=event.project_id,
        name=f"Diverse Population for {event.project_id}",
        event=EventFields.DIVERSE,
        image_ids=result,
    )
    return JSONResponse({ResponseFields.COLLECTION_ID: collection_id})


@app.event(Event.Projections, use_state=True)
@timeit
async def projections_event_endpoint(api: sly.Api, event: Event.Projections):
    # TODO: Remove this response and implement projections.
    return JSONResponse({ResponseFields.MESSAGE: "Projections are not supported yet."})
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

    sly.logger.info(
        f"Generating clusters for project {event.project_id}.",
        extra={
            "reduction_dimensions": event.reduction_dimensions,
            "image_ids": event.image_ids,
            "save": event.save,
        },
    )
    if event.image_ids:
        image_infos = await image_get_list_async(api, event.project_id, images_ids=event.image_ids)
    else:
        image_infos = await image_get_list_async(api, event.project_id)
    # image_hashes = [info.hash for info in image_infos]
    image_infos_result, vectors = await qdrant.get_items_by_id(
        qdrant.IMAGES_COLLECTION, image_infos, with_vectors=True
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


@server.post("/check_background_task_status")
async def check_task_status(request: Request):
    try:
        req_data = await request.json()
        task_id = req_data.get("task_id")

        if not task_id or task_id not in g.background_tasks:
            return JSONResponse(
                {
                    ResponseFields.STATUS: ResponseStatus.NOT_FOUND,
                    ResponseFields.MESSAGE: f"Task with ID {task_id} not found",
                }
            )

        task = g.background_tasks[task_id]

        if task.done():
            # Get the result and clean up
            result = task.result()
            # Optionally remove the task from active_tasks to free memory
            # del g.active_tasks[task_id]  # Uncomment if you want to clean up
            return JSONResponse(
                {ResponseFields.STATUS: ResponseStatus.COMPLETED, ResponseFields.RESULT: result}
            )
        else:
            return JSONResponse(
                {
                    ResponseFields.STATUS: ResponseStatus.IN_PROGRESS,
                    ResponseFields.MESSAGE: "Task is still running",
                }
            )

    except Exception as e:
        return JSONResponse(
            {ResponseFields.STATUS: ResponseStatus.ERROR, ResponseFields.MESSAGE: str(e)}
        )


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
