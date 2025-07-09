import asyncio
import datetime
from typing import Dict, List

import numpy as np
import supervisely as sly
from fastapi import Request
from fastapi.responses import JSONResponse
from supervisely.imaging.color import get_predefined_colors

import src.cas as cas
import src.globals as g
import src.qdrant as qdrant
from src.events import Event
from src.functions import process_images, update_embeddings
from src.pointcloud import upload as upload_pcd

# from src.search_cache import CollectionItem, SearchCache
from src.project_collection_manager import AiSearchCollectionManager, DiverseCollectionManager
from src.utils import (
    ClusteringMethods,
    ImageInfoLite,
    ResponseFields,
    ResponseStatus,
    create_lite_image_infos,
    embeddings_up_to_date,
    get_project_info,
    image_get_list_async,
    is_team_plan_sufficient,
    parse_timestamp,
    send_request,
    set_embeddings_in_progress,
    set_project_embeddings_updated_at,
    start_projections_service,
    timeit,
)
from src.visualization import (
    create_projections,
    get_or_create_projections_dataset,
    get_pcd_info,
    get_projections,
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
    Endpoint to create embeddings for images in a project.
    If `force` is True, it will delete the existing collection and create a new one.
    If `return_vectors` is True, it will return the vectors of the created embeddings.

    :param api: Supervisely API object.
    :param event: Event object containing parameters for creating embeddings.

    Examples of requests:
    1. Calculate embeddings for all images in the project.
    data = {"project_id": <your-project-id>, "team_id": <your-team-id>}
    2. Calculate embeddings for specific images.
    data = {"image_ids": [<image-id-1>, <image-id-2>, ...], "team_id": <your-team-id>}
    api.task.send_request(task_id, "embeddings", data, skip_response=True)
    """
    try:
        msg_prefix = f"[Project: {event.project_id}]"
        sly.logger.info(
            f"{msg_prefix} Started creating embeddings.",
            extra={
                "force": event.force,
                "return_vectors": event.return_vectors,
                "image_ids": event.image_ids,
            },
        )

        project_info: sly.ProjectInfo = await get_project_info(api, event.project_id)

        # --------------------- Step 0: Check Team Subscription Plan. --------------------- #
        if not await is_team_plan_sufficient(api, project_info.team_id):
            message = f"Team {project_info.team_id} with 'free' plan cannot create embeddings. "
            sly.logger.warning(message)
            return JSONResponse({ResponseFields.MESSAGE: message}, status_code=403)

        # ---------------------- Step 1-1: Check If Embeddings Enabled For Project. ---------------------- #

        if project_info.embeddings_enabled is not None and project_info.embeddings_enabled is False:
            message = f"{msg_prefix} Embeddings are disabled. Skipping."
            sly.logger.info(message)
            return JSONResponse({ResponseFields.MESSAGE: message}, status_code=200)

        # --------------------- Step 1-2: Check If Embeddings Are Already In Progress. --------------------- #

        if project_info.embeddings_in_progress:
            message = f"{msg_prefix} Embeddings creation is already in progress. Skipping."
            sly.logger.info(message)
            return JSONResponse({ResponseFields.MESSAGE: message}, status_code=200)

        # ---------------------- Step 2: Check If Embeddings Are Already Up To Date. --------------------- #

        if event.force:
            images_to_create = await image_get_list_async(api, event.project_id)
            images_to_delete = []
        elif event.image_ids:
            images_to_create = await image_get_list_async(
                api,
                event.project_id,
                image_ids=event.image_ids,
                wo_embeddings=True,
            )
            if project_info.embeddings_updated_at is not None:
                images_to_delete = await image_get_list_async(
                    api,
                    event.project_id,
                    image_ids=event.image_ids,
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
            message = f"{msg_prefix} Embeddings are up to date. Skipping."
            sly.logger.info(message)
            return JSONResponse({ResponseFields.MESSAGE: message}, status_code=200)

        async def execute():
            try:
                # ----------------------- Step 3: If Force Is True, Delete The Collection. ----------------------- #
                if event.force:
                    sly.logger.debug(f"{msg_prefix} Force enabled, deleting collection.")
                    await qdrant.delete_collection(event.project_id)

                # ---------------- Step 4: Process Images. Check And Create Collection If Needed. ---------------- #
                image_infos, vectors = await process_images(
                    api,
                    event.project_id,
                    to_create=images_to_create,
                    to_delete=images_to_delete,
                    return_vectors=event.return_vectors,
                )
                await set_project_embeddings_updated_at(api, event.project_id)
                if event.return_vectors:
                    sly.logger.info(
                        f"{msg_prefix} Embeddings creation has been completed. "
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
                    message = f"{msg_prefix} Embeddings creation has been completed."
                    sly.logger.info(message)
                    return JSONResponse({ResponseFields.MESSAGE: message})
            except Exception as e:
                message = f"{msg_prefix} Error while creating embeddings: {str(e)}"
                sly.logger.error(message, exc_info=True)
                await set_embeddings_in_progress(api, event.project_id, False)
                return JSONResponse({ResponseFields.MESSAGE: message}, status_code=500)

        await set_embeddings_in_progress(api, event.project_id, True)
        task_id = f"{event.project_id}"
        task = asyncio.create_task(execute())
        g.background_tasks[task_id] = task
        return JSONResponse(
            {
                ResponseFields.MESSAGE: f"{msg_prefix} Embeddings creation started.",
                ResponseFields.BACKGROUND_TASK_ID: task_id,
            }
        )
    except Exception as e:
        message = f"{msg_prefix} Error during embeddings creation: {str(e)}"
        sly.logger.error(message, exc_info=True)
        return JSONResponse({ResponseFields.MESSAGE: message}, status_code=500)


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
        msg_prefix = f"[Project: {event.project_id}]"
        sly.logger.info(
            f"{msg_prefix} Searching for similar images started.",
            extra={
                "prompt": event.prompt,
                "by_image_ids": event.by_image_ids,  # If True, search will be performed by image IDs.
                "limit": event.limit,
                "image_ids": event.image_ids,  # Image IDs to filter the search results.
                "dataset_id": event.dataset_id,
                "threshold": event.threshold,
            },
        )

        # ---------------------------- Step 0-1: Check Team Subscription Plan. --------------------------- #
        project_info: sly.ProjectInfo = await get_project_info(api, event.project_id)
        if not await is_team_plan_sufficient(api, project_info.team_id):
            message = (
                f"Team {project_info.team_id} with 'free' plan cannot use the AI search features."
            )
            sly.logger.error(message)
            return JSONResponse({ResponseFields.MESSAGE: message}, status_code=403)

        # ---------------------- Step 0-2: Check If Embeddings Enabled For Project. ---------------------- #

        if project_info.embeddings_enabled is not None and project_info.embeddings_enabled is False:
            message = f"{msg_prefix} Embeddings are disabled. Skipping."
            sly.logger.info(message)
            return JSONResponse({ResponseFields.MESSAGE: message}, status_code=200)

        # ----------------- Step 1: Initialise Collection Manager And List Of Image Infos ---------------- #
        if await qdrant.collection_exists(event.project_id) is False:
            message = f"{msg_prefix} Embeddings collection does not exist, search is not possible. Create embeddings first. Disabling AI Search."
            sly.logger.warning(message)
            api.project.disable_embeddings(project_info.id)
            return JSONResponse({ResponseFields.MESSAGE: message}, status_code=404)

        collection_manager = AiSearchCollectionManager(api, event.project_id)

        # -------------------- Step 2: Prepare Text Prompts And Image URLs For Search -------------------- #

        text_prompts = []
        if event.prompt:
            if isinstance(event.prompt, list):
                text_prompts = event.prompt
            else:
                # If prompt is a comma-separated string, split it into a list.
                text_prompts = event.prompt.split(",")

        sly.logger.debug(f"{msg_prefix} Formatted text prompts: {text_prompts}")

        image_urls = []
        # If request contains image IDs, get image URLs to add to the query.
        if event.by_image_ids:
            image_infos = await image_get_list_async(
                api, event.project_id, image_ids=event.by_image_ids
            )
            if image_infos is None or len(image_infos) == 0:
                return JSONResponse(
                    {ResponseFields.MESSAGE: "No images found for the provided IDs."}
                )

            lite_image_infos = await create_lite_image_infos(
                cas_size=g.IMAGE_SIZE_FOR_CLIP,
                image_infos=image_infos,
            )
            sly.logger.debug(
                f"{msg_prefix} Request contains image IDs, obtained {len(lite_image_infos)} image infos. Will use their URLs for the query.",
            )
            image_urls = [image_info.cas_url for image_info in lite_image_infos]

        # Combine text prompts and image URLs to create a query.
        queries = text_prompts + image_urls
        sly.logger.debug(f"{msg_prefix} Final search query: {queries}")

        if len(queries) == 0:
            return JSONResponse({ResponseFields.MESSAGE: "No queries provided."})

        # ----------- Step 3: Vectorize The Query Data (can Be A Text Prompt Or An Image URL). ----------- #

        query_vectors = await cas.get_vectors(queries)
        sly.logger.debug(
            f"{msg_prefix} The query has been vectorized and will be used for search. Number of vectors: {len(query_vectors)}.",
        )

        # -------------------------- Step 4: Search For Similar Images In Qdrant ------------------------- #

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

        for vector, query in zip(query_vectors, queries):
            tasks.append(
                asyncio.create_task(
                    _search_task(
                        collection=event.project_id,
                        vector=vector,
                        limit=event.limit,
                        query_filter=search_filter,
                        query=query,
                    )
                )
            )

        results = {}
        for task in asyncio.as_completed(tasks):
            search_results, query = await task
            items = search_results[qdrant.SearchResultField.ITEMS]
            if len(items) == 0:
                sly.logger.debug(f"{msg_prefix} No similar images found for query {query}")
                continue
            # items = [info.to_json() for info in items]
            if search_results.get(qdrant.SearchResultField.SCORES, None) is not None:
                for i, score in enumerate(search_results[qdrant.SearchResultField.SCORES]):
                    items[i].score = score
            else:
                for i in range(len(items)):
                    items[i].score = None
            # Update results dictionary, keeping only highest scoring items
            for item in items:
                item_id = item.id
                if item_id not in results:
                    # First occurrence of this item
                    results[item_id] = item
                else:
                    # Item already exists, compare scores
                    existing_score = results[item_id].score
                    new_score = item.score

                    # Update if new score is higher (handle None scores)
                    if existing_score is None and new_score is not None:
                        results[item_id] = item
                    elif (
                        existing_score is not None
                        and new_score is not None
                        and new_score > existing_score
                    ):
                        results[item_id] = item
                    # If both are None or new score is lower/equal, keep existing item
            sly.logger.info(
                f"{msg_prefix} Found {len(items)} similar images for a query '{query}'."
            )
        # Convert back to list for further processing
        results_list = list(results.values())
        if len(results_list) == 0:
            return JSONResponse({ResponseFields.MESSAGE: "No similar images found."})
        # Save the results to the Entities Collection.
        collection_id = await collection_manager.save(results_list)
        return JSONResponse({ResponseFields.COLLECTION_ID: collection_id})
    except Exception as e:
        sly.logger.error(f"{msg_prefix} Error during search: {str(e)}", exc_info=True)
        return JSONResponse({ResponseFields.MESSAGE: f"Search failed: {str(e)}"}, status_code=500)


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
    try:
        msg_prefix = f"[Project: {event.project_id}]"
        sly.logger.info(
            f"{msg_prefix} Generating diverse population.",
            extra={
                "sampling_method": event.sampling_method,
                "sample_size": event.sample_size,
                "clustering_method": event.clustering_method,
                "num_clusters": event.sample_size,
                "dataset_id": event.dataset_id,
                "image_ids": event.image_ids,
            },
        )

        # ---------------------------- Step 0-1: Check Team Subscription Plan. --------------------------- #
        project_info: sly.ProjectInfo = await get_project_info(api, event.project_id)
        if not await is_team_plan_sufficient(api, project_info.team_id):
            message = (
                f"Team {project_info.team_id} with 'free' plan cannot use the AI search features."
            )
            sly.logger.error(message)
            return JSONResponse({ResponseFields.MESSAGE: message}, status_code=403)

        # ---------------------- Step 0-2: Check If Embeddings Enabled For Project. ---------------------- #

        if project_info.embeddings_enabled is not None and project_info.embeddings_enabled is False:
            message = f"{msg_prefix} Embeddings are disabled. Skipping."
            sly.logger.info(message)
            return JSONResponse({ResponseFields.MESSAGE: message}, status_code=200)

        # ------------------------------ Step 1: Check If Collection Exists ------------------------------ #
        if await qdrant.collection_exists(event.project_id) is False:
            message = f"{msg_prefix} Embeddings collection does not exist, search is not possible. Create embeddings first. Disabling AI Search."
            sly.logger.warning(message)
            api.project.disable_embeddings(project_info.id)
            return JSONResponse({ResponseFields.MESSAGE: message}, status_code=404)

        # ------------------------------------ Step 2: Get Image Vectors From Qdrant ------------------------------------ #
        if event.image_ids:
            image_infos, vectors = await qdrant.get_items_by_id(
                event.project_id, event.image_ids, with_vectors=True
            )
        elif event.dataset_id:
            image_infos = await image_get_list_async(
                api, event.project_id, dataset_id=event.dataset_id
            )
            ids = [info.id for info in image_infos]
            image_infos, vectors = await qdrant.get_items_by_id(
                event.project_id, ids, with_vectors=True
            )
        else:
            image_infos, vectors = await qdrant.get_items(event.project_id, with_vectors=True)

        if len(vectors) == 0:
            return JSONResponse({ResponseFields.MESSAGE: "No vectors found."})

        # ------------------------- Step 3: Prepare Data For Projections Service ------------------------- #

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

        # -------------------------------- Step 4: Run Projections Service ------------------------------- #
        try:
            projections_service_task_id = await start_projections_service(api, event.project_id)
        except Exception as e:
            message = f"{msg_prefix} Failed to start projections service: {str(e)}"
            sly.logger.error(message, exc_info=True)
            return JSONResponse({ResponseFields.MESSAGE: message}, status_code=500)

        # --------------- Step 5: Send Request To Projections Service And Return The Result -------------- #
        samples = await send_request(
            api, task_id=projections_service_task_id, method="diverse", data=data
        )
        result = []
        sly.logger.debug(f"{msg_prefix} Generated diverse samples: {samples}.")
        for label, sample in samples.items():
            result.extend([image_infos[i] for i in sample])

        sly.logger.debug(f"{msg_prefix} Generated {len(result)} diverse images.")

        if len(result) == 0:
            return JSONResponse({ResponseFields.MESSAGE: "No diverse images found."})

        collection_manager = DiverseCollectionManager(api, event.project_id)
        collection_id = await collection_manager.save(result)

        return JSONResponse({ResponseFields.COLLECTION_ID: collection_id})

    except Exception as e:
        message = f"{msg_prefix} Error during diverse population generation: {str(e)}"
        sly.logger.error(message, exc_info=True)
        return JSONResponse({ResponseFields.MESSAGE: message}, status_code=500)


@app.event(Event.Clusters, use_state=True)
async def clusters_event_endpoint(api: sly.Api, event: Event.Clusters):
    # TODO: Remove this response to implement clustering.
    return JSONResponse({ResponseFields.MESSAGE: "Clustering are not supported yet."})
    sly.logger.info(
        f"Generating clusters for project {event.project_id}.",
        extra={
            "reduction_dimensions": event.reduction_dimensions,
            "image_ids": event.image_ids,
            "save": event.save,
        },
    )
    if event.image_ids:
        image_infos, vectors = await qdrant.get_items_by_id(
            event.project_id, event.image_ids, with_vectors=True
        )
    else:
        image_infos, vectors = await qdrant.get_items(event.project_id, with_vectors=True)

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
        image_infos, vectors = await get_projections(
            api, event.project_id, project_info=project_info
        )
        image_infos, vectors = zip(
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
            [info.id for info in image_infos],
            f"clusters_{datetime.datetime.now()}.pcd",
            dataset.id,
            cluster_ids=labels,
            colors=colors,
        )

    return [info.to_json() for info in image_infos], labels


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
        task_id = req_data.get("task_id")  # project_id

        if not task_id or task_id not in g.background_tasks:
            return JSONResponse(
                {
                    ResponseFields.STATUS: ResponseStatus.NOT_FOUND,
                    ResponseFields.MESSAGE: f"[Project: {task_id}] Processing task not found",
                }
            )
        task_id = str(task_id)  # Ensure task_id is a string
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
                    ResponseFields.MESSAGE: f"[Project: {task_id}] Task is still running",
                }
            )

    except Exception as e:
        return JSONResponse(
            {ResponseFields.STATUS: ResponseStatus.ERROR, ResponseFields.MESSAGE: str(e)}
        )


@server.get("/health")
async def health_check():
    status = "healthy"
    checks = {}
    status_code = 200
    try:
        # Check Qdrant connection
        try:
            await qdrant.client.info()
            checks["qdrant"] = "healthy"
        except Exception as e:
            checks["qdrant"] = f"unhealthy: {str(e)}"
            status = "degraded"
            status_code = 503

        # Check CLIP service availability
        try:
            if await cas.is_flow_ready():
                checks["clip"] = "healthy"
            else:
                checks["clip"] = "unhealthy: CLIP service is not ready"
                status = "degraded"
                status_code = 503
        except Exception as e:
            checks["clip"] = f"unhealthy: {str(e)}"
            status = "degraded"
            status_code = 503

    except Exception as e:
        status = "unhealthy"
        checks["general"] = f"error: {str(e)}"
        status_code = 500
    return JSONResponse(
        {
            ResponseFields.STATUS: status,
            "checks": checks,
            "timestamp": datetime.datetime.now(datetime.timezone.utc).strftime(
                "%Y-%m-%dT%H:%M:%S.%fZ"
            ),
        },
        status_code=status_code,
    )
