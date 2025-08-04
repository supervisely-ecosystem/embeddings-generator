import asyncio
import datetime
from typing import Dict, List, Optional

import numpy as np
import supervisely as sly
from fastapi import Request
from fastapi.responses import JSONResponse
from supervisely.imaging.color import get_predefined_colors

import src.cas as cas
import src.globals as g
import src.qdrant as qdrant
from src.autorestart import EmbeddingsTaskManager
from src.events import Event
from src.functions import Document, process_images, update_embeddings
from src.pointcloud import upload as upload_pcd

# from src.search_cache import CollectionItem, SearchCache
from src.project_collection_manager import AiSearchCollectionManager, DiverseCollectionManager
from src.utils import (
    ClusteringMethods,
    ImageInfoLite,
    ResponseFields,
    ResponseStatus,
    clean_image_embeddings_updated_at,
    cleanup_task_and_flags,
    clear_update_flag,
    create_current_timestamp,
    create_lite_image_infos,
    disable_embeddings,
    embeddings_up_to_date,
    get_all_processing_progress,
    get_processing_progress,
    get_project_info,
    image_get_list_async,
    send_request,
    set_embeddings_in_progress,
    set_project_embeddings_updated_at,
    set_update_flag,
    start_projections_service,
    timeit,
    validate_project_for_ai_features,
)
from src.visualization import (
    create_projections,
    get_or_create_projections,
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


@server.on_event("startup")
async def startup_event():
    """Reset state for stuck projects from previous session on service startup."""
    try:
        await EmbeddingsTaskManager.reset_state_stuck_projects(g.api)
    except Exception as e:
        sly.logger.error(
            f"[Embeddings Task Manager] Failed to reset state for stuck projects on startup: {e}",
            exc_info=True,
        )


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

    Send POST request to the service endpoint: /embeddings
    response = httpx.post(
        "http://<your-server-address>/embeddings",
        json=data,
        headers={"Authorization": f"Bearer {your_token}"}
    )
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

        async def cleanup_task_resources(error_message: Optional[str] = None):
            """Helper function to clean up task resources and reset project flags."""
            await cleanup_task_and_flags(api, event.project_id, error_message)
            # Remove task file when embeddings creation is complete or fails
            await EmbeddingsTaskManager.remove_task_file(event.project_id)

        project_info: sly.ProjectInfo = await get_project_info(api, event.project_id)

        # --------------------------- Step 0: Validate Project For AI Features --------------------------- #
        validation_error = await validate_project_for_ai_features(api, project_info, msg_prefix)
        if validation_error:
            return validation_error

        # --------------------- Step 1: Check If Embeddings Are Already In Progress. --------------------- #

        if project_info.embeddings_in_progress:
            message = f"{msg_prefix} Embeddings creation is already in progress. Skipping."
            sly.logger.info(message)
            return JSONResponse({ResponseFields.MESSAGE: message}, status_code=200)

        # ---------------------- Step 2: Check If Embeddings Are Already Up To Date. --------------------- #

        try:
            await set_embeddings_in_progress(api, event.project_id, True)

            timestamp = await create_current_timestamp()
            await set_update_flag(api, event.project_id, timestamp)
            # Create task file to track embeddings creation for crash recovery
            await EmbeddingsTaskManager.create_task_file(event.project_id, timestamp)

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
                images_to_create = await image_get_list_async(
                    api, event.project_id, wo_embeddings=True
                )
                if project_info.embeddings_updated_at is not None:
                    images_to_delete = await image_get_list_async(
                        api, event.project_id, deleted_after=project_info.embeddings_updated_at
                    )
                else:
                    images_to_delete = []

            if len(images_to_create) == 0 and len(images_to_delete) == 0:
                # Only reset flags, don't clean background tasks here since no task was created yet
                await set_embeddings_in_progress(api, event.project_id, False)
                await clear_update_flag(api, event.project_id)
                # Remove task file since no actual processing is needed
                await EmbeddingsTaskManager.remove_task_file(event.project_id)
                message = f"{msg_prefix} Nothing to update. Skipping."
                sly.logger.info(message)
                return JSONResponse({ResponseFields.MESSAGE: message}, status_code=200)
        except Exception as e:
            message = f"{msg_prefix} Error while fetching images: {str(e)}"
            sly.logger.error(message, exc_info=True)
            # Only reset flags, don't clean background tasks here since no task was created yet
            await set_embeddings_in_progress(
                api,
                event.project_id,
                False,
                error_message="Error while checking images for embeddings creation. Please contact instance administrator for more details.",
            )
            await clear_update_flag(api, event.project_id)
            # Remove task file on error
            await EmbeddingsTaskManager.remove_task_file(event.project_id)
            return JSONResponse({ResponseFields.MESSAGE: message}, status_code=500)

        async def execute():
            try:
                # ----------------------- Step 3: If Force Is True, Delete The Collection. ----------------------- #
                if event.force:
                    sly.logger.debug(f"{msg_prefix} Force enabled, deleting qdrant collection.")
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

                # Clean up resources before returning
                await cleanup_task_resources()

                if event.return_vectors:
                    sly.logger.info(
                        f"{msg_prefix} Embeddings creation has been completed. "
                        f"{len(image_infos)} images vectorized. {len(images_to_delete)} images deleted. {len(vectors)} vectors returned.",
                    )
                    return JSONResponse(
                        {
                            ResponseFields.IMAGE_IDS: [info.id for info in image_infos],
                            ResponseFields.VECTORS: vectors,
                        }
                    )
                else:
                    message = f"{msg_prefix} Embeddings creation has been completed."
                    sly.logger.info(message)
                    return JSONResponse({ResponseFields.MESSAGE: message})

            except asyncio.CancelledError:
                message = f"{msg_prefix} Embeddings creation was cancelled."
                sly.logger.info(message)
                await cleanup_task_resources()
                raise  # Re-raise the CancelledError
            except Exception as e:
                message = f"{msg_prefix} Error while creating embeddings: {str(e)}"
                sly.logger.error(message, exc_info=True)
                await cleanup_task_resources(
                    error_message="Error while creating embeddings. Please contact instance administrator for more details."
                )
                return JSONResponse({ResponseFields.MESSAGE: message}, status_code=500)

        task = asyncio.create_task(execute())
        task_id = int(event.project_id)
        g.background_tasks[task_id] = task
        return JSONResponse({ResponseFields.MESSAGE: f"{msg_prefix} Embeddings creation started."})
    except Exception as e:
        message = f"{msg_prefix} Error while creating embeddings: {str(e)}"
        sly.logger.error(message, exc_info=True)
        await cleanup_task_resources(
            error_message="Error while creating embeddings. Please contact instance administrator for more details."
        )
        return JSONResponse({ResponseFields.MESSAGE: message}, status_code=500)


@app.event(Event.Search, use_state=True)
@timeit
async def search(api: sly.Api, event: Event.Search) -> List[List[Dict]]:
    """
    Searches for similar images in a project using AI embeddings.
    This endpoint allows searching images by text prompts or by image IDs.

    Examples of requests:
    1. Search for similar images using text prompt.
    data = {"project_id": <your-project-id>, "limit": <limit>, "prompt": <text-prompt>}
    2. Search for similar images using image IDs.
    data = {"project_id": <your-project-id>, "limit": <limit>, "by_image_ids": [<image-id-1>, <image-id-2>, ...]}
    3. Both text prompt and image IDs can be provided at the same time.
    response = httpx.post(
        "http://<your-server-address>/search",
        json=data,
        headers={"Authorization": f"Bearer {your_token}"}
    )  # Do not skip response.
    returns a list of ImageInfoLite objects for each query.
    response: List[List[Dict]]
    """

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

        # --------------------------- Step 0: Validate Project For AI Features --------------------------- #
        project_info: sly.ProjectInfo = await get_project_info(api, event.project_id)
        validation_error = await validate_project_for_ai_features(api, project_info, msg_prefix)
        if validation_error:
            return validation_error

        # ----------------- Step 1: Initialise Collection Manager And List Of Image Infos ---------------- #
        if await qdrant.collection_exists(event.project_id) is False:
            message = f"{msg_prefix} Embeddings collection does not exist, search is not possible. Create embeddings first. Disabling AI Search."
            sly.logger.warning(message)
            await clean_image_embeddings_updated_at(api, project_info.id)
            await disable_embeddings(api, project_info.id)
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

        image_blobs = []
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
            image_ids = [image_info.id for image_info in lite_image_infos]
            image_bytes_list = await api.image.download_bytes_many_async(image_ids)

            # Create Document objects with blob data
            image_blobs = [Document(blob=image_bytes) for image_bytes in image_bytes_list]
        # Combine text prompts and image URLs to create a query.
        queries = text_prompts + image_blobs
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

    Examples of requests:
    1. Generate diverse population using KMeans method.
    data = {"project_id": <your-project-id>, "limit": <limit>, "method": "kmeans"}
    response = httpx.post(
        "http://<your-server-address>/diverse",
        json=data,
        headers={"Authorization": f"Bearer {your_token}"}
    )

    """
    try:
        msg_prefix = f"[Project: {event.project_id}]"
        sly.logger.info(
            f"{msg_prefix} Generating diverse population.",
            extra={
                "sampling_method": event.sampling_method,
                "sample_size": event.sample_size,
                "clustering_method": event.clustering_method,
                "num_clusters": event.num_clusters,
                "dataset_id": event.dataset_id,
                "image_ids": event.image_ids,
            },
        )

        # --------------------------- Step 0: Validate Project For AI Features --------------------------- #
        project_info: sly.ProjectInfo = await get_project_info(api, event.project_id)
        validation_error = await validate_project_for_ai_features(api, project_info, msg_prefix)
        if validation_error:
            return validation_error

        # ------------------------------ Step 1: Check If Collection Exists ------------------------------ #
        if await qdrant.collection_exists(event.project_id) is False:
            message = f"{msg_prefix} Embeddings collection does not exist, search is not possible. Create embeddings first. Disabling AI Search."
            sly.logger.warning(message)
            await clean_image_embeddings_updated_at(api, project_info.id)
            await disable_embeddings(api, project_info.id)
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

        if len(result) == 0:
            sly.logger.debug(f"{msg_prefix} No diverse images found.")
            return JSONResponse({ResponseFields.MESSAGE: "No diverse images found."})

        sly.logger.info(f"{msg_prefix} Generated {len(result)} diverse images.")

        collection_manager = DiverseCollectionManager(api, event.project_id)
        collection_id = await collection_manager.save(result)

        return JSONResponse({ResponseFields.COLLECTION_ID: collection_id})

    except Exception as e:
        message = f"{msg_prefix} Error during diverse population generation: {str(e)}"
        sly.logger.error(message, exc_info=True)
        return JSONResponse({ResponseFields.MESSAGE: message}, status_code=500)


@app.event(Event.CancelEmbeddings, use_state=True)
@timeit
async def cancel_embeddings(api: sly.Api, event: Event.CancelEmbeddings) -> JSONResponse:
    """
    Endpoint to cancel embeddings creation for a project.
    This will stop the running embeddings task if it exists.

    :param api: Supervisely API object.
    :param event: Event object containing project_id for which to cancel embeddings.

    Example request:
    data = {"project_id": <your-project-id>}
    response = httpx.post(
        "http://<your-server-address>/cancel_embeddings",
        json=data,
        headers={"Authorization": f"Bearer {your_token}"}
    )
    """

    try:
        msg_prefix = f"[Project: {event.project_id}]"
        sly.logger.info(f"{msg_prefix} Received request to cancel embeddings creation.")

        if api is None:
            api = g.api  # Use global API if not provided
            sly.logger.info(f"{msg_prefix} Using global API instance for cancel_embeddings.")

        task_id = int(event.project_id)

        # Check if there is a running task for this project
        if task_id not in g.background_tasks:
            message = f"{msg_prefix} No running embeddings task found."
            sly.logger.info(message)
            return JSONResponse({ResponseFields.MESSAGE: message}, status_code=200)

        # Get the task and cancel it
        task = g.background_tasks[task_id]

        if task.done():
            # Task is already finished
            if task.cancelled():
                message = f"{msg_prefix} Embeddings task was already cancelled."
            else:
                try:
                    task.result()  # Check if it completed successfully
                    message = f"{msg_prefix} Embeddings task was already completed."
                except Exception:
                    message = f"{msg_prefix} Embeddings task had already failed."
            sly.logger.info(message)
        else:
            # Task is still running, cancel it
            task.cancel()
            message = f"{msg_prefix} Embeddings task has been cancelled."
            sly.logger.info(message)

        # Clean up the task from the dictionary and reset project flags
        await cleanup_task_and_flags(api, event.project_id)

        return JSONResponse({ResponseFields.MESSAGE: message})

    except Exception as e:
        message = f"{msg_prefix} Error during embeddings cancellation: {str(e)}"
        sly.logger.error(message, exc_info=True)
        return JSONResponse({ResponseFields.MESSAGE: message}, status_code=500)


@app.event(Event.TaskStatus, use_state=True)
@timeit
async def task_status(api: sly.Api, event: Event.TaskStatus) -> JSONResponse:
    """
    Endpoint to check the status of embeddings creation task for a project.

    :param api: Supervisely API object.
    :param event: Event object containing project_id for which to check task status.

    Example request:
    data = {"project_id": <your-project-id>}
    response = httpx.post(
        "http://<your-server-address>/task_status",
        json=data,
        headers={"Authorization": f"Bearer {your_token}"}
    )
    """
    try:
        msg_prefix = f"[Project: {event.project_id}]"
        sly.logger.info(f"{msg_prefix} Received request to check task status.")

        task_id = int(event.project_id)

        # Check if there is a task for this project
        if task_id not in g.background_tasks:
            message = f"{msg_prefix} No task found."
            sly.logger.info(message)
            return JSONResponse(
                {
                    ResponseFields.MESSAGE: message,
                    ResponseFields.STATUS: ResponseStatus.NO_TASK,
                    ResponseFields.IS_RUNNING: False,
                }
            )

        # Get the task and check its status
        task = g.background_tasks[task_id]

        if task.done():
            # Task is finished, check if it was cancelled or completed
            if task.cancelled():
                status = ResponseStatus.CANCELLED
                is_running = False
                message = f"{msg_prefix} Task was cancelled."
            else:
                try:
                    result = task.result()  # This will raise an exception if the task failed
                    status = ResponseStatus.COMPLETED
                    is_running = False
                    message = f"{msg_prefix} Task completed successfully."
                except Exception as e:
                    status = ResponseStatus.FAILED
                    is_running = False
                    message = f"{msg_prefix} Task failed with error: {str(e)}"
        else:
            # Task is still running
            status = ResponseStatus.RUNNING
            is_running = True
            message = f"{msg_prefix} Task is currently running."

        sly.logger.info(message)
        return JSONResponse(
            {
                ResponseFields.MESSAGE: message,
                ResponseFields.STATUS: status,
                ResponseFields.IS_RUNNING: is_running,
            }
        )

    except Exception as e:
        message = f"{msg_prefix} Error while checking task status: {str(e)}"
        sly.logger.error(message, exc_info=True)
        return JSONResponse({ResponseFields.MESSAGE: message}, status_code=500)


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


# -------------- Method Below Is Deprecated And Will Be Removed In Future Versions. -------------- #


@server.post("/check_background_task_status")
async def check_task_status(request: Request):
    try:
        req_data = await request.json()
        task_id = req_data.get("task_id")  # project_id
        if not task_id:
            message = "Task ID is required"
            sly.logger.debug(message)
            return JSONResponse(
                {
                    ResponseFields.STATUS: ResponseStatus.ERROR,
                    ResponseFields.MESSAGE: message,
                },
                status_code=400,
            )
        else:
            task_id = int(task_id)

        if task_id not in g.background_tasks:
            message = f"[Project: {task_id}] Processing task not found"
            sly.logger.debug(message)
            return JSONResponse(
                {
                    ResponseFields.STATUS: ResponseStatus.NOT_FOUND,
                    ResponseFields.MESSAGE: message,
                }
            )
        else:
            message = f"[Project: {task_id}] Task is still running"
            sly.logger.debug(message)
            return JSONResponse(
                {
                    ResponseFields.STATUS: ResponseStatus.IN_PROGRESS,
                    ResponseFields.MESSAGE: message,
                }
            )

    except Exception as e:
        message = f"[Project: {task_id}] Error checking task status: {str(e)}"
        sly.logger.debug(message, exc_info=True)
        return JSONResponse(
            {
                ResponseFields.STATUS: ResponseStatus.ERROR,
                ResponseFields.MESSAGE: message,
            },
        )


# ------------------------------------ End Of Deprecated Code ------------------------------------ #


@app.event(Event.Clusters, use_state=True)
async def clusters_event_endpoint(api: sly.Api, event: Event.Clusters):
    # TODO: Remove this response to implement clustering.
    return JSONResponse({ResponseFields.MESSAGE: "Clustering are not supported yet."})

    msg_prefix = f"[Project: {event.project_id}]"

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

    try:
        projections_service_task_id = await start_projections_service(api, event.project_id)
    except Exception as e:
        message = f"{msg_prefix} Failed to start projections service: {str(e)}"
        sly.logger.error(message, exc_info=True)
        return JSONResponse({ResponseFields.MESSAGE: message}, status_code=500)

    labels = await send_request(
        api,
        projections_service_task_id,
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
    """
    Endpoint to get or create projections for a project.

    This endpoint creates 2D projections of image embeddings for visualization purposes.
    If projections already exist and are up to date, they will be returned.
    If projections are outdated or don't exist, new ones will be created.

    :param api: Supervisely API object.
    :param event: Event object containing project_id and optional image_ids for filtering.

    Example request:
    data = {"project_id": <your-project-id>, "image_ids": [<image-id-1>, <image-id-2>, ...]}
    response = httpx.post(
        "http://<your-server-address>/projections",
        json=data,
        headers={"Authorization": f"Bearer {your_token}"}
    )

    Returns:
    List[List[Dict]] - [image_infos, projections] where image_infos are filtered by event.image_ids if provided
    """
    try:
        msg_prefix = f"[Project: {event.project_id}]"
        sly.logger.info(
            f"{msg_prefix} Creating projections for project.",
            extra={
                "image_ids": event.image_ids,
            },
        )

        # --------------------------- Step 0: Validate Project For AI Features --------------------------- #
        project_info: sly.ProjectInfo = await get_project_info(api, event.project_id)
        validation_error = await validate_project_for_ai_features(api, project_info, msg_prefix)
        if validation_error:
            return validation_error

        # ------------------------------ Step 1: Check If Collection Exists ------------------------------ #
        if await qdrant.collection_exists(event.project_id) is False:
            message = f"{msg_prefix} Embeddings collection does not exist, projections are not possible. Create embeddings first. Disabling AI Search."
            sly.logger.warning(message)
            await clean_image_embeddings_updated_at(api, project_info.id)
            await disable_embeddings(api, project_info.id)
            return JSONResponse({ResponseFields.MESSAGE: message}, status_code=404)

        # ----------------------- Step 2: Update Embeddings If Needed ------------------------ #
        if not await embeddings_up_to_date(api, event.project_id):
            sly.logger.debug(
                f"{msg_prefix} Embeddings are not up to date. Updating embeddings before creating projections."
            )
            await update_embeddings(
                api,
                event.project_id,
                force=False,
                project_info=project_info,
            )

        # ----------------------- Step 3: Get Or Create Projections -------------------------- #
        image_infos, projections = await get_or_create_projections(
            api, event.project_id, project_info
        )
        if image_infos is None or projections is None:
            message = f"{msg_prefix} Projections could not be created or retrieved."
            sly.logger.error(message)
            return JSONResponse({ResponseFields.MESSAGE: message}, status_code=500)
        # ------------------------ Step 4: Filter Results By Image IDs ---------------------- #
        indexes = []
        for i, info in enumerate(image_infos):
            if event.image_ids is None or info.id in event.image_ids:
                indexes.append(i)

        sly.logger.info(f"{msg_prefix} Returning {len(indexes)} projections.")
        return [[image_infos[i].to_json() for i in indexes], [projections[i] for i in indexes]]

    except Exception as e:
        message = f"{msg_prefix} Error during projections creation: {str(e)}"
        sly.logger.error(message, exc_info=True)
        return JSONResponse({ResponseFields.MESSAGE: message}, status_code=500)


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


@app.event(Event.ProcessingProgress, use_state=True)
async def processing_progress_handler(event: Event.ProcessingProgress):
    """Get processing progress for a project or all projects."""
    try:
        if event.project_id is not None:
            # Get progress for specific project
            progress = await get_processing_progress(event.project_id)
            if progress is None:
                return JSONResponse(
                    {
                        ResponseFields.MESSAGE: f"No processing progress found for project {event.project_id}",
                        ResponseFields.PROGRESS: None,
                    },
                    status_code=200,
                )
            return JSONResponse({ResponseFields.PROGRESS: progress}, status_code=200)
        else:
            # Get progress for all projects
            all_progress = await get_all_processing_progress()
            return JSONResponse(
                {
                    ResponseFields.MESSAGE: "Project ID is not specified. Returning progress for all projects.",
                    ResponseFields.PROGRESS: all_progress,
                },
                status_code=200,
            )

    except Exception as e:
        message = f"Error getting processing progress: {str(e)}"
        sly.logger.error(message, exc_info=True)
        return JSONResponse({ResponseFields.MESSAGE: message}, status_code=500)
