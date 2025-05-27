# This script is used to remove AI Search collections from projects in Supervisely that have the embeddings update enabled.
# It also sets the embeddings updated at timestamp to None for the projects image randomly to reset isEmbeddingsUpdated flag.


import asyncio
from typing import List, Optional

import supervisely as sly
from supervisely.api.entities_collection_api import CollectionType, CollectionTypeFilter
from supervisely.api.module_api import ApiField
from tqdm import tqdm

from src.utils import batched, get_list_all_pages_async, set_embeddings_in_progress, to_thread, image_get_list_async

api = sly.Api.from_env()


async def get_all_projects(
    api: sly.Api,
    project_ids: Optional[List[int]] = None,
) -> List[sly.ProjectInfo]:
    """
    Get all projects from the Supervisely server that have a flag for automatic embeddings update.

    Fields that will be returned:
        - id
        - name
        - updated_at
        - embeddings_enabled
        - embeddings_in_progress
        - is_embeddings_updated
        - team_id
        - workspace_id
        - items_count

    """
    method = "projects.list.all"
    convert_json_info = api.project._convert_json_info
    fields = [
        ApiField.EMBEDDINGS_ENABLED,
        ApiField.EMBEDDINGS_IN_PROGRESS,
        ApiField.IS_EMBEDDINGS_UPDATED,
    ]
    data = {
        ApiField.SKIP_EXPORTED: True,
        ApiField.EXTRA_FIELDS: fields,
        ApiField.FILTER: [
            # {
            #     ApiField.FIELD: ApiField.EMBEDDINGS_ENABLED,
            #     ApiField.OPERATOR: "=",
            #     ApiField.VALUE: True,
            # },
            {
                ApiField.FIELD: ApiField.TYPE,
                ApiField.OPERATOR: "=",
                ApiField.VALUE: "images",
            }
        ],
    }
    tasks = []
    if project_ids is not None:
        for batch in batched(project_ids):
            data[ApiField.FILTER] = [
                {
                    ApiField.FIELD: ApiField.ID,
                    ApiField.OPERATOR: "in",
                    ApiField.VALUE: batch,
                },
                {
                    ApiField.FIELD: ApiField.EMBEDDINGS_ENABLED,
                    ApiField.OPERATOR: "=",
                    ApiField.VALUE: True,
                },
            ]
            tasks.append(
                get_list_all_pages_async(
                    api,
                    method,
                    data=data,
                    convert_json_info_cb=convert_json_info,
                    progress_cb=None,
                    limit=None,
                    return_first_response=False,
                )
            )
    else:
        tasks.append(
            get_list_all_pages_async(
                api,
                method,
                data=data,
                convert_json_info_cb=convert_json_info,
                progress_cb=None,
                limit=None,
                return_first_response=False,
            )
        )
    results = []
    for task in asyncio.as_completed(tasks):
        results.extend(await task)
    return results


@to_thread
def process_eua_for_project(
    project_info: sly.ProjectInfo,
    progress: tqdm,
):
    datasets = api.dataset.get_list(project_id=project_info.id)
    if len(datasets) == 0:
        progress.update(1)
        return
    dataset_id = None
    for dataset in datasets:
        if dataset.images_count != 0 or dataset.items_count != 0:
            dataset_id = dataset.id
            continue
    if dataset_id is None:
        progress.update(1)
        return
    try:
        images = api.image.get_list(dataset_id=dataset_id, limit=1)
    except Exception as e:
        sly.logger.warning(
            f"Failed to get images for project ID {project_info.id} dataset ID {dataset_id}: {e}"
        )
        progress.update(1)
        return
    api.image.set_embeddings_updated_at(ids=[images[0].id], timestamps=[None])
    sly.logger.debug(f"Set embeddings updated at to None for project {project_info.name}")
    progress.update(1)


@to_thread
def remove_collections(collection: sly.EntitiesCollectionInfo, progress: tqdm):
    api.entities_collection.remove(collection.id, force=True)
    sly.logger.debug(f"Collection {collection.name} removed from project {collection.project_id}")
    progress.update(1)


@to_thread
def get_collections(
    project_info: sly.ProjectInfo,
    collections: List[sly.EntitiesCollectionInfo],
    progress: tqdm,
):
    collections.extend(
        api.entities_collection.get_list(
            project_id=project_info.id, collection_type=CollectionType.AI_SEARCH
        )
    )
    progress.update(1)


@to_thread
def switch_off_auto_update(
    project_info: sly.ProjectInfo,
    progress: tqdm,
):
    api.project.disable_embeddings(project_info.id)
    sly.logger.debug(f"Switch off auto update for project ID: {project_info.id}")
    progress.update(1)


def main():
    project_infos = sly.run_coroutine(get_all_projects(api))

    if len(project_infos) == 0:
        sly.logger.info("No projects found with embeddings update enabled.")
        return

    collections = []
    progress_collections = tqdm(desc="Get collections", total=len(project_infos))
    tasks = []
    for project_info in project_infos:
        tasks.append(get_collections(project_info, collections, progress_collections))
    if len(tasks) > 0:
        sly.run_coroutine(asyncio.gather(*tasks))

    if len(collections) == 0:
        sly.logger.info("No collections found in projects.")
        return

    progress_remove = tqdm(desc="Remove collections", total=len(collections))
    tasks = []
    for collection in collections:
        tasks.append(remove_collections(collection, progress_remove))
    if len(tasks) > 0:
        sly.run_coroutine(asyncio.gather(*tasks))

    progress_updated_at = tqdm(desc="Set embeddings updated at to None", total=len(project_infos))
    tasks = []
    for project_info in project_infos:
        tasks.append(process_eua_for_project(project_info, progress_updated_at))
        tasks.append(set_embeddings_in_progress(api, project_info.id, False))
    if len(tasks) > 0:
        sly.run_coroutine(asyncio.gather(*tasks))

    progress_switch_off = tqdm(desc="Switch off auto update", total=len(project_infos))
    tasks = []
    for project_info in project_infos:
        if project_info.embeddings_enabled:
            tasks.append(switch_off_auto_update(project_info, progress_switch_off))
        else:
            progress_switch_off.update(1)
    if len(tasks) > 0:
        sly.run_coroutine(asyncio.gather(*tasks))
    sly.logger.info("All collections removed and auto update switched off.")


if __name__ == "__main__":
    sly.logger.info("Start removing AI Search collections from projects.")
    main()
