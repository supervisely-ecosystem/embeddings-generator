from typing import Dict, List

import supervisely as sly
from supervisely.api.entities_collection_api import CollectionItem, CollectionType

from src.utils import ImageInfoLite, get_key, to_thread


class SearchCache:
    """
    A cache for search results that can be persisted to storage.

    This class provides functionality to cache search results
    and check if cached results are still valid based on project update timestamps.
    The cache is automatically saved to storage when updated and loaded when initialized.
    """

    SYSTEM_NAME_PREFIX = "AI Search Collection: "

    @to_thread
    def __init__(self, api: sly.Api, project_id: int, prompt: str, settings: Dict):
        """
        Initialize the cache.

        :param api: Supervisely API object.
        :type api: sly.Api
        :param project_id: Project ID to cache results for.
        :type project_id: int
        :param prompt: Search prompt to cache results for.
        :type prompt: str
        :param settings: Settings for the cache.
        :type settings: Dict
        """
        self.api = api
        self.project_id = project_id
        self.project_info = api.project.get_info_by_id(project_id)
        self.team_id = self.project_info.team_id
        self.prompt_text = prompt
        self.settings = settings
        self.key = get_key(prompt, project_id, settings)
        self.cache_collection_name = self.SYSTEM_NAME_PREFIX + f"Unique Key - {self.key}"
        collection = self.api.entities_collection.get_info_by_ai_search_key(
            self.project_id, self.key
        )
        self.collection_id = collection.id if collection else None
        self.updated_at = collection.updated_at if collection else None

    @to_thread
    def save(self, results: List[ImageInfoLite]) -> int:
        """
        Save the current cache as Entities Collection with the given name and unique key.

        :param results: List of ImageInfoLite objects to be cached.
        :type results: List[ImageInfoLite]
        :return: The ID of Entities Collection associated with the cache.
        :rtype: int
        """
        collection_items = [
            CollectionItem(entity_id=info.id, meta=CollectionItem.Meta(score=info.score))
            for info in results
        ]
        self.collection_id = self.api.entities_collection.create(
            project_id=self.project_id,
            name=self.cache_collection_name,
            type=CollectionType.AI_SEARCH,
            ai_search_key=self.key,
        ).id
        self.api.entities_collection.add_items(self.collection_id, collection_items)

        return self.collection_id

    @to_thread
    def load(self):
        """
        Search the Enitites Collection with the unique key in the storage.
        If the Collection doesn't exist, the cache will be initialized as empty.
        """
        collection = self.api.entities_collection.get_info_by_ai_search_key(
            self.project_id, self.key
        )
        self.collection_id = collection.id if collection else None
        self.updated_at = collection.updated_at if collection else None

    @to_thread
    def clear(self):
        """Clear the cache and remove Entities Collection."""
        self.api.entities_collection.remove(self.collection_id, force=True)
        self.collection_id = None
        self.updated_at = None

    @staticmethod
    @to_thread
    def drop_cache(api: sly.Api, project_id: int):
        """Delete all AI Collections in the project."""

        collections = api.entities_collection.get_list(project_id, type=CollectionType.AI_SEARCH)
        for collection in collections:
            try:
                api.entities_collection.remove(collection.id, force=True)
            except Exception as e:
                sly.logger.warning(
                    f"Failed to remove collection {collection.name} ({collection.id})"
                    f" in project {project_id}: {e}"
                )
