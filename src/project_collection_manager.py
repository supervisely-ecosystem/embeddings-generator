from typing import Dict, List

import supervisely as sly
from supervisely.api.entities_collection_api import CollectionItem, CollectionType

from src.utils import ImageInfoLite, generate_key, to_thread


class ProjectCollectionManager:
    """
    Manager for Supervisely Entities Collection associated with AI Search or Diversity Search.
    This class handles the creation, saving, checking, and removal of collections
    that cache search results for a specific project.
    """

    SYSTEM_NAME_PREFIX = ""

    def __init__(self, api: sly.Api, project_id: int):
        """
        Initialize Collection Manager.

        :param api: Supervisely API object.
        :type api: sly.Api
        :param project_id: Project ID to cache results for.
        :type project_id: int
        :param params: Parameters for current user search request.
        :type params: Dict
        """
        self.api = api
        self.project_id = project_id
        self.project_info = api.project.get_info_by_id(project_id)
        self.key = generate_key(self.project_id, self.api.token)
        self.collection_name = self.SYSTEM_NAME_PREFIX + f"{self.key}"
        collection = self.api.entities_collection.get_info_by_ai_search_key(
            self.project_id, self.key
        )
        self.collection_id = collection.id if collection else None

    @to_thread
    def save(self, results: List[ImageInfoLite]) -> int:
        """
        Save the current search as Entities Collection with the given name and unique key.

        :param results: List of ImageInfoLite objects to be cached.
        :type results: List[ImageInfoLite]
        :return: The ID of Entities Collection associated with the cache.
        :rtype: int
        """
        if self.collection_id is not None:
            # If collection already exists, remove it first
            self.api.entities_collection.remove(self.collection_id, force=True)
            self.collection_id = None

        collection_items = []
        for info in results:
            meta = CollectionItem.Meta(score=info.score) if info.score is not None else None
            collection_items.append(CollectionItem(entity_id=info.id, meta=meta))

        self.collection_id = self.api.entities_collection.create(
            project_id=self.project_id,
            name=self.collection_name,
            type=CollectionType.AI_SEARCH,
            ai_search_key=self.key,
        ).id
        self.api.entities_collection.add_items(self.collection_id, collection_items)

        return self.collection_id

    @to_thread
    def check(self):
        """
        Check if the Entities Collection with the unique key exists.
        """
        collection = self.api.entities_collection.get_info_by_ai_search_key(
            self.project_id, self.key
        )
        self.collection_id = collection.id if collection else None

    @to_thread
    def remove(self):
        """Remove the Entities Collection associated with the unique key."""
        self.api.entities_collection.remove(self.collection_id, force=True)
        self.collection_id = None

    @staticmethod
    @to_thread
    def remove_all(api: sly.Api, project_id: int):
        """Delete all AI Collections in the project."""

        collections = api.entities_collection.get_list(
            project_id, collection_type=CollectionType.AI_SEARCH
        )
        for collection in collections:
            try:
                api.entities_collection.remove(collection.id, force=True)
            except Exception as e:
                sly.logger.warning(
                    f"Failed to remove collection {collection.name} ({collection.id})"
                    f" in project {project_id}: {e}"
                )


class AiSearchCollectionManager(ProjectCollectionManager):
    """
    Manager for Supervisely Entities Collection associated with AI Search.
    This class handles the creation, saving, checking, and removal of collections
    that cache search results for a specific project.
    """
    SYSTEM_NAME_PREFIX = "AI Search Collection: "

class DiverseCollectionManager(ProjectCollectionManager):
    """
    Manager for Supervisely Entities Collection associated with Diversity Search.
    This class handles the creation, saving, checking, and removal of collections
    that cache search results for a specific project.
    """
    SYSTEM_NAME_PREFIX = "Diverse Collection: "
