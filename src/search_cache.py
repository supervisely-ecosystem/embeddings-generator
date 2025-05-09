import hashlib
import json
from typing import TYPE_CHECKING, Dict, List

import supervisely as sly
from supervisely.api.entities_collection_api import CollectionItem, CollectionType

from src.utils import ImageInfoLite


class SearchCache:
    """
    A cache for search results that can be persisted to storage.

    This class provides functionality to cache search results
    and check if cached results are still valid based on project update timestamps.
    The cache is automatically saved to storage when updated and loaded when initialized.
    """

    SYSTEM_NAME_PREFIX = "AI Search Collection: "

    def __init__(self, api: sly.Api, project_id: int, prompt: str, settings: Dict):
        """
        Initialize the cache.
        """
        self.api = api
        self.project_id = project_id
        self.project_info = api.project.get_info_by_id(project_id)
        self.team_id = self.project_info.team_id
        self.collection_id: int = None
        self.timestamp: str = None
        self.prompt_text = prompt
        self.settings = settings
        self.key = self._get_key(prompt, project_id, settings)
        self.cache_collection_name = self.SYSTEM_NAME_PREFIX + f"Unique Key - {self.key}"
        self.load()

    def _get_key(self, prompt: str, project_id: str, settings: Dict) -> str:
        """
        Generate a unique hash key for a search request.

        Args:
            prompt (str): The search prompt text
            project_id (str): The ID of the project being searched
            settings (Dict): Search settings and parameters

        Returns:
            str: MD5 hash representing the unique key for this request
        """
        cache_data = {"prompt": prompt, "project_id": project_id, "settings": settings}
        serialized = json.dumps(cache_data, sort_keys=True)
        return hashlib.md5(serialized.encode()).hexdigest()

    def save(self, results: List[ImageInfoLite]) -> int:
        """
        Save the current cache as Entities Collection with the given name and unique key.

        Captures and logs any errors that occur during saving.

        Returns collection ID of the saved cache.

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

    def load(self):
        """
        Search the Enitites Collection with the unique key in the storage.

        If the Collection doesn't exist or there's an error loading it,
        the cache will be initialized as empty.
        """
        collection = self.api.entities_collection.get_info_by_ai_search_key(
            self.project_id, self.key
        )
        self.collection_id = collection.id if collection else None
        self.timestamp = collection.updated_at if collection else None

    def clear(self):
        """Clear the cache and remove Entities Collection."""
        self.api.entities_collection.remove(self.collection_id)
        self.collection_id = None
        self.timestamp = None
