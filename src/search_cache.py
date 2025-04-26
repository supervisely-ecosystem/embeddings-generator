import hashlib
import json
import pickle
import tempfile
import time
from typing import Any, Dict, Optional, Tuple

import supervisely as sly


class SearchCache:
    """
    A cache for search results that can be persisted to storage.

    This class provides functionality to cache search results
    and check if cached results are still valid based on project update timestamps.
    The cache is automatically saved to storage when updated and loaded when initialized.
    """

    SYSTEM_DIR = "/system/embeddings-search-cache"

    def __init__(self, api: sly.Api, team_id: int, project_id: int):
        """
        Initialize the cache.
        """
        self.api = api
        self.project_id = project_id
        self.team_id = team_id
        self.cache_file_path = self.SYSTEM_DIR + f"/{self.project_id}.pkl"
        # Dictionary mapping request keys to tuples of (cache_time, result)
        self.cache: Dict[str, Tuple[float, Any]] = {}
        # Dictionary mapping project IDs to their last update timestamps
        self.project_updates: Dict[str, float] = {}

        # Try to load existing cache data from storage
        self.load_cache()

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

    def update_project_timestamp(self, project_id: str, timestamp: Optional[float] = None):
        """
        Update the timestamp for when a project was last modified.

        Args:
            project_id (str): The ID of the project to update
            timestamp (float, optional): The timestamp to set. If None, current time is used
        """
        self.project_updates[project_id] = timestamp or time.time()
        # Save cache to storage after updating project timestamp
        self.save_cache()

    def get_cached_result(self, prompt: str, project_id: str, settings: Dict) -> Optional[Any]:
        """
        Try to retrieve a cached result for the given search parameters.

        Args:
            prompt (str): The search prompt text
            project_id (str): The ID of the project being searched
            settings (Dict): Search settings and parameters

        Returns:
            Any: The cached result if available and valid, None otherwise
        """
        key = self._get_key(prompt, project_id, settings)

        if key in self.cache:
            cache_time, result = self.cache[key]
            project_update_time = self.project_updates.get(project_id, 0)

            # If the project hasn't been updated since this result was cached,
            # the cached result is still valid
            if cache_time >= project_update_time:
                return result

        return None

    def cache_result(self, prompt: str, project_id: str, settings: Dict, result: Any):
        """
        Store a search result in the cache.

        Args:
            prompt (str): The search prompt text
            project_id (str): The ID of the project being searched
            settings (Dict): Search settings and parameters
            result (Any): The search result to cache
        """
        key = self._get_key(prompt, project_id, settings)
        current_time = time.time()
        self.cache[key] = (current_time, result)

        # If this is the first time we're seeing this project, initialize its update time
        if project_id not in self.project_updates:
            self.project_updates[project_id] = current_time

        # Save the updated cache to storage
        self.save_cache()

    def clear_cache(self):
        """Clear all cached data and save the empty cache to storage."""
        self.cache.clear()
        self.project_updates.clear()
        self.save_cache()

    def save_cache(self):
        """
        Save the current cache to a file using pickle serialization.

        Captures and logs any errors that occur during saving.
        """
        data_to_save = {"cache": self.cache, "project_updates": self.project_updates}

        try:
            temp_cache = tempfile.NamedTemporaryFile(
                "w+b", prefix=f"{self.project_id}", suffix=".pkl", delete=False
            )
            pickle.dump(data_to_save, temp_cache)
            temp_cache.close()
            sly.logger.debug("Uploading cache to storage", extra={"path": self.cache_file_path})
            self.api.file.upload(
                self.team_id,
                temp_cache.name,
                self.cache_file_path,
                progress=None,
            )
            sly.fs.silent_remove(temp_cache.name)
        except Exception as e:
            print(f"Error saving cache: {e}")

    def load_cache(self):
        """
        Load cache data from the file if it exists.

        If the file doesn't exist or there's an error loading it,
        the cache will be initialized as empty.
        """

        if not self.api.file.exists(self.team_id, self.cache_file_path):
            return

        try:
            with tempfile.TemporaryDirectory() as tmp_dir:
                local_path = tmp_dir + self.cache_file_path
                self.api.file.download(self.team_id, self.cache_file_path, local_path)
                with open(local_path, "rb") as f:
                    data = pickle.load(f)
                    self.cache = data.get("cache", {})
                    self.project_updates = data.get("project_updates", {})
        except Exception as e:
            print(f"Error loading cache: {e}")
            # Start with empty caches if there was an error
            self.cache = {}
            self.project_updates = {}
