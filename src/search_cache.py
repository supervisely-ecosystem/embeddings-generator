import hashlib
import json
import pickle
import tempfile
import time
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Tuple, Union

import supervisely as sly


class SearchCache:
    """
    A cache for search results that can be persisted to storage.

    This class provides functionality to cache search results
    and check if cached results are still valid based on project update timestamps.
    The cache is automatically saved to storage when updated and loaded when initialized.
    """

    SYSTEM_DIR = "/ai-search-cache"

    def __init__(self, api: sly.Api, project_id: int, prompt: str, settings: Dict):
        """
        Initialize the cache.
        """
        self.api = api
        self.project_id = project_id
        self.project_info = api.project.get_info_by_id(project_id)
        self.team_id = self.project_info.team_id
        self.results: Dict[str, Tuple[float, Any]] = {}
        self.timestamp: str = None
        self.prompt_text = prompt
        self.settings = settings
        self.key = self._get_key(prompt, project_id, settings)
        self.cache_file_path = self.SYSTEM_DIR + f"/{self.project_id}/{self.key}.pkl"
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

    def save(self, results: Any):
        """
        Save the current cache to a file using pickle serialization.

        Captures and logs any errors that occur during saving.
        """
        self.results = results
        self.timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")
        data_to_save = {"result": self.results, "timestamp": self.timestamp}

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
                progress_cb=None,
            )
            sly.fs.silent_remove(temp_cache.name)
        except Exception as e:
            print(f"Error saving cache: {e}")

    def load(self):
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
                    self.results = data.get("result", {})
                    self.timestamp = data.get("timestamp", None)
        except Exception as e:
            print(f"Error loading cache: {e}")
            # Start with empty caches if there was an error
            self.results = {}
            self.timestamp = None

    def clear(self):
        """Clear the cache and remove the file from storage."""
        self.results = {}
        self.timestamp = None
        self.api.file.remove(self.team_id, self.cache_file_path)
