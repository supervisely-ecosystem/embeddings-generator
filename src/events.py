from typing import Any, Dict, List, Optional

from src.utils import EventFields


class Event:
    class Embeddings:
        endpoint = "/embeddings"

        def __init__(self, project_id: int, force: Optional[bool], image_ids: Optional[List[int]]):
            self.project_id = project_id
            self.force = force
            self.image_ids = image_ids

        @classmethod
        def from_json(cls, data: Dict[str, Any]):
            return cls(
                data.get(EventFields.PROJECT_ID),
                data.get(EventFields.FORCE),
                data.get(EventFields.IMAGE_IDS),
            )

    class Search:
        endpoint = "/search"

        def __init__(
            self,
            project_id: int,
            limit: Optional[int] = None,
            prompt: Optional[str] = None,
            image_ids: Optional[List[int]] = None,
        ):
            self.project_id = project_id
            self.limit = limit
            self.prompt = prompt
            self.image_ids = image_ids

        @classmethod
        def from_json(cls, data: Dict[str, Any]):
            return cls(
                data.get(EventFields.PROJECT_ID),
                data.get(EventFields.LIMIT),
                data.get(EventFields.PROMPT),
                data.get(EventFields.IMAGE_IDS),
            )

    class Diverse:
        endpoint = "/diverse"

        def __init__(self, project_id: int, method: str, limit: int):
            self.project_id = project_id
            self.method = method
            self.limit = limit

        @classmethod
        def from_json(cls, data: Dict[str, Any]):
            return cls(
                data.get(EventFields.PROJECT_ID),
                data.get(EventFields.METHOD),
                data.get(EventFields.LIMIT),
            )

    class Projections:
        endpoint = "/projections"

        def __init__(self, project_id: int, image_ids: Optional[List[int]] = None):
            self.project_id = project_id
            self.image_ids = image_ids

        @classmethod
        def from_json(cls, data: Dict[str, Any]):
            return cls(
                data.get(EventFields.PROJECT_ID),
                data.get(EventFields.IMAGE_IDS),
            )
