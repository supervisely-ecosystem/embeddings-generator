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
            by_project_id: Optional[int] = None,
            by_dataset_id: Optional[int] = None,
            by_image_ids: Optional[List[int]] = None,
        ):
            self.project_id = project_id
            self.limit = limit
            self.prompt = prompt
            self.image_ids = image_ids
            self.by_project_id = by_project_id
            self.by_dataset_id = by_dataset_id
            self.by_image_ids = by_image_ids

        @classmethod
        def from_json(cls, data: Dict[str, Any]):
            return cls(
                data.get(EventFields.PROJECT_ID),
                data.get(EventFields.LIMIT),
                data.get(EventFields.PROMPT),
                data.get(EventFields.IMAGE_IDS),
                data.get(EventFields.BY_PROJECT_ID),
                data.get(EventFields.BY_DATASET_ID),
                data.get(EventFields.BY_IMAGE_IDS),
            )

    class Diverse:
        endpoint = "/diverse"

        def __init__(
            self,
            project_id: int,
            method: str,
            sample_size: int,
            image_ids: List[int],
        ):
            self.project_id = project_id
            self.method = method
            self.sample_size = sample_size
            self.image_ids = image_ids

        @classmethod
        def from_json(cls, data: Dict[str, Any]):
            return cls(
                data.get(EventFields.PROJECT_ID),
                data.get(EventFields.METHOD, "random"),
                data.get(EventFields.SAMPLE_SIZE),
                data.get(EventFields.IMAGE_IDS),
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

    class Clusters:
        endpoint = "/clusters"

        def __init__(
            self,
            project_id: int,
            image_ids: Optional[List[int]] = None,
            reduction_dimensions: Optional[int] = None,
            save: Optional[bool] = False,
        ):
            self.project_id = project_id
            self.image_ids = image_ids
            self.reduction_dimensions = reduction_dimensions
            self.save = save

        @classmethod
        def from_json(cls, data: Dict[str, Any]):
            return cls(
                data.get(EventFields.PROJECT_ID),
                data.get(EventFields.IMAGE_IDS),
                data.get(EventFields.REDUCTION_DIMENSIONS),
                data.get(EventFields.SAVE),
            )
