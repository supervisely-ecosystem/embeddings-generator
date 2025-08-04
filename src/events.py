from typing import Any, Dict, List, Optional

from src.utils import ClusteringMethods, EventFields, SamplingMethods


class Event:
    class Embeddings:
        endpoint = "/embeddings"

        def __init__(
            self,
            project_id: int,
            force: Optional[bool],
            image_ids: Optional[List[int]],
            return_vectors: Optional[bool] = False,
        ):
            self.project_id = project_id
            self.force = force
            self.image_ids = image_ids
            self.return_vectors = return_vectors

        @classmethod
        def from_json(cls, data: Dict[str, Any]):
            return cls(
                data.get(EventFields.PROJECT_ID),
                data.get(EventFields.FORCE),
                data.get(EventFields.IMAGE_IDS),
                data.get(EventFields.RETURN_VECTORS),
            )

    class Search:
        """
        Could be used for searching images by prompt or by image IDs.
        To search by image IDs, use the `by_image_ids` parameter.

        To limit selection by image IDs or dataset ID, use the corresponding parameters.
        If both image_ids and dataset_id are provided, the search will be limited to the specified images.

        """

        endpoint = "/search"

        def __init__(
            self,
            project_id: int,
            limit: Optional[int] = None,
            prompt: Optional[str] = None,
            by_image_ids: Optional[List[int]] = None,
            image_ids: Optional[List[int]] = None,
            dataset_id: Optional[int] = None,
            threshold: Optional[float] = None,
        ):
            self.project_id = project_id
            self.limit = limit
            self.prompt = prompt
            self.by_image_ids = by_image_ids
            self.image_ids = image_ids
            self.dataset_id = dataset_id
            self.threshold = threshold

        @classmethod
        def from_json(cls, data: Dict[str, Any]):
            return cls(
                data.get(EventFields.PROJECT_ID),
                data.get(EventFields.LIMIT),
                data.get(EventFields.PROMPT),
                data.get(EventFields.BY_IMAGE_IDS),
                data.get(EventFields.IMAGE_IDS),
                data.get(EventFields.DATASET_ID),
                data.get(EventFields.THRESHOLD),
            )

    class Diverse:
        """
        To limit selection by image IDs or dataset ID, use the corresponding parameters.
        If both image_ids and dataset_id are provided, the search will be limited to the specified images.
        """

        endpoint = "/diverse"

        def __init__(
            self,
            project_id: int,
            sampling_method: str,
            sample_size: int,
            clustering_method: str,
            num_clusters: Optional[int] = None,
            image_ids: Optional[List[int]] = None,
            dataset_id: Optional[int] = None,
        ):
            self.project_id = project_id
            self.sampling_method = sampling_method
            self.sample_size = sample_size
            self.clustering_method = clustering_method
            self.num_clusters = num_clusters
            self.image_ids = image_ids
            self.dataset_id = dataset_id

        @classmethod
        def from_json(cls, data: Dict[str, Any]):
            return cls(
                data.get(EventFields.PROJECT_ID),
                data.get(EventFields.SAMPLING_METHOD, SamplingMethods.RANDOM),
                data.get(EventFields.SAMPLE_SIZE),
                data.get(EventFields.CLUSTERING_METHOD, ClusteringMethods.DBSCAN),
                data.get(EventFields.NUM_CLUSTERS, 8),
                data.get(EventFields.IMAGE_IDS),
                data.get(EventFields.DATASET_ID),
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

    class CancelEmbeddings:
        endpoint = "/cancel_embeddings"

        def __init__(self, project_id: int):
            self.project_id = project_id

        @classmethod
        def from_json(cls, data: Dict[str, Any]):
            return cls(
                data.get(EventFields.PROJECT_ID),
            )

    class TaskStatus:
        endpoint = "/task_status"

        def __init__(self, project_id: int):
            self.project_id = project_id

        @classmethod
        def from_json(cls, data: Dict[str, Any]):
            return cls(
                data.get(EventFields.PROJECT_ID),
            )

    class ProcessingProgress:
        endpoint = "/processing_progress"

        def __init__(self, project_id: Optional[int] = None):
            self.project_id = project_id

        @classmethod
        def from_json(cls, data: Dict[str, Any]):
            return cls(
                data.get(EventFields.PROJECT_ID),
            )
