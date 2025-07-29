import os
import tempfile
from collections import namedtuple
from typing import Iterable, List, Union

import numpy as np
import supervisely as sly
from pypcd4 import Encoding, PointCloud

from src.utils import timeit, to_thread


class EmbeddingsVisPCD:
    """
    EmbeddingsVisPCD - A specialized point cloud class for visualizing embeddings.
    This class extends the conventional PCD (Point Cloud Data) format with additional fields
    for visualization and analysis of embeddings in 3D space.

    Unlike regular PCD, EmbeddingsVisPCD supports:
    - Color information in various formats (RGB, hex, uint32)
    - Reference IDs for source images, objects, and clusters
    - Atlas-related information for texture mapping
    - Automatic field type handling and validation
    The class provides an intuitive API for creating, manipulating, and saving point clouds
    with these extended attributes, making it particularly useful for visualizing embedding
    spaces, clustering results, and object relationships in 3D.

    Key differences from standard PCD:
    1. Built-in support for color conversion between different formats
    2. Metadata fields for tracking point origins and relationships
    3. Field validation and type checking
    4. Named field access through properties
    5. Simplified saving and loading with automatic field handling
    """

    _default_fields = ["x", "y", "z"]
    _default_fields_types = [np.float32, np.float32, np.float32]
    _extra_fields = ["rgb", "imageId", "objectId", "clusterId", "atlasId", "atlasIndex"]
    _extra_fields_types = [np.uint32, np.int32, np.int32, np.int32, np.int32, np.int32]
    _extra_fields_attrs = [
        "colors",
        "image_ids",
        "object_ids",
        "cluster_ids",
        "atlas_ids",
        "atlas_indices",
    ]
    Fields = namedtuple("Fields", ["x", "y", "z"] + _extra_fields)

    def __init__(
        self,
        points: np.ndarray,
        colors=None,
        image_ids=None,
        object_ids=None,
        cluster_ids=None,
        atlas_ids=None,
        atlas_indices=None,
    ):
        self._colors = None
        self._image_ids = None
        self._object_ids = None
        self._cluster_ids = None
        self._atlas_ids = None
        self._atlas_indices = None

        # required fields
        self.points = points
        self.size = points.shape[0]

        # optional fields
        self.colors = colors
        self.image_ids = image_ids
        self.object_ids = object_ids
        self.cluster_ids = cluster_ids
        self.atlas_ids = atlas_ids
        self.atlas_indices = atlas_indices

    def _hex_to_uint32(self, hex: str) -> np.uint32:
        if hex.startswith("#"):
            hex = hex[1:]
        return np.uint32(int(hex, 16))

    def _rgb_to_uint32(self, rgb) -> np.uint32:
        return np.uint32((rgb[0] << 16) + (rgb[1] << 8) + rgb[2])

    def _convert_color(self, color):
        if isinstance(color, np.uint32):
            return color
        elif color is None:
            return np.uint32(0)
        elif isinstance(color, (tuple, list, np.ndarray)):
            return self._rgb_to_uint32(color)
        elif isinstance(color, str):
            return self._hex_to_uint32(color)
        else:
            raise ValueError("Invalid color format")

    def _set_field(self, field_name, items, dtype=np.int32):
        if items is not None:
            if len(items) != self.size:
                raise ValueError(f"{field_name} and points must have the same length")
            if not isinstance(items, np.ndarray):
                items = np.array(items, dtype=dtype)
        self._image_ids = items
        setattr(self, f"_{field_name}", items)

    @property
    def colors(self) -> np.ndarray:
        return self._colors

    @colors.setter
    def colors(self, colors: Union[None, Iterable[Union[str, np.uint32, np.ndarray, List[int]]]]):
        if colors is None:
            self._colors = None
            return
        if len(colors) != self.size:
            raise ValueError("colors and points must have the same length")
        self._colors = np.array([self._convert_color(color) for color in colors], dtype=np.uint32)

    @property
    def image_ids(self) -> np.ndarray:
        return self._image_ids

    @image_ids.setter
    def image_ids(self, image_ids: Union[np.ndarray, List[int]]):
        self._set_field("image_ids", image_ids)

    @property
    def object_ids(self) -> np.ndarray:
        return self._object_ids

    @object_ids.setter
    def object_ids(self, object_ids: Union[np.ndarray, List[int]]):
        self._set_field("object_ids", object_ids)

    @property
    def cluster_ids(self) -> np.ndarray:
        return self._cluster_ids

    @cluster_ids.setter
    def cluster_ids(self, cluster_ids: Union[np.ndarray, List[int]]):
        self._set_field("cluster_ids", cluster_ids)

    @property
    def atlas_ids(self) -> np.ndarray:
        return self._atlas_ids

    @atlas_ids.setter
    def atlas_ids(self, atlas_ids: Union[np.ndarray, List[int]]):
        self._set_field("atlas_ids", atlas_ids)

    @property
    def atlas_indices(self) -> np.ndarray:
        return self._atlas_indices

    @atlas_indices.setter
    def atlas_indices(self, atlas_indices: Union[np.ndarray, List[int]]):
        self._set_field("atlas_indices", atlas_indices)

    def save(self, path: str):
        extras_items = []
        extras_fields = []
        extras_types = []
        for attr, field_name, dtype in zip(
            self._extra_fields_attrs, self._extra_fields, self._extra_fields_types
        ):
            items = getattr(self, attr)
            if items is None:
                continue
            extras_items.append(items)
            extras_fields.append(field_name)
            extras_types.append(dtype)
        points = self.points
        if len(extras_items) > 0:
            points = [points[:, i] for i in range(points.shape[1])]
            points.extend(extras_items)

        pcd = PointCloud.from_points(
            points,
            fields=self._default_fields + extras_fields,
            types=self._default_fields_types + extras_types,
        )
        pcd.save(path, encoding=Encoding.BINARY_COMPRESSED)

    @staticmethod
    def read(path: str):
        pcd = PointCloud.from_path(path)
        points = pcd.numpy(fields=EmbeddingsVisPCD._default_fields)
        pointcloud = EmbeddingsVisPCD(points)
        for attr, field_name in zip(
            EmbeddingsVisPCD._extra_fields_attrs, EmbeddingsVisPCD._extra_fields
        ):
            if field_name in pcd.fields:
                setattr(pointcloud, attr, np.array(pcd[field_name]))
        return pointcloud


@to_thread
@timeit
def upload(
    api: sly.Api,
    pointcloud: np.ndarray,
    image_ids: List[int],
    pcd_name: str,
    dataset_id: int,
    cluster_ids: List[int] = None,
    colors: List = None,
) -> sly.api.pointcloud_api.PointcloudInfo:
    if not isinstance(pointcloud, np.ndarray):
        pointcloud = np.array(pointcloud)
    if pointcloud.shape[1] == 2:
        pointcloud = np.hstack((pointcloud, np.zeros((pointcloud.shape[0], 1), dtype=np.float32)))

    pcd = EmbeddingsVisPCD(pointcloud, image_ids=image_ids, cluster_ids=cluster_ids, colors=colors)
    tmp = tempfile.NamedTemporaryFile("w+b", suffix=".pcd", delete=False)
    try:
        pcd.save(tmp.name)
        pcdinfo = api.pointcloud.upload_path(dataset_id, pcd_name, tmp.name)
    finally:
        tmp.close()
        os.remove(tmp.name)
    return pcdinfo


async def download(api: sly.Api, pcd_id: int) -> EmbeddingsVisPCD:
    tmp = tempfile.NamedTemporaryFile("w+b", suffix=".pcd", delete=False)
    try:
        await api.pointcloud.download_path_async(pcd_id, tmp.name)
        return EmbeddingsVisPCD.read(tmp.name)
    finally:
        tmp.close()
        os.remove(tmp.name)


@to_thread
@timeit
def remove_pcd_file(api: sly.Api, pcd_id: int):
    """
    Removes a point cloud file from the dataset.
    """
    try:
        api.pointcloud.remove(pcd_id)
        sly.logger.debug(f"Removed outdated PCD file with ID {pcd_id}")
    except Exception as e:
        sly.logger.warning(f"Failed to remove PCD file with ID {pcd_id}: {str(e)}")
