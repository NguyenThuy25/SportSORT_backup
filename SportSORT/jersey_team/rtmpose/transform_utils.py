import sys
import os
import math
from typing import Dict, List, Optional, Sequence, Tuple, Union
import warnings

import numpy as np
# from mmcv.transforms import LoadImageFromFile, BaseTransform
# from mmcv.transforms import BaseTransform
from mmengine.dist import get_dist_info

import cv2
from mmengine import is_seq_of
import torch
# Copyright (c) OpenMMLab. All rights reserved.
from typing import Union

from mmengine.structures import BaseDataElement, InstanceData, PixelData

# Copyright (c) OpenMMLab. All rights reserved.
from collections import abc
from typing import Any, Callable, List, Optional, Sequence, Tuple, Type, Union

import numpy as np
import torch
from mmengine.structures import BaseDataElement, PixelData
from mmengine.utils import is_list_of

from rtmpose.config import ConfigDict
from rtmpose.transforms.base_transform import BaseTransform
from rtmpose.transforms.loading import LoadImageFromFile

IndexType = Union[str, slice, int, list, torch.LongTensor,
                  torch.cuda.LongTensor, torch.BoolTensor,
                  torch.cuda.BoolTensor, np.ndarray]

ConfigType = Union[ConfigDict, dict]

MultiConfig = Union[ConfigType, List[ConfigType]]
# Copyright (c) OpenMMLab. All rights reserved.
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
from mmengine.registry import MODELS


def drop_path(x: torch.Tensor,
              drop_prob: float = 0.,
              training: bool = False) -> torch.Tensor:
    """Drop paths (Stochastic Depth) per sample (when applied in main path of
    residual blocks).

    We follow the implementation
    https://github.com/rwightman/pytorch-image-models/blob/a2727c1bf78ba0d7b5727f5f95e37fb7f8866b1f/timm/models/layers/drop.py  # noqa: E501
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    # handle tensors with different dimensions, not just 4D tensors.
    shape = (x.shape[0], ) + (1, ) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(
        shape, dtype=x.dtype, device=x.device)
    output = x.div(keep_prob) * random_tensor.floor()
    return output


@MODELS.register_module()
class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of
    residual blocks).

    We follow the implementation
    https://github.com/rwightman/pytorch-image-models/blob/a2727c1bf78ba0d7b5727f5f95e37fb7f8866b1f/timm/models/layers/drop.py  # noqa: E501

    Args:
        drop_prob (float): Probability of the path to be zeroed. Default: 0.1
    """

    def __init__(self, drop_prob: float = 0.1):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return drop_path(x, self.drop_prob, self.training)


@MODELS.register_module()
class Dropout(nn.Dropout):
    """A wrapper for ``torch.nn.Dropout``, We rename the ``p`` of
    ``torch.nn.Dropout`` to ``drop_prob`` so as to be consistent with
    ``DropPath``

    Args:
        drop_prob (float): Probability of the elements to be
            zeroed. Default: 0.5.
        inplace (bool):  Do the operation inplace or not. Default: False.
    """

    def __init__(self, drop_prob: float = 0.5, inplace: bool = False):
        super().__init__(p=drop_prob, inplace=inplace)


def build_dropout(cfg: Dict, default_args: Optional[Dict] = None) -> Any:
    """Builder for drop out layers."""
    return MODELS.build(cfg, default_args=default_args)

class MultilevelPixelData(BaseDataElement):
    """Data structure for multi-level pixel-wise annotations or predictions.

    All data items in ``data_fields`` of ``MultilevelPixelData`` are lists
    of np.ndarray or torch.Tensor, and should meet the following requirements:

    - Have the same length, which is the number of levels
    - At each level, the data should have 3 dimensions in order of channel,
        height and weight
    - At each level, the data should have the same height and weight

    Examples:
        >>> metainfo = dict(num_keypoints=17)
        >>> sizes = [(64, 48), (128, 96), (256, 192)]
        >>> heatmaps = [np.random.rand(17, h, w) for h, w in sizes]
        >>> masks = [torch.rand(1, h, w) for h, w in sizes]
        >>> data = MultilevelPixelData(metainfo=metainfo,
        ...                            heatmaps=heatmaps,
        ...                            masks=masks)

        >>> # get data item
        >>> heatmaps = data.heatmaps  # A list of 3 numpy.ndarrays
        >>> masks = data.masks  # A list of 3 torch.Tensors

        >>> # get level
        >>> data_l0 = data[0]  # PixelData with fields 'heatmaps' and 'masks'
        >>> data.nlevel
        3

        >>> # get shape
        >>> data.shape
        ((64, 48), (128, 96), (256, 192))

        >>> # set
        >>> offset_maps = [torch.rand(2, h, w) for h, w in sizes]
        >>> data.offset_maps = offset_maps
    """

    def __init__(self, *, metainfo: Optional[dict] = None, **kwargs) -> None:
        object.__setattr__(self, '_nlevel', None)
        super().__init__(metainfo=metainfo, **kwargs)

    @property
    def nlevel(self):
        """Return the level number.

        Returns:
            Optional[int]: The level number, or ``None`` if the data has not
            been assigned.
        """
        return self._nlevel

    def __getitem__(self, item: Union[int, str, list,
                                      slice]) -> Union[PixelData, Sequence]:
        if isinstance(item, int):
            if self.nlevel is None or item >= self.nlevel:
                raise IndexError(
                    f'Lcale index {item} out of range ({self.nlevel})')
            return self.get(f'_level_{item}')

        if isinstance(item, str):
            if item not in self:
                raise KeyError(item)
            return getattr(self, item)

        # TODO: support indexing by list and slice over levels
        raise NotImplementedError(
            f'{self.__class__.__name__} does not support index type '
            f'{type(item)}')

    def levels(self) -> List[PixelData]:
        if self.nlevel:
            return list(self[i] for i in range(self.nlevel))
        return []

    @property
    def shape(self) -> Optional[Tuple[Tuple]]:
        """Get the shape of multi-level pixel data.

        Returns:
            Optional[tuple]: A tuple of data shape at each level, or ``None``
            if the data has not been assigned.
        """
        if self.nlevel is None:
            return None

        return tuple(level.shape for level in self.levels())

    def set_data(self, data: dict) -> None:
        """Set or change key-value pairs in ``data_field`` by parameter
        ``data``.

        Args:
            data (dict): A dict contains annotations of image or
                model predictions.
        """
        assert isinstance(data,
                          dict), f'meta should be a `dict` but got {data}'
        for k, v in data.items():
            self.set_field(v, k, field_type='data')

    def set_field(self,
                  value: Any,
                  name: str,
                  dtype: Optional[Union[Type, Tuple[Type, ...]]] = None,
                  field_type: str = 'data') -> None:
        """Special method for set union field, used as property.setter
        functions."""
        assert field_type in ['metainfo', 'data']
        if dtype is not None:
            assert isinstance(
                value,
                dtype), f'{value} should be a {dtype} but got {type(value)}'

        if name.startswith('_level_'):
            raise AttributeError(
                f'Cannot set {name} to be a field because the pattern '
                '<_level_{n}> is reserved for inner data field')

        if field_type == 'metainfo':
            if name in self._data_fields:
                raise AttributeError(
                    f'Cannot set {name} to be a field of metainfo '
                    f'because {name} is already a data field')
            self._metainfo_fields.add(name)

        else:
            if name in self._metainfo_fields:
                raise AttributeError(
                    f'Cannot set {name} to be a field of data '
                    f'because {name} is already a metainfo field')

            if not isinstance(value, abc.Sequence):
                raise TypeError(
                    'The value should be a sequence (of numpy.ndarray or'
                    f'torch.Tesnor), but got a {type(value)}')

            if len(value) == 0:
                raise ValueError('Setting empty value is not allowed')

            if not isinstance(value[0], (torch.Tensor, np.ndarray)):
                raise TypeError(
                    'The value should be a sequence of numpy.ndarray or'
                    f'torch.Tesnor, but got a sequence of {type(value[0])}')

            if self.nlevel is not None:
                assert len(value) == self.nlevel, (
                    f'The length of the value ({len(value)}) should match the'
                    f'number of the levels ({self.nlevel})')
            else:
                object.__setattr__(self, '_nlevel', len(value))
                for i in range(self.nlevel):
                    object.__setattr__(self, f'_level_{i}', PixelData())

            for i, v in enumerate(value):
                self[i].set_field(v, name, field_type='data')

            self._data_fields.add(name)

        object.__setattr__(self, name, value)

    def __delattr__(self, item: str):
        """delete the item in dataelement.

        Args:
            item (str): The key to delete.
        """
        if item in ('_metainfo_fields', '_data_fields'):
            raise AttributeError(f'{item} has been used as a '
                                 'private attribute, which is immutable. ')

        if item in self._metainfo_fields:
            super().__delattr__(item)
        else:
            for level in self.levels():
                level.__delattr__(item)
            self._data_fields.remove(item)

    def __getattr__(self, name):
        if name in {'_data_fields', '_metainfo_fields'
                    } or name not in self._data_fields:
            raise AttributeError(
                f'\'{self.__class__.__name__}\' object has no attribute '
                f'\'{name}\'')

        return [getattr(level, name) for level in self.levels()]

    def pop(self, *args) -> Any:
        """pop property in data and metainfo as the same as python."""
        assert len(args) < 3, '``pop`` get more than 2 arguments'
        name = args[0]
        if name in self._metainfo_fields:
            self._metainfo_fields.remove(name)
            return self.__dict__.pop(*args)

        elif name in self._data_fields:
            self._data_fields.remove(name)
            return [level.pop(*args) for level in self.levels()]

        # with default value
        elif len(args) == 2:
            return args[1]
        else:
            # don't just use 'self.__dict__.pop(*args)' for only popping key in
            # metainfo or data
            raise KeyError(f'{args[0]} is not contained in metainfo or data')

    def _convert(self, apply_to: Type,
                 func: Callable[[Any], Any]) -> 'MultilevelPixelData':
        """Convert data items with the given function.

        Args:
            apply_to (Type): The type of data items to apply the conversion
            func (Callable): The conversion function that takes a data item
                as the input and return the converted result

        Returns:
            MultilevelPixelData: the converted data element.
        """
        new_data = self.new()
        for k, v in self.items():
            if is_list_of(v, apply_to):
                v = [func(_v) for _v in v]
                data = {k: v}
                new_data.set_data(data)
        return new_data

    def cpu(self) -> 'MultilevelPixelData':
        """Convert all tensors to CPU in data."""
        return self._convert(apply_to=torch.Tensor, func=lambda x: x.cpu())

    def cuda(self) -> 'MultilevelPixelData':
        """Convert all tensors to GPU in data."""
        return self._convert(apply_to=torch.Tensor, func=lambda x: x.cuda())

    def detach(self) -> 'MultilevelPixelData':
        """Detach all tensors in data."""
        return self._convert(apply_to=torch.Tensor, func=lambda x: x.detach())

    def numpy(self) -> 'MultilevelPixelData':
        """Convert all tensor to np.narray in data."""
        return self._convert(
            apply_to=torch.Tensor, func=lambda x: x.detach().cpu().numpy())

    def to_tensor(self) -> 'MultilevelPixelData':
        """Convert all tensor to np.narray in data."""
        return self._convert(
            apply_to=np.ndarray, func=lambda x: torch.from_numpy(x))

    # Tensor-like methods
    def to(self, *args, **kwargs) -> 'MultilevelPixelData':
        """Apply same name function to all tensors in data_fields."""
        new_data = self.new()
        for k, v in self.items():
            if hasattr(v[0], 'to'):
                v = [v_.to(*args, **kwargs) for v_ in v]
                data = {k: v}
                new_data.set_data(data)
        return new_data

class PoseDataSample(BaseDataElement):
    """The base data structure of MMPose that is used as the interface between
    modules.

    The attributes of ``PoseDataSample`` includes:

        - ``gt_instances``(InstanceData): Ground truth of instances with
            keypoint annotations
        - ``pred_instances``(InstanceData): Instances with keypoint
            predictions
        - ``gt_fields``(PixelData): Ground truth of spatial distribution
            annotations like keypoint heatmaps and part affine fields (PAF)
        - ``pred_fields``(PixelData): Predictions of spatial distributions

    Examples:
        >>> import torch
        >>> from mmengine.structures import InstanceData, PixelData
        >>> from mmpose.structures import PoseDataSample

        >>> pose_meta = dict(img_shape=(800, 1216),
        ...                  crop_size=(256, 192),
        ...                  heatmap_size=(64, 48))
        >>> gt_instances = InstanceData()
        >>> gt_instances.bboxes = torch.rand((1, 4))
        >>> gt_instances.keypoints = torch.rand((1, 17, 2))
        >>> gt_instances.keypoints_visible = torch.rand((1, 17, 1))
        >>> gt_fields = PixelData()
        >>> gt_fields.heatmaps = torch.rand((17, 64, 48))

        >>> data_sample = PoseDataSample(gt_instances=gt_instances,
        ...                              gt_fields=gt_fields,
        ...                              metainfo=pose_meta)
        >>> assert 'img_shape' in data_sample
        >>> len(data_sample.gt_instances)
        1
    """

    @property
    def gt_instances(self) -> InstanceData:
        return self._gt_instances

    @gt_instances.setter
    def gt_instances(self, value: InstanceData):
        self.set_field(value, '_gt_instances', dtype=InstanceData)

    @gt_instances.deleter
    def gt_instances(self):
        del self._gt_instances

    @property
    def gt_instance_labels(self) -> InstanceData:
        return self._gt_instance_labels

    @gt_instance_labels.setter
    def gt_instance_labels(self, value: InstanceData):
        self.set_field(value, '_gt_instance_labels', dtype=InstanceData)

    @gt_instance_labels.deleter
    def gt_instance_labels(self):
        del self._gt_instance_labels

    @property
    def pred_instances(self) -> InstanceData:
        return self._pred_instances

    @pred_instances.setter
    def pred_instances(self, value: InstanceData):
        self.set_field(value, '_pred_instances', dtype=InstanceData)

    @pred_instances.deleter
    def pred_instances(self):
        del self._pred_instances

    @property
    def gt_fields(self) -> Union[PixelData, MultilevelPixelData]:
        return self._gt_fields

    @gt_fields.setter
    def gt_fields(self, value: Union[PixelData, MultilevelPixelData]):
        self.set_field(value, '_gt_fields', dtype=type(value))

    @gt_fields.deleter
    def gt_fields(self):
        del self._gt_fields

    @property
    def pred_fields(self) -> PixelData:
        return self._pred_heatmaps

    @pred_fields.setter
    def pred_fields(self, value: PixelData):
        self.set_field(value, '_pred_heatmaps', dtype=PixelData)

    @pred_fields.deleter
    def pred_fields(self):
        del self._pred_heatmaps

def flip_keypoints(keypoints: np.ndarray,
                   keypoints_visible: Optional[np.ndarray],
                   image_size: Tuple[int, int],
                   flip_indices: List[int],
                   direction: str = 'horizontal'
                   ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Flip keypoints in the given direction.

    Note:

        - keypoint number: K
        - keypoint dimension: D

    Args:
        keypoints (np.ndarray): Keypoints in shape (..., K, D)
        keypoints_visible (np.ndarray, optional): The visibility of keypoints
            in shape (..., K, 1) or (..., K, 2). Set ``None`` if the keypoint
            visibility is unavailable
        image_size (tuple): The image shape in [w, h]
        flip_indices (List[int]): The indices of each keypoint's symmetric
            keypoint
        direction (str): The flip direction. Options are ``'horizontal'``,
            ``'vertical'`` and ``'diagonal'``. Defaults to ``'horizontal'``

    Returns:
        tuple:
        - keypoints_flipped (np.ndarray): Flipped keypoints in shape
            (..., K, D)
        - keypoints_visible_flipped (np.ndarray, optional): Flipped keypoints'
            visibility in shape (..., K, 1) or (..., K, 2). Return ``None`` if
            the input ``keypoints_visible`` is ``None``
    """

    ndim = keypoints.ndim
    assert keypoints.shape[:-1] == keypoints_visible.shape[:ndim - 1], (
        f'Mismatched shapes of keypoints {keypoints.shape} and '
        f'keypoints_visible {keypoints_visible.shape}')

    direction_options = {'horizontal', 'vertical', 'diagonal'}
    assert direction in direction_options, (
        f'Invalid flipping direction "{direction}". '
        f'Options are {direction_options}')

    # swap the symmetric keypoint pairs
    if direction == 'horizontal' or direction == 'vertical':
        keypoints = keypoints.take(flip_indices, axis=ndim - 2)
        if keypoints_visible is not None:
            keypoints_visible = keypoints_visible.take(
                flip_indices, axis=ndim - 2)

    # flip the keypoints
    w, h = image_size
    if direction == 'horizontal':
        keypoints[..., 0] = w - 1 - keypoints[..., 0]
    elif direction == 'vertical':
        keypoints[..., 1] = h - 1 - keypoints[..., 1]
    else:
        keypoints = [w, h] - keypoints - 1

    return keypoints, keypoints_visible
def image_to_tensor(img: Union[np.ndarray,
                               Sequence[np.ndarray]]) -> torch.torch.Tensor:
    """Translate image or sequence of images to tensor. Multiple image tensors
    will be stacked.

    Args:
        value (np.ndarray | Sequence[np.ndarray]): The original image or
            image sequence

    Returns:
        torch.Tensor: The output tensor.
    """

    if isinstance(img, np.ndarray):
        if len(img.shape) < 3:
            img = np.expand_dims(img, -1)

        img = np.ascontiguousarray(img)
        tensor = torch.from_numpy(img).permute(2, 0, 1).contiguous()
    else:
        assert is_seq_of(img, np.ndarray)
        tensor = torch.stack([image_to_tensor(_img) for _img in img])

    return tensor


def keypoints_to_tensor(keypoints: Union[np.ndarray, Sequence[np.ndarray]]
                        ) -> torch.torch.Tensor:
    """Translate keypoints or sequence of keypoints to tensor. Multiple
    keypoints tensors will be stacked.

    Args:
        keypoints (np.ndarray | Sequence[np.ndarray]): The keypoints or
            keypoints sequence.

    Returns:
        torch.Tensor: The output tensor.
    """
    if isinstance(keypoints, np.ndarray):
        keypoints = np.ascontiguousarray(keypoints)
        tensor = torch.from_numpy(keypoints).contiguous()
    else:
        assert is_seq_of(keypoints, np.ndarray)
        tensor = torch.stack(
            [keypoints_to_tensor(_keypoints) for _keypoints in keypoints])

    return tensor

def bbox_xyxy2cs(bbox: np.ndarray,
                 padding: float = 1.) -> Tuple[np.ndarray, np.ndarray]:
    """Transform the bbox format from (x,y,w,h) into (center, scale)

    Args:
        bbox (ndarray): Bounding box(es) in shape (4,) or (n, 4), formatted
            as (left, top, right, bottom)
        padding (float): BBox padding factor that will be multilied to scale.
            Default: 1.0

    Returns:
        tuple: A tuple containing center and scale.
        - np.ndarray[float32]: Center (x, y) of the bbox in shape (2,) or
            (n, 2)
        - np.ndarray[float32]: Scale (w, h) of the bbox in shape (2,) or
            (n, 2)
    """
    # convert single bbox from (4, ) to (1, 4)
    dim = bbox.ndim
    if dim == 1:
        bbox = bbox[None, :]

    scale = (bbox[..., 2:] - bbox[..., :2]) * padding
    center = (bbox[..., 2:] + bbox[..., :2]) * 0.5

    if dim == 1:
        center = center[0]
        scale = scale[0]

    return center, scale
def get_udp_warp_matrix(
    center: np.ndarray,
    scale: np.ndarray,
    rot: float,
    output_size: Tuple[int, int],
) -> np.ndarray:
    """Calculate the affine transformation matrix under the unbiased
    constraint. See `UDP (CVPR 2020)`_ for details.

    Note:

        - The bbox number: N

    Args:
        center (np.ndarray[2, ]): Center of the bounding box (x, y).
        scale (np.ndarray[2, ]): Scale of the bounding box
            wrt [width, height].
        rot (float): Rotation angle (degree).
        output_size (tuple): Size ([w, h]) of the output image

    Returns:
        np.ndarray: A 2x3 transformation matrix

    .. _`UDP (CVPR 2020)`: https://arxiv.org/abs/1911.07524
    """
    assert len(center) == 2
    assert len(scale) == 2
    assert len(output_size) == 2

    input_size = center * 2
    rot_rad = np.deg2rad(rot)
    warp_mat = np.zeros((2, 3), dtype=np.float32)
    scale_x = (output_size[0] - 1) / scale[0]
    scale_y = (output_size[1] - 1) / scale[1]
    warp_mat[0, 0] = math.cos(rot_rad) * scale_x
    warp_mat[0, 1] = -math.sin(rot_rad) * scale_x
    warp_mat[0, 2] = scale_x * (-0.5 * input_size[0] * math.cos(rot_rad) +
                                0.5 * input_size[1] * math.sin(rot_rad) +
                                0.5 * scale[0])
    warp_mat[1, 0] = math.sin(rot_rad) * scale_y
    warp_mat[1, 1] = math.cos(rot_rad) * scale_y
    warp_mat[1, 2] = scale_y * (-0.5 * input_size[0] * math.sin(rot_rad) -
                                0.5 * input_size[1] * math.cos(rot_rad) +
                                0.5 * scale[1])
    return warp_mat
def _rotate_point(pt: np.ndarray, angle_rad: float) -> np.ndarray:
    """Rotate a point by an angle.

    Args:
        pt (np.ndarray): 2D point coordinates (x, y) in shape (2, )
        angle_rad (float): rotation angle in radian

    Returns:
        np.ndarray: Rotated point in shape (2, )
    """

    sn, cs = np.sin(angle_rad), np.cos(angle_rad)
    rot_mat = np.array([[cs, -sn], [sn, cs]])
    return rot_mat @ pt
def _get_3rd_point(a: np.ndarray, b: np.ndarray):
    """To calculate the affine matrix, three pairs of points are required. This
    function is used to get the 3rd point, given 2D points a & b.

    The 3rd point is defined by rotating vector `a - b` by 90 degrees
    anticlockwise, using b as the rotation center.

    Args:
        a (np.ndarray): The 1st point (x,y) in shape (2, )
        b (np.ndarray): The 2nd point (x,y) in shape (2, )

    Returns:
        np.ndarray: The 3rd point.
    """
    direction = a - b
    c = b + np.r_[-direction[1], direction[0]]
    return c
def get_warp_matrix(
    center: np.ndarray,
    scale: np.ndarray,
    rot: float,
    output_size: Tuple[int, int],
    shift: Tuple[float, float] = (0., 0.),
    inv: bool = False,
    fix_aspect_ratio: bool = True,
) -> np.ndarray:
    """Calculate the affine transformation matrix that can warp the bbox area
    in the input image to the output size.

    Args:
        center (np.ndarray[2, ]): Center of the bounding box (x, y).
        scale (np.ndarray[2, ]): Scale of the bounding box
            wrt [width, height].
        rot (float): Rotation angle (degree).
        output_size (np.ndarray[2, ] | list(2,)): Size of the
            destination heatmaps.
        shift (0-100%): Shift translation ratio wrt the width/height.
            Default (0., 0.).
        inv (bool): Option to inverse the affine transform direction.
            (inv=False: src->dst or inv=True: dst->src)
        fix_aspect_ratio (bool): Whether to fix aspect ratio during transform.
            Defaults to True.

    Returns:
        np.ndarray: A 2x3 transformation matrix
    """
    assert len(center) == 2
    assert len(scale) == 2
    assert len(output_size) == 2
    assert len(shift) == 2

    shift = np.array(shift)
    src_w, src_h = scale[:2]
    dst_w, dst_h = output_size[:2]

    rot_rad = np.deg2rad(rot)
    src_dir = _rotate_point(np.array([src_w * -0.5, 0.]), rot_rad)
    dst_dir = np.array([dst_w * -0.5, 0.])

    src = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = center + scale * shift
    src[1, :] = center + src_dir + scale * shift

    dst = np.zeros((3, 2), dtype=np.float32)
    dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
    dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5]) + dst_dir

    if fix_aspect_ratio:
        src[2, :] = _get_3rd_point(src[0, :], src[1, :])
        dst[2, :] = _get_3rd_point(dst[0, :], dst[1, :])
    else:
        src_dir_2 = _rotate_point(np.array([0., src_h * -0.5]), rot_rad)
        dst_dir_2 = np.array([0., dst_h * -0.5])
        src[2, :] = center + src_dir_2 + scale * shift
        dst[2, :] = np.array([dst_w * 0.5, dst_h * 0.5]) + dst_dir_2

    if inv:
        warp_mat = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    else:
        warp_mat = cv2.getAffineTransform(np.float32(src), np.float32(dst))
    return warp_mat

def registry_transforms():
    from mmengine.registry import TRANSFORMS

    @TRANSFORMS.register_module()
    class LoadImage(LoadImageFromFile):
        def transform(self, results: dict) -> Optional[dict]:
            """The transform function of :class:`LoadImage`.

            Args:
                results (dict): The result dict

            Returns:
                dict: The result dict.
            """
            try:
                if 'img' not in results:
                    # Load image from file by :meth:`LoadImageFromFile.transform`
                    results = super().transform(results)
                else:
                    img = results['img']
                    assert isinstance(img, np.ndarray)
                    if self.to_float32:
                        img = img.astype(np.float32)

                    if 'img_path' not in results:
                        results['img_path'] = None
                    results['img_shape'] = img.shape[:2]
                    results['ori_shape'] = img.shape[:2]
            except Exception as e:
                e = type(e)(
                    f'`{str(e)}` occurs when loading `{results["img_path"]}`.'
                    'Please check whether the file exists.')
                raise e

            return results
    
    @TRANSFORMS.register_module()
    class GetBBoxCenterScale(BaseTransform):
        """Convert bboxes from [x, y, w, h] to center and scale.

        The center is the coordinates of the bbox center, and the scale is the
        bbox width and height normalized by a scale factor.

        Required Keys:

            - bbox

        Added Keys:

            - bbox_center
            - bbox_scale

        Args:
            padding (float): The bbox padding scale that will be multilied to
                `bbox_scale`. Defaults to 1.25
        """

        def __init__(self, padding: float = 1.25) -> None:
            super().__init__()

            self.padding = padding

        def transform(self, results: Dict) -> Optional[dict]:
            """The transform function of :class:`GetBBoxCenterScale`.

            See ``transform()`` method of :class:`BaseTransform` for details.

            Args:
                results (dict): The result dict

            Returns:
                dict: The result dict.
            """
            if 'bbox_center' in results and 'bbox_scale' in results:
                rank, _ = get_dist_info()
                if rank == 0:
                    warnings.warn('Use the existing "bbox_center" and "bbox_scale"'
                                '. The padding will still be applied.')
                results['bbox_scale'] = results['bbox_scale'] * self.padding

            else:
                bbox = results['bbox']
                center, scale = bbox_xyxy2cs(bbox, padding=self.padding)

                results['bbox_center'] = center
                results['bbox_scale'] = scale

            return results

    def __repr__(self) -> str:
        """print the basic information of the transform.

        Returns:
            str: Formatted string.
        """
        repr_str = self.__class__.__name__ + f'(padding={self.padding})'
        return repr_str
    
    @TRANSFORMS.register_module()
    class TopdownAffine(BaseTransform):
        """Get the bbox image as the model input by affine transform.

        Required Keys:

            - img
            - bbox_center
            - bbox_scale
            - bbox_rotation (optional)
            - keypoints (optional)

        Modified Keys:

            - img
            - bbox_scale

        Added Keys:

            - input_size
            - transformed_keypoints

        Args:
            input_size (Tuple[int, int]): The input image size of the model in
                [w, h]. The bbox region will be cropped and resize to `input_size`
            use_udp (bool): Whether use unbiased data processing. See
                `UDP (CVPR 2020)`_ for details. Defaults to ``False``

        .. _`UDP (CVPR 2020)`: https://arxiv.org/abs/1911.07524
        """

        def __init__(self,
                    input_size: Tuple[int, int],
                    use_udp: bool = False) -> None:
            super().__init__()

            assert is_seq_of(input_size, int) and len(input_size) == 2, (
                f'Invalid input_size {input_size}')

            self.input_size = input_size
            self.use_udp = use_udp

        @staticmethod
        def _fix_aspect_ratio(bbox_scale: np.ndarray, aspect_ratio: float):
            """Reshape the bbox to a fixed aspect ratio.

            Args:
                bbox_scale (np.ndarray): The bbox scales (w, h) in shape (n, 2)
                aspect_ratio (float): The ratio of ``w/h``

            Returns:
                np.darray: The reshaped bbox scales in (n, 2)
            """

            w, h = np.hsplit(bbox_scale, [1])
            bbox_scale = np.where(w > h * aspect_ratio,
                                np.hstack([w, w / aspect_ratio]),
                                np.hstack([h * aspect_ratio, h]))
            return bbox_scale

        def transform(self, results: Dict) -> Optional[dict]:
            """The transform function of :class:`TopdownAffine`.

            See ``transform()`` method of :class:`BaseTransform` for details.

            Args:
                results (dict): The result dict

            Returns:
                dict: The result dict.
            """

            w, h = self.input_size
            warp_size = (int(w), int(h))

            # reshape bbox to fixed aspect ratio
            results['bbox_scale'] = self._fix_aspect_ratio(
                results['bbox_scale'], aspect_ratio=w / h)

            # TODO: support multi-instance
            assert results['bbox_center'].shape[0] == 1, (
                'Top-down heatmap only supports single instance. Got invalid '
                f'shape of bbox_center {results["bbox_center"].shape}.')

            center = results['bbox_center'][0]
            scale = results['bbox_scale'][0]
            if 'bbox_rotation' in results:
                rot = results['bbox_rotation'][0]
            else:
                rot = 0.

            if self.use_udp:
                warp_mat = get_udp_warp_matrix(
                    center, scale, rot, output_size=(w, h))
            else:
                warp_mat = get_warp_matrix(center, scale, rot, output_size=(w, h))

            if isinstance(results['img'], list):
                results['img'] = [
                    cv2.warpAffine(
                        img, warp_mat, warp_size, flags=cv2.INTER_LINEAR)
                    for img in results['img']
                ]
            else:
                results['img'] = cv2.warpAffine(
                    results['img'], warp_mat, warp_size, flags=cv2.INTER_LINEAR)

            if results.get('keypoints', None) is not None:
                if results.get('transformed_keypoints', None) is not None:
                    transformed_keypoints = results['transformed_keypoints'].copy()
                else:
                    transformed_keypoints = results['keypoints'].copy()
                # Only transform (x, y) coordinates
                transformed_keypoints[..., :2] = cv2.transform(
                    results['keypoints'][..., :2], warp_mat)
                results['transformed_keypoints'] = transformed_keypoints

            results['input_size'] = (w, h)
            results['input_center'] = center
            results['input_scale'] = scale

            return results

        def __repr__(self) -> str:
            """print the basic information of the transform.

            Returns:
                str: Formatted string.
            """
            repr_str = self.__class__.__name__
            repr_str += f'(input_size={self.input_size}, '
            repr_str += f'use_udp={self.use_udp})'
            return repr_str
    @TRANSFORMS.register_module()
    class PackPoseInputs(BaseTransform):
        """Pack the inputs data for pose estimation.

        The ``img_meta`` item is always populated. The contents of the
        ``img_meta`` dictionary depends on ``meta_keys``. By default it includes:

            - ``id``: id of the data sample

            - ``img_id``: id of the image

            - ``'category_id'``: the id of the instance category

            - ``img_path``: path to the image file

            - ``crowd_index`` (optional): measure the crowding level of an image,
                defined in CrowdPose dataset

            - ``ori_shape``: original shape of the image as a tuple (h, w, c)

            - ``img_shape``: shape of the image input to the network as a tuple \
                (h, w).  Note that images may be zero padded on the \
                bottom/right if the batch tensor is larger than this shape.

            - ``input_size``: the input size to the network

            - ``flip``: a boolean indicating if image flip transform was used

            - ``flip_direction``: the flipping direction

            - ``flip_indices``: the indices of each keypoint's symmetric keypoint

            - ``raw_ann_info`` (optional): raw annotation of the instance(s)

        Args:
            meta_keys (Sequence[str], optional): Meta keys which will be stored in
                :obj: `PoseDataSample` as meta info. Defaults to ``('id',
                'img_id', 'img_path', 'category_id', 'crowd_index, 'ori_shape',
                'img_shape', 'input_size', 'input_center', 'input_scale', 'flip',
                'flip_direction', 'flip_indices', 'raw_ann_info')``
        """

        # items in `instance_mapping_table` will be directly packed into
        # PoseDataSample.gt_instances without converting to Tensor
        instance_mapping_table = dict(
            bbox='bboxes',
            bbox_score='bbox_scores',
            keypoints='keypoints',
            keypoints_cam='keypoints_cam',
            keypoints_visible='keypoints_visible',
            # In CocoMetric, the area of predicted instances will be calculated
            # using gt_instances.bbox_scales. To unsure correspondence with
            # previous version, this key is preserved here.
            bbox_scale='bbox_scales',
            # `head_size` is used for computing MpiiPCKAccuracy metric,
            # namely, PCKh
            head_size='head_size',
        )

        # items in `field_mapping_table` will be packed into
        # PoseDataSample.gt_fields and converted to Tensor. These items will be
        # used for computing losses
        field_mapping_table = dict(
            heatmaps='heatmaps',
            instance_heatmaps='instance_heatmaps',
            heatmap_mask='heatmap_mask',
            heatmap_weights='heatmap_weights',
            displacements='displacements',
            displacement_weights='displacement_weights')

        # items in `label_mapping_table` will be packed into
        # PoseDataSample.gt_instance_labels and converted to Tensor. These items
        # will be used for computing losses
        label_mapping_table = dict(
            keypoint_labels='keypoint_labels',
            keypoint_weights='keypoint_weights',
            keypoints_visible_weights='keypoints_visible_weights')

        def __init__(self,
                    meta_keys=('id', 'img_id', 'img_path', 'category_id',
                                'crowd_index', 'ori_shape', 'img_shape',
                                'input_size', 'input_center', 'input_scale',
                                'flip', 'flip_direction', 'flip_indices',
                                'raw_ann_info', 'dataset_name'),
                    pack_transformed=False):
            self.meta_keys = meta_keys
            self.pack_transformed = pack_transformed

        def transform(self, results: dict) -> dict:
            """Method to pack the input data.

            Args:
                results (dict): Result dict from the data pipeline.

            Returns:
                dict:

                - 'inputs' (obj:`torch.Tensor`): The forward data of models.
                - 'data_samples' (obj:`PoseDataSample`): The annotation info of the
                    sample.
            """
            # Pack image(s) for 2d pose estimation
            if 'img' in results:
                img = results['img']
                inputs_tensor = image_to_tensor(img)
            # Pack keypoints for 3d pose-lifting
            elif 'lifting_target' in results and 'keypoints' in results:
                if 'keypoint_labels' in results:
                    keypoints = results['keypoint_labels']
                else:
                    keypoints = results['keypoints']
                inputs_tensor = keypoints_to_tensor(keypoints)

            data_sample = PoseDataSample()

            # pack instance data
            gt_instances = InstanceData()
            _instance_mapping_table = results.get('instance_mapping_table',
                                                self.instance_mapping_table)
            for key, packed_key in _instance_mapping_table.items():
                if key in results:
                    gt_instances.set_field(results[key], packed_key)

            # pack `transformed_keypoints` for visualizing data transform
            # and augmentation results
            if self.pack_transformed and 'transformed_keypoints' in results:
                gt_instances.set_field(results['transformed_keypoints'],
                                    'transformed_keypoints')

            data_sample.gt_instances = gt_instances

            # pack instance labels
            gt_instance_labels = InstanceData()
            _label_mapping_table = results.get('label_mapping_table',
                                            self.label_mapping_table)
            for key, packed_key in _label_mapping_table.items():
                if key in results:
                    if isinstance(results[key], list):
                        # A list of labels is usually generated by combined
                        # multiple encoders (See ``GenerateTarget`` in
                        # mmpose/datasets/transforms/common_transforms.py)
                        # In this case, labels in list should have the same
                        # shape and will be stacked.
                        _labels = np.stack(results[key])
                        gt_instance_labels.set_field(_labels, packed_key)
                    else:
                        gt_instance_labels.set_field(results[key], packed_key)
            data_sample.gt_instance_labels = gt_instance_labels.to_tensor()

            # pack fields
            gt_fields = None
            _field_mapping_table = results.get('field_mapping_table',
                                            self.field_mapping_table)
            for key, packed_key in _field_mapping_table.items():
                if key in results:
                    if isinstance(results[key], list):
                        if gt_fields is None:
                            gt_fields = MultilevelPixelData()
                        else:
                            assert isinstance(
                                gt_fields, MultilevelPixelData
                            ), 'Got mixed single-level and multi-level pixel data.'
                    else:
                        if gt_fields is None:
                            gt_fields = PixelData()
                        else:
                            assert isinstance(
                                gt_fields, PixelData
                            ), 'Got mixed single-level and multi-level pixel data.'

                    gt_fields.set_field(results[key], packed_key)

            if gt_fields:
                data_sample.gt_fields = gt_fields.to_tensor()

            img_meta = {k: results[k] for k in self.meta_keys if k in results}
            data_sample.set_metainfo(img_meta)

            packed_results = dict()
            packed_results['inputs'] = inputs_tensor
            packed_results['data_samples'] = data_sample

            return packed_results

        def __repr__(self) -> str:
            """print the basic information of the transform.

            Returns:
                str: Formatted string.
            """
            repr_str = self.__class__.__name__
            repr_str += f'(meta_keys={self.meta_keys}, '
            repr_str += f'pack_transformed={self.pack_transformed})'
            return repr_str