from typing import List, Optional, Sequence, Tuple, Union
from itertools import product

import cv2
import numpy as np
import torch
from mmengine.utils import is_seq_of
from torch import Tensor
from typing import Sequence, Union
from mmengine.config import Config

import torch.nn as nn
from mmengine.utils import digit_version
from mmengine.utils.dl_utils import TORCH_VERSION
from mmengine.model import BaseModule


class ChannelAttention(BaseModule):
    """Channel attention Module.

    Args:
        channels (int): The input (and output) channels of the attention layer.
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Defaults to None
    """

    def __init__(self, channels: int, init_cfg = None) -> None:
        super().__init__(init_cfg=init_cfg)
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Conv2d(channels, channels, 1, 1, 0, bias=True)
        if digit_version(torch.__version__) < (1, 7, 0):
            self.act = nn.Hardsigmoid()
        else:
            self.act = nn.Hardsigmoid(inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        """Forward function for ChannelAttention."""
        with torch.cuda.amp.autocast(enabled=False):
            out = self.global_avgpool(x)
        out = self.fc(out)
        out = self.act(out)
        return x * out
    
class ScaleNorm(nn.Module):
    """Scale Norm.

    Args:
        dim (int): The dimension of the scale vector.
        eps (float, optional): The minimum value in clamp. Defaults to 1e-5.

    Reference:
        `Transformers without Tears: Improving the Normalization
        of Self-Attention <https://arxiv.org/abs/1910.05895>`_
    """

    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.scale = dim**-0.5
        self.eps = eps
        self.g = nn.Parameter(torch.ones(1))

    def forward(self, x):
        """Forward function.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: The tensor after applying scale norm.
        """

        if torch.onnx.is_in_onnx_export() and \
                digit_version(TORCH_VERSION) >= digit_version('1.12'):

            norm = torch.linalg.norm(x, dim=-1, keepdim=True)

        else:
            norm = torch.norm(x, dim=-1, keepdim=True)
        norm = norm * self.scale
        return x / norm.clamp(min=self.eps) * self.g
def get_simcc_maximum(simcc_x: np.ndarray,
                      simcc_y: np.ndarray,
                      apply_softmax: bool = False
                      ) -> Tuple[np.ndarray, np.ndarray]:
    """Get maximum response location and value from simcc representations.

    Note:
        instance number: N
        num_keypoints: K
        heatmap height: H
        heatmap width: W

    Args:
        simcc_x (np.ndarray): x-axis SimCC in shape (K, Wx) or (N, K, Wx)
        simcc_y (np.ndarray): y-axis SimCC in shape (K, Wy) or (N, K, Wy)
        apply_softmax (bool): whether to apply softmax on the heatmap.
            Defaults to False.

    Returns:
        tuple:
        - locs (np.ndarray): locations of maximum heatmap responses in shape
            (K, 2) or (N, K, 2)
        - vals (np.ndarray): values of maximum heatmap responses in shape
            (K,) or (N, K)
    """

    assert isinstance(simcc_x, np.ndarray), ('simcc_x should be numpy.ndarray')
    assert isinstance(simcc_y, np.ndarray), ('simcc_y should be numpy.ndarray')
    assert simcc_x.ndim == 2 or simcc_x.ndim == 3, (
        f'Invalid shape {simcc_x.shape}')
    assert simcc_y.ndim == 2 or simcc_y.ndim == 3, (
        f'Invalid shape {simcc_y.shape}')
    assert simcc_x.ndim == simcc_y.ndim, (
        f'{simcc_x.shape} != {simcc_y.shape}')

    if simcc_x.ndim == 3:
        N, K, Wx = simcc_x.shape
        simcc_x = simcc_x.reshape(N * K, -1)
        simcc_y = simcc_y.reshape(N * K, -1)
    else:
        N = None

    if apply_softmax:
        simcc_x = simcc_x - np.max(simcc_x, axis=1, keepdims=True)
        simcc_y = simcc_y - np.max(simcc_y, axis=1, keepdims=True)
        ex, ey = np.exp(simcc_x), np.exp(simcc_y)
        simcc_x = ex / np.sum(ex, axis=1, keepdims=True)
        simcc_y = ey / np.sum(ey, axis=1, keepdims=True)

    x_locs = np.argmax(simcc_x, axis=1)
    y_locs = np.argmax(simcc_y, axis=1)
    locs = np.stack((x_locs, y_locs), axis=-1).astype(np.float32)
    max_val_x = np.amax(simcc_x, axis=1)
    max_val_y = np.amax(simcc_y, axis=1)

    mask = max_val_x > max_val_y
    max_val_x[mask] = max_val_y[mask]
    vals = max_val_x
    locs[vals <= 0.] = -1

    if N:
        locs = locs.reshape(N, K, 2)
        vals = vals.reshape(N, K)

    return locs, vals

def to_numpy(x: Union[Tensor, Sequence[Tensor]],
             return_device: bool = False,
             unzip: bool = False) -> Union[np.ndarray, tuple]:
    """Convert torch tensor to numpy.ndarray.

    Args:
        x (Tensor | Sequence[Tensor]): A single tensor or a sequence of
            tensors
        return_device (bool): Whether return the tensor device. Defaults to
            ``False``
        unzip (bool): Whether unzip the input sequence. Defaults to ``False``

    Returns:
        np.ndarray | tuple: If ``return_device`` is ``True``, return a tuple
        of converted numpy array(s) and the device indicator; otherwise only
        return the numpy array(s)
    """

    if isinstance(x, Tensor):
        arrays = x.detach().cpu().numpy()
        device = x.device
    elif isinstance(x, np.ndarray) or is_seq_of(x, np.ndarray):
        arrays = x
        device = 'cpu'
    elif is_seq_of(x, Tensor):
        if unzip:
            # convert (A, B) -> [(A[0], B[0]), (A[1], B[1]), ...]
            arrays = [
                tuple(to_numpy(_x[None, :]) for _x in _each)
                for _each in zip(*x)
            ]
        else:
            arrays = [to_numpy(_x) for _x in x]

        device = x[0].device

    else:
        raise ValueError(f'Invalid input type {type(x)}')

    if return_device:
        return arrays, device
    else:
        return arrays

def flip_vectors(x_labels: Tensor, y_labels: Tensor, flip_indices: List[int]):
    """Flip instance-level labels in specific axis for test-time augmentation.

    Args:
        x_labels (Tensor): The vector labels in x-axis to flip. Should be
            a tensor in shape [B, C, Wx]
        y_labels (Tensor): The vector labels in y-axis to flip. Should be
            a tensor in shape [B, C, Wy]
        flip_indices (List[int]): The indices of each keypoint's symmetric
            keypoint
    """
    assert x_labels.ndim == 3 and y_labels.ndim == 3
    assert len(flip_indices) == x_labels.shape[1] and len(
        flip_indices) == y_labels.shape[1]
    x_labels = x_labels[:, flip_indices].flip(-1)
    y_labels = y_labels[:, flip_indices]

    return x_labels, y_labels

from typing import Tuple

import numpy as np
import torch
from torch import Tensor


def get_simcc_normalized(batch_pred_simcc, sigma=None):
    """Normalize the predicted SimCC.

    Args:
        batch_pred_simcc (torch.Tensor): The predicted SimCC.
        sigma (float): The sigma of the Gaussian distribution.

    Returns:
        torch.Tensor: The normalized SimCC.
    """
    B, K, _ = batch_pred_simcc.shape

    # Scale and clamp the tensor
    if sigma is not None:
        batch_pred_simcc = batch_pred_simcc / (sigma * np.sqrt(np.pi * 2))
    batch_pred_simcc = batch_pred_simcc.clamp(min=0)

    # Compute the binary mask
    mask = (batch_pred_simcc.amax(dim=-1) > 1).reshape(B, K, 1)

    # Normalize the tensor using the maximum value
    norm = (batch_pred_simcc / batch_pred_simcc.amax(dim=-1).reshape(B, K, 1))

    # Apply normalization
    batch_pred_simcc = torch.where(mask, norm, batch_pred_simcc)

    return batch_pred_simcc

def _calc_distances(preds: np.ndarray, gts: np.ndarray, mask: np.ndarray,
                    norm_factor: np.ndarray) -> np.ndarray:
    """Calculate the normalized distances between preds and target.

    Note:
        - instance number: N
        - keypoint number: K
        - keypoint dimension: D (normally, D=2 or D=3)

    Args:
        preds (np.ndarray[N, K, D]): Predicted keypoint location.
        gts (np.ndarray[N, K, D]): Groundtruth keypoint location.
        mask (np.ndarray[N, K]): Visibility of the target. False for invisible
            joints, and True for visible. Invisible joints will be ignored for
            accuracy calculation.
        norm_factor (np.ndarray[N, D]): Normalization factor.
            Typical value is heatmap_size.

    Returns:
        np.ndarray[K, N]: The normalized distances. \
            If target keypoints are missing, the distance is -1.
    """
    N, K, _ = preds.shape
    # set mask=0 when norm_factor==0
    _mask = mask.copy()
    _mask[np.where((norm_factor == 0).sum(1))[0], :] = False

    distances = np.full((N, K), -1, dtype=np.float32)
    # handle invalid values
    norm_factor[np.where(norm_factor <= 0)] = 1e6
    distances[_mask] = np.linalg.norm(
        ((preds - gts) / norm_factor[:, None, :])[_mask], axis=-1)
    return distances.T


def _distance_acc(distances: np.ndarray, thr: float = 0.5) -> float:
    """Return the percentage below the distance threshold, while ignoring
    distances values with -1.

    Note:
        - instance number: N

    Args:
        distances (np.ndarray[N, ]): The normalized distances.
        thr (float): Threshold of the distances.

    Returns:
        float: Percentage of distances below the threshold. \
            If all target keypoints are missing, return -1.
    """
    distance_valid = distances != -1
    num_distance_valid = distance_valid.sum()
    if num_distance_valid > 0:
        return (distances[distance_valid] < thr).sum() / num_distance_valid
    return -1

def keypoint_pck_accuracy(pred: np.ndarray, gt: np.ndarray, mask: np.ndarray,
                          thr: np.ndarray, norm_factor: np.ndarray) -> tuple:
    """Calculate the pose accuracy of PCK for each individual keypoint and the
    averaged accuracy across all keypoints for coordinates.

    Note:
        PCK metric measures accuracy of the localization of the body joints.
        The distances between predicted positions and the ground-truth ones
        are typically normalized by the bounding box size.
        The threshold (thr) of the normalized distance is commonly set
        as 0.05, 0.1 or 0.2 etc.

        - instance number: N
        - keypoint number: K

    Args:
        pred (np.ndarray[N, K, 2]): Predicted keypoint location.
        gt (np.ndarray[N, K, 2]): Groundtruth keypoint location.
        mask (np.ndarray[N, K]): Visibility of the target. False for invisible
            joints, and True for visible. Invisible joints will be ignored for
            accuracy calculation.
        thr (float): Threshold of PCK calculation.
        norm_factor (np.ndarray[N, 2]): Normalization factor for H&W.

    Returns:
        tuple: A tuple containing keypoint accuracy.

        - acc (np.ndarray[K]): Accuracy of each keypoint.
        - avg_acc (float): Averaged accuracy across all keypoints.
        - cnt (int): Number of valid keypoints.
    """
    distances = _calc_distances(pred, gt, mask, norm_factor)
    acc = np.array([_distance_acc(d, thr) for d in distances])
    valid_acc = acc[acc >= 0]
    cnt = len(valid_acc)
    avg_acc = valid_acc.mean() if cnt > 0 else 0.0
    return acc, avg_acc, cnt

def simcc_pck_accuracy(output: Tuple[np.ndarray, np.ndarray],
                       target: Tuple[np.ndarray, np.ndarray],
                       simcc_split_ratio: float,
                       mask: np.ndarray,
                       thr: float = 0.05,
                       normalize: Optional[np.ndarray] = None) -> tuple:
    """Calculate the pose accuracy of PCK for each individual keypoint and the
    averaged accuracy across all keypoints from SimCC.

    Note:
        PCK metric measures accuracy of the localization of the body joints.
        The distances between predicted positions and the ground-truth ones
        are typically normalized by the bounding box size.
        The threshold (thr) of the normalized distance is commonly set
        as 0.05, 0.1 or 0.2 etc.

        - instance number: N
        - keypoint number: K

    Args:
        output (Tuple[np.ndarray, np.ndarray]): Model predicted SimCC.
        target (Tuple[np.ndarray, np.ndarray]): Groundtruth SimCC.
        mask (np.ndarray[N, K]): Visibility of the target. False for invisible
            joints, and True for visible. Invisible joints will be ignored for
            accuracy calculation.
        thr (float): Threshold of PCK calculation. Default 0.05.
        normalize (np.ndarray[N, 2]): Normalization factor for H&W.

    Returns:
        tuple: A tuple containing keypoint accuracy.

        - np.ndarray[K]: Accuracy of each keypoint.
        - float: Averaged accuracy across all keypoints.
        - int: Number of valid keypoints.
    """
    pred_x, pred_y = output
    gt_x, gt_y = target

    N, _, Wx = pred_x.shape
    _, _, Wy = pred_y.shape
    W, H = int(Wx / simcc_split_ratio), int(Wy / simcc_split_ratio)

    if normalize is None:
        normalize = np.tile(np.array([[H, W]]), (N, 1))

    pred_coords, _ = get_simcc_maximum(pred_x, pred_y)
    pred_coords /= simcc_split_ratio
    gt_coords, _ = get_simcc_maximum(gt_x, gt_y)
    gt_coords /= simcc_split_ratio

    return keypoint_pck_accuracy(pred_coords, gt_coords, mask, thr, normalize)

def gaussian_blur1d(simcc: np.ndarray, kernel: int = 11) -> np.ndarray:
    """Modulate simcc distribution with Gaussian.

    Note:
        - num_keypoints: K
        - simcc length: Wx

    Args:
        simcc (np.ndarray[K, Wx]): model predicted simcc.
        kernel (int): Gaussian kernel size (K) for modulation, which should
            match the simcc gaussian sigma when training.
            K=17 for sigma=3 and k=11 for sigma=2.

    Returns:
        np.ndarray ([K, Wx]): Modulated simcc distribution.
    """
    assert kernel % 2 == 1

    border = (kernel - 1) // 2
    N, K, Wx = simcc.shape

    for n, k in product(range(N), range(K)):
        origin_max = np.max(simcc[n, k])
        dr = np.zeros((1, Wx + 2 * border), dtype=np.float32)
        dr[0, border:-border] = simcc[n, k].copy()
        dr = cv2.GaussianBlur(dr, (kernel, 1), 0)
        simcc[n, k] = dr[0, border:-border].copy()
        simcc[n, k] *= origin_max / np.max(simcc[n, k])
    return simcc

def refine_simcc_dark(keypoints: np.ndarray, simcc: np.ndarray,
                      blur_kernel_size: int) -> np.ndarray:
    """SimCC version. Refine keypoint predictions using distribution aware
    coordinate decoding for UDP. See `UDP`_ for details. The operation is in-
    place.

    Note:

        - instance number: N
        - keypoint number: K
        - keypoint dimension: D

    Args:
        keypoints (np.ndarray): The keypoint coordinates in shape (N, K, D)
        simcc (np.ndarray): The heatmaps in shape (N, K, Wx)
        blur_kernel_size (int): The Gaussian blur kernel size of the heatmap
            modulation

    Returns:
        np.ndarray: Refine keypoint coordinates in shape (N, K, D)

    .. _`UDP`: https://arxiv.org/abs/1911.07524
    """
    N = simcc.shape[0]

    # modulate simcc
    simcc = gaussian_blur1d(simcc, blur_kernel_size)
    np.clip(simcc, 1e-3, 50., simcc)
    np.log(simcc, simcc)

    simcc = np.pad(simcc, ((0, 0), (0, 0), (2, 2)), 'edge')

    for n in range(N):
        px = (keypoints[n] + 2.5).astype(np.int64).reshape(-1, 1)  # K, 1

        dx0 = np.take_along_axis(simcc[n], px, axis=1)  # K, 1
        dx1 = np.take_along_axis(simcc[n], px + 1, axis=1)
        dx_1 = np.take_along_axis(simcc[n], px - 1, axis=1)
        dx2 = np.take_along_axis(simcc[n], px + 2, axis=1)
        dx_2 = np.take_along_axis(simcc[n], px - 2, axis=1)

        dx = 0.5 * (dx1 - dx_1)
        dxx = 1e-9 + 0.25 * (dx2 - 2 * dx0 + dx_2)

        offset = dx / dxx
        keypoints[n] -= offset.reshape(-1)

    return keypoints

# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
import warnings

import numpy as np
from mmengine import Config


def parse_pose_metainfo(metainfo: dict):
    """Load meta information of pose dataset and check its integrity.

    Args:
        metainfo (dict): Raw data of pose meta information, which should
            contain following contents:

            - "dataset_name" (str): The name of the dataset
            - "keypoint_info" (dict): The keypoint-related meta information,
                e.g., name, upper/lower body, and symmetry
            - "skeleton_info" (dict): The skeleton-related meta information,
                e.g., start/end keypoint of limbs
            - "joint_weights" (list[float]): The loss weights of keypoints
            - "sigmas" (list[float]): The keypoint distribution parameters
                to calculate OKS score. See `COCO keypoint evaluation
                <https://cocodataset.org/#keypoints-eval>`__.

            An example of metainfo is shown as follows.

            .. code-block:: none
                {
                    "dataset_name": "coco",
                    "keypoint_info":
                    {
                        0:
                        {
                            "name": "nose",
                            "type": "upper",
                            "swap": "",
                            "color": [51, 153, 255],
                        },
                        1:
                        {
                            "name": "right_eye",
                            "type": "upper",
                            "swap": "left_eye",
                            "color": [51, 153, 255],
                        },
                        ...
                    },
                    "skeleton_info":
                    {
                        0:
                        {
                            "link": ("left_ankle", "left_knee"),
                            "color": [0, 255, 0],
                        },
                        ...
                    },
                    "joint_weights": [1., 1., ...],
                    "sigmas": [0.026, 0.025, ...],
                }


            A special case is that `metainfo` can have the key "from_file",
            which should be the path of a config file. In this case, the
            actual metainfo will be loaded by:

            .. code-block:: python
                metainfo = mmengine.Config.fromfile(metainfo['from_file'])

    Returns:
        Dict: pose meta information that contains following contents:

        - "dataset_name" (str): Same as ``"dataset_name"`` in the input
        - "num_keypoints" (int): Number of keypoints
        - "keypoint_id2name" (dict): Mapping from keypoint id to name
        - "keypoint_name2id" (dict): Mapping from keypoint name to id
        - "upper_body_ids" (list): Ids of upper-body keypoint
        - "lower_body_ids" (list): Ids of lower-body keypoint
        - "flip_indices" (list): The Id of each keypoint's symmetric keypoint
        - "flip_pairs" (list): The Ids of symmetric keypoint pairs
        - "keypoint_colors" (numpy.ndarray): The keypoint color matrix of
            shape [K, 3], where each row is the color of one keypint in bgr
        - "num_skeleton_links" (int): The number of links
        - "skeleton_links" (list): The links represented by Id pairs of start
             and end points
        - "skeleton_link_colors" (numpy.ndarray): The link color matrix
        - "dataset_keypoint_weights" (numpy.ndarray): Same as the
            ``"joint_weights"`` in the input
        - "sigmas" (numpy.ndarray): Same as the ``"sigmas"`` in the input
    """

    if 'from_file' in metainfo:
        cfg_file = metainfo['from_file']
        if not osp.isfile(cfg_file):
            # Search configs in 'mmpose/.mim/configs/' in case that mmpose
            # is installed in non-editable mode.
            mmpose_path = osp.dirname(__file__)
            _cfg_file = osp.join(mmpose_path, '.mim', 'configs', '_base_',
                                 'datasets', osp.basename(cfg_file))
            if osp.isfile(_cfg_file):
                warnings.warn(
                    f'The metainfo config file "{cfg_file}" does not exist. '
                    f'A matched config file "{_cfg_file}" will be used '
                    'instead.')
                cfg_file = _cfg_file
            else:
                raise FileNotFoundError(
                    f'The metainfo config file "{cfg_file}" does not exist.')

        # TODO: remove the nested structure of dataset_info
        # metainfo = Config.fromfile(metainfo['from_file'])
        metainfo = Config.fromfile(cfg_file).dataset_info

    # check data integrity
    assert 'dataset_name' in metainfo
    assert 'keypoint_info' in metainfo
    assert 'skeleton_info' in metainfo
    assert 'joint_weights' in metainfo
    assert 'sigmas' in metainfo

    # parse metainfo
    parsed = dict(
        dataset_name=None,
        num_keypoints=None,
        keypoint_id2name={},
        keypoint_name2id={},
        upper_body_ids=[],
        lower_body_ids=[],
        flip_indices=[],
        flip_pairs=[],
        keypoint_colors=[],
        num_skeleton_links=None,
        skeleton_links=[],
        skeleton_link_colors=[],
        dataset_keypoint_weights=None,
        sigmas=None,
    )

    parsed['dataset_name'] = metainfo['dataset_name']

    # parse keypoint information
    parsed['num_keypoints'] = len(metainfo['keypoint_info'])

    for kpt_id, kpt in metainfo['keypoint_info'].items():
        kpt_name = kpt['name']
        parsed['keypoint_id2name'][kpt_id] = kpt_name
        parsed['keypoint_name2id'][kpt_name] = kpt_id
        parsed['keypoint_colors'].append(kpt.get('color', [255, 128, 0]))

        kpt_type = kpt.get('type', '')
        if kpt_type == 'upper':
            parsed['upper_body_ids'].append(kpt_id)
        elif kpt_type == 'lower':
            parsed['lower_body_ids'].append(kpt_id)

        swap_kpt = kpt.get('swap', '')
        if swap_kpt == kpt_name or swap_kpt == '':
            parsed['flip_indices'].append(kpt_name)
        else:
            parsed['flip_indices'].append(swap_kpt)
            pair = (swap_kpt, kpt_name)
            if pair not in parsed['flip_pairs']:
                parsed['flip_pairs'].append(pair)

    # parse skeleton information
    parsed['num_skeleton_links'] = len(metainfo['skeleton_info'])
    for _, sk in metainfo['skeleton_info'].items():
        parsed['skeleton_links'].append(sk['link'])
        parsed['skeleton_link_colors'].append(sk.get('color', [96, 96, 255]))

    # parse extra information
    parsed['dataset_keypoint_weights'] = np.array(
        metainfo['joint_weights'], dtype=np.float32)
    parsed['sigmas'] = np.array(metainfo['sigmas'], dtype=np.float32)

    if 'stats_info' in metainfo:
        parsed['stats_info'] = {}
        for name, val in metainfo['stats_info'].items():
            parsed['stats_info'][name] = np.array(val, dtype=np.float32)

    # formatting
    def _map(src, mapping: dict):
        if isinstance(src, (list, tuple)):
            cls = type(src)
            return cls(_map(s, mapping) for s in src)
        else:
            return mapping[src]

    parsed['flip_pairs'] = _map(
        parsed['flip_pairs'], mapping=parsed['keypoint_name2id'])
    parsed['flip_indices'] = _map(
        parsed['flip_indices'], mapping=parsed['keypoint_name2id'])
    parsed['skeleton_links'] = _map(
        parsed['skeleton_links'], mapping=parsed['keypoint_name2id'])

    parsed['keypoint_colors'] = np.array(
        parsed['keypoint_colors'], dtype=np.uint8)
    parsed['skeleton_link_colors'] = np.array(
        parsed['skeleton_link_colors'], dtype=np.uint8)

    return parsed


def dataset_meta_from_config(config: Config,
                             dataset_mode: str = 'train') -> Optional[dict]:
    """Get dataset metainfo from the model config.

    Args:
        config (str, :obj:`Path`, or :obj:`mmengine.Config`): Config file path,
            :obj:`Path`, or the config object.
        dataset_mode (str): Specify the dataset of which to get the metainfo.
            Options are ``'train'``, ``'val'`` and ``'test'``. Defaults to
            ``'train'``

    Returns:
        dict, optional: The dataset metainfo. See
        ``mmpose.datasets.datasets.utils.parse_pose_metainfo`` for details.
        Return ``None`` if failing to get dataset metainfo from the config.
    """
    try:
        if dataset_mode == 'train':
            dataset_cfg = config.train_dataloader.dataset
        elif dataset_mode == 'val':
            dataset_cfg = config.val_dataloader.dataset
        elif dataset_mode == 'test':
            dataset_cfg = config.test_dataloader.dataset
        else:
            raise ValueError(
                f'Invalid dataset {dataset_mode} to get metainfo. '
                'Should be one of "train", "val", or "test".')

        if 'metainfo' in dataset_cfg:
            metainfo = dataset_cfg.metainfo
        else:
            # import mmpose.datasets.datasets  # noqa: F401, F403
            # from mmpose.registry import DATASETS
            from mmengine.registry import DATASETS
            dataset_class = dataset_cfg.type if isinstance(
                dataset_cfg.type, type) else DATASETS.get(dataset_cfg.type)
            metainfo = dataset_class.METAINFO

        metainfo = parse_pose_metainfo(metainfo)

    except AttributeError:
        metainfo = None

    return metainfo