import cv2
import numpy as np

from mmengine.registry import MODELS
from mmengine.utils.dl_utils.parrots_wrapper import SyncBatchNorm
import torch.nn as nn
from typing import List, Optional, Union

import torch
import torch.nn as nn
from mmengine.dataset import Compose, pseudo_collate
from mmengine.registry import init_default_scope
from PIL import Image
from rtmpose.transform_utils import registry_transforms, PoseDataSample

registry_transforms()
MODELS.register_module(module=nn.SiLU, name='SiLU')
MODELS.register_module('SyncBN', module=SyncBatchNorm)

def bbox_xywh2xyxy(bbox_xywh: np.ndarray) -> np.ndarray:
    """
    Transform the bbox format from xywh to x1y1x2y2.

    Args:
        bbox_xywh (ndarray): Bounding boxes (with scores),
            shaped (n, 4) or (n, 5). (left, top, width, height, [score])
    Returns:
        np.ndarray: Bounding boxes (with scores), shaped (n, 4) or
          (n, 5). (left, top, right, bottom, [score])
    """
    bbox_xyxy = bbox_xywh.copy()
    bbox_xyxy[:, 2] = bbox_xyxy[:, 2] + bbox_xyxy[:, 0]
    bbox_xyxy[:, 3] = bbox_xyxy[:, 3] + bbox_xyxy[:, 1]

    return bbox_xyxy

def inference_topdown(model: nn.Module,
                      img: Union[np.ndarray, str],
                      bboxes: Optional[Union[List, np.ndarray]] = None,
                      bbox_format: str = 'xyxy') -> List[PoseDataSample]:
    """Inference image with a top-down pose estimator.

    Args:
        model (nn.Module): The top-down pose estimator
        img (np.ndarray | str): The loaded image or image file to inference
        bboxes (np.ndarray, optional): The bboxes in shape (N, 4), each row
            represents a bbox. If not given, the entire image will be regarded
            as a single bbox area. Defaults to ``None``
        bbox_format (str): The bbox format indicator. Options are ``'xywh'``
            and ``'xyxy'``. Defaults to ``'xyxy'``

    Returns:
        List[:obj:`PoseDataSample`]: The inference results. Specifically, the
        predicted keypoints and scores are saved at
        ``data_sample.pred_instances.keypoints`` and
        ``data_sample.pred_instances.keypoint_scores``.
    """
    scope = model.cfg.get('default_scope', 'mmpose')
    if scope is not None:
        init_default_scope(scope)
    pipeline = Compose(model.cfg.test_dataloader.dataset.pipeline)

    if bboxes is None or len(bboxes) == 0:
        # get bbox from the image size
        if isinstance(img, str):
            w, h = Image.open(img).size
        else:
            h, w = img.shape[:2]

        bboxes = np.array([[0, 0, w, h]], dtype=np.float32)
    else:
        if isinstance(bboxes, list):
            bboxes = np.array(bboxes)

        assert bbox_format in {'xyxy', 'xywh'}, \
            f'Invalid bbox_format "{bbox_format}".'

        if bbox_format == 'xywh':
            bboxes = bbox_xywh2xyxy(bboxes)

    # construct batch data samples
    data_list = []
    for bbox in bboxes:
        if isinstance(img, str):
            data_info = dict(img_path=img)
        else:
            data_info = dict(img=img)
        data_info['bbox'] = bbox[None]  # shape (1, 4)
        data_info['bbox_score'] = np.ones(1, dtype=np.float32)  # shape (1,)
        data_info.update(model.dataset_meta)
        data_list.append(pipeline(data_info))

    if data_list:
        # collate data list into a batch, which is a dict with following keys:
        # batch['inputs']: a list of input images
        # batch['data_samples']: a list of :obj:`PoseDataSample`
        batch = pseudo_collate(data_list)
        with torch.no_grad():
            results = model.test_step(batch)
    else:
        results = []

    return results


# pose_config = '/mnt/banana/student/thuyntt/jersey-number-pipeline/pose/rtmpose/config/rtmpose-l_8xb256-420e_coco-256x192.py'
# pose_checkpoint = '/mnt/banana/student/thuyntt/jersey-number-pipeline/pose/rtmpose/ckpt/rtmpose-l_simcc-aic-coco_pt-aic-coco_420e-256x192-f016ffe0_20230126.pth'
# device = 'cuda:0'
# detection_file = '/mnt/banana/student/thuyntt/Deep-EIoU/cache/train/detection/v_-6Os86HzwCs_c001.npy'

# img_path = '/mnt/banana/student/thuyntt/data/sportsmot_publish/dataset/train/v_-6Os86HzwCs_c001/img1/000001.jpg'
# image = cv2.imread(img_path)

# all_detections = np.load(detection_file, allow_pickle=True)
# det = all_detections[0][ : , : 4]
# pose_model = init_model(config=pose_config, checkpoint=pose_checkpoint, device='cpu')


# pose_result = inference_topdown(pose_model, image, det, 'xyxy')


# kp_list = []
# score_list = []
# for pose in pose_result:
#     kp_list.append(pose.pred_instances.keypoints.squeeze(0))
#     score_list.append(pose.pred_instances.keypoint_scores.squeeze(0))
# # result = model(image, det)
# image = draw_skeleton(image, np.array(kp_list), np.array(score_list))