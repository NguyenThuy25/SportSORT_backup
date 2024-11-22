# mmpose and mmengine
from mmengine.config import Config
# from rtmpose.config import Config
# from mmpose.structures.bbox import bbox_xyxy2cs, get_warp_matrix
# from mmpose.models.utils.tta import flip_vectors
from mmengine.registry import MODELS

# CSPNeXt_RTMCC
from rtmpose.cspnext_rtmcc import CSPNeXtRTMCC
from rtmpose.func import flip_vectors
from rtmpose.transform_utils import bbox_xyxy2cs, get_warp_matrix
# torch, torchvision
import torch.nn as nn
import torch
from torchvision.transforms import functional as F

# cv2, numpy, os
import cv2
import numpy as np
import os
from typing import Tuple

@MODELS.register_module(name='CSPNeXt_RTMCC_Head_SelfImpl')
class CSPNeXt_RTMCC_Head_SelfImpl(object):    
    """CSPNeXt_RTMCC_Head_SelfImpl class is parent class of 
    CSPNeXt_RTMCC_Head_SelfImpl_PT and CSPNeXt_RTMCC_Head_SelfImpl_TensorRT.
    """
    def getBboxCenterScale(self, bbox, input_size, padding=1.25):
        """Get bbox center, scale from bounding boxes and input size.
        
        Args:
            bbox (list): bbox in format [x1, y1, x2, y2]
            input_size (tuple): image input size
            padding (float): padding of bbox
            
        Returns:
            center (np.ndarray): centers of bounding boxes
            scale (np.ndarray): scales of bounding boxes
        """
        
        # Get width, height of image input size
        w, h = input_size
        
        # Get aspect ratio
        aspect_ratio= w / h
        
        # Get center, scale from bounding boxes
        center, scale = bbox_xyxy2cs(bbox, padding)
        
        # Get scale from aspect ratio
        scale = self._fix_aspect_ratio(scale, aspect_ratio=aspect_ratio)
        
        return center, scale
    
    def getAffine(self, img, input_size, centers, scales, rot=0):
        """Affine image
        
        Args:
            img (np.ndarray): image
            input_size (tuple): image input size
            centers (np.ndarray): centers of bounding boxes
            scales (np.ndarray): scales of bounding boxes
            rot (float): rotation of image
        
        Returns:
            img_crop (np.ndarray): affine image
        """
        
        # Get width, height of image input size
        w, h = input_size
        
        # Get affine matrix
        warp_mat = get_warp_matrix(centers, scales, rot, output_size=(w, h))
        
        # Get affine image
        img_crop = cv2.warpAffine(img, warp_mat, (input_size[0], input_size[1]), flags=cv2.INTER_LINEAR)
        
        return img_crop

    def getTransform(self, img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        """Get transform image
        
        Args:
            img (np.ndarray): image
            mean (list): mean of image
            std (list): std of image
        
        Returns:
            img (torch.Tensor): transform image
        """
        
        # Transform image from numpy to device tensor, (H, W, C) -> (C, H, W), divied by 255.0
        img = torch.from_numpy(img.astype('float32')).permute(2, 0, 1).to(self.device).div_(255.0)
        
        # Normalize image by mean, std
        img = F.normalize(img, mean, std, inplace=True)
        
        return img
    
    def get_flip_pairs(self):
        """Get flip index pairs of whole body (133 keypoints).
        
        Returns:
            flip_pairs (list): list of flip index pairs
        """
        
        # define pairs of keypoints for body, foot, face, hand
        body = [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12],
                    [13, 14], [15, 16]]
        foot = [[17, 20], [18, 21], [19, 22]]

        face = [[23, 39], [24, 38], [25, 37], [26, 36], [27, 35], [28, 34],
                [29, 33], [30, 32], [40, 49], [41, 48], [42, 47], [43, 46],
                [44, 45], [54, 58], [55, 57], [59, 68], [60, 67], [61, 66],
                [62, 65], [63, 70], [64, 69], [71, 77], [72, 76], [73, 75],
                [78, 82], [79, 81], [83, 87], [84, 86], [88, 90]]

        hand = [[91, 112], [92, 113], [93, 114], [94, 115], [95, 116],
                [96, 117], [97, 118], [98, 119], [99, 120], [100, 121],
                [101, 122], [102, 123], [103, 124], [104, 125], [105, 126],
                [106, 127], [107, 128], [108, 129], [109, 130], [110, 131],
                [111, 132]]
        
        # Concat body, foot, face, hand
        flip_pairs = body + foot + face + hand
        
        return flip_pairs
    
    def get_flip_indices(self):
        """Get flip indices of whole body (133 keypoints).
        
        Example:
            >>> flip_indices = get_flip_indices()
            >>> flip_indices[0] 
            1
            >>> flip_indices[91]
            112
        
        Returns:
            flip_indices (np.ndarray): flip index of whole body (133 keypoints)
        """
        
        # Init flip indices
        flip_indices = np.zeros((133), dtype=np.int32)
        
        # Get flip pairs
        flip_pairs = self.get_flip_pairs()
        
        # For each pair in flip pairs to get flip indices
        for pair in flip_pairs:
            flip_indices[pair[0]] = pair[1]
            flip_indices[pair[1]] = pair[0]
        
        return flip_indices
        
    def _fix_aspect_ratio(self, bbox_scale, aspect_ratio):
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
    
    def post_process(self, batch_keypoints, batch_scores, centers, scales, input_size):
        """Post process for batch keypoints and batch scores (model output).
        
        Args:
            batch_keypoints (np.ndarray): keypoints of batches
            batch_scores (np.ndarray): scores of batches
            centers (np.ndarray): centers of bounding boxes
            scales (np.ndarray): scales of bounding boxes
            input_size (tuple): image input size
            
        Returns:
            pose_results (list): list of pose results after post process
        """
        
        # Convert input_size to numpy array
        input_size = np.array(input_size)
        
        # Init pose results
        pose_results = []
        
        # Get batch size, keypoints size = 133
        N = len(batch_keypoints)
        K = batch_keypoints[0].shape[1]
        
        # For each batch
        for i in range(N):
            # reshape batch keypoints, batch scores
            keypoints = batch_keypoints[i].reshape((K, 2))
            scores = batch_scores[i].reshape((K, 1))
            
            # Get center, scale of current image
            center = centers[i]
            scale = scales[i]
            
            # Decode keypoints from scale, center to image size keypoints
            keypoints = keypoints / input_size * scale + center - 0.5 * scale
            
            # Concat keypoints, scores
            pose_result = np.concatenate([keypoints, scores], axis=1)
            
            # Append pose result to pose results
            pose_results.append({'bbox': [0,0,0,0], 'keypoints': pose_result})
            
        return pose_results

@MODELS.register_module(name='CSPNeXt_RTMCC_Head_SelfImpl_PT')
class CSPNeXt_RTMCC_Head_SelfImpl_PT(CSPNeXt_RTMCC_Head_SelfImpl):
    """CSPNeXt_RTMCC_Head_SelfImpl_PT class is child class of CSPNeXt_RTMCC_Head_SelfImpl.
    CSPNeXt_RTMCC_Head_SelfImpl_PT class use pytorch to inference. 
    CSPNeXt_RTMCC_Head_SelfImpl_PT is a self implementation of RTMPose model.
    Original RTMPose paper can be found at https://arxiv.org/abs/2303.07399.
    """
    def __init__(self, pose_config, pose_checkpoint, device):
        """Init function: load pose config, pose checkpoint, device, flip indices.
        
        Args:
            pose_config (str): path to pose config
            pose_checkpoint (str): path to pose checkpoint
            device (str): device to inference
        """
        
        # Load pose config
        self.config = Config.fromfile(pose_config)
        # self.config = config
        # Load device
        self.device = device
        
        # Load flip indices
        self.flip_indices = self.get_flip_indices()
        
        # Load pose model
        self.pose_model = CSPNeXtRTMCC(backbone_cfg=self.config.model.backbone, 
                                       head_cfg=self.config.model.head, device=self.device)
        self.pose_model.load_state_dict(torch.load(pose_checkpoint, map_location=torch.device('cpu'))['state_dict'])
        self.pose_model.to(device)
        self.pose_model.eval()

        
    def infer(self, frame, person_results):
        """Infer function: contain preprocess, inference pose model, postprocess.
        
        Args:
            frame (np.ndarray): image
            person_results (list): list of person bounding boxes results
            
        Returns:
            results (list): list of pose results 
        """
        
        # Convert frame from BGR to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
         
        # Get input size from pose config
        input_size = self.config.codec.input_size
        
        # Init batch, get batch size
        batch = []
        batch_size = len(person_results)
        
        # Init centers, scales
        centers = np.zeros((batch_size, 2), dtype=np.float32)
        scales = np.zeros((batch_size, 2), dtype=np.float32)
        
        # For each person in person results
        for i, bbox_result in enumerate(person_results):
            
            # Get center, scale from bounding box
            # center, scale = self.getBboxCenterScale(bbox=bbox_result['bbox'][:4], input_size=input_size)
            center, scale = self.getBboxCenterScale(bbox=bbox_result, input_size=input_size)
            
            # Get affine image
            img = self.getAffine(frame, input_size, center, scale)
            
            # Get transform image
            img = self.getTransform(img)
            
            # Append image to batch
            batch.append(img)
            
            # Append center, scale to centers, scales
            centers[i, :] = center
            scales[i, :] = scale
        
        if len(batch) == 0:
            return []
        # Stack batch to infer model
        batch = torch.stack(batch, dim=0)
        
        # Inference model and pose processing
        results = self.infer_model(batch, centers, scales, input_size)
        
        return results
    
    @torch.no_grad()
    def infer_model(self, img, centers, scales, input_size):
        """ Inference pose model function
        
        Args:
            img (torch.Tensor): image
            centers (np.ndarray): centers of bounding boxes
            scales (np.ndarray): scales of bounding boxes
            input_size (tuple): image input size
        
        Returns:
            pose_results (list): list of pose results
        """
        
        # infer model with batch of image
        _batch_pred_x, _batch_pred_y = self.pose_model(img)
        # infer model with batch of image flipped
        _batch_pred_x_flip, _batch_pred_y_flip = self.pose_model(img.flip(3))
        # flip coordinates of batch of image flipped
        _batch_pred_x_flip, _batch_pred_y_flip = flip_vectors(_batch_pred_x_flip, _batch_pred_y_flip, self.flip_indices)
        
        # Get results from inference result and flip inference result by average
        batch_pred_x = (_batch_pred_x + _batch_pred_x_flip) * 0.5
        batch_pred_y = (_batch_pred_y + _batch_pred_y_flip) * 0.5

        # Decode keypoints from batch_pred_x, batch_pred_y
        batch_keypoints, batch_scores = self.pose_model.head.decode((batch_pred_x, batch_pred_y))
        
        # Post process for batch_keypoints, batch_scores
        pose_results = self.post_process(batch_keypoints, batch_scores, centers, scales, input_size)
        
        return pose_results
    
    def infer_tracker_results(self, frame, tracker_results):
        """Infer function: contain preprocess, inference pose model, postprocess.
        
        Args:
            frame (np.ndarray): image
            tracker_results (np.ndarray): list of person bounding boxes results
            
        Returns:
            results (list): list of pose results 
        """
        
        # Convert frame from BGR to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
         
        # Get input size from pose config
        input_size = self.config.codec.input_size
        
        # Init batch, get batch size
        batch = []
        batch_size = tracker_results.shape[0]
        
        # Init centers, scales
        centers = np.zeros((batch_size, 2), dtype=np.float32)
        scales = np.zeros((batch_size, 2), dtype=np.float32)
        
        # For each person in person results
        for i, bbox_result in enumerate(tracker_results):
            
            # Get center, scale from bounding box
            center, scale = self.getBboxCenterScale(bbox=bbox_result[:4], input_size=input_size)
            
            # Get affine image
            img = self.getAffine(frame, input_size, center, scale)
            
            # Get transform image
            img = self.getTransform(img)
            
            # Append image to batch
            batch.append(img)
            
            # Append center, scale to centers, scales
            centers[i, :] = center
            scales[i, :] = scale
        
        # Stack batch to infer model
        if len(batch) == 0:
            return []
        batch = torch.stack(batch, dim=0)
        
        # Inference model and pose processing
        results = self.infer_model(batch, centers, scales, input_size)
        
        return results
    
