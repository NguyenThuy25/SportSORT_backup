import cv2
import numpy as np
import torch
from torchvision.transforms import functional as F
import warnings
from ultralytics import YOLO
import time
from ultralytics.nn.autobackend import AutoBackend
# from ultralytics.yolo.utils import DEFAULT_CFG, ROOT, ops
# from ultralytics.yolo.utils.torch_utils import smart_inference_mode 

class YOLOv8(object):
    """YOLOv8 detector model with OCSORT tracker
    """
    def __init__(self, det_checkpoint, device="cpu", use_tracker=False):
        """Init YOLOv8 detector model
        
        Args:
            det_checkpoint (str): path to detector checkpoint
            device (str, optional): device to run model. Defaults to "cpu".
            use_tracker (bool, optional): use tracker or not. Defaults to False.
        """    

        # Init detector model by YOLO from ultralytics
        self.detect_model = YOLO(det_checkpoint)
        # Load device, use_tracker to class
        self.device = device
        self.use_tracker = use_tracker
        
        # If use_tracker is True, init tracker
        # if self.use_tracker:
        #     from boxmot import OCSORT
        #     self.tracker = OCSORT(det_thresh=0.2,
        #                           asso_func='giou',
        #                           delta_t=1, 
        #                           inertia=0.3941737016672115, 
        #                           iou_threshold=0.22136877277096445, 
        #                           max_age=50,
        #                           min_hits=1, 
        #                           use_byte=False)
            
    def infer(self, frame, imgsz=1280, conf=0.25, iou=0.5, half=True):
        """Inference detector model
        
        Args:
            frame (np.array): input image
            imgsz (int, optional): input image size to inference. Defaults to 1280.
            conf (float, optional): confidence threshold. Defaults to 0.25.
            iou (float, optional): iou threshold. Defaults to 0.5.
            half (bool, optional): use half precision or not. Defaults to True.
        
        Returns:
            result (list): list of person's bounding boxes. Each bounding box is a dict with format {'bbox': [x1, y1, x2, y2, conf]}
        """
        
        # Inference detector model
        detector_output = self.detect_model(frame, imgsz=imgsz, conf=conf, device=self.device, half=half, iou=iou)
        
        # Get bounding boxes with class 0 (person)
        dets = detector_output[0].boxes.data # (x, y, x, y, conf, cls)
        dets = dets[dets[:, 5] == 0]

        # If use_tracker is False, convert dets to numpy and remove class column
        if not self.use_tracker:
            dets = dets.cpu().detach().numpy()
            dets = dets[:, :5]
        
        # If use_tracker is True, update tracker, remove class column
        if self.use_tracker:
            dets = self.tracker.update(dets.cpu(), frame) # (x, y, x, y, id, conf, cls)
            dets = dets[:, [0, 1, 2, 3, 5]]

        # Convert dets to list of dict
        result = []
        for i in range(dets.shape[0]):
            result.append({'bbox': dets[i]})
        return result