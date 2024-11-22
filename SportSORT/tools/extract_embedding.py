import argparse
import json
import os
import os.path as osp
import numpy as np
import time
import cv2
import torch
import sys
sys.path.append('.')

from loguru import logger

from yolox.data.data_augment import preproc
from yolox.exp import get_exp
from yolox.utils import fuse_model, get_model_info, postprocess
from yolox.utils.visualize import plot_tracking
from yolox.tracking_utils.timer import Timer

from tracker.Deep_EIoU_2 import Deep_EIoU
from reid.torchreid.utils import FeatureExtractor
import torchvision.transforms as T

def get_crop(image, bboxes):
    crops = []
    for bbox in bboxes:
        x1, y1, x2, y2 = bbox[:4]
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        crop = image[y1:y2, x1:x2]
        crops.append(crop)
    return crops

def save_embs(ckpt_path, cache_folder, img_folder):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    feature_extractor = FeatureExtractor(
        model_name='osnet_x1_0',
        model_path=ckpt_path,
        device=device
    )

    detection_folder = os.path.join(cache_folder, "fix_detection")
    embedding_dir = os.path.join(cache_folder, "embedding_sports")
    os.makedirs(embedding_dir, exist_ok=True)
    npy_list = os.listdir(detection_folder)
    video_list = [x.split(".")[0] for x in npy_list]

    for video_name in video_list:
        detection_path = os.path.join(detection_folder, f"{video_name}.npy")
        video_detection = np.load(detection_path, allow_pickle=True)
        video_length = video_detection.shape[0]
        all_embeddings = []
        
        for i in range(video_length):
            frame_detection = video_detection[i]
            frame = cv2.imread(os.path.join(img_folder, video_name, "img1", f"{str(i+1).zfill(6)}.jpg"))
            crops = get_crop(frame, frame_detection)
            embs = feature_extractor(crops)
            embs = embs.cpu().detach().numpy()
            all_embeddings.append(embs)
        
        embedding_file = os.path.join(embedding_dir, f"{video_name}.npy")
        np.save(embedding_file, np.array(all_embeddings, dtype=object))

        print(f"Saved embeddings for video {video_name}")

def main():
    img_folders = [
        # "/mnt/banana/student/thuyntt/data/sportsmot_publish/dataset/train/",
        # "/mnt/banana/student/thuyntt/data/sportsmot_publish/dataset/val/",
        "/mnt/banana/student/thuyntt/data/sportsmot_publish/dataset/test/"
    ]

    cache_folders = [   
        # "/mnt/banana/student/thuyntt/Deep-EIoU/cache/train/",
        # "/mnt/banana/student/thuyntt/Deep-EIoU/cache/val/",
        "/mnt/banana/student/thuyntt/Deep-EIoU/cache/test/"
    ]

    ckpt_path = "/mnt/banana/student/thuyntt/Deep-EIoU/Deep-EIoU/checkpoints/model.osnet.pth.tar-10"

    for cache_folder, img_folder in zip(cache_folders, img_folders):
        save_embs(ckpt_path, cache_folder, img_folder)

if __name__ == "__main__":
    main()