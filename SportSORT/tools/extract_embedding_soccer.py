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

    # detection_folder = os.path.join(cache_folder, "fix_detection")
    embedding_dir = os.path.join(cache_folder, "embedding_sports")
    os.makedirs(embedding_dir, exist_ok=True)
    # npy_list = os.listdir(detection_folder)
    # video_list = [x.split(".")[0] for x in npy_list]
    video_list = os.listdir(img_folder)

    for video_name in video_list:
        # detection_path = os.path.join(detection_folder, f"{video_name}.npy")
        detection_path = os.path.join(img_folder, video_name, "det", "det.txt")
        video_detection = []
        with open(detection_path, "r") as file:
            for line in file:
                data = line.strip().split(',')
                frame_id = int(data[0])
                x_min, y_min, width, height = map(int, data[2:6])
                x_max = x_min + width
                y_max = y_min + height
                bbox = [x_min, y_min, x_max, y_max]
                
                # Ensure the list is long enough to accommodate this frame
                while len(video_detection) < frame_id:
                    video_detection.append([])  # Append empty lists for each frame
                video_detection[frame_id - 1].append(bbox)

        video_length = len(video_detection)
        all_embeddings = []
        
        for i in range(video_length):
            frame_detection = video_detection[i]
            frame_path = os.path.join(img_folder, video_name, "img1", f"{str(i + 1).zfill(6)}.jpg")
            frame = cv2.imread(frame_path)

            # Get crops of detections and compute embeddings
            crops = get_crop(frame, frame_detection)
            embs = feature_extractor(crops)
            embs = embs.cpu().detach().numpy()
            all_embeddings.append(embs)

        # Save embeddings as .npy file
        embedding_file = os.path.join(embedding_dir, f"{video_name}.npy")
        np.save(embedding_file, np.array(all_embeddings, dtype=object))

        print(f"Saved embeddings for video {video_name}")

def main():
    img_folders = [
        # "/mnt/banana/student/thuyntt/data/sportsmot_publish/dataset/train/",
        # "/mnt/banana/student/thuyntt/data/sportsmot_publish/dataset/val/",
        # "/mnt/banana/student/thuyntt/data/sportsmot_publish/dataset/test/"
        "/mnt/banana/student/thuyntt/data/SoccerNet/tracking/test"
    ]

    cache_folders = [   
        # "/mnt/banana/student/thuyntt/Deep-EIoU/cache/train/",
        # "/mnt/banana/student/thuyntt/Deep-EIoU/cache/val/",
        # "/mnt/banana/student/thuyntt/Deep-EIoU/cache/test/"
        "/mnt/banana/student/thuyntt/Deep-EIoU/cache_soccer/test"
    ]

    ckpt_path = "/mnt/banana/student/thuyntt/Deep-EIoU/Deep-EIoU/checkpoints/model.osnet.pth.tar-10"

    for cache_folder, img_folder in zip(cache_folders, img_folders):
        save_embs(ckpt_path, cache_folder, img_folder)

if __name__ == "__main__":
    main()