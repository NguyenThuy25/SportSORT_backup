import json
import os
import random
import time
import numpy as np
import cv2
import sys
# ROOT = './str/parseq/'
# sys.path.append(str(ROOT))
import torch
import legibility_classifier as lc
import configuration as config
from PIL import Image, ImageDraw
from networks import LegibilityClassifier, LegibilityClassifier34, LegibilityClassifierTransformer
import torch.backends.cudnn as cudnn
from strhub.models.utils import load_from_checkpoint, parse_model_args
from str import run_inference
import string

from rtmpose.top_down import init_model
from rtmpose.utils.infer_utils import inference_topdown
import rtmpose.utils.register

# from ..pose.rtmpose.top_down import init_model
PADDING = 5
CONFIDENCE_THRESHOLD = 0.4
STR_THRESHOLD = 0.7
# TS = 2.367
HEIGHT_MIN = 35
WIDTH_MIN = 30
RUN_TYPE = 'HOCKEY'

def get_points(kps):
    if len(kps) < 12:
        #print("not enough points")
        return []
    relevant = [kps[6], kps[5], kps[11], kps[12]]
    result = []
    for r in relevant:
        if r[2] < CONFIDENCE_THRESHOLD:
            #print(f"confidence {r[2]}")
            return []
        result.append(r[:2])
    return result

class JerseyNumberPipeline:
    def __init__(self, video_path, visualize_out_path, cache_path=None, gt_jersey_path=None):
        self.video_path = video_path
        self.cache_path = cache_path
        # self.legibile_out_path = legibile_out_path
        self.visualize_out_path = visualize_out_path
        self.gt_jersey_path = gt_jersey_path
        
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        cudnn.benchmark = True
        # device = torch.device("cpu")

        #load model
        if RUN_TYPE == 'HOCKEY':
            legible_model_path = config.dataset['Hockey']['legibility_model']
            # arch = config.dataset['Hockey']['legibility_model_arch']
            arch = 'resnet34'
        else:
            legible_model_path = config.dataset['SoccerNet']['legibility_model']
            arch = config.dataset['SoccerNet']['legibility_model_arch']
        state_dict = torch.load(legible_model_path, map_location=device)
        

        if arch == 'resnet18':
            self.legible_model = LegibilityClassifier()
        elif arch == 'vit':
            self.legible_model = LegibilityClassifierTransformer()
        else:
            self.legible_model = LegibilityClassifier34()
        if hasattr(state_dict, '_metadata'):
            del state_dict._metadata
        self.legible_model.load_state_dict(state_dict)
        self.legible_model = self.legible_model.to(device)
        self.legible_model.eval()

        self.pose_model = init_model(config=config.dataset['SoccerNet']['pose_config'], checkpoint=config.dataset['SoccerNet']['pose_ckpt'], device=device)
    
        batch_size=1
        charset_test = string.digits # + string.ascii_lowercase
        inference = True
        if RUN_TYPE == 'HOCKEY':
            self.str_model = load_from_checkpoint(config.dataset['Hockey']['str_model'], charset_test=charset_test, batch_size=batch_size, inference = inference).eval().to(device)
        else:
            self.str_model = load_from_checkpoint(config.dataset['SoccerNet']['str_model'], charset_test=charset_test, batch_size=batch_size, inference = inference).eval().to(device)


    def pose_estimation(self, img):
        
        pose_result = inference_topdown(self.pose_model, img)
        kps = pose_result[0].pred_instances.keypoints.squeeze(0)
        kp_scores = pose_result[0].pred_instances.keypoint_scores.squeeze(0)
        kps_with_conf = np.hstack([kps, kp_scores[:, np.newaxis]])
        filtered_points = get_points(kps_with_conf)
        if len(filtered_points) == 0:
            #TODO: better approach then skipping
            return None            
            
        height, width, _ = img.shape
        x_min = min([p[0] for p  in filtered_points]) - PADDING
        x_max = max([p[0] for p  in filtered_points]) + PADDING
        y_min = min([p[1] for p  in filtered_points]) - PADDING
        y_max = max([p[1] for p  in filtered_points])
        x1 = int(0 if x_min < 0 else x_min)
        y1 = int(0 if y_min < 0 else y_min)
        x2 = int(width - 1 if x_max > width else x_max)
        y2 = int(height -1 if y_max > height else y_max)

        crop = img[y1:y2, x1:x2, :]
        return crop
    
    def infer_one_image(self, image, detection, frame_name, video_name, save_img=False, save_cache=True):
        
        for person_idx, bbox in enumerate(detection):
            x1, y1, x2, y2 = map(int, bbox[:4])  # Extract bounding box
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cropped = image[y1:y2, x1:x2]  # Crop the image
            cropped_list = [cropped]
            if RUN_TYPE == 'HOCKEY':
                ## ADD ##
                # if cropped.shape[0] != 0:
                ## ADD ##
                try:
                    track_results = lc.run(cropped_list, self.legible_model, arch=config.dataset['Hockey']['legibility_model'], threshold=0.5)
                except:
                    import IPython; IPython.embed()
                    time.sleep(0.6)
            else:
                track_results = lc.run(cropped_list, self.legible_model, arch=config.dataset['SoccerNet']['legibility_model_arch'], threshold=0.5)
            results = None
            if track_results[0] == 1:
                pose_cropped = self.pose_estimation(cropped)
                if pose_cropped is not None:
                    
                    pose_cropped = Image.fromarray(pose_cropped).convert('RGB')
                    # reslts = {'label':, 'confidence':}
                    results = run_inference(self.str_model, pose_cropped, self.str_model.hparams.img_size)
                    # if all(x > STR_THRESHOLD for x in results['confidence']):
                    if type(results['confidence']) == list:
                        min_conf = min(results['confidence'])
                    else:
                        min_conf = results['confidence']
                        if results['label'] == '':
                            results['label'] = -1
                    text = f'{results["label"]}, {min_conf:.2f}'
                    cv2.putText(image, text, (x1 , y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                else:
                    print(f"skipping {frame_name}, unreliable points")
            if save_cache:
                cache_file = os.path.join(self.cache_path, f'{video_name}.txt')
                pred = results['label'] if results is not None else -1
                pred_conf = round(min_conf, 2) if results is not None else -1
                cache = frame_name, x1, y1, x2, y2, pred, pred_conf
                # cache = frame_id, person_id, x, y, w, h, pred, pred_conf, jersey_num
                with open(cache_file, 'a') as file:
                    file.write(','.join(map(str, cache)) + '\n')

                      
        if save_img:
            cv2.imwrite(f'{self.visualize_out_path}/{video_name}/{frame_name}', image)

        
    def run_all_vid_infer(self, save_cache=True):
        # scale = min(1440/1280, 800/720)
        all_videos = os.listdir(self.video_path)
        # import IPython; IPython.embed()
        # time.sleep(0.6)
        for i, video in enumerate(all_videos):
            visualize_out_path = os.path.join(self.visualize_out_path, video)
            os.makedirs(visualize_out_path, exist_ok=True)
            img_folder = os.path.join(self.video_path, video, 'img1')
            detection_file = os.path.join(self.video_path, video, 'det', 'det.txt')

            # Load detections
            detections = open(detection_file, 'r')
            detections_lines = detections.readlines()
            parsed_data = [list(map(float, line.split(','))) for line in detections_lines]
            # format [frame_id, person_id, x, y, w, h, 1,-1,-1,-1]
            detections = {}
            for row in parsed_data:
                frame_id = int(row[0])
                if frame_id not in detections:
                    detections[frame_id] = []
                # convert to x1, y1, x2, y2
                row[2] = int(row[2])
                row[3] = int(row[3])
                row[4] = int(row[2] + row[4])
                row[5] = int(row[3] + row[5])
                detections[frame_id].append(row[2:])
            
            frames = sorted([f for f in os.listdir(img_folder)])
            
            for frame in frames:
                frame_index = int(frame.split('.')[0])
                detection = detections.get(frame_index)
                img_path = os.path.join(img_folder, frame)
                image = cv2.imread(img_path)
                # self.infer_one_image(image, detection, frame, video, save_img=False, save_cache=save_cache)
                self.infer_one_image(image, detection, int(frame.split('.')[0]), video, save_cache=save_cache)

            print(f"Done video {video}")


    
    def eval_one_image(self, image, gt, frame_name, video_name, save_img=False, save_cache=True):
        
        for frame_id, person_id, x, y, w, h, jersey_num in gt:
            # print("frame", frame_id)
            # x1, y1, x2, y2 = map(int, bbox[:4])  # Extract bounding box
            
            cropped = image[y:y + h, x:x + w]  # Crop the image
            cropped_list = [cropped]
            # cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)


            track_results = lc.run(cropped_list, self.legible_model, arch=config.dataset['SoccerNet']['legibility_model_arch'], threshold=0.5)
            results = None
            if track_results[0] == 1:
                pose_cropped = self.pose_estimation(cropped)

                if pose_cropped is not None:
                    # cv2.imwrite(f'{self.visualize_out_path}/{video_name}/{frame_id}_{person_id}.jpg', pose_cropped)
                    pose_cropped = Image.fromarray(pose_cropped).convert('RGB')
                    # reslts = {'label':, 'confidence':}
                    results = run_inference(self.str_model, pose_cropped, self.str_model.hparams.img_size)
                    # if all(x > STR_THRESHOLD for x in results['confidence']):
                    if type(results['confidence']) == list:
                        min_conf = min(results['confidence'])
                    else:
                        min_conf = results['confidence']
                        if results['label'] == '':
                            results['label'] = -1
                    text = f'{results["label"]}, {min_conf:.2f}'
                    cv2.putText(image, text, (x , y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                else:
                    print(f"skipping {frame_name}, unreliable points")

            if save_cache:
                cache_file = os.path.join(self.cache_path, f'{video_name}.txt')
                # os.makedirs(cache_file, exist_ok=True)
                pred = results['label'] if results is not None else -1
                # pred_conf = min(results['confidence']) if results is not None else -1
                pred_conf = round(min_conf, 2) if results is not None else -1
                cache = frame_id, person_id, x, y, w, h, pred, pred_conf, jersey_num
                with open(cache_file, 'a') as file:
                    file.write(','.join(map(str, cache)) + '\n')
                
        if save_img:
            cv2.imwrite(f'{self.visualize_out_path}/{video_name}/{frame_name}', image)

    def run_all_vid_eval(self, save_cache=True):
        all_videos = os.listdir(self.video_path)
        index = 0
        for video in all_videos:
            if index <= 33:
                print(f"Skip video {video}")
            else:
                visualize_out_path = os.path.join(self.visualize_out_path, video)
                os.makedirs(visualize_out_path, exist_ok=True)
                img_folder = os.path.join(self.video_path, video, 'img1')
                gt_file = os.path.join(self.video_path, video, 'gt', 'gt.txt')
                jersey_gt_file = os.path.join(self.gt_jersey_path, f'{video}.txt')
                # Load frames and detections
                # frames = sorted([f for f in os.listdir(img_folder) if f.endswith('.jpg')])
                frames = sorted([f for f in os.listdir(img_folder)])
                gt = open(gt_file, 'r')
                gt_lines = gt.readlines()
                with open(jersey_gt_file, 'r') as file:
                    jersey_gt = {}
                    for line in file:
                        key, value = line.strip().split(':')
                        jersey_gt[int(key)] = tuple(map(int, value.strip().split()))

                # lines = gt_lines.strip().split('\n')
                parsed_data = [list(map(int, line.split(','))) for line in gt_lines]

                # Organize data by frame
                gt_frames = {}
                for row in parsed_data:
                    frame_id = row[0]
                    if frame_id not in gt_frames:
                        gt_frames[frame_id] = []
                    gt_frames[frame_id].append(row[1:])

                for frame in frames:
                    frame_index = int(frame.split('.')[0])
                    gts = gt_frames.get(frame_index)
                    gt = []
                    for g in gts:
                        id, x, y, w, h, _, _, _= g
                        id, x, y, w, h = map(int, [id, x, y, w, h])
                        if id in jersey_gt.keys():
                            jersey = jersey_gt[id][0] # only get jersey, ignore team
                            gt.append([frame_index, id, x, y, w, h, jersey])
                        else:
                            gt.append([frame_index, id, x, y, w, h, -1])
                    img_path = os.path.join(img_folder, frame)
                    image = cv2.imread(img_path)
                    # image = Image.open(img_path).convert('RGB')
                    self.eval_one_image(image, gt, frame, video, save_img=False, save_cache=save_cache)
            index += 1

    def run_one_vid_eval(self, video_path, save_cache=True):
        # all_videos = os.listdir(self.video_path)
        # for video in all_videos:
        video_name = video_path.split('/')[-1]
        visualize_out_path = os.path.join(self.visualize_out_path, video_name)
        os.makedirs(visualize_out_path, exist_ok=True)
        img_folder = os.path.join(self.video_path, video_name, 'img1')
        gt_file = os.path.join(self.video_path, video_name, 'gt', 'gt.txt')
        jersey_gt_file = os.path.join(self.gt_jersey_path, f'{video_name}.txt')
        # Load frames and detections
        # frames = sorted([f for f in os.listdir(img_folder) if f.endswith('.jpg')])
        frames = sorted([f for f in os.listdir(img_folder)])
        gt = open(gt_file, 'r')
        gt_lines = gt.readlines()
        with open(jersey_gt_file, 'r') as file:
            jersey_gt = {}
            for line in file:
                key, value = line.strip().split(':')
                jersey_gt[int(key)] = tuple(map(int, value.strip().split()))

        # lines = gt_lines.strip().split('\n')
        parsed_data = [list(map(int, line.split(','))) for line in gt_lines]

        # Organize data by frame
        gt_frames = {}
        for row in parsed_data:
            frame_id = row[0]
            if frame_id not in gt_frames:
                gt_frames[frame_id] = []
            gt_frames[frame_id].append(row[1:])

        for frame in frames:
            frame_index = int(frame.split('.')[0])
            gts = gt_frames.get(frame_index)
            gt = []
            for g in gts:
                id, x, y, w, h, _, _, _= g
                id, x, y, w, h = map(int, [id, x, y, w, h])
                if id in jersey_gt.keys():
                    jersey = jersey_gt[id][0] # only get jersey, ignore team
                    gt.append([frame_index, id, x, y, w, h, jersey])
                else:
                    gt.append([frame_index, id, x, y, w, h, -1])
            img_path = os.path.join(img_folder, frame)
            image = cv2.imread(img_path)
            # image = Image.open(img_path).convert('RGB')
            self.eval_one_image(image, gt, frame, video_name, save_img=False, save_cache=save_cache)

    # run pipeline for batch for faster processing
    def run_batch(self):
        all_videos = os.listdir(self.video_path)
        for video in all_videos:
            visualize_out_path = os.path.join(self.visualize_out_path, video)
            os.makedirs(visualize_out_path, exist_ok=True)
            img_folder = os.path.join(self.video_path, video, 'img1')
            gt_file = os.path.join(self.video_path, video, 'gt', 'gt.txt')
            jersey_gt_file = os.path.join(self.gt_jersey_path, f'{video}.txt')
            # Load frames and detections
            # frames = sorted([f for f in os.listdir(img_folder) if f.endswith('.jpg')])
            frames = sorted([f for f in os.listdir(img_folder)])
            gt = open(gt_file, 'r')
            gt_lines = gt.readlines()
            with open(jersey_gt_file, 'r') as file:
                jersey_gt = {}
                for line in file:
                    key, value = line.strip().split(':')
                    jersey_gt[int(key)] = tuple(map(int, value.strip().split()))

            # lines = gt_lines.strip().split('\n')
            parsed_data = [list(map(int, line.split(','))) for line in gt_lines]

            # Organize data by frame
            gt_frames = {}
            for row in parsed_data:
                frame_id = row[0]
                if frame_id not in gt_frames:
                    gt_frames[frame_id] = []
                gt_frames[frame_id].append(row[1:])
            
            for frame in frames:
                frame_index = int(frame.split('.')[0])
                gts = gt_frames.get(frame_index)
                gt = []
                for g in gts:
                    id, x, y, w, h, _, _, _= g
                    id, x, y, w, h = map(int, [id, x, y, w, h])
                    if id in jersey_gt.keys():
                        jersey = jersey_gt[id][0]
                        gt.append([frame_index, id, x, y, w, h, jersey])
                    else:
                        gt.append([frame_index, id, x, y, w, h, -1])
                img_path = os.path.join(img_folder, frame)
                image = cv2.imread(img_path)
            self.eval_all_vid()        
        

if __name__ == '__main__':

    video_path = '/mnt/banana/student/thuyntt/data/SoccerNet/tracking/test'
    # det_path = '/mnt/banana/student/thuyntt/Deep-EIoU/cache/test/detection'
    visualize_out_path = '/mnt/banana/student/thuyntt/jersey-number-pipeline/test/output/test/'
    cache_path = '/mnt/banana/student/thuyntt/Deep-EIoU/cache_soccer/test/jersey_soccer'
    os.makedirs(cache_path, exist_ok=True)
    # gt_jersey_path = '/mnt/banana/student/thuyntt/data/sportsmot_publish/jersey_label/train'
    pipeline = JerseyNumberPipeline(video_path, visualize_out_path, cache_path)
    pipeline.run_all_vid_infer()

        