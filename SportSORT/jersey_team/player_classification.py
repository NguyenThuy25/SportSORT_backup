import os
import time
from sklearn.cluster import KMeans
import numpy as np
import cv2

ROOT = './jersey_team/'
import sys
sys.path.append(ROOT)
from rtmpose.infer import inference_topdown
# from rtmpose.top_down import init_model

# try:
#     from rtmpose.top_down import init_model
#     from rtmpose.infer import inference_topdown
# except:
#     pass
# import color_extraction
# from color_extraction import CODE_BOOK, COLOR_NAMES, vq

class PlayerClassification:
    def __init__(self):
        self.kmeans = KMeans(n_clusters=2)
        self.train_features = None
    
    def get_crop(self, image, bboxes, type='xyxy'):
        crops = []
        for bbox in bboxes:
            if type == 'xywh':
                x1, y1, w, h = bbox[:4]
                x1, y1, w, h = int(x1), int(y1), int(w), int(h)
                crop = image[y1:y1+h, x1:x1+w]
            elif type == 'xyxy':
                x1, y1, x2, y2 = bbox[:4]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                crop = image[y1:y2, x1:x2]
            if crop is not None:
                crops.append(crop)
        return crops

    def get_hist_features(self, images, bins=8):
        X = []
        for image in images:
            img_hist = []
            non_black_pixels_mask = np.any(image != [0, 0, 0], axis=-1)

            for k in range(image.shape[2]):
                temp = image[:,:,k]
                new = temp[non_black_pixels_mask].flatten()
                hist, _ = np.histogram(new, bins)
                
                if np.sum(hist) == 0:
                    hist = hist
                else:
                    hist = hist/(1.0*np.sum(hist))
                
                img_hist = img_hist + hist.tolist()
            
            img_hist = np.array(img_hist)
            X.append(img_hist)
            
        X = np.array(X)
        return X

    def update(self, embeddings):
        if len(embeddings.shape) == 1:
            embeddings = embeddings.reshape(1, -1)
        if self.train_features is None:
            self.train_features = embeddings
        else:
            self.train_features = np.vstack((self.train_features, embeddings))
    
    def fit(self):
        self.kmeans.fit(self.train_features)

    def classify(self, features):
        centroids = self.kmeans.cluster_centers_

        labels = []
        confs = []
        for j, feature in enumerate(features):
            dist_0 = np.linalg.norm(feature - centroids[0])
            dist_1 = np.linalg.norm(feature - centroids[1])
            if dist_0 < dist_1:
                labels.append(0)
                confs.append(1 - dist_0 / (dist_0 + dist_1))
            else:
                labels.append(1)
                confs.append(1 - dist_1 / (dist_0 + dist_1))
        
        return labels, confs
    
    def get_masked_image(self, crops, pose_results, use_shoulder_hip=True, use_knee_hip=True, frame_detection=None, xyxy=True):
        if not use_shoulder_hip and not use_knee_hip:
            return crops
        
        crop_poses = []
        back_crop_poses = []
        for j, crop in enumerate(crops):                                                                                                                                                                                                            
            pose_result = pose_results[j]
            
            if frame_detection is None:
                x_min, y_min = 0, 0
            else:
                x_min, y_min = frame_detection[j][:2]
                
            kp = pose_result.pred_instances.keypoints.squeeze(0)
            
            mask = np.zeros(crop.shape[:2], dtype=np.uint8)
        
            if use_shoulder_hip:
                shoulder_hip = kp[[5, 6, 12, 11]] # shoulder, hip
                shoulder_hip = (shoulder_hip - np.array([x_min, y_min])).astype(np.int32)
                cv2.fillPoly(mask, [shoulder_hip], 1)
            
            if use_knee_hip:
                knee_hip = kp[[13, 14, 12, 11]] # knee ,hip
                knee_hip = (knee_hip - np.array([x_min, y_min])).astype(np.int32)
                cv2.fillPoly(mask, [knee_hip], 1)

        
            # get min x, y of shoulder hip
            # x_min, y_min = np.min(shoulder_hip, axis=0)
            # x_max, y_max = np.max(shoulder_hip, axis=0)
            # b_crop = crop[y_min:y_max, x_min:x_max]
            # back_crop_poses.append(b_crop)
            masked_image = cv2.bitwise_and(crop, crop, mask=mask)
            
            # turn the background of masked_image to white
            masked_image[mask == 0] = 255
            crop_poses.append(masked_image)
        
        # return crop_poses, back_crop_poses
        return crop_poses
    
    def infer_one_image(self, pose_model, frame, detection_data, team_names):
        # Read the frame
        # team_folders = {team_name: os.path.join(cache_folder, team_name) for team_name in team_names}
        # Run pose estimation on the detections
        pose_results = inference_topdown(pose_model, frame, detection_data[:, :4], 'xyxy')

        # Extract crops for detected players
        crops = self.get_crop(frame, detection_data)

        use_shoulder_hip = True
        use_knee_hip = True

        # Masked images based on pose keypoints
        masked_images = self.get_masked_image(
            crops, pose_results, use_shoulder_hip=use_shoulder_hip, use_knee_hip=use_knee_hip, frame_detection=detection_data
        )

        # Extract histogram features from the masked images
        features = self.get_hist_features(masked_images)

        # Store the features for the team
        # features_dict[team_names] = features

        return features

def draw_keypoints(image, keypoints):
    for kp in keypoints:
        x, y = kp[:2]
        image = cv2.circle(image, (int(x), int(y)), 5, (0, 255, 0), -1)
    return image    

def draw_bbox(image, bboxes, infos, confs):
    for bbox, info, conf in zip(bboxes, infos, confs):
        # x1, y1, x2, y2 = bbox[:4]
        x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
        color = (0, 255, 0) if info == 0 else (255, 0, 0)
        image = cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        image = cv2.putText(image, f"{conf.round(2)}", (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    return image

# def infer_one_frame(color_classifier, pose_model, frame, frame_detection, team_names):
#     crops = color_classifier.get_crop(frame, frame_detection)
#     pose_results = inference_topdown(pose_model, frame, frame_detection[:, :4], 'xyxy')
#     use_shoulder_hip = team_names["use_shoulder_hip"]
#     use_knee_hip = team_names["use_knee_hip"]
#     crop_poses = color_classifier.get_masked_image(crops, pose_results, use_shoulder_hip, use_knee_hip, frame_detection)
#     features = color_classifier.get_hist_features(crop_poses)
#     return features

def save_data_pred(pose_model, cache_folder, img_folder, team_names):
    detection_folder = os.path.join(cache_folder, "detection")
    # detection_folder = os.path.join(cache_folder, "fix_detection")
    team_folders = {team_name: os.path.join(cache_folder, team_name) for team_name in team_names}
    for team_name, team_folder in team_folders.items():
        if not os.path.exists(team_folder):
            os.makedirs(team_folder)

    npy_list = os.listdir(detection_folder)
    video_list = [x.split(".")[0] for x in npy_list]

    for video_name in video_list:
        if video_name == 'v_0kUtTtmLaJA_c006':
            detection_path = os.path.join(detection_folder, f"{video_name}.npy")
            video_detection = np.load(detection_path, allow_pickle=True)

            video_length = video_detection.shape[0]
            
            video_team_data = {}
            for team_name, team_folder in team_folders.items():
                video_team_data[team_name] = np.empty_like(video_detection)

            # scale = min(1440/1280, 800/720)
            pc = PlayerClassification()
            for i in range(video_length):
                frame_detection = video_detection[i]

                # ## ADD ##
                frame_detection = np.array(frame_detection)
                # ## ADD ##
                # frame_detection /= scale
                frame = cv2.imread(os.path.join(img_folder, video_name, "img1", f"{str(i+1).zfill(6)}.jpg"))
                pose_results = inference_topdown(pose_model, frame, frame_detection[:, :4], 'xyxy')
                crops = pc.get_crop(frame, frame_detection)
                # import IPython; IPython.embed()
                # time.sleep(0.6)
                
                for team_name, team_folder in team_folders.items():
                    use_shoulder_hip = team_names[team_name]["use_shoulder_hip"]
                    use_knee_hip = team_names[team_name]["use_knee_hip"]
                    crop_poses = pc.get_masked_image(crops, pose_results, use_shoulder_hip, use_knee_hip, frame_detection)

                    features = pc.get_hist_features(crop_poses)
                    video_team_data[team_name][i] = features

            # save with allow pickle
            for team_name, team_folder in team_folders.items():
                np.save(os.path.join(team_folder, f"{video_name}.npy"), video_team_data[team_name], allow_pickle=True)
        
        # print(f"Saved {video_name} in {team_folders}")
        
            # pc.update(features)
                
        # pc.fit()
        
        # for ...
            # pc.classify(features)

def save_data_gt(pose_model, cache_folder, img_folder, team_names):
    team_folders = {team_name: os.path.join(cache_folder, team_name) for team_name in team_names}
    for team_name, team_folder in team_folders.items():
        if not os.path.exists(team_folder):
            os.makedirs(team_folder)

    video_list = os.listdir(img_folder)

    for video_name in video_list:
        detection_path = os.path.join(os.path.join(img_folder, video_name, "gt", "gt.txt"))
        video_detection = load_gt_txt(detection_path)
        video_length = len(video_detection)
        
        video_team_data = {}
        for team_name, team_folder in team_folders.items():
            video_team_data[team_name] = open(os.path.join(team_folder, f"{video_name}.txt"), "w")
        
        pc = PlayerClassification()
        for i in range(video_length):
            raw_frame_detection = video_detection[i]
            frame_detection = np.array([[x[1], x[2], x[1] + x[3], x[2] + x[4]] for x in raw_frame_detection])
            frame = cv2.imread(os.path.join(img_folder, video_name, "img1", f"{str(i+1).zfill(6)}.jpg"))
            pose_results = inference_topdown(pose_model, frame, frame_detection[:, :4], 'xyxy')
            crops = pc.get_crop(frame, frame_detection)
            
            # if i == 0:
            #     for j, crop in enumerate(crops):
            #         cv2.imwrite(f"crop_{j}.jpg", crop)
            
            
            for team_name, team_folder in team_folders.items():
                use_shoulder_hip = team_names[team_name]["use_shoulder_hip"]
                use_knee_hip = team_names[team_name]["use_knee_hip"]
                crop_poses = pc.get_masked_image(crops, pose_results, use_shoulder_hip, use_knee_hip, frame_detection)
                
                
                features = pc.get_hist_features(crop_poses)

                for j, feature in enumerate(features):
                    video_team_data[team_name].write(str(i+1) + ", ")
                    video_team_data[team_name].write(", ".join([str(x) for x in raw_frame_detection[j]]))
                    video_team_data[team_name].write(", ")
                    video_team_data[team_name].write(", ".join([str(x) for x in feature]) + "\n")
            
        for team_name, team_folder in team_folders.items():
            video_team_data[team_name].close()
        
        print(f"Saved {video_name}")

def load_gt_txt(gt_txt_path):
    with open(gt_txt_path, "r") as f:
        lines = f.readlines()
    last_frame = int(lines[-1].split(",")[0])
    video_detection = []
    current_frame = 1
    current_data = []
    for line in lines:
        line = line.strip().split(",")
        frame_idx = int(line[0])
        
        if current_frame != frame_idx:
            video_detection.append(current_data)
            current_data = []
            current_frame = frame_idx
        
        current_data.append([int(x) for x in line[1:]])
        
    if frame_idx == last_frame:
        video_detection.append(current_data)
        
    return video_detection

def load_team_pred_txt(gt_txt_path):
    with open(gt_txt_path, "r") as f:
        lines = f.readlines()
    last_frame = int(lines[-1].split(",")[0])
    video_detection = []
    video_team = []
    current_frame = 1
    current_data = []
    current_team = []
    for line in lines:
        line = line.strip().split(",")
        frame_idx = int(line[0])
        
        if current_frame != frame_idx:
            video_detection.append(current_data)
            video_team.append(np.array(current_team))
            current_data, current_team = [], []
            current_frame = frame_idx
        
        current_data.append([int(x) for x in line[1:9]])
        current_team.append(np.array([float(x) for x in line[9:]]))
        
    if frame_idx == last_frame:
        video_detection.append(current_data)
        video_team.append(np.array(current_team))
        
    return video_detection, video_team

def load_team_jersey_gt_txt(gt_txt_path):
    team_jersey = {}
    with open(gt_txt_path, "r") as f:
        lines = f.readlines()
        for line in lines:
            id, jersey, team = line.strip().split(" ")
            id, jersey, team = int(id.strip(":")), int(jersey), int(team)
            team_jersey[id] = {"jersey": jersey, "team": team}
    
    return team_jersey
    

def test_load_pred(cache_folder, img_folder, team_name, result_folder):
    video_list = os.listdir(img_folder)

    for video_name in video_list:
        if video_name == 'SNMOT-196':
            video_result_folder = os.path.join(result_folder, video_name)
            os.makedirs(video_result_folder, exist_ok=True)

            detection_folder = os.path.join(cache_folder, "detection")
            team_folder = os.path.join(cache_folder, team_name)

            detection_path = os.path.join(detection_folder, f"{video_name}.npy")
            video_detection = np.load(detection_path, allow_pickle=True)
            team_path = os.path.join(team_folder, f"{video_name}.npy")
            team_data = np.load(team_path, allow_pickle=True)
            train_length = 30
            video_length = video_detection.shape[0]
            
            pc = PlayerClassification()
            for i in range(train_length):
                frame_detection = video_detection[i]
                frame_team_data = team_data[i]
                pc.update(frame_team_data)
            
            pc.fit()

            for i in range(video_length):
                if i % 10 != 0:
                    continue
                frame_detection = video_detection[i]
                frame_team_data = team_data[i]
                frame = cv2.imread(os.path.join(img_folder, video_name, "img1", f"{str(i+1).zfill(6)}.jpg"))
                crops = pc.get_crop(frame, frame_detection)
                labels, confs = pc.classify(frame_team_data)
                frame = draw_bbox(frame, frame_detection, labels, confs)
                # import IPython; IPython.embed()
                # time.sleep(0.6)
                cv2.imwrite(os.path.join(video_result_folder, f"{str(i+1).zfill(6)}.jpg"), frame)
            
            print(f"Saved {video_name}")

def benchmark(predict_team_folder, gt_team_folder, conf_threshold=0.5, train_length = 1):
    gt_video_list = os.listdir(gt_team_folder)
    predict_video_list = os.listdir(predict_team_folder)
    assert len(gt_video_list) == len(predict_video_list)
    
    total_precision = 0
    total_recall = 0
    
    for video_name in predict_video_list:
        try:
            gt_path = os.path.join(gt_team_folder, video_name)
            predict_path = os.path.join(predict_team_folder, video_name)
        except:
            continue
        gt_team_jersey = load_team_jersey_gt_txt(gt_path)
        pred_data, pred_jersey_feature = load_team_pred_txt(predict_path)
        
        true_count = 0
        total_count = 0
        total_bbox = 0
        
        # TRAIN
        pc = PlayerClassification()
        for i in range(train_length):
            pc.update(pred_jersey_feature[i])
        pc.fit()
        
        # INFER
        video_length = len(pred_data)
        for i in range(video_length):
            pred_team, confs = pc.classify(pred_jersey_feature[i])
            gt_team = []
            for j in range(len(pred_data[i])):
                id = pred_data[i][j][0]
                try:
                    gt_team.append(gt_team_jersey[id]["team"])
                except:
                    gt_team.append(0)
            
            true = 0
            total = 0
            for j in range(len(pred_team)):
                if confs[j] < conf_threshold:
                    continue
                if pred_team[j] == gt_team[j]:
                    true += 1
                total += 1
            
            if true < total - true:
                true = total - true
            true_count += true
            total_count += total
            total_bbox += len(pred_team)
            # print(f"{video_name}: Frame {i+1} - Accuracy = {temp/len(pred_team)}")
        
        total_precision += true_count/total_count
        total_recall += total_count/total_bbox
        # print(f"{video_name}: Precision = {true_count/total_count}, Recall = {total_count/total_bbox}, {true_count}-{total_count}-{total_bbox}")
    
    print(f"Conf: {conf_threshold}, Precision: {total_precision/len(predict_video_list)}, Recall: {total_recall/len(predict_video_list)}")
                
def add_sport_type():
    sport_type_mapping = {
        "6Os86HzwCs": "Volleyball",
        "1LwtoLPw2TU": "Volleyball",
        "1yHWGw8DH4A": "Football",
        "2j7kLB-vEEk": "Basketball",
        "4LXTUim5anY": "Basketball",
        "ApPxnw_Jffg": "Volleyball",
        "CW0mQbgYIF4": "Volleyball",
        "dChHNGIfm4Y": "Volleyball",
        "Dk3EpDDa3o0": "Volleyball",
        "gQNyhv8y0QY": "Football",
        "HdiyOtliFiw": "Football",
        "iIxMOsCGH58": "Football",
        "00HRwkvvjtQ": "Basketball",
        "0kUtTtmLaJA": "Volleyball",
        "2QhNRucNC7E": "Football",
        "EmEtrturE": "Volleyball",
        "4r8QL_wglzQ": "Basketball",
        "5ekaksddqrc": "Basketball",
        "9MHDmAMxO5I": "Volleyball",
        "BgwzTUxJaeU": "Basketball",
        "cC2mHWqMcjk": "Volleyball",
        "dw7LOz17Omg": "Football",
        "G-vNjfx1GGc": "Football",
        "L4qquVg0": "Football",
        "ITo3sCnpw": "Football",
        "9kabh1K8UA": "Basketball",
        "hhDbvY5aAM": "Football",
        "1UDUODIBSsc": "Football",
        "2BhBRkkAqbQ": "Basketball",
        "2ChiYdg5bxI": "Football",
        "2Dnx8BpgUEs": "Volleyball",
        "2Dw9QNH5KtU": "Volleyball",
        "6OLC1-bhioc": "Basketball",
        "6oTxfzKdG6Q": "Basketball",
        "7FTsO8S3h88": "Basketball",
        "8rG1vjmJHr4": "Volleyball",
        "9p0i81kAEwE": "Volleyball",
        "42VrMbd68Zg": "Basketball",
        "A4OJhlI6hgc": "Basketball",
        "aAb0psypDj4": "Volleyball",
        "aVfJwdQxCsU": "Basketball",
        "BdD9xu0E2H4": "Basketball",
        "bhWjlAEICp8": "Volleyball",
        "bQNDRprvpus": "Basketball",
        "czYZnO9QxYQ": "Volleyball",
        "DjtFlW2eHFI": "Football",
    }
    
    seqinfo_folders = [
        "./data/sportsmot_publish/dataset/train/",
        "./data/sportsmot_publish/dataset/val/",
        "./data/sportsmot_publish/dataset/test/",
    ]
    
    for seqinfo_folder in seqinfo_folders:
        video_list = os.listdir(seqinfo_folder)
        for video_name in video_list:
            sport_type = None
            for sport_id in sport_type_mapping:
                if sport_id in video_name:
                    sport_type = sport_type_mapping[sport_id]
                    
            if sport_type is not None:
                seqinfo_path = os.path.join(seqinfo_folder, video_name, "seqinfo.ini")
                sport_type_line = f"\nsportType = {sport_type}"
                with open(seqinfo_path, "a") as f:
                    f.write(sport_type_line)


if __name__ == "__main__":
    img_folders = [
        # "./data/sportsmot_publish/dataset/train/",
        "./data/sportsmot_publish/dataset/val/",
        # "./data/sportsmot_publish/dataset/test/"
        # "./data/sportsmot_publish/dataset/train"
    ]
    
    cache_folders = [
        "./SportSORT/cache/val/",
    ]
    
    ## INIT POSE ESTIMATION
    pose_config = './SportSORT/SportSORT/jersey_team/rtmpose/config/rtmpose-l_8xb256-420e_coco-256x192.py'
    pose_checkpoint = './SportSORT/SportSORT/jersey_team/rtmpose/ckpt/rtmpose-l_simcc-aic-coco_pt-aic-coco_420e-256x192-f016ffe0_20230126.pth'
    device = 'cuda:0'
    pose_model = init_model(config=pose_config, checkpoint=pose_checkpoint, device=device)
    
    ###############################
    team_names = {
        "team_raw": {"use_shoulder_hip": False, "use_knee_hip": False},
        "team_upper": {"use_shoulder_hip": True, "use_knee_hip": False},
        "team_lower": {"use_shoulder_hip": False, "use_knee_hip": True},
        "team_full": {"use_shoulder_hip": True, "use_knee_hip": True},
    }
    
    # PREDICTION BOX
    for cache_folder, img_folder in zip(cache_folders, img_folders):
        save_data_pred(pose_model, cache_folder, img_folder, team_names)
    ################################

    ################################
    # # # GROUND TRUTH BOX
    # team_names = {
    #     "gt_team_raw": {"use_shoulder_hip": False, "use_knee_hip": False},
    #     "gt_team_upper": {"use_shoulder_hip": True, "use_knee_hip": False},
    #     "gt_team_lower": {"use_shoulder_hip": False, "use_knee_hip": True},
    #     "gt_team_full": {"use_shoulder_hip": True, "use_knee_hip": True},
    # }
    
    # # gt/gt.txt
    # for cache_folder, img_folder in zip(cache_folders, img_folders):
    #     save_data_gt(pose_model, cache_folder, img_folder, team_names)
    ################################

    ################################
    # # TEST PREDICTION
    # result_folder = "./SportSORT/cache/team_classification6"
    # cache_folder = "./SportSORT/cache/train/"
    # img_folder = "./data/sportsmot_publish/dataset/train/"
    # team_name = "team_full"
    # test_load_pred(cache_folder, img_folder, team_name, result_folder)
    ################################

    # result_folder = "./data/SoccerNet/tracking/visualize_team"
    # os.makedirs(result_folder, exist_ok=True)
    # cache_folder = "./SportSORT/cache_soccer/test"
    # img_folder = "./data/SoccerNet/tracking/test"
    # team_name = "team_full"
    # test_load_pred(cache_folder, img_folder, team_name, result_folder)
    ################################
    # # BENCMARK GROUND TRUTH
    # gt_team_folder = "./data/sportsmot_publish/jersey_label"
    # predict_team_folder = "./SportSORT/cache/train/gt_team_upper"
    # for conf in [0.5, 0.52, 0.54, 0.56, 0.58, 0.6, 0.62, 0.64, 0.66, 0.68, 0.7]:
    #     benchmark(predict_team_folder, gt_team_folder, conf)
    ################################
    
    ################################
    # # ADD SPORT TYPE TO SEQINFO.INI
    # add_sport_type()


        
            
            

            
# color_dict = pc.get_counts(cv2.cvtColor(crop, cv2.COLOR_RGB2BGR), mask)
# color_list = []
# for color in sorted(color_dict.keys()):
#     color_list.append(color_dict[color])
# color_list = np.array(color_list)
# color_list = color_list / np.sum(color_list)
# color_lists.append(color_list)            
            
# def get_counts(self, image, mask=None):
#     w, h, d = image.shape
#     image_flatten = np.reshape(image[:,:,:3], (w * h, 3))
#     if mask is not None:
#         mask_flatten = np.reshape(mask, (w * h)) # 0 or 1
#         image_flatten = image_flatten[mask_flatten == 1]
#     bool_arrays = self.get_bool_arrays(image_flatten)
#     counts = dict()
#     for color_name in bool_arrays:
#         counts[color_name] = np.sum(bool_arrays[color_name])
#     return counts

# def get_bool_arrays(self, image_flatten):
#     shape = image_flatten.shape[0]
#     code_book, color_names = CODE_BOOK, COLOR_NAMES
#     labels, _ = vq(image_flatten, code_book)
#     img_labels = np.empty((shape), dtype=object)
#     for idx, label in np.ndenumerate(labels):
#         img_labels[idx] = color_names[label]

#     # Create color bool arrays and store in dictionary...
#     bool_arrays = dict()
#     for color_name in set(color_names):
#         bool_array = img_labels == color_name
#         bool_arrays[color_name] = bool_array
#     return bool_arrays

