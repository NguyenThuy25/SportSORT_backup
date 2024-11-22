import argparse
import os
import os.path as osp
import numpy as np
import time
import cv2
import torch
import sys
sys.path.append('.')
sys.path.append('./SportSORT/')
sys.path.append('./SportSORT/SportSPORT')
sys.path.append('./SportSORT/SportSPORT/reid')

from loguru import logger

from yolox.data.data_augment import preproc
from yolox.exp import get_exp
from yolox.utils import fuse_model, get_model_info, postprocess
from yolox.tracking_utils.timer import Timer
from yolox.utils.visualize import plot_tracking
from tracker.SportSORT import SportSORT
from reid.torchreid.utils import FeatureExtractor
from jersey_team.jersey_pipeline import JerseyNumberPipeline
# from jersey_team.rtmpose.top_down import init_model
from jersey_team.player_classification import PlayerClassification
# from jersey_team import configuration as config
IMAGE_EXT = [".jpg", ".jpeg", ".webp", ".bmp", ".png"]

def make_parser():
    parser = argparse.ArgumentParser("SportSORT Demo")
    parser.add_argument("-expn", "--experiment-name", type=str, default=None)
    parser.add_argument("-n", "--name", type=str, default=None, help="model name")
    parser.add_argument("--path", default="./data/sportsmot_publish/dataset/", help="path to folder containing images")

    parser.add_argument(
        "--output_dir", 
        default="./SportSORT/evaluation/TrackEval/data/res/",

        type=str, 
        help="Directory to save output results"
    )
    parser.add_argument(
        "--cache_dir", 
        default="./SportSORT/cache/", 
        # default="./SportSORT/cache/test", 

        type=str, 
        help="Directory to save output results"
    )
    parser.add_argument("--save_result", default=True, help="whether to save the inference result of images")

    # exp file
    parser.add_argument("-f", "--exp_file", default="yolox/yolox_x_ch_sportsmot.py", type=str, help="experiment description file")
    parser.add_argument("-c", "--ckpt", default=None, type=str, help="ckpt for eval")
    parser.add_argument("--device", default="gpu", type=str, help="device to run our model, can either be cpu or gpu")
    parser.add_argument("--conf", default=None, type=float, help="test conf")
    parser.add_argument("--nms", default=None, type=float, help="test nms threshold")
    parser.add_argument("--tsize", default=None, type=int, help="test img size")
    parser.add_argument("--fps", default=30, type=int, help="frame rate (fps)")
    parser.add_argument("--fp16", dest="fp16", default=False, action="store_true", help="Adopting mix precision evaluating.")
    parser.add_argument("--fuse", dest="fuse", default=False, action="store_true", help="Fuse conv and bn for testing.")
    parser.add_argument("--trt", dest="trt", default=False, action="store_true", help="Using TensorRT model for testing.")

    # tracking args
    parser.add_argument("--track_high_thresh", type=float, default=0.6, help="tracking confidence threshold")
    parser.add_argument("--track_low_thresh", default=0.1, type=float, help="lowest detection threshold valid for tracks")
    parser.add_argument("--new_track_thresh", default=0.7, type=float, help="new track thresh")
    parser.add_argument("--track_buffer", type=int, default=60, help="the frames for keeping lost tracks")
    parser.add_argument("--match_thresh", type=float, default=0.8, help="matching threshold for tracking")
    parser.add_argument("--aspect_ratio_thresh", type=float, default=1.6, help="threshold for filtering out boxes of which aspect ratio are above the given value.")
    parser.add_argument("--min_box_area", type=float, default=10, help="filter out tiny boxes")
    parser.add_argument("--nms_thres", type=float, default=0.7, help="nms threshold")
    parser.add_argument("--mot20", dest="mot20", default=False, action="store_true", help="test mot20.")

    parser.add_argument("--with-reid", dest="with_reid", default=True, action="store_true", help="use Re-ID flag.")
    parser.add_argument("--proximity_thresh", type=float, default=0.5, help="threshold for rejecting low overlap reid matches")
    parser.add_argument("--appearance_thresh", type=float, default=0.25, help="threshold for rejecting low appearance similarity reid matches")
    parser.add_argument("--iou_thres", type=float, default=0.5, help="filter out overlapping boxes")
    parser.add_argument("--jersey_iou_thres", type=float, default=None, help="filter out overlapping boxes for jersey")

    parser.add_argument("--init_expand_scale", type=float, default=0.7, help="initial expand scale")
    parser.add_argument("--expand_scale_step", type=float, default=0.1, help="expand scale step")
    parser.add_argument("--num_iteration", type=int, default=2, help="number of iteration")
    parser.add_argument("--init_team_frame_thres", type=int, default=10, help="number of frame to start infer team")
    parser.add_argument("--use_first_association_team", action="store_true", help="use first score association team")
    parser.add_argument("--use_first_association_jersey", action="store_true", help="use first score association jersey")

    parser.add_argument("--ensemble_metric", type=str, default="bot", help="ensemble metric")
    parser.add_argument("--use_appearance_thresh", type=bool, default=False, help="use appearance threshold") # Default is False for Harmonic Mean

    # parser.add_argument("--cache_detection_name", type=str, default="detection", help="cache detection name")
    parser.add_argument("--cache_detection_name", type=str, default=None, help="cache detection name (leave empty to not load cache)")

    # parser.add_argument("--cache_embedding_name", type=str, default="embedding", help="cache embedding name") # embedding_sports
    parser.add_argument("--cache_embedding_name", type=str, default=None, help="cache embedding name (leave empty to not load cache)")
    parser.add_argument("--cache_jersey_name", type=str, default="jersey_num_infer", help="cache jersey name") # jersey_num_hockey
    parser.add_argument("--cache_team_name", type=str, default="team_full", help="cache team name")
    parser.add_argument("--use_fourth_association", action="store_true", help="use fourth association")
    parser.add_argument("--use_fourth_association_corner", action="store_true", help="use fourth association corner")
    parser.add_argument("--corner_ratio", type=float, default=0.15, help="corner ratio")
    parser.add_argument("--use_fourth_association_team", action="store_true", help="use fourth association team")
    parser.add_argument("--use_fourth_association_jersey", action="store_true", help="use fourth association jersey")
    parser.add_argument("--use_fourth_association_same_corner", action="store_true", help="use fourth association same corner")
    parser.add_argument("--emb_match_thresh", type=float, default=0.3, help="embedding matching threshold")
    
    parser.add_argument("--team_thres", type=float, default=0.7, help="thresh for predict team")
    parser.add_argument("--jersey_thres", type=float, default=0.7, help="thresh for predict jersey")
    parser.add_argument("--team_factor", type=float, default=None, help="multiple factor to dist when not match team")
    parser.add_argument("--team_factor_conf", action="store_true", help="multiple conf to dist when not match team") 
    parser.add_argument("--jersey_factor", type=float, default=None, help="multiple to dist when not match jersey")
    parser.add_argument("--jersey_factor_conf", action="store_true", help="multiple conf to dist when not match jersey")
    
    parser.add_argument("--fixed_team_thresh", type=int, default=10, help="fixed team thresh")
    parser.add_argument("--fixed_jersey_thresh", type=int, default=30, help="fixed jersey thresh")
    parser.add_argument("--max_new_len_thresh", type=int, default=10, help="max new len thresh")
    parser.add_argument("--split", type=str, default="train", help="split for using train/val/test")
    parser.add_argument("--exp_name", type=str, default=None, help="experiment name")
    parser.add_argument("--adt_team", default=False, action="store_true", help="use adaptive team flag.")
    parser.add_argument("--adt_jersey", default=False, action="store_true", help="use adaptive jersey flag.")
    parser.add_argument("--adt_alpha", type=float, default=0, help="adaptive alpha")

    parser.add_argument("--online", default=False, action="store_true", help="online flag.")

    return parser


def get_image_list(path):
    image_names = []
    for maindir, subdir, file_name_list in os.walk(path):
        for filename in file_name_list:
            apath = osp.join(maindir, filename)
            ext = osp.splitext(apath)[1]
            if ext in IMAGE_EXT:
                image_names.append(apath)
    return sorted(image_names)


def write_results(filename, results):
    save_format = '{frame},{id},{x1},{y1},{w},{h},{s},-1,-1,-1\n'
    with open(filename, 'w') as f:
        for frame_id, tlwhs, track_ids, scores in results:
            for tlwh, track_id, score in zip(tlwhs, track_ids, scores):
                if track_id < 0:
                    continue
                x1, y1, w, h = tlwh
                line = save_format.format(frame=frame_id, id=track_id, x1=round(x1, 1), y1=round(y1, 1), w=round(w, 1), h=round(h, 1), s=round(score, 2))
                f.write(line)
    logger.info('save results to {}'.format(filename))


class Predictor(object):
    def __init__(
        self,
        model,
        exp,
        trt_file=None,
        decoder=None,
        device=torch.device("cpu"),
        fp16=False
    ):
        self.model = model
        self.decoder = decoder
        self.num_classes = exp.num_classes
        self.confthre = exp.test_conf
        self.nmsthre = exp.nmsthre
        self.test_size = exp.test_size
        self.device = device
        self.fp16 = fp16
        if trt_file is not None:
            from torch2trt import TRTModule

            model_trt = TRTModule()
            model_trt.load_state_dict(torch.load(trt_file))

            x = torch.ones((1, 3, exp.test_size[0], exp.test_size[1]), device=device)
            self.model(x)
            self.model = model_trt
        self.rgb_means = (0.485, 0.456, 0.406)
        self.std = (0.229, 0.224, 0.225)

    def inference(self, img, timer):
        img_info = {"id": 0}
        if isinstance(img, str):
            img_info["file_name"] = osp.basename(img)
            img = cv2.imread(img)
        else:
            img_info["file_name"] = None

        height, width = img.shape[:2]
        img_info["height"] = height
        img_info["width"] = width
        img_info["raw_img"] = img

        img, ratio = preproc(img, self.test_size, self.rgb_means, self.std)
        img_info["ratio"] = ratio
        img = torch.from_numpy(img).unsqueeze(0).float().to(self.device)
        if self.fp16:
            img = img.half()  # to FP16

        with torch.no_grad():
            timer.tic()
            outputs = self.model(img)
            if self.decoder is not None:
                outputs = self.decoder(outputs, dtype=outputs.type())
            outputs = postprocess(
                outputs, self.num_classes, self.confthre, self.nmsthre
            )
        return outputs, img_info

def load_cached_data(video_name, args):
    """
    Load cached embeddings and detections if they exist.
    Returns loaded detections, embeddings, jersey data, and team data.
    """
    # Initialize file paths based on provided arguments
    detection_file = (
        osp.join(args.cache_dir, args.cache_detection_name, f"{video_name}.npy")
        if args.cache_detection_name
        else None
    )
    embedding_file = (
        osp.join(args.cache_dir, args.cache_embedding_name, f"{video_name}.npy")
        if args.cache_embedding_name
        else None
    )
    jersey_file = (
        osp.join(args.cache_dir, args.cache_jersey_name, f"{video_name}.txt")
        if args.use_first_association_jersey or args.use_fourth_association_jersey
        else None
    )
    team_file = (
        osp.join(args.cache_dir, args.cache_team_name, f"{video_name}.npy")
        if args.use_first_association_team or args.use_fourth_association_team
        else None
    )

    # Attempt to load cached files if they exist
    all_detections = None
    all_embeddings = None
    all_jerseys = None
    all_teams = None

    if detection_file and osp.exists(detection_file):
        logger.info(f"Loading cached detections from {detection_file}")
        all_detections = np.load(detection_file, allow_pickle=True)
    elif detection_file:
        logger.warning(f"Detection cache not found at {detection_file}")

    if embedding_file and osp.exists(embedding_file):
        logger.info(f"Loading cached embeddings from {embedding_file}")
        all_embeddings = np.load(embedding_file, allow_pickle=True)
    elif embedding_file:
        logger.warning(f"Embedding cache not found at {embedding_file}")

    if jersey_file and osp.exists(jersey_file):
        logger.info(f"Loading cached jersey data from {jersey_file}")
        all_jerseys = np.loadtxt(jersey_file, delimiter=',')
    elif jersey_file:
        logger.warning(f"Jersey cache not found at {jersey_file}")

    if team_file and osp.exists(team_file):
        logger.info(f"Loading cached team data from {team_file}")
        all_teams = np.load(team_file, allow_pickle=True)
    elif team_file:
        logger.warning(f"Team cache not found at {team_file}")

    return all_detections, all_embeddings, all_jerseys, all_teams



def save_cache(embedding_file, detection_file, all_embeddings, all_detections):
    """Save embeddings and detections to cache."""
    os.makedirs(osp.dirname(embedding_file), exist_ok=True)
    os.makedirs(osp.dirname(detection_file), exist_ok=True)
    np.save(embedding_file, np.array(all_embeddings, dtype=object))
    np.save(detection_file, np.array(all_detections, dtype=object))
    logger.info(f"Saved embeddings to {embedding_file}")
    logger.info(f"Saved detections to {detection_file}")


def process_frame(frame, predictor, extractor, timer, img_info):
    """Process a single frame to extract detections and embeddings."""
    outputs, img_info = predictor.inference(frame, timer)
    if outputs[0] is None:
        return None, None

    det = outputs[0].cpu().detach().numpy()
    scale = min(1440 / 1280, 800 / 720)
    det /= scale
    det = det[~np.any(det[:, :4] < 1, axis=1)]  # Remove invalid rows

    cropped_imgs = []
    invalid_indices = []
    for i, (x1, y1, x2, y2, *rest) in enumerate(det):
        x1, y1 = max(0, int(x1)), max(0, int(y1))
        x2, y2 = min(img_info["width"], int(x2)), min(img_info["height"], int(y2))
        if x2 > x1 and y2 > y1:
            cropped_imgs.append(frame[y1:y2, x1:x2])
        else:
            invalid_indices.append(i)

    det = np.delete(det, invalid_indices, axis=0)
    embs = extractor(cropped_imgs).cpu().detach().numpy()
    return det, embs


def track_objects(tracker, detections, embeddings, teams=None, jerseys=None, frame_id=1, args=None):
    """Track objects using detections and embeddings."""
    results = []
    det_with_jerseys = np.concatenate((detections, jerseys), axis=1) if jerseys is not None else detections

    if len(embeddings.shape) == 3:
        embeddings = np.array([e[0] for e in embeddings])

    online_targets = tracker.update(det_with_jerseys, embeddings, teams)
    online_tlwhs, online_ids, online_scores = [], [], []

    for t in online_targets:
        tlwh, tid, score = t.last_tlwh, t.track_id, t.score
        if tlwh[2] * tlwh[3] > args.min_box_area:
            online_tlwhs.append(tlwh)
            online_ids.append(tid)
            online_scores.append(score)
            results.append(f"{frame_id},{tid},{tlwh[0]:.2f},{tlwh[1]:.2f},{tlwh[2]:.2f},{tlwh[3]:.2f},{score:.2f},-1,-1,-1\n")
    return results


def process_image_folder(predictor, extractor, jersey_recognizer, pose_model, color_classifier, vis_folder, args, image_folder):
    """Main function to process an image folder."""
    image_names = get_image_list(image_folder)
    logger.info(f"Processing folder: {image_folder}")
    video_name = osp.basename(image_folder)
    tracker = SportSORT(args, frame_rate=30)
    timer = Timer()
    frame_id, results = 1, []



    # If cached data is available, use it
    if not args.online:
        logger.info("Using cached data")
        all_detections, all_embeddings, all_jerseys, all_teams = load_cached_data(video_name, args)
        for frame_id, (det, embs, team_embs) in enumerate(zip(all_detections, all_embeddings, all_teams if all_teams is not None else []), start=1):
            jerseys = [j[5:] for j in all_jerseys if j[0] == frame_id] if all_jerseys is not None else None
            results += track_objects(tracker, det, embs, team_embs, jerseys, frame_id, args)
    else:
        logger.info("Processing images from folder")
        all_detections, all_embeddings = [], []
        det_emb_total_time = 0
        jersey_total_time = 0
        team_total_time = 0
        frame_time = 0
        hungarian_time = 0
        for img_path in sorted(image_names):
            frame_time_start = time.time()
            if frame_id % 30 == 0:
                logger.info(f'Processing frame {frame_id} ({1. / max(1e-5, timer.average_time):.2f} fps)')
            frame = cv2.imread(img_path)
            if frame is None:
                break
            
            det_emb_start = time.time()
            det, embs = process_frame(frame, predictor, extractor, timer, img_info=None)
            det_emb_total_time += time.time() - det_emb_start

            jersey_start = time.time()
            jerseys = jersey_recognizer.infer_one_image(frame, det, frame_id, video_name=None, save_img=False, save_cache=False)
            jersey_total_time += time.time() - jersey_start

            team_start = time.time()
            team_embs = color_classifier.infer_one_image(pose_model, frame, det, team_names="team_full")
            team_total_time += time.time() - team_start

            if det is not None:
                all_detections.append(det)
                all_embeddings.append(embs)
                jerseys = [j[5:] for j in jerseys] if jerseys is not None else None
                jerseys = [np.array([float(item) for item in sublist]) for sublist in jerseys]
                
                hungarian_start = time.time()
                results += track_objects(tracker, det, embs, team_embs, jerseys, frame_id=frame_id, args=args)
                hungarian_time += time.time() - hungarian_start
                # results += track_objects(tracker, det, embs, team_embs, jerseys=None, frame_id=frame_id, args=args)
            frame_id += 1
            frame_time += time.time() - frame_time_start
    
        team_fps = frame_id / team_total_time
        jersey_fps = frame_id / jersey_total_time
        det_emb_fps = frame_id / det_emb_total_time
        hungarian_fps = frame_id / hungarian_time
        frame_fps = frame_id / frame_time
        logger.info(f"Team FPS: {team_fps:.2f}")
        logger.info(f"Jersey FPS: {jersey_fps:.2f}")
        logger.info(f"Det/Emb FPS: {det_emb_fps:.2f}")
        logger.info(f"Hungarian FPS: {hungarian_fps:.2f}")
        logger.info(f"Frame FPS: {frame_fps:.2f}")


    # Save results
    if args.save_result:
        res_file = osp.join(vis_folder, f"{video_name}.txt")
        with open(res_file, 'w') as f:
            f.writelines(results)
        logger.info(f"Saved results to {res_file}")

def main(exp, args):

    if args.split == 'test':
        args.output_dir = os.path.join(args.output_dir, f'result_{args.split}', args.exp_name)
    else:
        args.output_dir = os.path.join(args.output_dir, f'result_{args.split}', args.exp_name, f'sportsmot-{args.split}', 'tracker_to_eval', 'data')

    os.makedirs(args.output_dir, exist_ok=True)

    args.cache_dir = os.path.join(args.cache_dir, args.split)
    # vis_folder = osp.join(args.output_dir, "track_vis")
    vis_folder = args.output_dir
    os.makedirs(vis_folder, exist_ok=True)

    args.device = torch.device("cuda" if args.device == "gpu" else "cpu")

    logger.info("Args: {}".format(args))

    if args.conf is not None:
        exp.test_conf = args.conf
    if args.nms is not None:
        exp.nmsthre = args.nms
    if args.tsize is not None:
        exp.test_size = (args.tsize, args.tsize)

    model = exp.get_model().to(args.device)
    
    logger.info("Model Summary: {}".format(get_model_info(model, exp.test_size)))
    model.eval()

    if not args.trt:
        ckpt_file = args.ckpt or "checkpoints/best_ckpt.pth.tar"
        logger.info("Loading checkpoint")
        ckpt = torch.load(ckpt_file, map_location="cpu")
        model.load_state_dict(ckpt["model"])
        logger.info("Loaded checkpoint done.")

    if args.fuse:
        logger.info("\tFusing model...")
        model = fuse_model(model)

    if args.fp16:
        model = model.half()

    if args.trt:
        trt_file = osp.join(args.output_dir, "model_trt.pth")
        assert osp.exists(trt_file), "TensorRT model is not found!"
        model.head.decode_in_inference = False
        decoder = model.head.decode_outputs
        logger.info("Using TensorRT to inference")
    else:
        trt_file, decoder = None, None

    predictor = Predictor(model, exp, trt_file, decoder, args.device, args.fp16)
    current_time = time.localtime()
    
    extractor = FeatureExtractor(
        model_name='osnet_x1_0',
        model_path='checkpoints/sports_model.pth.tar-60',
        device='cuda'
        # device='cpu'
    )

    jersey_recognizer = JerseyNumberPipeline()

    pose_model = jersey_recognizer.pose_model
    color_classifier = PlayerClassification()
    # 
    args.path = osp.join(args.path, args.split)
    # Get all video folders inside the parent path
    video_folders = [osp.join(args.path, d) for d in os.listdir(args.path) if osp.isdir(osp.join(args.path, d))]

    for video_folder in video_folders:
        process_image_folder(predictor, extractor, jersey_recognizer, pose_model, color_classifier, vis_folder, args, video_folder)


if __name__ == "__main__":
    args = make_parser().parse_args()
    exp = get_exp(args.exp_file, args.name)

    main(exp, args)