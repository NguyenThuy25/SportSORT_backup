import numpy as np


pred_gt_path = '/mnt/banana/student/thuyntt/Deep-EIoU/cache/train/jersey_num/v_-6Os86HzwCs_c003.txt'
pred_gts = np.loadtxt(pred_gt_path, delimiter=',')
conf_thresh = 0.95

ids_list = pred_gts[:, 1]
ids_list = set(int(id) for id in ids_list)
print(ids_list)
# for pred_gt in pred_gts:
#     frame, id, x, y, w, h, jersey_num, jersey_conf, jersey_gt = pred_gt
#     if jersey_conf >= conf_thresh:
        