#!/bin/bash

# # Định nghĩa các tham số tùy chọn
# # 2
# iou_thres_04="--iou_thres 0.4"
# iou_thres_05="--iou_thres 0.5"

# # 3
# init_expand_scale_05="--init_expand_scale 0.5 --expand_scale_step 0.3"
# init_expand_scale_04="--init_expand_scale 0.4 --expand_scale_step 0.4"
# init_expand_scale_03="--init_expand_scale 0.3 --expand_scale_step 0.5"

# # 3
# proximity_thresh_06="--proximity_thresh 0.6"
# proximity_thresh_07="--proximity_thresh 0.7"
# proximity_thresh_08="--proximity_thresh 0.8"

# # 4
# team_thresh_06="--team_thresh 0.6"
# team_thresh_07="--team_thresh 0.7"
# team_thresh_08="--team_thresh 0.8"
# team_thresh_09="--team_thresh 0.9"

# # 2
# team_factor_iou_2="--team_factor_iou 2"
# team_factor_iou_3="--team_factor_iou 3"

# # 2
# team_factor_emb_2="--team_factor_emb 2"
# team_factor_emb_4="--team_factor_emb 4"



# # Mảng chứa các tham số tùy chọn
# options=("$iou_thres_04" "$iou_thres_05" "$init_expand_scale_05" "$init_expand_scale_04" "$init_expand_scale_03" "$proximity_thresh_06" "$proximity_thresh_07" "$proximity_thresh_08" "$team_thresh_06" "$team_thresh_07" "$team_thresh_08" "$team_thresh_09" "$team_factor_iou_2" "$team_factor_iou_3" "$team_factor_emb_2" "$team_factor_emb_4")
# # options=("$iou_thres_03" "$iou_thres_035" "$iou_thres_04" "$iou_thres_045" "$iou_thres_05" "$init_expand_scale_07" "$init_expand_scale_06" "$init_expand_scale_05" "$init_expand_scale_04" "$init_expand_scale_03" "$proximity_thresh_05" "$proximity_thresh_06" "$proximity_thresh_07" "$proximity_thresh_08" "$team_thresh_06" "$team_thresh_065" "$team_thresh_07" "$team_thresh_075" "$team_thresh_08" "$team_thresh_085" "$team_thresh_09" "$team_thresh_095" "$team_factor_iou_15" "$team_factor_iou_2" "$team_factor_iou_25" "$team_factor_iou_3" "$team_factor_emb_15" "$team_factor_emb_2" "$team_factor_emb_25" "$team_factor_emb_3" "$team_factor_emb_4")
# # options=("$iou_thres_03" "$iou_thres_035" "$iou_thres_04" "$iou_thres_045" "$iou_thres_05" "$init_expand_scale_07" "$init_expand_scale_06" "$init_expand_scale_05" "$init_expand_scale_04" "$init_expand_scale_03" "$proximity_thresh_05" "$proximity_thresh_06" "$proximity_thresh_07" "$proximity_thresh_08" "$team_thresh_06" "$team_thresh_065" "$team_thresh_07" "$team_thresh_075" "$team_thresh_08" "$team_thresh_085" "$team_thresh_09" "$team_thresh_095" "$team_factor_iou_15" "$team_factor_iou_2" "$team_factor_iou_25" "$team_factor_iou_3" "$team_factor_emb_15" "$team_factor_emb_2" "$team_factor_emb_25" "$team_factor_emb_3" "$team_factor_emb_4")

# # fix error tools/only_team.sh: 44: Syntax error: "(" unexpected
# # Tổng số các tham số tùy chọn
# num_options=${#options[@]}

# Đường dẫn thư mục log chung
log_dir="/mnt/banana/student/thuyntt/Deep-EIoU/evaluation/TrackEval/eval_logs/new_only_team2"

# Tạo thư mục log nếu chưa tồn tại
mkdir -p "$log_dir"


iou_thres=("0.4")                                 # 1 options
init_expand_scale=("0.5 0.3" "0.4 0.4" "0.3 0.5") # 3 options
proximity_thresh=("0.6" "0.7" "0.8")              # 3 options
team_thresh=("0.6" "0.7" "0.8" "0.9")             # 4 options
team_factor_iou=("2" "3")                         # 2 options
team_factor_emb=("2" "4")                         # 2 options

for iou in "${iou_thres[@]}"; do
  for scale in "${init_expand_scale[@]}"; do
    for proximity in "${proximity_thresh[@]}"; do
      for team in "${team_thresh[@]}"; do
        for iou_factor in "${team_factor_iou[@]}"; do
          for emb_factor in "${team_factor_emb[@]}"; do

            # Construct the argument string for the current parameter set
            args="--iou_thres $iou --init_expand_scale ${scale% *} --expand_scale_step ${scale#* } --proximity_thresh $proximity --team_thres $team --team_factor_iou $iou_factor --team_factor_emb $emb_factor"
            
            # Construct a unique log filename
            log_filename="only_team_iou_${iou}_scale_${scale// /_}_prox_${proximity}_team_${team}_ioufac_${iou_factor}_embfac_${emb_factor}"

            # Change to Deep-EIoU directory and run inference
            echo "Running inference with parameters: $args"
            cd /mnt/banana/student/thuyntt/Deep-EIoU/Deep-EIoU
            python tools/infer_2.py $args
            wait  # Ensure inference completes before continuing
            
            # Move to TrackEval directory for evaluation
            echo "Running evaluation and saving logs in $log_dir/$log_filename.txt"
            cd /mnt/banana/student/thuyntt/Deep-EIoU/evaluation/TrackEval
            python ./scripts/run_mot_challenge2.py --BENCHMARK sportsmot --SPLIT_TO_EVAL train --METRICS HOTA CLEAR2 Identity VACE --USE_PARALLEL False --PRINT_CONFIG True --GT_FOLDER ./data/ref --TRACKERS_FOLDER ./data/thuy_res --OUTPUT_FOLDER ./output/ > "$log_dir/$log_filename.txt"
            wait  # Ensure evaluation completes before moving to the next combination

          done
        done
      done
    done
  done
done

# for ((i=0; i< (1 << num_options); i++)); do
#     args=""
#     log_filename="ony_team"

#     # Xây dựng chuỗi tham số cho mỗi trường hợp
#     for ((j=0; j<num_options; j++)); do
#         if (( (i & (1 << j)) != 0 )); then
#             args+=" ${options[j]}"
#             # Thêm phần tham số vào tên file log
#             log_filename+="_${options[j]// /_}"
#         fi
#     done
#     # Chạy lệnh infer_3.py với các tham số đã chọn
    # cd /mnt/banana/student/thuyntt/Deep-EIoU/Deep-EIoU
    # python tools/infer_2.py $args

    # # Chạy lệnh đánh giá trong TrackEval và lưu kết quả vào file log
    # echo "Running evaluation and saving logs in $log_dir/$log_filename.txt"
    # cd /mnt/banana/student/thuyntt/Deep-EIoU/evaluation/TrackEval
    # python ./scripts/run_mot_challenge2.py --BENCHMARK sportsmot --SPLIT_TO_EVAL train --METRICS HOTA CLEAR2 Identity VACE --USE_PARALLEL False --PRINT_CONFIG True --GT_FOLDER ./data/ref --TRACKERS_FOLDER ./data/thuy_res --OUTPUT_FOLDER ./output/ > "$log_dir/$log_filename.txt"
#     echo "Running infer_3.py with args: $args"
#     echo "--------------------------------------------"
#     echo "--------------------------------------------"
#     echo "--------------------------------------------"
#     echo "--------------------------------------------"
#     echo "--------------------------------------------"
# done

