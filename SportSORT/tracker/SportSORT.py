import time
import numpy as np
from collections import deque

from tracker import matching
from tracker.basetrack import BaseTrack, TrackState
from tracker.kalman_filter import KalmanFilter
from jersey_team.player_classification import PlayerClassification

from collections import defaultdict
# FIXED_JERSEY_THRESH = 30
# FIXED_TEAM_THRESH = 10
# MAX_NEW_LEN_THRESH = 10

class STrack(BaseTrack):
    shared_kalman = KalmanFilter()
    

    def __init__(self, tlwh, score, feat=None, feat_history=30):

        # wait activate
        self._tlwh = np.asarray(tlwh, dtype=np.float)
        self.kalman_filter = None
        self.mean, self.covariance = None, None
        self.is_activated = False

        self.last_tlwh = self._tlwh

        self.score = score
        self.tracklet_len = 0

        self.smooth_feat = None
        self.curr_feat = None
        if feat is not None:
            self.update_features(feat)
        self.features = deque([], maxlen=feat_history)
        self.features = []
        self.times = []
        self.alpha = 0.9

        self.jersey_list = {} # count for time jersey is detected
        self.fix_jersey = -1 # jersey fix for track after 20 frame
        self.det_jersey = -1 # jersey detected in the current frame (for new det only)
        self.jersey_conf = -1

        self.team_list = {}
        self.fix_team = -1
        self.det_team = -1
        self.team_conf = -1
        
        self.is_new_track = True
        self.reborn = False

    def update_features(self, feat):
        feat /= np.linalg.norm(feat)
        self.curr_feat = feat
        if self.smooth_feat is None:
            self.smooth_feat = feat
        else:
            self.smooth_feat = self.alpha * self.smooth_feat + (1 - self.alpha) * feat
        self.smooth_feat /= np.linalg.norm(self.smooth_feat)

    def predict(self):
        mean_state = self.mean.copy()
        if self.state != TrackState.Tracked:
            mean_state[6] = 0
            mean_state[7] = 0

        self.mean, self.covariance = self.kalman_filter.predict(mean_state, self.covariance)

    @staticmethod
    def multi_predict(stracks):
        if len(stracks) > 0:
            multi_mean = np.asarray([st.mean.copy() for st in stracks])
            multi_covariance = np.asarray([st.covariance for st in stracks])
            for i, st in enumerate(stracks):
                if st.state != TrackState.Tracked:
                    multi_mean[i][6] = 0
                    multi_mean[i][7] = 0
            multi_mean, multi_covariance = STrack.shared_kalman.multi_predict(multi_mean, multi_covariance)
            for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
                stracks[i].mean = mean
                stracks[i].covariance = cov

    @staticmethod
    def multi_gmc(stracks, H=np.eye(2, 3)):
        if len(stracks) > 0:
            multi_mean = np.asarray([st.mean.copy() for st in stracks])
            multi_covariance = np.asarray([st.covariance for st in stracks])

            R = H[:2, :2]
            R8x8 = np.kron(np.eye(4, dtype=float), R)
            t = H[:2, 2]

            for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
                mean = R8x8.dot(mean)
                mean[:2] += t
                cov = R8x8.dot(cov).dot(R8x8.transpose())

                stracks[i].mean = mean
                stracks[i].covariance = cov

    def activate(self, kalman_filter, frame_id):
        """Start a new tracklet"""
        self.kalman_filter = kalman_filter
        self.track_id = self.next_id()
        self.mean, self.covariance = self.kalman_filter.initiate(self.tlwh_to_xywh(self._tlwh))

        self.tracklet_len = 0
        self.state = TrackState.Tracked
        if frame_id == 1:
            self.is_activated = True
        self.frame_id = frame_id
        self.start_frame = frame_id

    def re_activate(self, new_track, frame_id, new_id=False):

        self.mean, self.covariance = self.kalman_filter.update(self.mean, self.covariance, self.tlwh_to_xywh(new_track.tlwh))

        self.last_tlwh = new_track.tlwh

        if new_track.curr_feat is not None:
            self.update_features(new_track.curr_feat)
            self.features.append(new_track.curr_feat)
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        self.is_activated = True
        self.frame_id = frame_id
        if new_id:
            self.track_id = self.next_id()
        self.score = new_track.score

    def update(self, new_track, frame_id):
        """
        Update a matched track
        :type new_track: STrack
        :type frame_id: int
        :type update_feature: bool
        :return:
        """
        self.frame_id = frame_id
        self.tracklet_len += 1
        if self.is_new_track == True and self.tracklet_len > SportSORT.MAX_NEW_LEN_THRESH:
            self.is_new_track = False

        new_tlwh = new_track.tlwh

        self.last_tlwh = new_tlwh

        if new_track.curr_feat is not None:
            self.update_features(new_track.curr_feat)
            self.features.append(new_track.curr_feat)
            self.times.append(frame_id)

        self.state = TrackState.Tracked
        self.is_activated = True

        self.score = new_track.score

        if new_track.det_jersey != -1:
            if new_track.det_jersey in self.jersey_list:
                self.jersey_list[new_track.det_jersey] += 1
            else:
                self.jersey_list[new_track.det_jersey] = 1

            if self.fix_jersey == -1:
                if self.tracklet_len > 10:
                    max_jersey = max(self.jersey_list, key=self.jersey_list.get)
                    if self.jersey_list[max_jersey] > SportSORT.FIXED_JERSEY_THRESH:
                        self.fix_jersey = max_jersey
            else:
                max_jersey = max(self.jersey_list, key=self.jersey_list.get)
                self.fix_jersey = max_jersey
        
        if new_track.det_team != -1:
            if new_track.det_team in self.team_list:
                self.team_list[new_track.det_team] += 1
            else:
                self.team_list[new_track.det_team] = 1

            if self.fix_team == -1:
                if self.tracklet_len > 10:
                    max_team = max(self.team_list, key=self.team_list.get)
                    if self.team_list[max_team] > SportSORT.FIXED_TEAM_THRESH:
                        self.fix_team = max_team
            else:
                max_team = max(self.team_list, key=self.team_list.get)
                self.fix_team = max_team


    @property
    def tlwh(self):
        """Get current position in bounding box format `(top left x, top left y,
                width, height)`.
        """
        if self.mean is None:
            return self._tlwh.copy()
        ret = self.mean[:4].copy()
        ret[:2] -= ret[2:] / 2
        return ret

    @property
    def tlbr(self):
        """Convert bounding box to format `(min x, min y, max x, max y)`, i.e.,
        `(top left, bottom right)`.
        """
        ret = self.tlwh.copy()
        ret[2:] += ret[:2]
        return ret
    
    @property
    def last_tlbr(self):
        ret = self.last_tlwh.copy()
        ret[2:] += ret[:2]
        return ret

    @property
    def xywh(self):
        """Convert bounding box to format `(min x, min y, max x, max y)`, i.e.,
        `(top left, bottom right)`.
        """
        ret = self.tlwh.copy()
        ret[:2] += ret[2:] / 2.0
        return ret

    @staticmethod
    def tlwh_to_xyah(tlwh):
        """Convert bounding box to format `(center x, center y, aspect ratio,
        height)`, where the aspect ratio is `width / height`.
        """
        ret = np.asarray(tlwh).copy()
        ret[:2] += ret[2:] / 2
        ret[2] /= ret[3]
        return ret

    @staticmethod
    def tlwh_to_xywh(tlwh):
        """Convert bounding box to format `(center x, center y, width,
        height)`.
        """
        ret = np.asarray(tlwh).copy()
        ret[:2] += ret[2:] / 2
        return ret

    def to_xywh(self):
        return self.tlwh_to_xywh(self.tlwh)

    @staticmethod
    def tlbr_to_tlwh(tlbr):
        ret = np.asarray(tlbr).copy()
        ret[2:] -= ret[:2]
        return ret

    @staticmethod
    def tlwh_to_tlbr(tlwh):
        ret = np.asarray(tlwh).copy()
        ret[2:] += ret[:2]
        return ret

    def __repr__(self):
        return 'OT_{}_({}-{})'.format(self.track_id, self.start_frame, self.end_frame)


class SportSORT(object):
    def __init__(self, args, frame_rate=30):

        self.tracked_stracks = []  # type: list[STrack]
        self.lost_stracks = []  # type: list[STrack]
        self.removed_stracks = []  # type: list[STrack]
        BaseTrack.clear_count()

        self.frame_id = 0
        self.args = args

        self.track_high_thresh = args.track_high_thresh
        self.track_low_thresh = args.track_low_thresh
        self.new_track_thresh = args.new_track_thresh

        self.buffer_size = int(frame_rate / 30.0 * args.track_buffer)
        self.max_time_lost = self.buffer_size
        self.kalman_filter = KalmanFilter()
        self.player_classifier = PlayerClassification()

        # ReID module
        self.proximity_thresh = args.proximity_thresh
        self.appearance_thresh = args.appearance_thresh

        self.ensemble_metric = args.ensemble_metric
        self.use_appearance_thresh = args.use_appearance_thresh

        self.init_expand_scale = args.init_expand_scale
        self.expand_scale_step = args.expand_scale_step
        self.num_iteration = args.num_iteration
        self.init_team_frame_thres = args.init_team_frame_thres
        self.use_first_association_team = args.use_first_association_team
        self.use_first_association_jersey = args.use_first_association_jersey

        self.use_fourth_association = args.use_fourth_association
        self.use_fourth_association_corner = args.use_fourth_association_corner
        self.corner_ratio = args.corner_ratio
        self.use_fourth_association_team = args.use_fourth_association_team
        self.use_fourth_association_jersey = args.use_fourth_association_jersey
        self.use_fourth_association_same_corner = args.use_fourth_association_same_corner
        self.emb_match_thresh = args.emb_match_thresh
        self.iou_thres = args.iou_thres
        self.jersey_iou_thres = args.jersey_iou_thres
        self.team_thres = args.team_thres
        self.jersey_thres = args.jersey_thres
        self.team_factor = args.team_factor
        self.jersey_factor = args.jersey_factor
        self.team_factor_conf = args.team_factor_conf
        self.jersey_factor_conf = args.jersey_factor_conf

        self.adt_team = args.adt_team
        self.adt_jersey = args.adt_jersey
        self.adt_alpha = args.adt_alpha
        
        SportSORT.FIXED_JERSEY_THRESH = args.fixed_jersey_thresh
        SportSORT.FIXED_TEAM_THRESH = args.fixed_team_thresh
        SportSORT.MAX_NEW_LEN_THRESH = args.max_new_len_thresh

    def update(self, output_results, embedding, team_embs=None):
        
        '''
        output_results : [x1,y1,x2,y2,score] type:ndarray
        embdding : [emb1,emb2,...] dim:512
        '''
        if self.use_first_association_team or self.use_fourth_association_team:
            team_preds = np.ones(len(output_results)) * -1
            team_confs = np.ones(len(output_results)) * -1
            if self.frame_id < self.init_team_frame_thres:
                self.player_classifier.update(team_embs)
            if self.frame_id == self.init_team_frame_thres:
                self.player_classifier.fit()
            if self.frame_id >= self.init_team_frame_thres:
                team_preds, team_confs = self.player_classifier.classify(team_embs)

                for i, team in enumerate(team_preds):
                    team_preds[i] = team_preds[i] if team_confs[i] >= self.team_thres else -1
                team_preds = np.array(team_preds)
                team_confs = np.array(team_confs)
                
                all_bboxes = output_results[:, :4]
                
                for i, bbox in enumerate(all_bboxes):
                    for j in range(i+1, len(all_bboxes)):
                        if iou(bbox, all_bboxes[j]) > self.iou_thres:
                            team_preds[i] = -1
                            team_preds[j] = -1


        self.frame_id += 1
        activated_starcks = []
        refind_stracks = []
        lost_stracks = []
        removed_stracks = []

        if len(output_results):
            if output_results.shape[1] == 7:
                scores = output_results[:,4]
                bboxes = output_results[:, :4]  # x1y1x2y2
            elif output_results.shape[1] == 9:
                scores = output_results[:, 4] * output_results[:, 5]
                bboxes = output_results[:, :4]  # x1y1x2y2
                if self.use_first_association_jersey or self.use_fourth_association_jersey:
                    jerseys = output_results[:, 7]
                    jersey_confs = output_results[:, 8]
            else:
                raise ValueError('Wrong detection size {}'.format(output_results.shape[1]))
            
            if self.use_first_association_jersey or self.use_fourth_association_jersey:
                if self.jersey_iou_thres is not None:
                    for i, bbox in enumerate(bboxes):
                        for j in range(i+1, len(bboxes)):
                            if iou(bbox, bboxes[j]) > self.jersey_iou_thres:
                                jerseys[i] = -1
                                jerseys[j] = -1
            
            # Remove bad detections
            lowest_inds = scores > self.track_low_thresh
            bboxes = bboxes[lowest_inds]
            scores = scores[lowest_inds]
            if self.use_first_association_jersey or self.use_fourth_association_jersey:
                try:
                    jerseys = jerseys[lowest_inds]
                    jersey_confs = jersey_confs[lowest_inds]
                except:
                    import IPython; IPython.embed()
                    time.sleep(0.6)
            
            if self.use_first_association_team or self.use_fourth_association_team:
                team_preds = team_preds[lowest_inds]
                team_confs = team_confs[lowest_inds]



            # Find high threshold detections
            remain_inds = scores > self.args.track_high_thresh
            dets = bboxes[remain_inds]
            scores_keep = scores[remain_inds]
            
            if self.args.with_reid:
                embedding = embedding[lowest_inds]
                features_keep = embedding[remain_inds]

        else:
            bboxes =   []
            scores = []
            dets = []
            scores_keep = []
            features_keep = []      

        if len(dets) > 0:
            '''Detections'''
            if self.args.with_reid:
                detections = [STrack(STrack.tlbr_to_tlwh(tlbr), s, f) for
                              (tlbr, s, f) in zip(dets, scores_keep, features_keep)]
                # CURRENTLY COMMENTED OUT
                for i, det in enumerate(detections):
                    if self.use_first_association_jersey or self.use_fourth_association_jersey:
                        if jerseys[i] != -1 and jersey_confs[i] >= self.jersey_thres:
                            det.det_jersey = jerseys[i]
                            det.jersey_conf = jersey_confs[i]
                    if self.use_first_association_team or self.use_fourth_association_team:
                        if team_preds[i] != -1:
                            det.det_team = team_preds[i]    
                            det.team_conf = team_confs[i]
            
            else:
                detections = [STrack(STrack.tlbr_to_tlwh(tlbr), s) for
                              (tlbr, s) in zip(dets, scores_keep)]
        else:
            detections = []
        

        ''' Add newly detected tracklets to tracked_stracks'''
        unconfirmed = []
        tracked_stracks = []  # type: list[STrack]
        for track in self.tracked_stracks:
            if not track.is_activated:
                unconfirmed.append(track)
            else:
                tracked_stracks.append(track)

        ''' Step 2: First association, with high score detection boxes'''
        strack_pool = joint_stracks(tracked_stracks, self.lost_stracks)

        # Associate with high score detection boxes
        num_iteration = self.num_iteration

        # init_expand_scale = self.init_expand_scale
        # expand_scale_step = self.expand_scale_step
        # check if team_preds is all -1
        # team_detected = True
        if (self.use_first_association_team or self.use_fourth_association_team) or (self.use_first_association_jersey or self.use_fourth_association_jersey):

            team_preds = np.array([det.det_team for det in detections])
            jerseys = np.array([det.det_jersey for det in detections])
            if self.adt_team:
                team_detected_ids = np.where(team_preds != -1)[0]
            else:
                team_detected_ids = np.array([])
            if self.adt_jersey:
                jersey_detected_ids = np.where(jerseys != -1)[0]
            else:
                jersey_detected_ids = np.array([])
            
            team_jersey_detected_ids = np.unique(np.concatenate((team_detected_ids, jersey_detected_ids)))
            scale = len(team_jersey_detected_ids) / len(detections)

            init_expand_scale = 0.7 - self.adt_alpha * scale
            expand_scale_step = 0.1 + self.adt_alpha * scale

        else:
            init_expand_scale = self.init_expand_scale
            expand_scale_step = self.expand_scale_step
                
        # print('init_expand_scale: {}, expand_scale_step: {}, scale: {}'.format(init_expand_scale, expand_scale_step, scale))
        for iteration in range(num_iteration):
            cur_expand_scale = init_expand_scale + expand_scale_step*iteration

        #     if self.use_first_association_team or self.use_fourth_association_team:

        #         proxomity_thresh = np.ones((len(strack_pool), len(detections))) * 0.5
        #         for i, track in enumerate(strack_pool):
        #             flag = False
        #             for j, det in enumerate(detections):
        #                 if track.fix_team != -1 and det.det_team != -1:
        #                     # proxomity_thresh[i, j] = 0.6
        #                     flag = True
        #             proxomity_thresh[i, :] = 0.6 if flag else 0.5


            if self.ensemble_metric == 'harmonic':
                appearance_thresh = self.appearance_thresh if self.use_appearance_thresh else None
                dists = harmonic_dists(strack_pool, detections, cur_expand_scale, 
                                       self.args.with_reid, appearance_thresh)
            else: # default is bot

                dists = bot_dists(strack_pool, detections, cur_expand_scale, 
                    self.proximity_thresh, 
                    # proxomity_thresh,
                    self.appearance_thresh, self.args.with_reid,
                    self.use_first_association_team, self.use_first_association_jersey,
                    self.team_factor, self.jersey_factor, self.team_factor_conf, self.jersey_factor_conf)
            
            matches, u_track, u_detection = matching.linear_assignment(dists, thresh=self.args.match_thresh) 

            for itracked, idet in matches:
                track = strack_pool[itracked]
                det = detections[idet]
                if track.state == TrackState.Tracked:
                    track.update(detections[idet], self.frame_id)
                    activated_starcks.append(track)
                else:
                    track.re_activate(det, self.frame_id, new_id=False)
                    refind_stracks.append(track)
            
            strack_pool = [strack_pool[i] for i in u_track if strack_pool[i].state == TrackState.Tracked]
            detections = [detections[i] for i in u_detection]

        ''' Step 3: Second association, with low score detection boxes'''
        if len(scores):
            inds_high = scores < self.args.track_high_thresh
            inds_low = scores > self.args.track_low_thresh
            inds_second = np.logical_and(inds_low, inds_high)
            dets_second = bboxes[inds_second]
            scores_second = scores[inds_second]
            if self.args.with_reid:
                features_second = embedding[inds_second]
        else:
            dets_second = []
            scores_second = []
            features_second = []

        # association the untrack to the low score detections
        if len(dets_second) > 0:
            '''Detections'''
            if self.args.with_reid:
                detections_second = [STrack(STrack.tlbr_to_tlwh(tlbr), s, f) for
                                    (tlbr, s, f) in zip(dets_second, scores_second, features_second)]
                # CURRENTLY COMMENTED OUT
                # for i, det in enumerate(detections_second):
                #     if jerseys[i] != -1 and jersey_confs[i] >= self.jersey_thres:
                #         det.det_jersey = jerseys[i]
            else:
                detections_second = [STrack(STrack.tlbr_to_tlwh(tlbr), s) for
                                    (tlbr, s) in zip(dets_second, scores_second)]
        else:
            detections_second = []

        r_tracked_stracks = strack_pool

        dists = matching.eiou_distance(r_tracked_stracks, detections_second, expand=0.5)
        matches, u_track, u_detection_second = matching.linear_assignment(dists, thresh=0.5)
        for itracked, idet in matches:
            track = r_tracked_stracks[itracked]
            det = detections_second[idet]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id)
                activated_starcks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)
        for it in u_track:
            track = r_tracked_stracks[it]

            if not track.state == TrackState.Lost:
                track.mark_lost()
                lost_stracks.append(track)

        '''Deal with unconfirmed tracks, usually tracks with only one beginning frame'''
        if self.ensemble_metric == 'harmonic':
            appearance_thresh = self.appearance_thresh if self.use_appearance_thresh else None
            dists = harmonic_dists(unconfirmed, detections, 0.5, with_reid=self.args.with_reid, appearance_thresh=appearance_thresh)
        else: # default is bot
            dists = bot_dists(unconfirmed, detections, cur_expand_scale=0.5,  
                proximity_thresh=self.proximity_thresh, appearance_thresh=self.appearance_thresh, with_reid=self.args.with_reid)

        matches, u_unconfirmed, u_detection = matching.linear_assignment(dists, thresh=0.7)
        
        # indices_to_remove = []
        for itracked, idet in matches:
            unconfirmed[itracked].update(detections[idet], self.frame_id)
            activated_starcks.append(unconfirmed[itracked])
            # indices_to_remove.append(itracked)

        for it in u_unconfirmed:
            track = unconfirmed[it]
            track.mark_removed()
            removed_stracks.append(track)
            
        """ Step 4: Init new stracks"""
        for inew in u_detection:
            track = detections[inew]
            if track.score < self.new_track_thresh:
                continue
            
            track.activate(self.kalman_filter, self.frame_id)
            activated_starcks.append(track)
        
        ######################################
        if self.args.with_reid and self.use_fourth_association:
            if self.use_fourth_association_corner:
                idx_keep_born_track = get_tracks_corner(activated_starcks, 1280, 720, ratio=self.corner_ratio)
                born_track = [(i, track) for i, track in enumerate(activated_starcks) if track.is_new_track == True and i in idx_keep_born_track]
                idx_keep_removed_stracks = get_tracks_corner(self.removed_stracks, 1280, 720, ratio=self.corner_ratio)
                self.removed_stracks = [track for i, track in enumerate(self.removed_stracks) if i in idx_keep_removed_stracks]
            else:
                born_track = [(i, track) for i, track in enumerate(activated_starcks) if track.is_new_track == True]
            dists = matching.embedding_distance_tracks(self.removed_stracks, [track for _, track in born_track])

            if self.use_fourth_association_team:
                dists = add_team_to_dists(dists, self.removed_stracks, [track for _, track in born_track], factor=self.team_factor)
            if self.use_fourth_association_jersey:
                dists = add_det_jersey_to_dists(dists, self.removed_stracks, [track for _, track in born_track], factor=self.jersey_factor)
            if self.use_fourth_association_corner and self.use_fourth_association_same_corner:
                dists = filter_same_corner(dists, self.removed_stracks, [track for _, track in born_track])
            
            matches, u_removed, u_born = matching.linear_assignment(dists, thresh=self.emb_match_thresh)
            
            for iremoved, iborn in matches:
                original_index = born_track[iborn][0]
                activated_starcks[original_index].track_id = self.removed_stracks[iremoved].track_id
                activated_starcks[original_index].is_new_track = False
                self.removed_stracks[iremoved].reborn = True
            
            # print(self.frame_id, len(matches), len(u_born), len(u_removed))
            self.removed_stracks = [track for track in self.removed_stracks if track.reborn == False]
        #######################################

        """ Step 5: Update state"""
        for track in self.lost_stracks:
            if self.frame_id - track.end_frame > self.max_time_lost:
                track.mark_removed()
                removed_stracks.append(track)
        
        self.lost_stracks = [track for track in self.lost_stracks if track.state != TrackState.Removed]

        """ Merge """
        self.tracked_stracks = [t for t in self.tracked_stracks if t.state == TrackState.Tracked]
        self.tracked_stracks = joint_stracks(self.tracked_stracks, activated_starcks)
        self.tracked_stracks = joint_stracks(self.tracked_stracks, refind_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.tracked_stracks)
        self.lost_stracks.extend(lost_stracks)
        
        ## ADD
        # self.lost_stracks.extend(unconfirmed)
        ##

        self.lost_stracks = sub_stracks(self.lost_stracks, self.removed_stracks)
        self.removed_stracks.extend(removed_stracks)
        self.removed_stracks = remove_duplicate_removed_stracks(self.removed_stracks)
        self.tracked_stracks, self.lost_stracks = remove_duplicate_stracks(self.tracked_stracks, self.lost_stracks)
        output_stracks = [track for track in self.tracked_stracks]
        
        # check = remove_duplicate_removed_stracks(output_stracks, check_only=True)
        # if check == False:
        #     import IPython; IPython.embed()
        
        return output_stracks

def add_team_to_dists(dists, tracks, detections, factor=None, factor_conf=False):
    if len(tracks) == 0 or len(detections) == 0:
        return dists
    
    for i, det in enumerate(detections):
        if det.det_team != -1:    
            for j, track in enumerate(tracks):
                if track.fix_team != -1 and det.det_team != track.fix_team:
                    if factor_conf:
                        dists[j, i] = dists[j, i] * (1 + det.team_conf)
                    if factor is None:
                        dists[j, i] = 1
                    else:
                        dists[j, i] = dists[j, i] * factor
    return dists

def add_det_jersey_to_dists(dists, tracks, detections, factor=None, factor_conf=False):
    if len(tracks) == 0 or len(detections) == 0:
        return dists    
    
    for i, det in enumerate(detections):
        if det.det_jersey != -1:
            for j, track in enumerate(tracks):
                # if det.det_jersey in track.jersey_list and track.fix_jersey != -1:
                if track.fix_jersey != -1:
                    if det.det_jersey != track.fix_jersey:
                        if factor_conf:
                            dists[j, i] = dists[j, i] * (1 + det.jersey_conf)
                        if factor is None:
                            dists[j, i] = 1
                        else:
                            dists[j, i] = dists[j, i] * factor
                    # else:
                    #     if factor is None:
                    #         dists[j, i] = 0
                    #     elif factor == -1:
                    #         dists[j, i] = dists[j, i] / (1 + det.jersey_conf)
                    #     else:
                    #         dists[j, i] = dists[j, i] / factor
                
    return dists

def get_tracks_corner(tracks, w, h, ratio=0.1, last_tlbr=True):
    idx_keep = []
    for i, track in enumerate(tracks):
        x1, y1, x2, y2 = track.tlbr if not last_tlbr else track.last_tlbr
        corner_check = []
        corner_check.append(1) if (x1 < w*ratio or x2 < w*ratio) else corner_check.append(0)
        corner_check.append(1) if (y1 < h*ratio or y2 < h*ratio) else corner_check.append(0)
        corner_check.append(1) if (x1 > w*(1-ratio) or x2 > w*(1-ratio)) else corner_check.append(0)
        corner_check.append(1) if (y1 > h*(1-ratio) or y2 > h*(1-ratio)) else corner_check.append(0)
        
        if sum(corner_check) >= 1:
            idx_keep.append(i)
        tracks[i].corner_check = corner_check    

    return idx_keep

def filter_same_corner(dists, tracks, detections):
    for i, det in enumerate(detections):
        for j, track in enumerate(tracks):
            sum_check = 0
            for k in range(len(track.corner_check)):
                if det.corner_check[k] == track.corner_check[k]:
                    sum_check += 1
            if sum_check == 0:
                dists[j, i] = 1
    return dists

def bot_dists(tracks, detections, cur_expand_scale, 
              proximity_thresh, appearance_thresh, 
              with_reid=True, use_team=False, 
              use_jersey=False, team_factor=None, 
              jersey_factor=None, team_factor_conf=False, 
              jersey_factor_conf=False, embedding_scale=2.0):
    

    ious_dists = matching.eiou_distance(tracks, detections, cur_expand_scale)
    # ious_dists_mask = (ious_dists > proximity_thresh)
    ious_dists_mask = ious_dists > proximity_thresh
    
    if use_team:
        ious_dists = add_team_to_dists(ious_dists, tracks, detections, team_factor, team_factor_conf)
    if use_jersey:
        ious_dists = add_det_jersey_to_dists(ious_dists, tracks, detections, jersey_factor, jersey_factor_conf)
    
    if with_reid:
        emb_dists = matching.embedding_distance(tracks, detections) / embedding_scale
        emb_dists[emb_dists > appearance_thresh] = 1.0
        emb_dists[ious_dists_mask] = 1.0
        
        if use_team:
            emb_dists = add_team_to_dists(emb_dists, tracks, detections, team_factor, team_factor_conf)
        if use_jersey:
            emb_dists = add_det_jersey_to_dists(emb_dists, tracks, detections, jersey_factor, jersey_factor_conf)
        
        dists = np.minimum(ious_dists, emb_dists)
    else:
        dists = ious_dists
    return dists

def harmonic_dists(tracks, detections, cur_expand_scale, 
                   with_reid=True, appearance_thresh=None,
                   embedding_scale=2.0):
    ious_dists = matching.eiou_distance(tracks, detections, cur_expand_scale)
    if with_reid:
        emb_dists = matching.embedding_distance(tracks, detections) / embedding_scale
        if appearance_thresh is not None:
            emb_dists[emb_dists > appearance_thresh] = 1.0
        dists = 2 * ious_dists * emb_dists / (ious_dists + emb_dists)
    else:
        dists = ious_dists
    return dists

def joint_stracks(tlista, tlistb):
    exists = {}
    res = []
    for t in tlista:
        exists[t.track_id] = 1
        res.append(t)
    for t in tlistb:
        tid = t.track_id
        if not exists.get(tid, 0):
            exists[tid] = 1
            res.append(t)
    return res

def sub_stracks(tlista, tlistb):
    stracks = {}
    for t in tlista:
        stracks[t.track_id] = t
    for t in tlistb:
        tid = t.track_id
        if stracks.get(tid, 0):
            del stracks[tid]
    return list(stracks.values())

def remove_duplicate_removed_stracks(removed_stracks, check_only=False):
    # check equal track_id
    res = []
    for i, t in enumerate(removed_stracks):
        if t.track_id not in [x.track_id for x in removed_stracks[i+1:]]:
            res.append(t)
        else:
            if check_only:
                return False
    if check_only:
        return True
    return res

def remove_duplicate_stracks(stracksa, stracksb):
    pdist = matching.iou_distance(stracksa, stracksb)
    pairs = np.where(pdist < 0.15)
    dupa, dupb = list(), list()
    for p, q in zip(*pairs):
        timep = stracksa[p].frame_id - stracksa[p].start_frame
        timeq = stracksb[q].frame_id - stracksb[q].start_frame
        if timep > timeq:
            dupb.append(q)
        else:
            dupa.append(p)
    resa = [t for i, t in enumerate(stracksa) if not i in dupa]
    resb = [t for i, t in enumerate(stracksb) if not i in dupb]
    return resa, resb

def iou(bbox1, bbox2):
    
    # Unpack the bounding boxes
    x1_min, y1_min, x1_max, y1_max = bbox1
    x2_min, y2_min, x2_max, y2_max = bbox2

    # Calculate the coordinates of the intersection rectangle
    inter_x1 = max(x1_min, x2_min)
    inter_y1 = max(y1_min, y2_min)
    inter_x2 = min(x1_max, x2_max)
    inter_y2 = min(y1_max, y2_max)

    # Calculate the area of the intersection rectangle
    inter_width = max(0, inter_x2 - inter_x1)
    inter_height = max(0, inter_y2 - inter_y1)
    inter_area = inter_width * inter_height

    # Calculate the area of each bounding box
    area1 = (x1_max - x1_min) * (y1_max - y1_min)
    area2 = (x2_max - x2_min) * (y2_max - y2_min)

    # Calculate the union area
    union_area = area1 + area2 - inter_area

    # Calculate the IoU
    iou_value = inter_area / union_area if union_area > 0 else 0

    return iou_value

# idsw: /mnt/banana/student/thuyntt/Deep-EIoU/evaluation/TrackEval/output/tracker_to_eval/idsw_train
# eval full video: python ./scripts/run_mot_challenge.py --BENCHMARK sportsmot --SPLIT_TO_EVAL train --METRICS HOTA CLEAR Identity VACE --USE_PARALLEL False --PRINT_CONFIG True --GT_FOLDER ./data/ref --TRACKERS_FOLDER ./data/res --OUTPUT_FOLDER ./output/
# eval 1 video: python ./scripts/run_mot_challenge.py --BENCHMARK sportsmot --SPLIT_TO_EVAL train --METRICS HOTA CLEAR Identity VACE --USE_PARALLEL False --PRINT_CONFIG True --GT_FOLDER ./data/test_jersey/ref --TRACKERS_FOLDER ./data/test_jersey/res --OUTPUT_FOLDER ./output/
# eval infer 2 full video: python ./scripts/run_mot_challenge2.py --BENCHMARK sportsmot --SPLIT_TO_EVAL val --METRICS HOTA CLEAR2 Identity VACE --USE_PARALLEL False --PRINT_CONFIG True --GT_FOLDER ./data/ref --TRACKERS_FOLDER ./data/thuy_res --OUTPUT_FOLDER ./output/
# A 4
# B 3

# A 3
# B 4

# A 3 10
# B 4 11

# Track đi lang thang ghép nhầm
# Ghép IoU
# Không mở rộng được thuật toán

# corner
# emb: tăng
# team: giảm
# jersey: giảm

# python tools/infer_2.py --iou_thres 0.4 --init_expand_scale 0.4 --expand_scale_step 0.4 --proximity_thresh 0.6 --team_thres 0.6 --team_factor 2 --jersey_factor 2 --use_first_association_team --use_first_association_jersey 

# python tools/infer_3.py --team_factor 2 --jersey_factor 2 --use_first_association_team --use_first_association_jersey --track_low_thresh 0.4 --iou_thres 0.4 --proximity_thresh 0.6 --cache_embedding_name embedding_sports --use_fourth_association_team --use_fourth_association_jersey --use_fourth_association --cache_jersey_name jersey_num_hockey