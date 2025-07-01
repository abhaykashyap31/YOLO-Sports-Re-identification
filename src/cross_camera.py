import os
import cv2
import numpy as np
from src.detector import PlayerDetector
from src.tracking import PlayerTracker, compute_similarity
from src.utils import read_video, draw_boxes
from scipy.optimize import linear_sum_assignment

def run_cross_camera(broadcast_path, tacticam_path, model_path, out_broadcast, out_tacticam):
    print('Loading videos...')
    broadcast_frames = read_video(broadcast_path)
    tacticam_frames = read_video(tacticam_path)
    print(f'Broadcast frames: {len(broadcast_frames)}, Tacticam frames: {len(tacticam_frames)}')
    detector = PlayerDetector(model_path)
    tracker_b = PlayerTracker()
    tracker_t = PlayerTracker()
    os.makedirs(os.path.dirname(out_broadcast), exist_ok=True)
    os.makedirs(os.path.dirname(out_tacticam), exist_ok=True)
    if broadcast_frames:
        height, width = broadcast_frames[0].shape[:2]
        b_writer = cv2.VideoWriter(out_broadcast, cv2.VideoWriter_fourcc(*'mp4v'), 25, (width, height))
    else:
        b_writer = None
    if tacticam_frames:
        height, width = tacticam_frames[0].shape[:2]
        t_writer = cv2.VideoWriter(out_tacticam, cv2.VideoWriter_fourcc(*'mp4v'), 25, (width, height))
    else:
        t_writer = None
    global_id_counter = 0
    b_to_global = {}
    t_to_global = {}
    for idx, (b_frame, t_frame) in enumerate(zip(broadcast_frames, tacticam_frames)):
        b_dets = detector.detect_players(b_frame)
        t_dets = detector.detect_players(t_frame)
        b_id_map = tracker_b.update(b_dets, b_frame, idx)
        t_id_map = tracker_t.update(t_dets, t_frame, idx)
        # Extract features for all tracks
        b_tracks = tracker_b.tracks
        t_tracks = tracker_t.tracks
        if b_tracks and t_tracks:
            # Build cost matrix using feature similarity
            cost_matrix = np.zeros((len(t_tracks), len(b_tracks)))
            for i, t_tr in enumerate(t_tracks):
                for j, b_tr in enumerate(b_tracks):
                    cost_matrix[i, j] = -compute_similarity(t_tr.features, b_tr.features)  # negative for Hungarian
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            for t_idx, b_idx in zip(row_ind, col_ind):
                if -cost_matrix[t_idx, b_idx] > 0.5:  # similarity threshold
                    t_id = t_tracks[t_idx].id
                    b_id = b_tracks[b_idx].id
                    if b_id in b_to_global:
                        global_id = b_to_global[b_id]
                        t_to_global[t_id] = global_id
                    elif t_id in t_to_global:
                        global_id = t_to_global[t_id]
                        b_to_global[b_id] = global_id
                    else:
                        global_id = global_id_counter
                        global_id_counter += 1
                        b_to_global[b_id] = global_id
                        t_to_global[t_id] = global_id
        # Prepare ID maps for drawing
        b_draw_map = {i: b_to_global.get(b_id_map.get(i, -1), -1) for i in range(len(b_dets))}
        t_draw_map = {i: t_to_global.get(t_id_map.get(i, -1), -1) for i in range(len(t_dets))}
        b_out = draw_boxes(b_frame.copy(), b_dets, b_draw_map)
        t_out = draw_boxes(t_frame.copy(), t_dets, t_draw_map)
        if b_writer is not None:
            b_writer.write(b_out)
        if t_writer is not None:
            t_writer.write(t_out)
        if idx % 10 == 0:
            print(f'Processed frame {idx+1}/{min(len(broadcast_frames), len(tacticam_frames))}')
    if b_writer is not None:
        b_writer.release()
    if t_writer is not None:
        t_writer.release()
    print(f'Output saved to {out_broadcast} and {out_tacticam}') 