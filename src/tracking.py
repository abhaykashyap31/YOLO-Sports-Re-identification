import cv2
import numpy as np
from src.features import extract_color_histogram, extract_hog_features

def iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    return interArea / float(boxAArea + boxBArea - interArea + 1e-6)

def compute_similarity(f1, f2):
    color_sim = 1 - cv2.compareHist(f1['color'], f2['color'], cv2.HISTCMP_BHATTACHARYYA)
    hog1, hog2 = f1['hog'], f2['hog']
    if np.linalg.norm(hog1) == 0 or np.linalg.norm(hog2) == 0:
        hog_sim = 0
    else:
        hog_sim = np.dot(hog1, hog2) / (np.linalg.norm(hog1) * np.linalg.norm(hog2) + 1e-8)
    return 0.6 * color_sim + 0.4 * hog_sim

class PlayerTrack:
    def __init__(self, id, bbox, features, frame_idx):
        self.id = id
        self.bbox = bbox
        self.features = features
        self.last_seen = frame_idx
        self.missed = 0

class PlayerTracker:
    def __init__(self, iou_thresh=0.3, sim_thresh=0.5, max_missed=30):
        self.next_id = 0
        self.tracks = []
        self.iou_thresh = iou_thresh
        self.sim_thresh = sim_thresh
        self.max_missed = max_missed
    def update(self, detections, frame, frame_idx):
        det_features = []
        for det in detections:
            color = extract_color_histogram(frame, det['bbox'])
            hog = extract_hog_features(frame, det['bbox'])
            det_features.append({'color': color, 'hog': hog})
        assigned = set()
        for track in self.tracks:
            best_idx, best_score = -1, -1
            for i, det in enumerate(detections):
                if i in assigned:
                    continue
                iou_score = iou(track.bbox, det['bbox'])
                sim_score = compute_similarity(track.features, det_features[i])
                score = 0.5 * iou_score + 0.5 * sim_score
                if score > best_score:
                    best_score = score
                    best_idx = i
            if best_score > self.iou_thresh:
                track.bbox = detections[best_idx]['bbox']
                track.features = det_features[best_idx]
                track.last_seen = frame_idx
                track.missed = 0
                assigned.add(best_idx)
            else:
                track.missed += 1
        for i, det in enumerate(detections):
            if i not in assigned:
                self.tracks.append(PlayerTrack(self.next_id, det['bbox'], det_features[i], frame_idx))
                self.next_id += 1
        self.tracks = [t for t in self.tracks if t.missed <= self.max_missed]
        id_map = {}
        for i, det in enumerate(detections):
            for t in self.tracks:
                if t.bbox == det['bbox'] and t.last_seen == frame_idx:
                    id_map[i] = t.id
        return id_map 