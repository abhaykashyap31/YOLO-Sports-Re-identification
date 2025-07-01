import sys
import os
import argparse
import cv2
import numpy as np
from utils import download_from_info, read_video, draw_boxes
from single_feed import run_single_feed
from cross_camera import run_cross_camera

# -------------------- Detection --------------------
from ultralytics import YOLO
class PlayerDetector:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
    def detect_players(self, frame):
        results = self.model(frame)
        players = []
        for r in results:
            for box in r.boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                xyxy = box.xyxy[0]
                if not (hasattr(xyxy, '__iter__') and len(xyxy) == 4):
                    continue
                x1, y1, x2, y2 = map(int, xyxy)
                players.append({'bbox': [x1, y1, x2, y2], 'conf': conf, 'class': cls})
        return players

# -------------------- Feature Extraction --------------------
def extract_color_histogram(frame, bbox):
    x1, y1, x2, y2 = bbox
    patch = frame[y1:y2, x1:x2]
    if patch.size == 0:
        return np.zeros((256,))
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [16, 16], [0, 180, 0, 256])
    return cv2.normalize(hist, hist).flatten()

def extract_hog_features(frame, bbox):
    x1, y1, x2, y2 = bbox
    patch = frame[y1:y2, x1:x2]
    if patch.size == 0:
        return np.zeros((3780,))
    gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    hog = cv2.HOGDescriptor()
    features = hog.compute(gray)
    return features.flatten() if features is not None else np.zeros((3780,))

# -------------------- Similarity --------------------
def compute_similarity(f1, f2):
    # Color histogram: Bhattacharyya, HOG: cosine
    color_sim = 1 - cv2.compareHist(f1['color'], f2['color'], cv2.HISTCMP_BHATTACHARYYA)
    hog1, hog2 = f1['hog'], f2['hog']
    if np.linalg.norm(hog1) == 0 or np.linalg.norm(hog2) == 0:
        hog_sim = 0
    else:
        hog_sim = np.dot(hog1, hog2) / (np.linalg.norm(hog1) * np.linalg.norm(hog2) + 1e-8)
    return 0.6 * color_sim + 0.4 * hog_sim

# -------------------- Tracking & Re-ID --------------------
def iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    return interArea / float(boxAArea + boxBArea - interArea + 1e-6)

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
        # Extract features for detections
        det_features = []
        for det in detections:
            color = extract_color_histogram(frame, det['bbox'])
            hog = extract_hog_features(frame, det['bbox'])
            det_features.append({'color': color, 'hog': hog})
        assigned = set()
        # Match to existing tracks
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
                # Update track
                track.bbox = detections[best_idx]['bbox']
                track.features = det_features[best_idx]
                track.last_seen = frame_idx
                track.missed = 0
                assigned.add(best_idx)
            else:
                track.missed += 1
        # Add new tracks
        for i, det in enumerate(detections):
            if i not in assigned:
                self.tracks.append(PlayerTrack(self.next_id, det['bbox'], det_features[i], frame_idx))
                self.next_id += 1
        # Remove old tracks
        self.tracks = [t for t in self.tracks if t.missed <= self.max_missed]
        # Prepare ID map for drawing
        id_map = {}
        for i, det in enumerate(detections):
            for t in self.tracks:
                if t.bbox == det['bbox'] and t.last_seen == frame_idx:
                    id_map[i] = t.id
        return id_map

# -------------------- Main Pipeline --------------------
def run_pipeline(input_path, model_path, output_path):
    print('Loading video...')
    frames = read_video(input_path)
    print(f'Total frames: {len(frames)}')
    detector = PlayerDetector(model_path)
    tracker = PlayerTracker()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    if frames:
        height, width = frames[0].shape[:2]
        writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), 25, (width, height))
    else:
        writer = None
    for idx, frame in enumerate(frames):
        detections = detector.detect_players(frame)
        id_map = tracker.update(detections, frame, idx)
        out_frame = draw_boxes(frame.copy(), detections, id_map)
        if writer is not None:
            writer.write(out_frame)
        if idx % 10 == 0:
            print(f'Processed frame {idx+1}/{len(frames)}')
    if writer is not None:
        writer.release()
    print(f'Output saved to {output_path}')

# -------------------- CLI --------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--download', action='store_true', help='Download all required files from Info.txt')
    parser.add_argument('--task', type=int, choices=[1,2], help='Task 1: Cross-camera mapping, Task 2: Single-feed re-ID')
    parser.add_argument('--input', type=str, help='Input video path (for task 2)')
    parser.add_argument('--model', type=str, default='models/model.pt', help='YOLOv11 model path')
    parser.add_argument('--output', type=str, help='Output video path (for task 2)')
    parser.add_argument('--broadcast', type=str, default='videos/broadcast.mp4', help='Broadcast video path (for task 1)')
    parser.add_argument('--tacticam', type=str, default='videos/tacticam.mp4', help='Tacticam video path (for task 1)')
    parser.add_argument('--out_broadcast', type=str, default='output/cross_camera_broadcast_output.mp4', help='Output path for broadcast (task 1)')
    parser.add_argument('--out_tacticam', type=str, default='output/cross_camera_tacticam_output.mp4', help='Output path for tacticam (task 1)')
    args = parser.parse_args()
    if args.download:
        download_from_info()
        print('Download complete.')
        sys.exit(0)
    if args.task == 2:
        input_path = args.input if args.input else 'videos/15sec_input_720.mp4'
        output_path = args.output if args.output else 'output/single_feed_output.mp4'
        print(f'Using input: {input_path}')
        print(f'Using output: {output_path}')
        run_single_feed(input_path, args.model, output_path)
        sys.exit(0)
    if args.task == 1:
        run_cross_camera(args.broadcast, args.tacticam, args.model, args.out_broadcast, args.out_tacticam)
        sys.exit(0)
    print('No action specified. Use --download or --task 1/2.')

if __name__ == '__main__':
    main() 