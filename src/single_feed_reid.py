# Assignment Task 2: Single-Feed Player Re-Identification
# Implements player re-identification in a single video feed as described in instructions.txt
# Usage: python src/single_feed_reid.py [--input ...] [--model ...] [--download]

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import cv2
from src.detector import YOLOv11Detector
from src.utils import read_video, draw_boxes, download_from_info
import numpy as np
from deep_sort_realtime.deepsort_tracker import DeepSort

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='videos/15sec_input_720.mp4')
    parser.add_argument('--model', default='models/model.pt')
    parser.add_argument('--download', action='store_true', help='Download all required files from Info.txt')
    args = parser.parse_args()

    if args.download:
        download_from_info()
        sys.exit(0)

    detector = YOLOv11Detector(args.model)
    frames = read_video(args.input)

    # Initialize DeepSORT tracker
    tracker = DeepSort(max_age=30, n_init=2, nms_max_overlap=1.0, embedder="mobilenet", half=True)

    # Prepare output directory and video writer
    os.makedirs('output', exist_ok=True)
    if frames:
        height, width = frames[0].shape[:2]
        writer = cv2.VideoWriter('output/single_feed_output.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 25, (width, height))
    else:
        writer = None

    for idx, frame in enumerate(frames):
        detections = detector.detect(frame)
        # Prepare detections for DeepSORT: [[x1, y1, x2, y2, confidence], ...]
        dets_for_sort = []
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            conf = det['conf']
            dets_for_sort.append([x1, y1, x2 - x1, y2 - y1, conf])
        # DeepSORT expects [x, y, w, h, conf]
        tracks = tracker.update_tracks(dets_for_sort, frame=frame)
        # Prepare ID map for drawing
        id_map = {}
        for i, det in enumerate(detections):
            # Find the track that matches this detection (by IOU)
            best_iou = 0
            best_id = None
            for track in tracks:
                if not track.is_confirmed():
                    continue
                t_x, t_y, t_w, t_h = track.to_ltrb()
                iou = (
                    max(0, min(x2, t_x + t_w) - max(x1, t_x)) *
                    max(0, min(y2, t_y + t_h) - max(y1, t_y))
                ) / float((x2 - x1) * (y2 - y1) + t_w * t_h - max(0, min(x2, t_x + t_w) - max(x1, t_x)) * max(0, min(y2, t_y + t_h) - max(y1, t_y)) + 1e-6)
                if iou > best_iou and iou > 0.3:
                    best_iou = iou
                    best_id = track.track_id
            if best_id is not None:
                id_map[i] = best_id
            else:
                id_map[i] = -1  # Unmatched
        out_frame = draw_boxes(frame.copy(), detections, id_map)
        if writer is not None:
            writer.write(out_frame)
    if writer is not None:
        writer.release()

if __name__ == '__main__':
    main() 