import os
import cv2
from src.detector import PlayerDetector
from src.tracking import PlayerTracker
from src.utils import read_video, draw_boxes

def run_single_feed(input_path, model_path, output_path):
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