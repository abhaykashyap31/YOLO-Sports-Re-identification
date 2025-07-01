import cv2
import numpy as np
import os
import gdown
import sys
import argparse
from scipy.optimize import linear_sum_assignment
import shutil

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def read_video(path):
    print(f"Trying to open video: {path}")
    cap = cv2.VideoCapture(path)
    frames = []
    if cap.isOpened():
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        cap.release()
        if len(frames) > 0:
            print(f"Read {len(frames)} frames from {path} using OpenCV.")
            return frames
        else:
            print("OpenCV opened the file but found 0 frames, trying imageio fallback...")
    else:
        print("OpenCV failed to open the file, trying imageio fallback...")
    # Fallback: use imageio
    try:
        import imageio.v3 as iio
        frames = []
        for frame in iio.imiter(path):
            frames.append(np.array(frame))
        print(f"Read {len(frames)} frames from {path} using imageio.")
        return frames
    except Exception as e:
        print(f"imageio fallback failed: {e}")
        return []

def draw_boxes(frame, detections, id_map=None):
    for i, det in enumerate(detections):
        x1, y1, x2, y2 = det['bbox']
        label = str(det.get('class', ''))
        if id_map is not None:
            label = f"ID {id_map.get(i, i)}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
        cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
    return frame

# --- Download utilities ---
def parse_info_txt(info_path='Info.txt'):
    files = {}
    with open(info_path, 'r') as f:
        for line in f:
            if '=' in line:
                key, url = line.strip().split('=', 1)
                files[key.strip()] = url.strip()
    return files

def safe_download_file(url, output_path, max_retries=3):
    """
    Safely download a single file with multiple fallback strategies
    """
    for attempt in range(max_retries):
        try:
            # Try with fuzzy=True first
            gdown.download(url, output_path, quiet=False, use_cookies=False, fuzzy=True)
            return True
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            
            if attempt < max_retries - 1:
                # Try different approaches
                try:
                    # Extract file ID and try direct download
                    if 'id=' in url:
                        file_id = url.split('id=')[-1].split('&')[0]
                        gdown.download(id=file_id, output=output_path, quiet=False, use_cookies=False, fuzzy=True)
                        return True
                except Exception as e2:
                    print(f"Direct ID download failed: {e2}")
                    continue
            else:
                print(f"Failed to download {url} after {max_retries} attempts")
                return False
    return False

def safe_download_folder(folder_url, output_dir, max_retries=2):
    """
    Safely download folder with fallback to individual file downloads
    """
    for attempt in range(max_retries):
        try:
            print(f"Attempting folder download (attempt {attempt + 1})...")
            gdown.download_folder(folder_url, output=output_dir, quiet=False, use_cookies=False)
            return True
        except Exception as e:
            print(f"Folder download attempt {attempt + 1} failed: {e}")
            
            if attempt < max_retries - 1:
                print("Retrying folder download...")
                continue
            else:
                print("Folder download failed. Falling back to individual file downloads...")
                return False
    return False

def download_all_files(info_path='Info.txt', video_dir='videos', model_dir='models', data_dir='Data'):
    """
    Download all files (videos and model) listed in Info.txt to the correct folders.
    If a Google Drive folder link is present, download the whole folder to 'Data/'.
    After download, copy model and videos to the expected locations.
    """
    files = parse_info_txt(info_path)
    # If a folder key is present, try to use gdown.download_folder
    folder_url = files.get('folder')
    if folder_url:
        if os.path.exists(data_dir):
            print(f"{data_dir} directory already exists, skipping folder download.")
        else:
            print(f"Downloading all files from folder: {folder_url} to {data_dir}/ ...")
            if safe_download_folder(folder_url, data_dir):
                print("Folder download completed successfully!")
            else:
                print("Folder download failed. Attempting individual file downloads...")
    # Create directories
    os.makedirs(video_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs('output', exist_ok=True)
    # Copy model and videos from Data/ to expected locations
    for fname in os.listdir(data_dir):
        src = os.path.join(data_dir, fname)
        if fname.endswith('.pt'):
            dst = os.path.join(model_dir, 'model.pt')
            print(f"Copying {src} to {dst}")
            shutil.copy2(src, dst)
        elif fname.endswith('.mp4'):
            dst = os.path.join(video_dir, fname)
            print(f"Copying {src} to {dst}")
            shutil.copy2(src, dst)
    # Download individual files if needed (fallback)
    for key, url in files.items():
        if key == 'folder':
            continue
        # Determine output path
        if key.endswith('.mp4'):
            out_path = os.path.join(video_dir, key)
        elif key.endswith('.pt'):
            out_path = os.path.join(model_dir, 'model.pt')
        else:
            out_path = os.path.join(data_dir, key)
        if not os.path.exists(out_path):
            print(f"Downloading {key} to {out_path} ...")
            if not safe_download_file(url, out_path):
                print(f"WARNING: Failed to download {key}. You may need to download it manually.")
        else:
            print(f"{out_path} already exists, skipping download.")

# For backward compatibility
def download_from_info(info_path='Info.txt', video_dir='videos', model_dir='models', data_dir='Data'):
    files = parse_info_txt(info_path)
    os.makedirs(video_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)
    for key, url in files.items():
        if key.endswith('.mp4'):
            out_path = os.path.join(video_dir, key)
        elif key.endswith('.pt'):
            out_path = os.path.join(model_dir, 'model.pt')
        else:
            out_path = os.path.join(data_dir, key)
        if not os.path.exists(out_path):
            print(f"Downloading {key} to {out_path} ...")
            gdown.download(url, out_path, quiet=False, use_cookies=False, fuzzy=True)
        else:
            print(f"{out_path} already exists, skipping download.")

def iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    
    iou = interArea / float(boxAArea + boxBArea - interArea + 1e-6)
    return iou

def assign_ids(detections, prev_tracks, iou_thresh=0.5):
    assigned = {}
    used = set()
    
    for i, det in enumerate(detections):
        best_iou = 0
        best_id = None
        
        for tid, tbox in prev_tracks.items():
            score = iou(det['bbox'], tbox)
            if score > best_iou and score > iou_thresh and tid not in used:
                best_iou = score
                best_id = tid
        
        if best_id is not None:
            assigned[i] = best_id
            used.add(best_id)
        else:
            assigned[i] = max(prev_tracks.keys(), default=-1) + 1 + len(assigned)
    
    return assigned

def extract_histogram(frame, bbox):
    x1, y1, x2, y2 = bbox
    patch = frame[y1:y2, x1:x2]
    
    if patch.size == 0:
        return np.zeros((32,))
    
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [8, 4], [0, 180, 0, 256])
    hist = cv2.normalize(hist, hist).flatten()
    return hist

def match_players(b_dets, t_dets, b_frame, t_frame):
    if not b_dets or not t_dets:
        return {}, {}
    
    b_hists = [extract_histogram(b_frame, det['bbox']) for det in b_dets]
    t_hists = [extract_histogram(t_frame, det['bbox']) for det in t_dets]
    
    cost_matrix = np.zeros((len(t_hists), len(b_hists)))
    
    for i, t_hist in enumerate(t_hists):
        for j, b_hist in enumerate(b_hists):
            cost_matrix[i, j] = cv2.compareHist(t_hist.astype('float32'), b_hist.astype('float32'), cv2.HISTCMP_BHATTACHARYYA)
    
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    
    t_to_b = {}
    b_to_t = {}
    
    for t_idx, b_idx in zip(row_ind, col_ind):
        if cost_matrix[t_idx, b_idx] < 0.5:  # threshold for similarity
            t_to_b[t_idx] = b_idx
            b_to_t[b_idx] = t_idx
    
    return t_to_b, b_to_t

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--download', action='store_true', help='Download all required files from Info.txt')
    args = parser.parse_args()
    
    if args.download:
        try:
            download_all_files()
            print('Download complete.')
        except Exception as e:
            print(f'Download failed with error: {e}')
            print('You may need to download the files manually from the Google Drive links.')
    else:
        print('No action specified. Use --download to fetch all required files.')
