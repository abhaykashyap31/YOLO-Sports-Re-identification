import cv2
import numpy as np

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
    # Resize patch to (64, 128) as required by HOGDescriptor
    try:
        patch_resized = cv2.resize(patch, (64, 128))
    except Exception:
        return np.zeros((3780,))
    gray = cv2.cvtColor(patch_resized, cv2.COLOR_BGR2GRAY)
    hog = cv2.HOGDescriptor()
    features = hog.compute(gray)
    return features.flatten() if features is not None else np.zeros((3780,)) 