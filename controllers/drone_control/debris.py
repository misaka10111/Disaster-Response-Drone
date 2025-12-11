import cv2
import numpy as np
from scipy.ndimage import gaussian_filter


def detect_debris_unsupervised(image_rgb):
    """
    Unsupervised rubble segmentation for disaster scenes.
    Returns a binary mask (0/1) of debris regions.
    """
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (11, 11), 0)

    # Texture contrast
    entropy = cv2.absdiff(gray, blur).astype(float)
    entropy = (entropy - entropy.min()) / (entropy.max() - entropy.min() + 1e-6)
    m1 = (entropy > 0.25).astype(np.uint8)

    # Edge density
    edges = cv2.Canny(gray, 50, 150)
    edges_s = gaussian_filter(edges.astype(float), sigma=2)
    edges_s = (edges_s - edges_s.min()) / (edges_s.max() - edges_s.min() + 1e-6)
    m2 = (edges_s > 0.20).astype(np.uint8)

    # Local variance
    local_mean = cv2.GaussianBlur(gray.astype(float), (15, 15), 0)
    local_mean2 = cv2.GaussianBlur(gray.astype(float)**2, (15, 15), 0)
    local_std = np.sqrt(np.abs(local_mean2 - local_mean**2))
    local_std = (local_std - local_std.min()) / (local_std.max() - local_std.min() + 1e-6)
    m3 = (local_std > 0.3).astype(np.uint8)

    combined = (m1 | m2 | m3)
    combined = cv2.dilate(combined, np.ones((25, 25), np.uint8))

    return combined
