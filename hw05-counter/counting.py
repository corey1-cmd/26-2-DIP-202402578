"""HW5 object counting pipeline (notebook -> reusable functions)."""
import numpy as np
import cv2
import scipy.ndimage as nd


def preprocess(gray, crop_top=0, median_k=3, alpha=0.8):
    """Crop -> median filter -> Sobel-based edge enhancement (HW5 step 1-2)."""
    f = gray[crop_top:, :] if crop_top > 0 else gray
    f_den = cv2.medianBlur(f, median_k)

    fd = f_den.astype(np.float64)
    gx = cv2.Sobel(fd, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(fd, cv2.CV_64F, 0, 1, ksize=3)
    grad = np.sqrt(gx ** 2 + gy ** 2)
    grad_norm = grad / (grad.max() + 1e-8)
    blurred = cv2.GaussianBlur(fd, (3, 3), 0)
    f_enh = fd + alpha * grad_norm * (fd - blurred)
    f_enh = np.clip(f_enh, 0, 255).astype(np.uint8)
    return f, f_den, f_enh


def make_mask(f_den, th_mode="manual", th_val=128, bright_objects=True,
              k_size=5, erosion_iter=2, dilation_iter=1):
    """Threshold -> erosion -> dilation (HW5 step 3)."""
    if th_mode == "otsu":
        th_val, _ = cv2.threshold(f_den, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    th_val = int(th_val)
    mask = (f_den > th_val) if bright_objects else (f_den <= th_val)
    mask = mask.astype(np.uint8)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size))
    eroded = cv2.erode(mask, kernel, iterations=erosion_iter)
    dilated = cv2.dilate(eroded, kernel, iterations=dilation_iter)
    return th_val, mask, eroded, dilated


def label_components(binary, min_area=0):
    """Connected components + pixel counts (HW5 step 4).

    Returns relabeled map (1..N), areas array (len N), centroids (N,2 as y,x).
    Note: counts ALL N labels (the notebook loop `range(1, nums)` skipped the last one).
    """
    labels, n = nd.label(binary)
    if n == 0:
        return labels, np.array([], dtype=int), np.zeros((0, 2))
    areas = np.bincount(labels.ravel(), minlength=n + 1)[1:]
    keep = np.where(areas >= min_area)[0] + 1  # old label ids to keep

    new_labels = np.zeros_like(labels)
    for new_id, old_id in enumerate(keep, start=1):
        new_labels[labels == old_id] = new_id
    areas = areas[keep - 1]
    if len(keep) == 0:
        return new_labels, areas, np.zeros((0, 2))
    cents = np.array(nd.center_of_mass(new_labels > 0, new_labels,
                                       range(1, len(keep) + 1)))
    return new_labels, areas, cents


def colorize(labels):
    """Random-color label map as RGB uint8 (background black)."""
    n = labels.max()
    rng = np.random.RandomState(0)
    palette = np.vstack([[0, 0, 0], rng.randint(60, 255, size=(n, 3))]).astype(np.uint8)
    return palette[labels]


def overlay_numbers(gray, labels, cents):
    """Original image with object contours and index numbers."""
    rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    edges = (labels > 0).astype(np.uint8)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(rgb, contours, -1, (255, 60, 60), 1)
    scale = max(0.35, min(rgb.shape[:2]) / 1200)
    for i, (cy, cx) in enumerate(cents, start=1):
        cv2.putText(rgb, str(i), (int(cx), int(cy)), cv2.FONT_HERSHEY_SIMPLEX,
                    scale, (0, 255, 0), 1, cv2.LINE_AA)
    return rgb
