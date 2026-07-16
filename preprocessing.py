import cv2
import numpy as np

COCO_KEYPOINT_CONNECTIONS = [
    (0, 1), (0, 2), (1, 3), (2, 4), (5, 6),
    (5, 7), (7, 9), (6, 8), (8, 10), (11, 12),
    (5, 11), (6, 12), (11, 13), (13, 15),
    (12, 14), (14, 16),
]

DETECTOR_CONF = 0.25
DETECTOR_IMGSZ = 640
POSE_CONF = 0.10
POSE_IMGSZ = 640
POSE_IOU = 0.50
POSE_MAX_DET = 5

TRACK_MIN_IOU = 0.05
TRACK_MAX_CENTER_DISTANCE = 0.20
TRACK_IOU_WEIGHT = 0.65
TRACK_CENTER_WEIGHT = 0.35
BBOX_SMOOTH_WINDOW = 5
MAX_MISSING_BOX = 5
YOLO_PAD_X = 0.45
YOLO_PAD_Y = 0.45

NUM_KEYPOINT = 17
NUM_FEATURE_PER_KEYPOINT = 3
FEATURE_DIM = NUM_KEYPOINT * NUM_FEATURE_PER_KEYPOINT
SEQUENCE_LENGTH = 90

KEYPOINT_CONF_THRESHOLD = 0.25
MIN_VALID_KEYPOINTS = 8
MIN_CORE_KEYPOINTS = 4
CORE_KEYPOINT_INDEX = [5, 6, 11, 12, 13, 14, 15, 16]
KEYPOINT_MARGIN_RATIO = 0.10
KEYPOINT_MIN_SPAN_RATIO = 0.015
KEYPOINT_MAX_SPAN_RATIO = 1.50

LEFT_SHOULDER = 5
RIGHT_SHOULDER = 6
LEFT_HIP = 11
RIGHT_HIP = 12
EPSILON = 1e-6


def bbox_iou(a, b):
    if a is None or b is None:
        return 0.0
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def bbox_center(box):
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def center_distance_normalized(a, b, w, h):
    ax, ay = bbox_center(a)
    bx, by = bbox_center(b)
    return float(np.hypot(ax - bx, ay - by) / max(np.hypot(w, h), 1e-6))


def get_person_candidates(frame, detector):
    dets = detector.detect(frame)
    candidates = []
    for d in dets:
        x1, y1, x2, y2 = d['box']
        if x2 > x1 and y2 > y1 and d['class_id'] == 0:
            candidates.append({'box': d['box'], 'conf': d['conf']})
    return candidates


def select_initial_target(candidates, frame):
    if not candidates:
        return None
    h, w = frame.shape[:2]
    fc = (w / 2.0, h / 2.0)
    diagonal = max(np.hypot(w, h), 1e-6)
    best = None
    best_score = -np.inf
    for item in candidates:
        x1, y1, x2, y2 = item['box']
        bw = x2 - x1
        bh = y2 - y1
        area_norm = (bw * bh) / max(w * h, 1)
        cx, cy = bbox_center(item['box'])
        dist = np.hypot(cx - fc[0], cy - fc[1]) / diagonal
        score = 0.50 * item['conf'] + 0.35 * (1 - dist) + 0.15 * min(area_norm * 20.0, 1.0)
        if score > best_score:
            best_score = score
            best = item
    return best


def select_next_target(candidates, previous_box, frame):
    if not candidates:
        return None
    if previous_box is None:
        return select_initial_target(candidates, frame)
    h, w = frame.shape[:2]
    best = None
    best_score = -np.inf
    for item in candidates:
        iou = bbox_iou(previous_box, item['box'])
        dist = center_distance_normalized(previous_box, item['box'], w, h)
        if iou < TRACK_MIN_IOU and dist > TRACK_MAX_CENTER_DISTANCE:
            continue
        score = TRACK_IOU_WEIGHT * iou + TRACK_CENTER_WEIGHT * (1 - dist)
        if score > best_score:
            best_score = score
            best = item
    return best


def smooth_box(history):
    valid = [b for b in list(history)[-BBOX_SMOOTH_WINDOW:] if b is not None]
    if not valid:
        return None
    return tuple(np.median(np.asarray(valid), axis=0).astype(int))


def determine_crop_box(raw_box, history, last_valid_box, missing_count):
    if raw_box is not None:
        history.append(raw_box)
        chosen = smooth_box(history) or raw_box
        return chosen, chosen, 0, True
    missing_count += 1
    if last_valid_box is not None and missing_count <= MAX_MISSING_BOX:
        return last_valid_box, last_valid_box, missing_count, True
    return None, None, missing_count, False


def crop_with_padding(frame, box):
    if box is None:
        return None, None
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = box
    bw = max(1, x2 - x1)
    bh = max(1, y2 - y1)
    px = int(round(bw * YOLO_PAD_X))
    py = int(round(bh * YOLO_PAD_Y))
    cx1 = max(0, x1 - px)
    cy1 = max(0, y1 - py)
    cx2 = min(w, x2 + px)
    cy2 = min(h, y2 + py)
    if cx2 <= cx1 or cy2 <= cy1:
        return None, None
    return frame[cy1:cy2, cx1:cx2].copy(), (cx1, cy1, cx2, cy2)


def run_pose_on_crop(crop, pose_model):
    if crop is None or crop.size == 0:
        return None
    poses = pose_model.estimate_pose(crop)
    if not poses:
        return None
    idx = int(np.argmax([p['conf'] for p in poses]))
    best = poses[idx]
    return {
        'bbox_conf': best['conf'],
        'keypoints_crop': best['keypoints'],
    }


def map_keypoints_crop_to_frame(keypoints_crop, crop_box):
    arr = np.asarray(keypoints_crop, dtype=np.float32)
    if arr.shape != (NUM_KEYPOINT, NUM_FEATURE_PER_KEYPOINT):
        return None
    x1, y1, _, _ = crop_box
    mapped = arr.copy()
    mapped[:, 0] += float(x1)
    mapped[:, 1] += float(y1)
    return mapped


def validate_coco_keypoints(keypoints, frame_shape):
    if keypoints is None:
        return False
    arr = np.asarray(keypoints, dtype=np.float32)
    if arr.shape != (NUM_KEYPOINT, NUM_FEATURE_PER_KEYPOINT):
        return False
    if not np.isfinite(arr).all():
        return False
    conf = arr[:, 2]
    if conf.min() < -0.01 or conf.max() > 1.01:
        return False
    valid = conf >= KEYPOINT_CONF_THRESHOLD
    if int(valid.sum()) < MIN_VALID_KEYPOINTS:
        return False
    if int(valid[CORE_KEYPOINT_INDEX].sum()) < MIN_CORE_KEYPOINTS:
        return False
    h, w = frame_shape[:2]
    xy = arr[valid, :2]
    inside = (
        (xy[:, 0] >= -KEYPOINT_MARGIN_RATIO * w)
        & (xy[:, 0] <= (1 + KEYPOINT_MARGIN_RATIO) * w)
        & (xy[:, 1] >= -KEYPOINT_MARGIN_RATIO * h)
        & (xy[:, 1] <= (1 + KEYPOINT_MARGIN_RATIO) * h)
    )
    if float(np.mean(inside)) < 0.8:
        return False
    safe = xy[inside]
    if len(safe) < MIN_CORE_KEYPOINTS:
        return False
    span = np.hypot(float(np.ptp(safe[:, 0])), float(np.ptp(safe[:, 1]))) / max(np.hypot(w, h), 1e-6)
    if span < KEYPOINT_MIN_SPAN_RATIO:
        return False
    if span > KEYPOINT_MAX_SPAN_RATIO:
        return False
    return True


def normalize_coco17_sequences(X):
    X = np.asarray(X, dtype=np.float32)
    reshaped = X.reshape(-1, SEQUENCE_LENGTH, 17, 3)
    output = np.zeros_like(reshaped, dtype=np.float32)

    for seq_idx in range(len(reshaped)):
        for frame_idx in range(SEQUENCE_LENGTH):
            frame = reshaped[seq_idx, frame_idx]
            xy = frame[:, :2]
            confidence = frame[:, 2]
            if np.allclose(xy, 0.0) and np.allclose(confidence, 0.0):
                continue
            valid = confidence > 0.0
            if valid.sum() < 2:
                continue
            left_hip_valid = confidence[LEFT_HIP] > 0
            right_hip_valid = confidence[RIGHT_HIP] > 0
            left_shoulder_valid = confidence[LEFT_SHOULDER] > 0
            right_shoulder_valid = confidence[RIGHT_SHOULDER] > 0
            if left_hip_valid and right_hip_valid:
                center = (xy[LEFT_HIP] + xy[RIGHT_HIP]) / 2.0
            elif left_shoulder_valid and right_shoulder_valid:
                center = (xy[LEFT_SHOULDER] + xy[RIGHT_SHOULDER]) / 2.0
            else:
                center = np.mean(xy[valid], axis=0)
            scale = 0.0
            if left_hip_valid and right_hip_valid and left_shoulder_valid and right_shoulder_valid:
                hip_center = (xy[LEFT_HIP] + xy[RIGHT_HIP]) / 2.0
                shoulder_center = (xy[LEFT_SHOULDER] + xy[RIGHT_SHOULDER]) / 2.0
                scale = float(np.linalg.norm(shoulder_center - hip_center))
            if scale <= EPSILON:
                valid_xy = xy[valid]
                width = float(np.ptp(valid_xy[:, 0]))
                height = float(np.ptp(valid_xy[:, 1]))
                scale = float(np.hypot(width, height))
            if scale <= EPSILON:
                scale = 1.0
            normalized_xy = (xy - center) / scale
            normalized_xy[~valid] = 0.0
            output[seq_idx, frame_idx, :, :2] = normalized_xy
            output[seq_idx, frame_idx, :, 2] = confidence

    output[..., :2] = np.clip(output[..., :2], -5.0, 5.0)
    output = output.reshape(-1, SEQUENCE_LENGTH, FEATURE_DIM)
    output = np.nan_to_num(output, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    return output


def draw_coco_skeleton(frame, keypoints):
    image = frame.copy()
    if keypoints is None:
        return image
    conf = keypoints[:, 2]
    points = [(int(round(p[0])), int(round(p[1]))) for p in keypoints]
    for pa, pb in COCO_KEYPOINT_CONNECTIONS:
        if conf[pa] >= KEYPOINT_CONF_THRESHOLD and conf[pb] >= KEYPOINT_CONF_THRESHOLD:
            cv2.line(image, points[pa], points[pb], (0, 255, 255), 2, cv2.LINE_AA)
    for idx, (x, y) in enumerate(points):
        if conf[idx] >= KEYPOINT_CONF_THRESHOLD and 0 <= x < image.shape[1] and 0 <= y < image.shape[0]:
            cv2.circle(image, (x, y), 4, (0, 0, 255), -1, cv2.LINE_AA)
    return image
