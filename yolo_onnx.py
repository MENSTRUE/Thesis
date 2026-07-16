import urllib.request
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

MODEL_URLS = {
    "yolov8s.onnx": "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolov8s.onnx",
    "yolo26s-pose.onnx": "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26s-pose.onnx",
}


def _download_if_missing(path):
    path = Path(path)
    name = path.name
    if path.exists():
        return
    url = MODEL_URLS.get(name)
    if url is None:
        raise FileNotFoundError(f"No download URL for {name}")
    print(f"Downloading {name}...")
    urllib.request.urlretrieve(url, path)
    print(f"Downloaded {name}")


class YOLOONNX:
    def __init__(self, model_path, conf_threshold=0.25, iou_threshold=0.45):
        _download_if_missing(model_path)
        self.session = ort.InferenceSession(model_path)
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        inp = self.session.get_inputs()[0]
        self.input_size = (inp.shape[2], inp.shape[3])

    def _letterbox(self, img, color=(114, 114, 114)):
        h, w = img.shape[:2]
        target_w, target_h = self.input_size
        scale = min(target_w / w, target_h / h)
        new_w = int(round(w * scale))
        new_h = int(round(h * scale))

        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        pad_w = target_w - new_w
        pad_h = target_h - new_h
        left = pad_w // 2
        top = pad_h // 2

        canvas = np.full((target_h, target_w, 3), color, dtype=np.uint8)
        canvas[top:top + new_h, left:left + new_w] = resized

        return canvas, left, top, scale

    def _preprocess(self, img):
        canvas, left, top, scale = self._letterbox(img)
        blob = canvas.astype(np.float32) / 255.0
        blob = np.transpose(blob, (2, 0, 1))
        blob = np.expand_dims(blob, axis=0).astype(np.float32)
        return blob, left, top, scale

    def _map_to_original(self, x, y, left, top, scale):
        return (x - left) / scale, (y - top) / scale

    @staticmethod
    def _sigmoid(x):
        return 1.0 / (1.0 + np.exp(-x))

    def detect(self, img):
        h, w = img.shape[:2]
        blob, left, top, scale = self._preprocess(img)
        outputs = self.session.run(None, {self.session.get_inputs()[0].name: blob})
        pred = outputs[0][0].T

        boxes = []
        for i in range(pred.shape[0]):
            cx, cy, bw, bh = pred[i, :4]
            scores = self._sigmoid(pred[i, 4:])
            conf = float(np.max(scores))
            if conf < self.conf_threshold:
                continue
            class_id = int(np.argmax(scores))

            x1 = cx - bw / 2
            y1 = cy - bh / 2
            x2 = cx + bw / 2
            y2 = cy + bh / 2

            ox1, oy1 = self._map_to_original(x1, y1, left, top, scale)
            ox2, oy2 = self._map_to_original(x2, y2, left, top, scale)

            ox1 = max(0, ox1)
            oy1 = max(0, oy1)
            ox2 = min(w, ox2)
            oy2 = min(h, oy2)

            if ox2 <= ox1 or oy2 <= oy1:
                continue

            boxes.append([ox1, oy1, ox2, oy2, conf, class_id])

        if not boxes:
            return []

        boxes = np.array(boxes, dtype=np.float32)
        keep = self._nms(boxes[:, :4], boxes[:, 4])
        boxes = boxes[keep]

        results = []
        for b in boxes:
            results.append({
                "box": (int(b[0]), int(b[1]), int(b[2]), int(b[3])),
                "conf": float(b[4]),
                "class_id": int(b[5]),
            })
        return results

    def estimate_pose(self, img):
        h, w = img.shape[:2]
        blob, left, top, scale = self._preprocess(img)
        outputs = self.session.run(None, {self.session.get_inputs()[0].name: blob})
        pred = outputs[0][0]

        results = []
        for i in range(pred.shape[0]):
            x1, y1, x2, y2 = pred[i, :4]
            conf = float(pred[i, 4])
            class_id = int(pred[i, 5])
            if conf < self.conf_threshold or class_id != 0:
                continue

            ox1, oy1 = self._map_to_original(x1, y1, left, top, scale)
            ox2, oy2 = self._map_to_original(x2, y2, left, top, scale)

            ox1 = max(0, ox1)
            oy1 = max(0, oy1)
            ox2 = min(w, ox2)
            oy2 = min(h, oy2)

            kpts = pred[i, 6:].reshape(17, 3).copy()
            for j in range(17):
                kpts[j, 0], kpts[j, 1] = self._map_to_original(
                    kpts[j, 0], kpts[j, 1], left, top, scale
                )
                kpts[j, 0] = float(np.clip(kpts[j, 0], 0, w))
                kpts[j, 1] = float(np.clip(kpts[j, 1], 0, h))

            results.append({
                "box": (int(ox1), int(oy1), int(ox2), int(oy2)),
                "conf": conf,
                "keypoints": kpts.astype(np.float32),
            })

        return results

    @staticmethod
    def _nms(boxes, scores, iou_threshold=0.45):
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        areas = (x2 - x1) * (y2 - y1)
        order = np.argsort(scores)[::-1]

        keep = []
        while len(order) > 0:
            i = order[0]
            keep.append(i)
            if len(order) == 1:
                break

            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            inter_w = np.maximum(0, xx2 - xx1)
            inter_h = np.maximum(0, yy2 - yy1)
            inter = inter_w * inter_h

            iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-7)
            inds = np.where(iou <= iou_threshold)[0]
            order = order[inds + 1]

        return keep
