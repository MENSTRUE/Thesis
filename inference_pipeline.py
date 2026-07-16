import json
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np

from preprocessing import (
    get_person_candidates,
    select_initial_target,
    select_next_target,
    crop_with_padding,
    run_pose_on_crop,
    map_keypoints_crop_to_frame,
    validate_coco_keypoints,
    normalize_coco17_sequences,
    draw_coco_skeleton,
    SEQUENCE_LENGTH,
    FEATURE_DIM,
)
from yolo_onnx import YOLOONNX


class InferencePipeline:
    def __init__(self, model_dir, skip=4):
        self.skip = skip
        self.skip_counter = 0
        self.buffer = deque(maxlen=SEQUENCE_LENGTH)
        self.buffer_filled = False
        self.current_keypoints = None
        self.current_box = None
        self.current_crop_box = None

        self.previous_box = None
        self.last_valid_box = None
        self.missing_count = 0
        self.box_history = deque(maxlen=5)

        self.current_class = None
        self.current_confidence = 0.0
        self.current_probs = None

        self.frame_count = 0
        self.processed_count = 0
        self.prediction_count = 0
        self.pipeline_time = 0.0

        model_dir = Path(model_dir)
        best_model_path = str(model_dir / "best_model.keras")
        class_mapping_path = model_dir / "class_mapping.json"

        import tensorflow as tf
        self.model = tf.keras.models.load_model(best_model_path, compile=False)

        with open(class_mapping_path) as f:
            raw = json.load(f)
        self.classes = [raw[str(i)] for i in range(len(raw))]

        self.detector = YOLOONNX("yolov8s.onnx")
        self.pose_model = YOLOONNX("yolo26s-pose.onnx")

    def _process_frame(self, frame):
        self.pipeline_time = time.time()
        candidates = get_person_candidates(frame, self.detector)

        if self.previous_box is None:
            target = select_initial_target(candidates, frame)
        else:
            target = select_next_target(candidates, self.previous_box, frame)

        raw_box = target["box"] if target is not None else None

        if raw_box is not None:
            self.previous_box = raw_box
        else:
            self.previous_box = None

        valid_frame = False
        keypoints = None
        crop_box = None

        from preprocessing import determine_crop_box

        crop_source_box, self.last_valid_box, self.missing_count, box_valid = (
            determine_crop_box(raw_box, self.box_history, self.last_valid_box, self.missing_count)
        )

        if not box_valid:
            self.previous_box = None

        if box_valid and crop_source_box is not None:
            crop, crop_box = crop_with_padding(frame, crop_source_box)
            if crop is not None and crop_box is not None:
                pose_result = run_pose_on_crop(crop, self.pose_model)
                if pose_result is not None:
                    keypoints = map_keypoints_crop_to_frame(pose_result["keypoints_crop"], crop_box)
                    if keypoints is not None:
                        valid_frame = validate_coco_keypoints(keypoints, frame.shape)

        self.current_keypoints = keypoints if valid_frame else None
        self.current_box = crop_source_box if box_valid else None
        self.current_crop_box = crop_box if valid_frame else None

        feature = np.zeros((17, 3), dtype=np.float32)
        if valid_frame and keypoints is not None:
            feature = keypoints.astype(np.float32)
        self.buffer.append(feature.reshape(FEATURE_DIM))

        self.pipeline_time = time.time() - self.pipeline_time

    def update(self, frame):
        self.frame_count += 1
        self.skip_counter += 1

        if self.skip_counter >= self.skip:
            self.skip_counter = 0
            self._process_frame(frame)
            self.processed_count += 1

        if len(self.buffer) == SEQUENCE_LENGTH and not self.buffer_filled:
            self._run_prediction()
            self.buffer_filled = True

    def _run_prediction(self):
        seq = np.array(self.buffer, dtype=np.float32).reshape(1, SEQUENCE_LENGTH, FEATURE_DIM)
        normalized = normalize_coco17_sequences(seq)
        probs = self.model.predict(normalized, verbose=0)[0]
        self.current_probs = probs
        idx = int(np.argmax(probs))
        self.current_class = self.classes[idx]
        self.current_confidence = float(probs[idx])

    def reset(self):
        self.buffer.clear()
        self.buffer_filled = False
        self.current_keypoints = None
        self.current_box = None
        self.current_crop_box = None
        self.current_class = None
        self.current_confidence = 0.0
        self.current_probs = None
        self.previous_box = None
        self.last_valid_box = None
        self.missing_count = 0
        self.box_history.clear()
        self.skip_counter = 0

    def get_overlay(self, frame, fps=0.0):
        overlay = frame.copy()
        h, w = overlay.shape[:2]

        if self.current_crop_box is not None:
            x1, y1, x2, y2 = self.current_crop_box
            color = (0, 255, 0) if self.current_class is not None else (0, 0, 255)
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)

        if self.current_keypoints is not None:
            overlay = draw_coco_skeleton(overlay, self.current_keypoints)

        if self.buffer_filled and self.current_class is not None:
            text = f"{self.current_class} {self.current_confidence * 100:.1f}%"
            cv2.putText(overlay, text, (15, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)

            if self.current_probs is not None:
                bar_x = 15
                bar_y = h - 130
                bar_w = 180
                bar_h = 20
                bar_gap = 4
                for i, (cls_name, prob) in enumerate(zip(self.classes, self.current_probs)):
                    y_pos = bar_y + i * (bar_h + bar_gap)
                    fill_w = int(bar_w * prob)
                    cv2.rectangle(overlay, (bar_x, y_pos), (bar_x + bar_w, y_pos + bar_h), (50, 50, 50), -1)
                    cv2.rectangle(overlay, (bar_x, y_pos), (bar_x + fill_w, y_pos + bar_h), (0, 200, 0), -1)
                    cv2.putText(overlay, f"{cls_name}: {prob * 100:.0f}%", (bar_x + bar_w + 8, y_pos + 15),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
        else:
            n = len(self.buffer)
            text = f"INITIALIZING... {n}/{SEQUENCE_LENGTH}"
            cv2.putText(overlay, text, (15, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2, cv2.LINE_AA)

        if fps > 0:
            cv2.putText(overlay, f"{fps:.1f} FPS", (w - 140, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

        if w > 960:
            scale = 960 / w
            nh = int(h * scale)
            overlay = cv2.resize(overlay, (960, nh), interpolation=cv2.INTER_AREA)

        return overlay
