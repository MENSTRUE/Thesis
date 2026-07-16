import argparse
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np


def create_session_dir(base="recordings"):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_dir = Path(base) / timestamp
    session_dir.mkdir(parents=True, exist_ok=True)
    return session_dir


def main():
    parser = argparse.ArgumentParser(description="UAV-Human Real-time Inference")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--webcam", action="store_true", help="Use webcam")
    group.add_argument("--drone", type=str, metavar="RTSP_URL", help="Use drone RTSP stream")
    args = parser.parse_args()

    if not args.webcam and not args.drone:
        print("Usage: uv run main.py --webcam | --drone RTSP_URL")
        sys.exit(1)

    session_dir = create_session_dir()
    print(f"[INFO] Session dir: {session_dir}")

    if args.webcam:
        from webcam_camera import cameraDroneThread as CameraThread
        camera = CameraThread(src=1).start()
    else:
        from rtsp_camera import cameraDroneThread as CameraThread
        camera = CameraThread(args.drone).start()

    from inference_pipeline import InferencePipeline

    model_dir = Path(__file__).parent / "training_results" / "UAV_HUMAN_5CLASS_H4_CNN_BILSTM_ANTI_OVERFITTING"
    if not (model_dir / "best_model.keras").exists():
        print(f"[ERROR] Model not found: {model_dir / 'best_model.keras'}")
        sys.exit(1)

    pipeline = InferencePipeline(model_dir=model_dir, skip=4)

    writer = None
    is_recording = False
    record_counter = 0
    overlay_written = 0
    fps_start = time.time()
    fps_frames = 0
    fps_display = 0.0

    print("[INFO] Press R to start/stop recording, Q to quit")

    while True:
        frame = camera.read()
        if frame is None:
            time.sleep(0.01)
            continue

        pipeline.update(frame)

        fps_frames += 1
        elapsed = time.time() - fps_start
        if elapsed >= 1.0:
            fps_display = fps_frames / elapsed
            fps_frames = 0
            fps_start = time.time()

        overlay = pipeline.get_overlay(frame, fps=fps_display)

        if is_recording:
            if overlay_written == 0:
                h, w = overlay.shape[:2]
                w = w if w % 2 == 0 else w - 1
                h = h if h % 2 == 0 else h - 1
                fname = session_dir / f"inference_{record_counter:03d}.avi"
                writer = cv2.VideoWriter(
                    str(fname),
                    cv2.VideoWriter_fourcc(*"MJPG"),
                    20.0,
                    (w, h),
                )
                print(f"[REC] Started: {fname}")
            if writer is not None:
                h2, w2 = overlay.shape[:2]
                w2 = w2 if w2 % 2 == 0 else w2 - 1
                h2 = h2 if h2 % 2 == 0 else h2 - 1
                if w2 != overlay.shape[1] or h2 != overlay.shape[0]:
                    frame_out = cv2.resize(overlay, (w2, h2), interpolation=cv2.INTER_AREA)
                else:
                    frame_out = overlay
                writer.write(frame_out)
                overlay_written += 1

            cv2.putText(overlay, "REC", (overlay.shape[1] - 100, overlay.shape[0] - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)

        cv2.imshow("UAV-Human Inference", overlay)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break
        elif key == ord("r"):
            if is_recording:
                if writer is not None:
                    writer.release()
                    writer = None
                    print(f"[REC] Stopped: {record_counter} frames written")
                is_recording = False
            else:
                is_recording = True
                overlay_written = 0
                record_counter += 1

    camera.stop()
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()
    print("[INFO] Exited")


if __name__ == "__main__":
    main()
