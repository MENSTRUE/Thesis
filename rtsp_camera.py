import cv2
import threading
import os

os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
    "rtsp_transport;tcp"
    "|fflags;nobuffer"
    "|flags;low_delay"
    "|analyzeduration;1000000"
    "|buffer_size;1024000"
)


class cameraDroneThread:
    def __init__(self, rtsp_url):
        self.url = rtsp_url
        print("[INFO]: Connecting drone...")
        self.stream = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
        self.stream.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.stream.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 10000)
        self.stream.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 10000)

        for _ in range(5):
            self.stream.read()

        (self.grabbed, self.frame) = self.stream.read()
        self.stopped = False

    def start(self):
        threading.Thread(target=self.update, args=(), daemon=True).start()
        return self

    def update(self):
        stale_counter = 0
        while not self.stopped:
            grabbed, frame = self.stream.read()
            if not grabbed or frame is None:
                stale_counter += 1
                if stale_counter > 30:
                    self.stream.release()
                    self.stream = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG)
                    self.stream.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    stale_counter = 0
                continue
            stale_counter = 0
            self.grabbed, self.frame = True, frame

    def read(self):
        if self.frame is not None:
            return cv2.rotate(self.frame, cv2.ROTATE_90_CLOCKWISE)
        return None

    def stop(self):
        self.stopped = True
