import cv2
import time
import threading
import os

# FFMPEG options ini bisa dibiarkan saja, tidak akan berefek buruk saat pakai webcam lokal
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;udp|fflags;nobuffer|flags;low_delay"

class cameraDroneThread:
    def __init__(self, src=1): # Parameter diubah namanya jadi 'src' agar lebih umum
        print(f"[INFO]: Connecting to camera {src}...")
        
        # Hapus cv2.CAP_FFMPEG agar bisa dipakai untuk webcam lokal
        self.stream = cv2.VideoCapture(src) 
        self.stream.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        # read first frame
        (self.grabbed, self.frame) = self.stream.read()
        self.stopped = False

    def start(self):
        # Thread starting
        threading.Thread(target=self.update, args=(), daemon=True).start()
        return self

    def update(self):
        # Looping
        while True:
            if self.stopped:
                self.stream.release()
                return
            # read next frame
            self.grabbed, self.frame = self.stream.read()

    def read(self):
        # Call looping 
        if self.frame is not None:
            # Hapus rotasi 90 derajat karena webcam lokal biasanya sudah lurus
            return self.frame 
        return None

    def stop(self):
        self.stopped = True