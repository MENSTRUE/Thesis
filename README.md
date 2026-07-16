# UAV-Human Real-time Inference

Pipeline _real-time_ untuk **UAV-Human 5-class activity recognition** menggunakan CNN-BiLSTM, dengan sumber kamera webcam atau drone RTSP.

Model menerima input urutan skeleton 90 frame (COCO-17 keypoints) yang telah dinormalisasi (_hip-centered torso-scale_), dan mengklasifikasikan ke 5 aktivitas:

| Kelas | Aktivitas |
| ----- | --------- |
| 0     | Walking   |
| 1     | Running   |
| 2     | Waving    |
| 3     | Clapping  |
| 4     | Sitting   |

## Requirements

- Python **3.12** (TensorFlow tidak support >3.12)
- [UV](https://docs.astral.sh/uv/) — package manager

## Instalasi

```bash
uv sync
```

Perintah di atas akan membuat virtual environment dan menginstall semua dependensi (TensorFlow, ONNX Runtime, OpenCV, dll).

## Penggunaan

### Webcam

```bash
uv run main.py --webcam
```

### Drone RTSP

```bash
uv run main.py --drone rtsp://192.168.1.1:7070/webcam
```

Ganti URL RTSP sesuai dengan drone Anda.

## Cara Kerja

Pipeline memproses frame sebagai berikut:

- **Frame skip** — 1 dari 4 frame diproses untuk menjaga FPS
- **Deteksi orang** — YOLOv8s (via ONNX Runtime) mendeteksi orang di frame
- **Tracking** —IOU + center-distance untuk melacak orang yang sama antar frame
- **Crop & padding** — Region orang di-crop dengan padding 45%
- **Pose estimation** — YOLO26s-Pose mengekstrak 17 keypoints COCO
- **Validasi** — Keypoints divalidasi (jumlah, posisi, span)
- **Buffer** — 90 frame valid terakhir disimpan di deque
- **Prediksi** — Setelah buffer terisi, CNN-BiLSTM memprediksi aktivitas
- **Overlay** — Class name, confidence, probability bars, skeleton, bounding box, FPS

## Kontrol (saat running)

| Tombol | Fungsi                                                                          |
| ------ | ------------------------------------------------------------------------------- |
| `R`    | Mulai/stop rekaman (disimpan ke `recordings/YYYYMMDD_HHMMSS/inference_XXX.avi`) |
| `Q`    | Keluar                                                                          |

## Struktur Proyek

```
├── main.py                      # Entry point CLI
├── inference_pipeline.py        # Class InferencePipeline (buffer, prediksi, overlay)
├── preprocessing.py             # YOLO detection, crop, pose, validasi, normalisasi, drawing
├── yolo_onnx.py                 # YOLO ONNX Runtime wrapper (detect & pose)
├── webcam_camera.py             # Kamera webcam thread
├── rtsp_camera.py               # Kamera drone RTSP thread dengan auto-reconnect
├── pyproject.toml               # UV project config & dependencies
├── training_results/
│   └── UAV_HUMAN_5CLASS_H4_CNN_BILSTM_ANTI_OVERFITTING/
│       ├── best_model.keras     # Model terlatih (CNN-BiLSTM)
│       └── class_mapping.json   # Mapping index → nama kelas
├── recordings/                  # Hasil rekaman (auto-generated)
└── *.onnx                       # YOLO ONNX models (auto-download)
```

## Model YOLO (ONNX)

Model YOLO (detection & pose) akan **otomatis didownload** dari GitHub Ultralytics saat pertama kali pipeline dijalankan, tidak perlu di-download manual.

## Catatan

- Pipeline otomatis memilih CPU sebagai backend. Jika GPU diperlukan, install `onnxruntime-gpu` sebagai pengganti `onnxruntime`.
- TensorFlow membutuhkan waktu ~25 detik untuk inisialisasi pertama.
