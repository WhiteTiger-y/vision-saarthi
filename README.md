# Vision Saarthi

**An AI assistant for visually impaired people.** It watches a live camera feed, detects obstacles, estimates how far away they are and which side they are on, and speaks an alert out loud.

The repo has two parts:

1. **Obstacle alerts (Streamlit apps):** YOLO object detection on a live camera stream, with distance and direction estimates and spoken alerts through text-to-speech.
2. **Scene description (notebooks):** BLIP image captioning that describes what the camera sees, with an LLM that rewrites captions into guidance meant for a blind user.

---

## How it works

```
Camera (Raspberry Pi / IP stream)
        │  frames over HTTP
        ▼
YOLOv7 / YOLOv5 detector ──► confidence filter (> 0.5)
        │
        ├─► distance estimate   (pinhole model: real width × focal length ÷ box width)
        ├─► direction           (frame split into left / in front / right thirds)
        ▼
Alert text  e.g. "person at 1.8 m in front of you"
        │
        ├─► Streamlit UI (annotated video + last 5 detections)
        └─► gTTS speech output
```

### Scene description pipeline (notebooks)
1. Sample one frame per second from a video.
2. Caption each frame with **BLIP** (`Salesforce/blip-image-captioning-base` / `-large`).
3. Skip redundant captions using object movement (Faster R-CNN boxes), a background-change check and caption similarity (Jaccard similarity on words, with stop words removed).
4. Rewrite each caption with **Phi-3.5-mini-instruct** into an alert-style description for a blind user.
5. Optionally **fine-tune BLIP on Flickr8k** with Hugging Face `Trainer`, early stopping and Albumentations augmentation.

---

## Repository structure

| File | What it does |
|---|---|
| `v7_dist.py` | Main app: YOLOv7 detection, distance and direction estimates, annotated video, spoken alerts |
| `v7_non.py` | Lighter YOLOv7 app: direction only (left / middle / right) with spoken alerts |
| `yolo_distance.py` | Minimal YOLOv7 distance-measurement viewer |
| `hackcbs.py` | Early prototype using YOLOv5s and spoken alerts |
| `Image to Text Improved with LLM.ipynb` | BLIP captioning with Phi-3.5-mini refinement for blind-friendly descriptions |
| `FineTuning BLIP.ipynb` | Video frame captioning, redundancy filtering and BLIP fine-tuning on Flickr8k |
| `yolov7.pt`, `yolov5s.pt` | Pretrained detector weights (COCO) |

---

## Tech stack

**Detection:** YOLOv7, YOLOv5 (PyTorch Hub), Faster R-CNN (torchvision)
**Vision-language:** BLIP (Hugging Face Transformers), Phi-3.5-mini-instruct
**App and audio:** Streamlit, OpenCV, gTTS
**Hardware:** Raspberry Pi camera streaming over the local network

---

## Getting started

```bash
git clone https://github.com/WhiteTiger-y/vision-saarthi.git
cd vision-saarthi
pip install streamlit opencv-python torch torchvision gtts pandas numpy
```

1. Start a camera stream, for example on a Raspberry Pi, and note its URL.
2. Set `video_url` at the top of the script to that URL (default `http://<pi-ip>:8081/`).
3. Run the app:

```bash
streamlit run v7_dist.py
```

Turn on **Enable Audio Alerts** in the sidebar to hear the detections.

> **Distance calibration:** `calculate_distance()` uses `focal_length=615` and `actual_width=0.5` m. Calibrate the focal length for your camera, and the reference width for the objects you care about, to get meaningful distances.

> **Audio playback:** the scripts call `os.system("start detection.mp3")`, which only works on Windows. On macOS use `afplay`, and on Linux use `mpg123` or a similar player.

The notebooks are written for Google Colab with a GPU. The fine-tuning notebook expects the Flickr8k dataset from Kaggle (`adityajn105/flickr8k`).

---

## Limitations and next steps

- Detection runs on a fixed interval (every 5–10 s) to keep the speech readable. A next step is to trigger alerts on new or approaching objects.
- Distances use a single reference width for every class. Per-class widths or a monocular depth model would be more accurate.
- The detector uses COCO classes, so hazards such as curbs, stairs and potholes need a custom-trained model.
- The detector and the captioning pipeline are separate today. The goal is one app that gives quick obstacle alerts plus on-demand scene descriptions.
