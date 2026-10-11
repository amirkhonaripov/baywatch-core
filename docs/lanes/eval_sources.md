# Lane G (evaluation): data, models and price sources

| name | URL | licence | used for | faces / plates |
|---|---|---|---|---|
| Seen jury set v1 (96 inputs: 82 photos, 7 videos, 7 frames cut from those videos) | Wikimedia Commons; one file page per item in `eval/jury/jury.jsonl` | per file, recorded in `eval/jury/jury.jsonl`: CC BY 2.0/3.0/4.0, CC BY-SA 2.0/3.0/4.0, CC0, public domain | test only (jury-style end-to-end test); held out in `eval/manifests/jury.jsonl` | yes / yes (some plates were blurred by the uploaders) |
| RF-DETR base, COCO-pretrained (via `seen.baseline`, rfdetr 1.11.2) | https://github.com/roboflow/rf-detr | Apache-2.0 (code and weights, per repo) | speed benchmark (PyTorch, ONNX Runtime, TensorRT); baseline pipeline under test | n/a |
| fast-alpr 0.4.0: `yolo-v9-t-384-license-plate-end2end` detector + `cct-xs-v2-global-model` OCR | https://github.com/ankandrew/fast-alpr | library MIT; model weights: licence not stated on the hub page (not checked) | speed benchmark; baseline pipeline under test | n/a |
| OpenCV YuNet face detector `face_detection_yunet_2023mar.onnx` | https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet | MIT | extra face blur on every failure-gallery image | n/a |
| OpenCV Haar cascades (frontal, profile), bundled with opencv-python | https://github.com/opencv/opencv/tree/4.x/data/haarcascades | Intel open-source licence (BSD-style), per the XML headers | extra face blur on every failure-gallery image | n/a |
| NVIDIA TensorRT 10.15 libraries (`tensorrt_cu13_libs` wheel, only libnvinfer/plugin/onnxparser + sm120/ptx builder resources kept, on the NVMe at `seen_fast/eval/trt` (1.2 GB)) | https://pypi.nvidia.com/tensorrt-cu13-libs/ | NVIDIA TensorRT licence (proprietary, redistribution restricted) | speed table only (ONNX Runtime TensorRT EP, FP16) | n/a |

## Price and spec sources used by `eval/cost.py`

| item | value | source (read 2026-10-10) |
|---|---|---|
| Modal L4 GPU | $0.000222 per second | https://modal.com/pricing |
| Modal CPU / memory | $0.0000131 per physical core per second; $0.00000222 per GiB per second | https://modal.com/pricing |
| Modal idle window | 60 s by default, and idle time is billed | https://modal.com/docs/guide/cold-start |
| Electricity, legal entities, Uzbekistan | 1,100 UZS per kWh from 1 Jun 2026 | https://kun.uz/en/news/2026/05/15/uzbekistan-to-raise-electricity-and-natural-gas-tariffs-from-june-1 (Interfax, 15 May 2026, agrees) |
| UZS/USD | 11,778.45 | Central Bank of Uzbekistan official rate for 6 Oct 2026 (reported by zamin.uz) |
| RTX 4060 Ti PC in Tashkent | about $925–1,345 (used); about $1,020 (new i5-13400F build) | `research/competitors_legal.md` (OLX.uz, 10 Oct 2026) |
| RTX 4060 Ti | 22.06 TFLOPS FP32 (derived: 4,352 cores × 2 × 2.535 GHz); 288 GB/s; 160–165 W | technical.city, Puget Systems |
| NVIDIA L4 | 30.3 TFLOPS FP32; 300 GB/s; 72 W | NVIDIA L4 datasheet |
| RTX PRO 5000 Blackwell | about 66.9 TFLOPS FP32 (derived: 14,080 cores × 2 × 2.377 GHz); 1,344 GB/s | PNY/NVIDIA datasheet (bandwidth), Puget Systems (cores) |
