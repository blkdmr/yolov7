
# YOLOv7 + OpenCV: Step-by-Step with Explanations

[YOLOv7: Trainable bag-of-freebies sets new state-of-the-art for real-time object detectors](https://arxiv.org/abs/2207.02696)
---

## ✅ Step 1: Install Required Libraries

Install PyTorch and OpenCV:

```bash
pip install torch torchvision opencv-python
```

---

## ✅ Step 2: Clone the YOLOv7 Repository

YOLOv7 needs its own architecture code to load the `.pt` weights:

```bash
git clone https://github.com/WongKinYiu/yolov7.git
cd yolov7
pip install -r requirements.txt
```

---

## ✅ Step 3: Download the Pretrained YOLOv7 Model

Place the `.pt` file in the root of the repo:

```bash
wget https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7.pt
```

---

## ✅ Step 4: Capture a Frame from the Camera

```python
import cv2

cap = cv2.VideoCapture(0)
ret, frame = cap.read()
cap.release()

if ret:
    cv2.imshow("Captured", frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("Camera error")
```

---

## ✅ Step 5: Preprocess the Frame for YOLOv7

```python
from utils.datasets import letterbox
import numpy as np

img = letterbox(frame, new_shape=640)[0]
img = img[:, :, ::-1].transpose(2, 0, 1)
img = np.ascontiguousarray(img, dtype=np.float32)
img /= 255.0
```

---

## ✅ Step 6: Convert Image to Tensor with Batch Dimension

```python
import torch

img_tensor = torch.from_numpy(img).to(torch.float32)
img_tensor = img_tensor.unsqueeze(0)
```

---

## ✅ Step 7: Load the YOLOv7 Model

```python
from models.experimental import attempt_load
from utils.torch_utils import select_device

device = select_device('')
model = attempt_load('yolov7.pt', map_location=device)
model.eval()
img_tensor = img_tensor.to(device)
```

---

## ✅ Step 8: Run Inference on the Image

```python
with torch.no_grad():
    pred = model(img_tensor)[0]
```

---

## ✅ Step 9: Apply Non-Max Suppression (Filter Detections)

```python
from utils.general import non_max_suppression

detections = non_max_suppression(pred, conf_thres=0.25, iou_thres=0.45)
```

---

## ✅ Step 10: Scale Coordinates Back to Original Image Size

```python
from utils.general import scale_coords

det = detections[0]
if det is not None and len(det):
    det[:, :4] = scale_coords(img_tensor.shape[2:], det[:, :4], frame.shape).round()
```

---

## ✅ Step 11: Draw Bounding Boxes and Labels on the Image

```python
names = model.names

for *xyxy, conf, cls in det:
    x1, y1, x2, y2 = map(int, xyxy)
    label = f'{names[int(cls)]} {conf:.2f}'
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(frame, label, (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
```

---

## ✅ Step 12: Show the Final Image with Detections

```python
cv2.imshow("YOLOv7 Detection", frame)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

---

## ✅ Step 13: Real-Time Detection Loop (Optional)

```python
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Repeat steps 5–11 here

    cv2.imshow("YOLOv7 Webcam", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

---
