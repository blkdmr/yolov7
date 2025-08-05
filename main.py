import sys
import os

sys.path.insert(0, os.path.abspath("yolov7"))

import cv2
from yolov7.utils.datasets import letterbox
from yolov7.utils.torch_utils import select_device
from yolov7.utils.general import non_max_suppression, scale_coords
from yolov7.models.experimental import attempt_load
import numpy as np
import torch
import joblib
import timm
from torchvision import transforms
from PIL import Image

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

classes = ['healthy', 'rotten']

def export(cropped_images):
    os.makedirs('export', exist_ok=True)
    for i, img in enumerate(cropped_images):
        cv2.imwrite(f'export/cropped_{i}.png', img)

if __name__ == "__main__":

    frame = cv2.imread('pot.jpg')
    image = cv2.imread('pot.jpg')

    device = select_device('') # Automatically select GPU or CPU

    model = attempt_load('weights/best.pt', map_location=device) # Load the model
    model.eval() # Set model to evaluation mode - e.g. Dropout layers are disabled

    backbone = timm.create_model('resnet50', pretrained=True)
    backbone.to(device)

    # Freeze the model
    backbone.eval()  # disables dropout, batchnorm updates
    for param in backbone.parameters():
        param.requires_grad = False

    clf = joblib.load("resnet50_potato_classifier.joblib")

    names = model.names

    np_img = letterbox(frame, new_shape=640)[0] # Resize frame to 640x640 and pad
    np_img = np_img[:, :, ::-1].transpose(2, 0, 1) # BGR to RGB and HWC to CHW (channels first) H: Height, W: Width, C: Channels
    np_img = np.ascontiguousarray(np_img) # make sure it's contiguous in memory

    torch_img = torch.from_numpy(np_img).float() # Convert to torch tensor
    torch_img /= 255.0 # Normalize to [0, 1]
    torch_img = torch_img.unsqueeze(0) # Add batch dimension

    with torch.no_grad():  # Disable gradient calculation for inference
        torch_img = torch_img.to(device)
        pred = model(torch_img)[0]

    # YOLO predicts multiple overlapping boxes per object.
    # So we use Non-Max Suppression (NMS) to:
    #   1. Remove duplicate detections,
    #   2. Keep only high-confidence boxes.
    # In particular,
    #   1. conf_thres: filters out low-confidence detections
    #   2. iou_thres: controls how much overlap is allowed between boxes (for duplicate removal)

    detections = non_max_suppression(pred, conf_thres=0.45, iou_thres=0.45)

    # Now detections is a list (1 item, since we passed 1 image), and inside that:
    # Each row is a detection: [x1, y1, x2, y2, confidence, class]

    det = detections[0] # Get the first (and only) image detection

    cropped_images = []  # List to store cropped images

    if det is not None and len(det): # Rescale boxes to match original image size
        det[:, :4] = scale_coords(torch_img.shape[2:], det[:, :4], frame.shape).round()

        print(f'Detected {len(det)} objects')

        for i, (*xyxy, conf, cls) in enumerate(det):  # Loop through each detection

            class_id = int(cls.item())

            x1, y1, x2, y2 = map(int, xyxy)

            cropped = image[y1:y2, x1:x2]

            # Converti da BGR a RGB
            image_rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)

            # Crea un oggetto PIL.Image dalla matrice numpy
            image_pil = Image.fromarray(image_rgb)
            image_pil = transform(image_pil)
            # Add batch dimension
            image_pil = image_pil.unsqueeze(0)

            image_pil = image_pil.to(device)
            feats = backbone.forward_features(image_pil)
            pooled = feats.mean(dim=[2, 3])  # global average pooling to [B, 2048]

            health = clf.predict(pooled.cpu().numpy())
            #cropped_images.append(cropped)

            label = f'{classes[health[0]]} {names[class_id]}'# Class and confidence

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1) # Draw rectangle
            cv2.putText(frame, label, (x1, y1 +10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        cv2.imwrite('result.png', frame)
        #export(cropped_images)
    else:
        print("No detections found.")