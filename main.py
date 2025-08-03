import sys
import os

sys.path.insert(0, os.path.abspath("yolov7"))

import cv2
from yolov7.utils.datasets import letterbox
from yolov7.utils.torch_utils import select_device
from yolov7.utils.general import non_max_suppression, scale_coords
from yolov7.models.experimental import attempt_load
from time import sleep
import numpy as np
import torch

if __name__ == "__main__":

    cap = cv2.VideoCapture(0)

    device = select_device('') # Automatically select GPU or CPU

    model = attempt_load('yolov7.pt', map_location=device) # Load the model
    model.eval() # Set model to evaluation mode - e.g. Dropout layes are disabled

    names = model.names

    while True:

        ret, frame = cap.read()

        if ret:
            np_img = letterbox(frame, new_shape=640)[0] # Resize frame to 640x640 and pad
            np_img = np_img[:, :, ::-1].transpose(2, 0, 1) # BGR to RGB and HWC to CHW (channels first) H: Height, W: Width, C: Channels
            np_img = np.ascontiguousarray(np_img) # make sure it's contiguous in memory

        else:
            print("Failed to capture frame from camera.")
            break

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

        if det is not None and len(det): # Rescale boxes to match original image size
            det[:, :4] = scale_coords(torch_img.shape[2:], det[:, :4], frame.shape).round()

            for *xyxy, conf, cls in det:  # Loop through each detection

                class_id = int(cls.item())

                if class_id != 0:  # Detect only 'person' class (class_id 0)
                    continue

                x1, y1, x2, y2 = map(int, xyxy)

                label = f'{names[class_id]} {conf:.2f}'# Class and confidence

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2) # Draw rectangle
                cv2.putText(frame, label, (x1, y1 -10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

                cv2.imshow("YOLOv7 Webcam", frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

                # Show the result

                #cv2.imwrite('result.png', frame)
                #cv2.imshow('YOLO Detection', frame)
                #cv2.waitKey(0)
                #cv2.destroyAllWindows()

        sleep(0.1)

    cap.release()
    cv2.destroyAllWindows()