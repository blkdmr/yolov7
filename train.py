import sys
import os
sys.path.insert(0, os.path.abspath("yolov7"))
# ====================================================== #

import torch
from yolov7.utils.torch_utils import select_device
from yolov7.models.yolo import Model

device = select_device('')
cfg_path = "dataset/rock_paper_scissors/data.yaml"
num_classes = 3

# ch = 3 for RGB images
model = Model(cfg_path, ch=3, nc=num_classes)

checkpoint = torch.load("yolov7.pt", map_location=device, weights_only=False)
model.load_state_dict(checkpoint['model'], strict=False)

model.train()
