import sys
import os

import joblib

sys.path.insert(0, os.path.abspath("yolov7"))

import cv2
import timm
import torch

from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from yolov7.utils.torch_utils import select_device
from yolov7.models.experimental import attempt_load

# ========================================================= #
# Dataset for loading potato images
import os
import pandas as pd
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset

class PotatoDataset(Dataset):

    def __init__(self, csv_path, image_dir, transform=None):
        df = pd.read_csv(csv_path)
        self.image_dir = image_dir
        self.transform = transform

        device = select_device('')  # Automatically select GPU or CPU
        self._model = attempt_load('yolov7/weights/best.pt', map_location=device) # Load the model
        self._model.eval() # Set model to evaluation mode - e.g. Dropout layers are disabled

        # Convert one-hot to class ID
        self.samples = [
            (row['filename'], 0 if row['healthy'] == 1 else 1)
            for _, row in df.iterrows()
        ]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        filename, label = self.samples[idx]
        path = os.path.join(self.image_dir, filename)
        image = Image.open(path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label

# ========================================================= #

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = timm.create_model('resnet50', pretrained=True)
model.to(device)

# Freeze the model
model.eval()  # disables dropout, batchnorm updates
for param in model.parameters():
    param.requires_grad = False


from torch.utils.data import DataLoader

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

train_dataset = PotatoDataset(
    csv_path='dataset/rotten_healthy/prp_train/_classes.csv',
    image_dir='dataset/rotten_healthy/prp_train',
    transform=transform
)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False)

test_dataset = PotatoDataset(
    csv_path='dataset/rotten_healthy/prp_test/_classes.csv',
    image_dir='dataset/rotten_healthy/prp_test',
    transform=transform
)

test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

train_features = []
train_labels = []

with torch.no_grad():
    for imgs, labels in train_loader:
        imgs = imgs.to(device)
        feats = model.forward_features(imgs)
        pooled = feats.mean(dim=[2, 3])  # global average pooling to [B, 2048]

        train_features.append(pooled.cpu())
        train_labels.append(labels)

X_train = torch.cat(train_features).numpy()
y_train = torch.cat(train_labels).numpy()

test_features = []
test_labels = []

with torch.no_grad():
    for imgs, labels in test_loader:
        imgs = imgs.to(device)
        feats = model.forward_features(imgs)
        pooled = feats.mean(dim=[2, 3])

        test_features.append(pooled.cpu())
        test_labels.append(labels)

X_test = torch.cat(test_features).numpy()
y_test = torch.cat(test_labels).numpy()


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

clf = LogisticRegression(max_iter=1000)
clf.fit(X_train, y_train)

from sklearn.metrics import accuracy_score, classification_report

test_preds = clf.predict(X_test)
print(f"Test Accuracy: {accuracy_score(y_test, test_preds):.2%}")
print(classification_report(y_test, test_preds, target_names=["Healthy", "Rotten"]))

joblib.dump(clf, 'rotten_healthy_classifier.pkl')