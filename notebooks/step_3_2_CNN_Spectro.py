# L 4-23-25 [milestone: cnn_inference_ready]
# notebooks/step_3_2_CNN_Spectro.py

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support
import numpy as np
import pandas as pd
from PIL import Image

# Constants
IMG_SIZE = 128
BATCH_SIZE = 32
EPOCHS = 5

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "../spectrograms")
REPORT_DIR = os.path.join(BASE_DIR, "../reports/3_CNN_Spectrogram_Classifier")
MODEL_DIR = os.path.join(BASE_DIR, "../models")
MODEL_PATH = os.path.join(MODEL_DIR, "cnn_model.pth")
BUNDLE_PATH = os.path.join(MODEL_DIR, "cnn_inference_bundle.pth")


class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super(SimpleCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 16 * 16, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


def cnn_predict_folder(bundle_path, image_dir):
    bundle = torch.load(bundle_path, map_location=torch.device(
        'cpu'), weights_only=False)

    model = SimpleCNN(num_classes=len(bundle['class_names']))
    model.load_state_dict(bundle['model_state_dict'])
    model.eval()

    transform = bundle['transform']

    class_names = bundle['class_names']

    predictions = []

    for genre in os.listdir(image_dir):
        genre_path = os.path.join(image_dir, genre)
        if not os.path.isdir(genre_path):
            continue

        for img_file in os.listdir(genre_path):
            if not img_file.endswith(".png"):
                continue

            img_path = os.path.join(genre_path, img_file)
            try:
                image = Image.open(img_path).convert("RGB")
                input_tensor = transform(image).unsqueeze(0)  # add batch dim
                with torch.no_grad():
                    outputs = model(input_tensor)
                    _, pred = torch.max(outputs, 1)
                    predictions.append({
                        "file": img_file,
                        "true_genre": genre,
                        "CNN": class_names[pred.item()]
                    })
            except Exception as e:
                print(f"[ERROR] CNN inference failed for {img_path}: {e}")

    return pd.DataFrame(predictions)


def run():
    os.makedirs(REPORT_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Training CNN on: {DATA_DIR}")

    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])

    dataset = datasets.ImageFolder(DATA_DIR, transform=transform)
    class_names = dataset.classes

    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, test_size])
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = SimpleCNN(num_classes=len(class_names)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        avg_loss = running_loss / len(train_loader)
        print(f"[INFO] Epoch {epoch+1}, Loss: {avg_loss:.4f}")

    # Save model and bundle
    torch.save(model.state_dict(), MODEL_PATH)
    torch.save({
        'model_state_dict': model.state_dict(),
        'class_names': class_names,
        'transform': transform,
        'img_size': IMG_SIZE,
    }, BUNDLE_PATH)

    # Evaluation
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            y_true.extend(labels.numpy())
            y_pred.extend(preds.cpu().numpy())

    report = classification_report(y_true, y_pred, target_names=class_names)
    with open(os.path.join(REPORT_DIR, "cnn_classification_report.txt"), "w") as f:
        f.write(report)
    print(report)

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, cmap="Blues", interpolation="nearest")
    plt.title("Confusion Matrix - CNN")
    plt.colorbar()
    plt.xticks(np.arange(len(class_names)), class_names, rotation=90)
    plt.yticks(np.arange(len(class_names)), class_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(os.path.join(REPORT_DIR, "cnn_confusion_matrix.png"))
    plt.close()

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=range(len(class_names)), zero_division=0)
    metrics = {"Precision": precision, "Recall": recall, "F1-Score": f1}
    for metric_name, values in metrics.items():
        plt.figure(figsize=(10, 5))
        plt.bar(class_names, values)
        plt.title(f"{metric_name} per Genre - CNN")
        plt.ylabel(metric_name)
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        fname = f"cnn_{metric_name.lower().replace('-', '_')}_bar.png"
        plt.savefig(os.path.join(REPORT_DIR, fname))
        plt.close()


if __name__ == "__main__":
    run()
