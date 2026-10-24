"""
ELEC 475: Lab 2 - Held-out Tiny ImageNet evaluation of saved checkpoints.

Evaluates saved weights on the official 10,000-image Tiny ImageNet validation split
(val/images + val/val_annotations.txt). It does not modify or import the loader in utils/dataset.py,
so the starter code stays untouched. Run from the project root:

    python eval_tiny_heldout.py --ckpt-dir /content/drive/MyDrive/lab2_out --data-dir ./data
"""
import argparse
import os

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, Dataset

from models.alexnet import AlexNet64
from models.lenet import LeNet5
from models.lenet_modern import LeNetModern
from models.zoo import get_torchvision_alexnet
from utils.transforms import get_test_transforms

NUM_CLASSES = 200

# name -> (checkpoint file, model constructor)
MODELS = {
    "Base LeNet-5":     ("lenet_tiny.pth",             lambda: LeNet5(NUM_CLASSES)),
    "Modernized LeNet": ("lenet_modern_tiny.pth",      lambda: LeNetModern(NUM_CLASSES)),
    "AlexNet + BN":     ("alexnet_tiny_imagenet.pth",  lambda: AlexNet64(NUM_CLASSES, 3, "bn")),
    "Zoo AlexNet":      ("zoo_alexnet_tiny.pth",       lambda: get_torchvision_alexnet(NUM_CLASSES)),
    "Zoo (lr 1e-4)":    ("zoo_alexnet_tiny_lr1e-4.pth", lambda: get_torchvision_alexnet(NUM_CLASSES)),
}


class TinyVal(Dataset):
    def __init__(self, tiny_dir, transform):
        train_dir = os.path.join(tiny_dir, "train")
        # same class order as torchvision ImageFolder: sorted class folder names
        wnids = sorted(d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d)))
        class_to_idx = {w: i for i, w in enumerate(wnids)}
        img_dir = os.path.join(tiny_dir, "val", "images")
        self.samples = []
        with open(os.path.join(tiny_dir, "val", "val_annotations.txt")) as f:
            for line in f:
                parts = line.split()
                if len(parts) >= 2:
                    self.samples.append((os.path.join(img_dir, parts[0]), class_to_idx[parts[1]]))
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        path, label = self.samples[i]
        return self.transform(Image.open(path).convert("RGB")), label


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    preds, labels, loss_sum = [], [], 0.0
    ce = torch.nn.CrossEntropyLoss(reduction="sum")
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        out = model(x)
        loss_sum += ce(out, y).item()
        preds.append(out.argmax(1).cpu().numpy())
        labels.append(y.cpu().numpy())
    preds, labels = np.concatenate(preds), np.concatenate(labels)
    acc = 100.0 * (preds == labels).mean()
    f1 = 100.0 * f1_score(labels, preds, labels=list(range(NUM_CLASSES)), average="macro", zero_division=0)
    return loss_sum / len(labels), acc, f1, len(labels)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-dir", default="checkpoints")
    ap.add_argument("--data-dir", default="./data")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tiny_dir = os.path.join(args.data_dir, "tiny-imagenet-200")
    val_set = TinyVal(tiny_dir, get_test_transforms("tiny_imagenet", target_size=64))
    loader = DataLoader(val_set, batch_size=256, shuffle=False, num_workers=2)
    print(f"Held-out Tiny ImageNet validation images: {len(val_set)}  (device: {device})")
    print(f"{'Model':18s} {'Val loss':>9s} {'Acc (%)':>9s} {'Macro-F1 (%)':>13s}")
    for name, (ckpt, make) in MODELS.items():
        path = os.path.join(args.ckpt_dir, ckpt)
        if not os.path.exists(path):
            print(f"{name:18s} missing checkpoint: {path}")
            continue
        model = make().to(device)
        model.load_state_dict(torch.load(path, map_location=device))
        loss, acc, f1, n = evaluate(model, loader, device)
        print(f"{name:18s} {loss:9.4f} {acc:9.2f} {f1:13.2f}")


if __name__ == "__main__":
    main()
