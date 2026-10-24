"""
ELEC 475: Lab 2 - Batch-1 inference latency benchmark (order-independent).

Warms every model up first, then measures them in interleaved rounds in random order and
reports the median, so GPU warm-up and clock ramp-up do not favour whichever model is timed
last. Run from the project root:   python latency_benchmark.py
"""
import random
import statistics as st
import time

import torch

from models.alexnet import AlexNet64
from models.lenet import LeNet5
from models.lenet_modern import LeNetModern
from models.zoo import get_torchvision_alexnet

dev = "cuda" if torch.cuda.is_available() else "cpu"
makers = {
    "LeNet-5":      lambda: LeNet5(10),
    "LeNet-Modern": lambda: LeNetModern(10),
    "AlexNet none": lambda: AlexNet64(10, 3, "none"),
    "AlexNet LRN":  lambda: AlexNet64(10, 3, "lrn"),
    "AlexNet BN":   lambda: AlexNet64(10, 3, "bn"),
    "Zoo AlexNet":  lambda: get_torchvision_alexnet(10),
}
models = {k: f().to(dev).eval() for k, f in makers.items()}
x = torch.randn(1, 3, 64, 64, device=dev)


def sync():
    if dev == "cuda":
        torch.cuda.synchronize()


def timed(m, n):
    sync()
    t = time.perf_counter()
    with torch.no_grad():
        for _ in range(n):
            m(x)
    sync()
    return (time.perf_counter() - t) / n * 1000  # ms per forward pass


if __name__ == "__main__":
    print(f"Device: {dev}")
    for m in models.values():          # warm-up
        timed(m, 200)

    res = {k: [] for k in models}
    for _ in range(15):                # interleaved rounds, random order
        keys = list(models)
        random.shuffle(keys)
        for k in keys:
            res[k].append(timed(models[k], 100))

    for k, v in res.items():
        print(f"{k:14s} median {st.median(v):.2f} ms   (min {min(v):.2f}, max {max(v):.2f})")
