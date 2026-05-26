# MNIST Conditional Diffusion (Toy)

Simple MNIST conditional score model experiments.

Quick usage:

1. Train (example):

```bash
python train.py
```

This saves `mnist_cfg_model.pth` in the project root.

2. Generate a digit (after training or with random weights):

```bash
python generate.py --digit 7 --out gen7.png --steps 50
```

`generate.py` now requires `mnist_cfg_model.pth`. If the checkpoint is missing, it exits with a clear error instead of sampling from untrained weights.

Requirements:
- Python 3.8+
- torch
- torchvision

This project is a minimal educational example and not tuned for high-quality generation.
