import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import datasets, transforms

from model import MNISTCFGBlock
from generate import reverse_sde_sample


PROJECT_ROOT = Path(__file__).resolve().parent
IMAGE_SIZE = 560
FPS = 5


def _mnist_example(digit):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])
    dataset = datasets.MNIST(
        root=str(PROJECT_ROOT / "data"),
        train=True,
        download=True,
        transform=transform,
    )
    for image, label in dataset:
        if label == digit:
            return image.unsqueeze(0)
    raise RuntimeError(f"No MNIST training example found for digit {digit}")


def _display_image(tensor):
    image = (tensor.detach().cpu().squeeze().clamp(-1, 1) + 1) / 2
    pixels = (image.numpy() * 255).astype(np.uint8)
    return np.asarray(
        Image.fromarray(pixels, mode="L").resize(
            (IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.NEAREST
        )
    )


def build_animation(model, digit, device, steps=200, seed=123, cfg_scale=3.0):
    forward_generator = torch.Generator(device=device).manual_seed(seed)
    clean = _mnist_example(digit).to(device)

    frames = []
    forward_frames = FPS * 6
    reverse_frames = FPS * 22
    final_frames = FPS * 2

    forward_states = [clean.clone()]
    x_forward = clean.clone()
    dt = 1.0 / steps
    for step in range(steps):
        t_value = step / steps
        g = (2.0 * t_value) ** 0.5
        noise = torch.randn(
            x_forward.shape, device=device, dtype=x_forward.dtype,
            generator=forward_generator,
        )
        x_forward = x_forward + g * (dt ** 0.5) * noise
        forward_states.append(x_forward.clone())

    for frame_idx in range(forward_frames):
        state_idx = frame_idx * steps // (forward_frames - 1)
        frames.append(_display_image(forward_states[state_idx][0]))

    reverse_states = []
    model.eval()
    reverse_sde_sample(
        model,
        digit,
        device,
        steps=steps,
        seed=seed,
        cfg_scale=cfg_scale,
        step_callback=lambda _step, image: reverse_states.append(image.clone()),
    )

    for frame_idx in range(reverse_frames):
        state_idx = min(
            len(reverse_states) - 1,
            frame_idx * len(reverse_states) // reverse_frames,
        )
        frames.append(_display_image(reverse_states[state_idx][0]))

    final_image = reverse_states[-1][0]
    for _ in range(final_frames):
        frames.append(_display_image(final_image))
    return frames


def main():
    parser = argparse.ArgumentParser(
        description="Create a 30-second GIF of forward and reverse SDE sampling."
    )
    parser.add_argument("--digit", type=int, default=7, choices=range(10))
    parser.add_argument("--model", default="mnist_cfg_model.pth")
    parser.add_argument("--out", default="digit7_diffusion_30s.gif")
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--cfg-scale", type=float, default=3.0)
    args = parser.parse_args()
    if args.steps < 1:
        parser.error("--steps must be at least 1")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MNISTCFGBlock().to(device)
    checkpoint = Path(args.model)
    if not checkpoint.is_absolute():
        checkpoint = PROJECT_ROOT / checkpoint
    if not checkpoint.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint}. Train first with train.py."
        )
    model.load_state_dict(torch.load(checkpoint, map_location=device))

    frames = build_animation(
        model, args.digit, device, steps=args.steps, seed=args.seed,
        cfg_scale=args.cfg_scale,
    )
    output = Path(args.out)
    if not output.is_absolute():
        output = Path.cwd() / output
    output.parent.mkdir(parents=True, exist_ok=True)
    frame_images = [Image.fromarray(frame) for frame in frames]
    frame_images[0].save(
        output,
        save_all=True,
        append_images=frame_images[1:],
        duration=1000 // FPS,
        loop=0,
        optimize=False,
        disposal=2,
    )
    print(f"Saved {len(frames) / FPS:.1f}-second, {IMAGE_SIZE}x{IMAGE_SIZE} GIF to {output}")


if __name__ == "__main__":
    main()
