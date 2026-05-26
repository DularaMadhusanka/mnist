import argparse
from pathlib import Path

import torch
from torchvision.utils import save_image
from model import MNISTCFGBlock


def sample(model, label, device, steps=200, seed=None):
    if seed is not None:
        torch.manual_seed(seed)
    model.eval()
    x = torch.randn(1, 1, 28, 28, device=device)
    for i in reversed(range(steps)):
        t_scalar = (i + 1) / steps  # in (0, 1]
        t = torch.full((1,), t_scalar, device=device)
        y = torch.tensor([label], device=device)
        with torch.no_grad():
            pred = model(x, t, y)

        # x0_hat = x + t^2 * pred
        x0_hat = x + (t_scalar * t_scalar) * pred


        mix = 1.0 / float(i + 1)
        x = (1.0 - mix) * x + mix * x0_hat

        if i > 0:
            x = x + 0.02 * (t_scalar / steps) * torch.randn_like(x)
    return x


def ddim_sample(model, label, device, steps=200, seed=None, eta=0.0):

    if seed is not None:
        torch.manual_seed(seed)
    model.eval()
    x = torch.randn(1, 1, 28, 28, device=device)
    y = torch.tensor([label], device=device)

    for i in reversed(range(steps)):
        t = float(i + 1) / float(steps)
        t_next = float(i) / float(steps)
        t_tensor = torch.full((1,), t, device=device)
        with torch.no_grad():
            pred = model(x, t_tensor, y)

        x0_hat = x + (t * t) * pred

        if t > 0:
            x = x0_hat + (t_next / t) * (x - x0_hat)
        else:
            x = x0_hat

        if eta and t_next > 0:
            x = x + eta * t_next * torch.randn_like(x)

    return x


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--digit', type=int, required=True, help='Digit class to condition on (0-9)')
    parser.add_argument('--model', type=str, default='mnist_cfg_model.pth', help='Path to model weights')
    parser.add_argument('--out', type=str, default='gen.png', help='Output image path')
    parser.add_argument('--steps', type=int, default=200, help='Number of sampling steps')
    parser.add_argument('--sampler', type=str, default='ddim', choices=['ddim', 'euler'], help='Sampler to use')
    parser.add_argument('--eta', type=float, default=0.0, help='Noise scale for DDIM (0 = deterministic)')
    parser.add_argument('--seed', type=int, default=None, help='Optional random seed')
    args = parser.parse_args()

    if not (0 <= args.digit <= 9):
        raise ValueError('`--digit` must be in [0, 9]')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MNISTCFGBlock().to(device)

    model_path = Path(args.model)
    if not model_path.exists():
        raise FileNotFoundError(
            f'Checkpoint not found: {model_path}. Train first with train.py to create mnist_cfg_model.pth.'
        )

    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    print(f'Loaded model weights from {model_path}')

    if args.sampler == 'ddim':
        img = ddim_sample(model, args.digit, device, steps=args.steps, seed=args.seed, eta=args.eta)
    else:
        img = sample(model, args.digit, device, steps=args.steps, seed=args.seed)
    # model expects normalized images in [-1, 1]; convert to [0,1] for saving
    img = (img.clamp(-1, 1) + 1) / 2.0
    save_image(img, args.out)
    print(f'Generated image saved to {args.out}')
