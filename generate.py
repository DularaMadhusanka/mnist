import argparse
from pathlib import Path

import torch
from torchvision.utils import save_image
from model import MNISTCFGBlock

PROJECT_ROOT = Path(__file__).resolve().parent


def ddim_sample(
    model, label, device, steps=200, seed=None, eta=0.0, cfg_scale=1.0,
    step_callback=None,
):

    if seed is not None:
        torch.manual_seed(seed)
    model.eval()
    x = torch.randn(1, 1, 28, 28, device=device)
    y = torch.tensor([label], device=device)
    y_uncond = torch.tensor([10], device=device)

    for i in reversed(range(steps)):
        t = float(i + 1) / float(steps)
        t_next = float(i) / float(steps)
        t_tensor = torch.full((1,), t, device=device)
        with torch.no_grad():
            if cfg_scale != 1.0:
                # classifier-free guidance: get unconditional and conditional preds
                pred_uncond = model(x, t_tensor, y_uncond)
                pred_cond = model(x, t_tensor, y)
                pred = pred_uncond + cfg_scale * (pred_cond - pred_uncond)
            else:
                pred = model(x, t_tensor, y)

        x0_hat = x + (t * t) * pred

        if t > 0:
            x = x0_hat + (t_next / t) * (x - x0_hat)
        else:
            x = x0_hat

        if eta and t_next > 0:
            x = x + eta * t_next * torch.randn_like(x)
        if step_callback is not None:
            step_callback(i, x)

    return x


def reverse_sde_sample(
    model, label, device, steps=200, seed=None, cfg_scale=1.0,
    step_callback=None,
):
    if steps < 1:
        raise ValueError("steps must be at least 1")

    generator = None
    if seed is not None:
        generator = torch.Generator(device=device).manual_seed(seed)

    model.eval()
    x = torch.randn((1, 1, 28, 28), device=device, generator=generator)
    y = torch.tensor([label], device=device)
    y_uncond = torch.tensor([10], device=device)
    dt = 1.0 / steps

    # For the forward VE SDE f=0, g(t)=sqrt(2t). Integrate
    # dx=[-g(t)^2 score]dt+g(t)dW_bar with dt=-step_size.
    for i in reversed(range(steps)):
        t_value = (i + 1) / steps
        t = torch.full((1,), t_value, device=device)
        with torch.no_grad():
            if cfg_scale != 1.0:
                score_uncond = model(x, t, y_uncond)
                score_cond = model(x, t, y)
                score = score_uncond + cfg_scale * (score_cond - score_uncond)
            else:
                score = model(x, t, y)

        g_squared = 2.0 * t_value
        noise = torch.randn(
            x.shape, device=device, dtype=x.dtype, generator=generator
        )
        x = x + g_squared * score * dt + (g_squared * dt) ** 0.5 * noise
        if step_callback is not None:
            step_callback(i, x)

    return x


def _denoise_step(model, x, t_scalar, label, device, cfg_scale=1.0):
    # Helper used by legacy 'sample' behavior: a deterministic denoising step with CFG
    t = torch.full((1,), t_scalar, device=device)
    y = torch.tensor([label], device=device)
    y_uncond = torch.tensor([10], device=device)
    with torch.no_grad():
        if cfg_scale != 1.0:
            pred_uncond = model(x, t, y_uncond)
            pred_cond = model(x, t, y)
            pred = pred_uncond + cfg_scale * (pred_cond - pred_uncond)
        else:
            pred = model(x, t, y)
    x0_hat = x + (t_scalar * t_scalar) * pred
    return x0_hat


def sample(model, label, device, steps=200, seed=None, cfg_scale=1.0):
    if seed is not None:
        torch.manual_seed(seed)
    model.eval()
    x = torch.randn(1, 1, 28, 28, device=device)
    for i in reversed(range(steps)):
        t_scalar = (i + 1) / steps  # in (0, 1]
        x0_hat = _denoise_step(model, x, t_scalar, label, device, cfg_scale=cfg_scale)

        mix = 1.0 / float(i + 1)
        x = (1.0 - mix) * x + mix * x0_hat

        if i > 0:
            x = x + 0.02 * (t_scalar / steps) * torch.randn_like(x)
    return x


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--digit', type=int, required=True, help='Digit class to condition on (0-9)')
    parser.add_argument('--model', type=str, default='mnist_cfg_model.pth', help='Path to model weights')
    parser.add_argument('--out', type=str, default='gen.png', help='Output image path')
    parser.add_argument('--steps', type=int, default=200, help='Number of sampling steps')
    parser.add_argument('--sampler', type=str, default='ddim', choices=['ddim', 'euler'], help='Sampler to use')
    parser.add_argument('--eta', type=float, default=0.0, help='Noise scale for DDIM (0 = deterministic)')
    parser.add_argument('--cfg-scale', type=float, default=3.0, help='Classifier-free guidance scale (1.0 = off)')
    parser.add_argument('--no-ema', action='store_true', help='Do not load EMA weights even if available')
    parser.add_argument('--seed', type=int, default=None, help='Optional random seed')
    args = parser.parse_args()

    if not (0 <= args.digit <= 9):
        raise ValueError('`--digit` must be in [0, 9]')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MNISTCFGBlock().to(device)

    model_path = Path(args.model)
    if not model_path.is_absolute():
        model_path = PROJECT_ROOT / model_path
    if not model_path.exists():
        raise FileNotFoundError(
            f'Checkpoint not found: {model_path}. Train first with train.py to create mnist_cfg_model.pth.'
        )

    # Load EMA if available and not disabled; fallback to standard checkpoint
    use_ema = not args.no_ema
    ema_path = PROJECT_ROOT / 'mnist_cfg_model_ema.pth'
    if use_ema and ema_path.exists():
        state = torch.load(ema_path, map_location=device)
        model.load_state_dict(state)
        print(f'Loaded EMA weights from {ema_path}')
    else:
        state = torch.load(model_path, map_location=device)
        model.load_state_dict(state)
        print(f'Loaded model weights from {model_path}')

    if args.sampler == 'ddim':
        img = ddim_sample(model, args.digit, device, steps=args.steps, seed=args.seed, eta=args.eta, cfg_scale=args.cfg_scale)
    else:
        img = reverse_sde_sample(
            model, args.digit, device, steps=args.steps, seed=args.seed,
            cfg_scale=args.cfg_scale,
        )
    # model expects normalized images in [-1, 1]; convert to [0,1] for saving
    img = (img.clamp(-1, 1) + 1) / 2.0
    output_path = Path(args.out)
    if not output_path.is_absolute():
        output_path = Path.cwd() / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_image(img, output_path)
    print(f'Generated image saved to {output_path}')
