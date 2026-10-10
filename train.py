import argparse
from pathlib import Path

import torch
import torch.optim as optim
from dataset import get_mnist_dataloader
from model import MNISTCFGBlock

PROJECT_ROOT = Path(__file__).resolve().parent


def train(epochs=5, ema_decay=0.0):
    # Setup device (use GPU if you have one, otherwise CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

    # Hyperparameters
    batch_size = 128
    lr = 2e-4
    cfg_dropout_rate = 0.2  # 20% chance to drop the prompt

    # Initialize Data, Model, and Optimizer
    dataloader = get_mnist_dataloader(batch_size)
    model = MNISTCFGBlock().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # EMA setup
    use_ema = ema_decay and ema_decay > 0.0
    ema_model = None
    if use_ema:
        ema_model = MNISTCFGBlock().to(device)
        ema_model.load_state_dict(model.state_dict())
        for p in ema_model.parameters():
            p.requires_grad = False

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for batch_idx, (images, labels) in enumerate(dataloader):
            images = images.to(device)
            labels = labels.to(device)
            current_batch_size = images.shape[0]

            # 1. Sample random time steps between 0.001 and 1.0
            t = torch.rand(current_batch_size, device=device) * 0.999 + 0.001

            # 2. CFG Random Dropout: Replace 20% of labels with 10 (Blank Token)
            dropout_mask = torch.rand(current_batch_size, device=device) < cfg_dropout_rate
            labels[dropout_mask] = 10

            # VE SDE: f=0, g(t)=sqrt(2t), giving sigma_t=t and
            # x_t = x_0 + sigma_t * epsilon.
            sigma_t = t.view(-1, 1, 1, 1)
            noise = torch.randn_like(images)
            noisy_images = images + sigma_t * noise

            # 4. Predict the Score using our Noisy Image, Time, and (sometimes dropped) Label
            optimizer.zero_grad()
            predicted_score = model(noisy_images, t, labels)

            # E[||sigma_t * s_theta(x_t, t) + epsilon||^2]
            loss = torch.mean(
                torch.sum((sigma_t * predicted_score + noise) ** 2, dim=[1, 2, 3])
            )

            # 6. Backpropagation
            loss.backward()
            optimizer.step()

            # EMA update
            if use_ema:
                with torch.no_grad():
                    for p, ema_p in zip(model.parameters(), ema_model.parameters()):
                        ema_p.data.mul_(ema_decay).add_(p.data, alpha=1.0 - ema_decay)

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch + 1}/{epochs} | Average Loss: {avg_loss:.4f}")

    # Save the trained brain to disk
    model_path = PROJECT_ROOT / "mnist_cfg_model.pth"
    torch.save(model.state_dict(), model_path)
    print(f"Model successfully saved to {model_path}")
    if use_ema:
        ema_path = PROJECT_ROOT / "mnist_cfg_model_ema.pth"
        torch.save(ema_model.state_dict(), ema_path)
        print(f"EMA model successfully saved to {ema_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=5, help='Number of training epochs')
    parser.add_argument('--ema-decay', type=float, default=0.0, help='EMA decay (0 to disable)')
    args = parser.parse_args()
    train(epochs=args.epochs, ema_decay=args.ema_decay)
