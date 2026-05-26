import torch
from dataset import get_mnist_dataloader
from model import MNISTCFGBlock

# tiny smoke test
device = torch.device('cpu')
loader = get_mnist_dataloader(batch_size=4)
images, labels = next(iter(loader))
model = MNISTCFGBlock().to(device)
images = images.to(device)
labels = labels.to(device)

# sample t
t = torch.rand(images.shape[0]) * 0.999 + 0.001
# cfg dropout
dropout_mask = torch.rand(images.shape[0]) < 0.2
labels[dropout_mask] = 10

std = t.view(-1,1,1,1)
noise = torch.randn_like(images)
noisy = images + std * noise
pred = model(noisy, t, labels)
loss = torch.mean(torch.sum((std * pred + noise) ** 2, dim=[1,2,3]))
print('Smoke test loss:', loss.item())
loss.backward()
print('Backward succeeded')
