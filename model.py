import torch
import torch.nn as nn
import numpy as np

class TimeEmbedding(nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim)
        )

    def forward(self, t):
        device = t.device
        half_dim = self.dim // 2
        embeddings = np.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = t[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return self.mlp(embeddings)

class UNetBlock(nn.Module):

    def __init__(self, in_ch, out_ch, embed_dim):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.SiLU(),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.SiLU()
        )
        self.context_mlp = nn.Linear(embed_dim, out_ch)

    def forward(self, x, context_embed):
        h = self.conv(x)
        bias = self.context_mlp(context_embed).unsqueeze(-1).unsqueeze(-1)
        return h + bias

class MNISTCFGBlock(nn.Module):

    def __init__(self, embed_dim=128):
        super().__init__()
        self.embed_dim = embed_dim
        
        # Embeddings
        self.time_embed = TimeEmbedding(embed_dim)
        self.label_embed = nn.Embedding(11, embed_dim) 
        
        # U-Net Downward Path
        self.init_conv = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.down1 = UNetBlock(32, 64, embed_dim)
        self.pool = nn.MaxPool2d(2)
        
        # Bottleneck Middle Path
        self.mid = UNetBlock(64, 64, embed_dim)
        
        # U-Net Upward Path
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.up1 = UNetBlock(64 + 64, 32, embed_dim)  # includes skip connection
        self.final_conv = nn.Conv2d(32, 1, kernel_size=3, padding=1)

    def forward(self, x, t, y):
        t_emb = self.time_embed(t)
        y_emb = self.label_embed(y)
        context = t_emb + y_emb
        
        x1 = self.init_conv(x)
        x2 = self.down1(x1, context)
        x_pooled = self.pool(x2)
        
        mid_feat = self.mid(x_pooled, context)
        
        up_feat = self.upsample(mid_feat)
        merged = torch.cat([up_feat, x2], dim=1) # Skip connection from x2
        x3 = self.up1(merged, context)
        
        return self.final_conv(x3)

if __name__ == "__main__":
    # Quick sanity check test
    model = MNISTCFGBlock()
    test_img = torch.randn(2, 1, 28, 28)
    test_t = torch.tensor([0.1, 0.9])
    test_y = torch.tensor([5, 10]) 
    
    out = model(test_img, test_t, test_y)
    print(f"Input shape:  {test_img.shape}")
    print(f"Output shape: {out.shape}") 