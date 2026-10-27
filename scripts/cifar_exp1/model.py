import torch, torch.nn as nn

class PatchEmbedding(nn.Module):
    def __init__(self, in_ch, dim, patch):
        super().__init__()
        self.proj = nn.Conv2d(in_ch, dim, kernel_size=patch, stride=patch)
    def forward(self, x):
        return self.proj(x).flatten(2).transpose(1, 2)          # (B, N, D)

class Block(nn.Module):
    def __init__(self, dim, heads, mlp_dim, drop):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, dropout=drop, batch_first=True)
        self.ln2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, mlp_dim), nn.GELU(), nn.Dropout(drop),
                                 nn.Linear(mlp_dim, dim), nn.Dropout(drop))
    def forward(self, x, return_attn=False):
        h = self.ln1(x)
        a, w = self.attn(h, h, h, need_weights=return_attn, average_attn_weights=False)
        x = x + a
        x = x + self.mlp(self.ln2(x))
        return (x, w) if return_attn else x

class ViT(nn.Module):
    def __init__(self, img_size=32, patch=4, in_ch=3, num_classes=10, dim=192,
                 depth=6, heads=6, mlp_dim=384, drop=0.1):
        super().__init__()
        n = (img_size // patch) ** 2
        self.patch_embedding = PatchEmbedding(in_ch, dim, patch)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos = nn.Parameter(torch.zeros(1, n + 1, dim))
        nn.init.trunc_normal_(self.pos, std=0.02); nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.drop = nn.Dropout(drop)
        self.blocks = nn.ModuleList([Block(dim, heads, mlp_dim, drop) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, num_classes)
        self.apply(self._init)
    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None: nn.init.zeros_(m.bias)
    def embed(self, x):
        x = self.patch_embedding(x)
        x = torch.cat([self.cls_token.expand(x.size(0), -1, -1), x], 1) + self.pos
        return self.drop(x)
    def forward(self, x):
        x = self.embed(x)
        for b in self.blocks: x = b(x)
        return self.head(self.norm(x)[:, 0])
