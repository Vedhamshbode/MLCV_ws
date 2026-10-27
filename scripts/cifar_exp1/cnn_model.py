import torch, torch.nn as nn, torch.nn.functional as F, torchvision

CIFAR_MEAN, CIFAR_STD = (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)
IMNET_MEAN, IMNET_STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
CLASSES = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']

# Per-architecture defaults. "_cifar" = trained from scratch at 32x32; the rest = ImageNet-pretrained, fine-tuned.
PRESETS = {
    "convnext_tiny":   dict(pretrained=True,  img=224, epochs=10, opt="adamw", lr=1e-4, wd=0.05, bs=64,  smoothing=0.1, warmup=1),
    "efficientnet_b0": dict(pretrained=True,  img=224, epochs=10, opt="adamw", lr=3e-4, wd=0.01, bs=64,  smoothing=0.1, warmup=1),
    "resnet50":        dict(pretrained=True,  img=224, epochs=10, opt="adamw", lr=2e-4, wd=0.05, bs=64,  smoothing=0.1, warmup=1),
    "resnet18_cifar":  dict(pretrained=False, img=32,  epochs=60, opt="sgd",   lr=0.1,  wd=5e-4, bs=128, smoothing=0.0, warmup=2),
}

def _replace_last_linear(m, n_out):
    last = None
    for name, mod in m.named_modules():
        if isinstance(mod, nn.Linear): last = (name, mod)
    parent = m; parts = last[0].split(".")
    for p in parts[:-1]: parent = getattr(parent, p)
    setattr(parent, parts[-1], nn.Linear(last[1].in_features, n_out))

def build_model(arch, pretrained=False, num_classes=10):
    base = arch.replace("_cifar", "")
    m = torchvision.models.get_model(base, weights="DEFAULT" if pretrained else None)
    if arch.endswith("_cifar"):                     # CIFAR stem: 3x3 stride-1 conv, no max-pool
        m.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False); m.maxpool = nn.Identity()
    _replace_last_linear(m, num_classes)
    return m

def prepare(x01, img, mean, std):
    """x01: float (B,3,32,32) in [0,1] -> resized + normalised network input."""
    if img != x01.shape[-1]:
        x01 = F.interpolate(x01, size=(img, img), mode="bicubic", align_corners=False).clamp(0, 1)
    m = torch.tensor(mean, device=x01.device).view(1, 3, 1, 1); s = torch.tensor(std, device=x01.device).view(1, 3, 1, 1)
    return ((x01 - m) / s).contiguous(memory_format=torch.channels_last)

def load_checkpoint(path, device="cpu"):
    ck = torch.load(path, map_location=device)
    m = build_model(ck["arch"], pretrained=False).to(device)
    m.load_state_dict(ck["state_dict"]); m.eval()
    return m, ck
