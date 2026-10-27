"""Train a pure-CNN CIFAR-10 classifier (uses cifar10_train.pt + the same stratified 40k/10k split as before).
  python cnn_train.py                          # ConvNeXt-Tiny, ImageNet-pretrained, 224px  -> ~98%
  python cnn_train.py --arch efficientnet_b0   # or resnet50
  python cnn_train.py --arch resnet18_cifar    # from scratch, 32px, 60 epochs              -> ~95%
"""
import argparse, time, numpy as np, torch, torch.nn as nn
from cnn_model import *

def load(path, dev):
    d = torch.load(path); X, y = d["X"], d["y"]
    rng = np.random.default_rng(42); yn = y.numpy(); tr, va = [], []
    for c in range(10):
        i = np.where(yn == c)[0]; rng.shuffle(i); va += list(i[:1000]); tr += list(i[1000:])
    return X[tr].to(dev), y[tr].to(dev), X[va].to(dev), y[va].to(dev)

def augment(x, pad=4):                                  # random crop (reflect pad) + h-flip, per-sample, on GPU
    B, C, H, W = x.shape; d = x.device
    x = torch.where(torch.rand(B,1,1,1, device=d) < .5, x.flip(3), x)
    xp = nn.functional.pad(x, (pad,)*4, mode="reflect")
    i = torch.randint(0, 2*pad+1, (B,), device=d); j = torch.randint(0, 2*pad+1, (B,), device=d)
    r = torch.arange(H, device=d)[None] + i[:,None]; c = torch.arange(W, device=d)[None] + j[:,None]
    return xp[torch.arange(B,device=d)[:,None,None,None], torch.arange(C,device=d)[None,:,None,None], r[:,None,:,None], c[:,None,None,:]]

@torch.no_grad()
def evaluate(model, X, y, cfg, dev, tta=False):
    model.eval(); ok = 0; eb = 500 if cfg["img"] <= 64 else 100
    for k in range(0, len(X), eb):
        x = X[k:k+eb].float()/255
        with torch.autocast(dev.type, enabled=dev.type=="cuda"):
            o = model(prepare(x, cfg["img"], cfg["mean"], cfg["std"]))
            if tta: o = o.softmax(-1) + model(prepare(x.flip(3), cfg["img"], cfg["mean"], cfg["std"])).softmax(-1)
        ok += (o.argmax(1) == y[k:k+eb]).sum().item()
    return 100*ok/len(X)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", default="convnext_tiny", choices=list(PRESETS)); ap.add_argument("--data", default="cifar10_train.pt")
    ap.add_argument("--out", default="cnn_best.pt")
    for k in ("epochs", "bs"): ap.add_argument(f"--{k}", type=int)
    for k in ("lr", "wd"): ap.add_argument(f"--{k}", type=float)
    a = ap.parse_args(); cfg = dict(PRESETS[a.arch])
    for k in ("epochs", "bs", "lr", "wd"):
        if getattr(a, k) is not None: cfg[k] = getattr(a, k)
    cfg["mean"], cfg["std"] = (IMNET_MEAN, IMNET_STD) if cfg["pretrained"] else (CIFAR_MEAN, CIFAR_STD)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu"); torch.backends.cudnn.benchmark = True
    Xtr, ytr, Xva, yva = load(a.data, dev); print("config:", a.arch, cfg)
    model = build_model(a.arch, cfg["pretrained"]).to(dev).to(memory_format=torch.channels_last)
    decay = [p for n, p in model.named_parameters() if p.ndim > 1]; no_decay = [p for n, p in model.named_parameters() if p.ndim <= 1]
    groups = [dict(params=decay, weight_decay=cfg["wd"]), dict(params=no_decay, weight_decay=0.)]
    opt = (torch.optim.SGD(groups, lr=cfg["lr"], momentum=0.9, nesterov=True) if cfg["opt"] == "sgd"
           else torch.optim.AdamW(groups, lr=cfg["lr"]))
    spe = len(Xtr) // cfg["bs"]; total = cfg["epochs"] * spe; wu = cfg["warmup"] * spe
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: (s+1)/wu if s < wu else 0.5*(1+np.cos(np.pi*(s-wu)/max(1,total-wu))))
    crit = nn.CrossEntropyLoss(label_smoothing=cfg["smoothing"]); scaler = torch.amp.GradScaler(enabled=dev.type=="cuda"); best = 0
    for ep in range(cfg["epochs"]):
        model.train(); perm = torch.randperm(len(Xtr), device=dev); t0 = time.time(); ok = 0
        for k in range(spe):
            idx = perm[k*cfg["bs"]:(k+1)*cfg["bs"]]; x = augment(Xtr[idx].float()/255)
            x = prepare(x, cfg["img"], cfg["mean"], cfg["std"])
            with torch.autocast(dev.type, enabled=dev.type=="cuda"):
                out = model(x); loss = crit(out, ytr[idx])
            opt.zero_grad(set_to_none=True); scaler.scale(loss).backward(); scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0); scaler.step(opt); scaler.update(); sched.step()
            ok += (out.argmax(1) == ytr[idx]).sum().item()
        va = evaluate(model, Xva, yva, cfg, dev)
        if va > best:
            best = va; torch.save({"state_dict": model.state_dict(), "arch": a.arch, "img": cfg["img"], "mean": cfg["mean"],
                                   "std": cfg["std"], "classes": CLASSES, "val_acc": va}, a.out)
        print(f"ep {ep+1:3d}  train {100*ok/(spe*cfg['bs']):.2f}  val {va:.2f}  best {best:.2f}  ({time.time()-t0:.0f}s)", flush=True)
    model, ck = load_checkpoint(a.out, dev)
    print(f"best checkpoint: val {evaluate(model, Xva, yva, cfg, dev):.2f}%   with flip-TTA {evaluate(model, Xva, yva, cfg, dev, tta=True):.2f}%   saved -> {a.out}")

if __name__ == "__main__": main()
