"""Train ViT on CIFAR-10 (uses the cifar10_train.pt your notebook creates).
  python train.py                      # best-known config, 100 epochs
  python train.py --sweep              # compare configs for a few epochs each
"""
import argparse, json, time, numpy as np, torch, torch.nn as nn
from model import ViT

MEAN = (0.4914, 0.4822, 0.4465); STD = (0.2470, 0.2435, 0.2616)
CLASSES = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']
BASE = dict(patch=4, dim=192, depth=6, heads=6, mlp_dim=384, drop=0.1,
            lr=1e-3, wd=0.05, bs=128, epochs=100, smoothing=0.1, aug=True, warmup=5)

def load(path, dev):
    d = torch.load(path); X, y = d["X"], d["y"]
    rng = np.random.default_rng(42); yn = y.numpy(); tr, va = [], []
    for c in range(10):                       # same stratified split as your notebook
        i = np.where(yn == c)[0]; rng.shuffle(i); va += list(i[:1000]); tr += list(i[1000:])
    return X[tr].to(dev), y[tr].to(dev), X[va].to(dev), y[va].to(dev)

def prep(x, dev):
    m = torch.tensor(MEAN, device=dev).view(1,3,1,1); s = torch.tensor(STD, device=dev).view(1,3,1,1)
    return (x.float()/255 - m) / s

def augment(x, pad=4):                        # random crop + horizontal flip, on GPU
    B, C, H, W = x.shape
    x = torch.where(torch.rand(B,1,1,1, device=x.device) < .5, x.flip(3), x)
    xp = nn.functional.pad(x, (pad,)*4, mode="reflect")
    i = torch.randint(0, 2*pad+1, (B,), device=x.device); j = torch.randint(0, 2*pad+1, (B,), device=x.device)
    r = (torch.arange(H, device=x.device)[None] + i[:,None]); c = (torch.arange(W, device=x.device)[None] + j[:,None])
    return xp[torch.arange(B,device=x.device)[:,None,None,None], torch.arange(C,device=x.device)[None,:,None,None],
              r[:,None,:,None], c[:,None,None,:]]

@torch.no_grad()
def evaluate(model, X, y, dev):
    model.eval(); ok = 0
    for k in range(0, len(X), 500):
        with torch.autocast(dev.type, enabled=dev.type=="cuda"):
            ok += (model(prep(X[k:k+500], dev)).argmax(1) == y[k:k+500]).sum().item()
    return 100*ok/len(X)

def run(cfg, data, dev, save=None, verbose=True):
    Xtr, ytr, Xva, yva = data
    model = ViT(patch=cfg["patch"], dim=cfg["dim"], depth=cfg["depth"], heads=cfg["heads"],
                mlp_dim=cfg["mlp_dim"], drop=cfg["drop"]).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"])
    steps = cfg["epochs"] * (len(Xtr)//cfg["bs"]); wu = cfg["warmup"] * (len(Xtr)//cfg["bs"])
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: (s+1)/wu if s < wu else
            0.5*(1+np.cos(np.pi*(s-wu)/max(1,steps-wu))))
    crit = nn.CrossEntropyLoss(label_smoothing=cfg["smoothing"])
    scaler = torch.amp.GradScaler(enabled=dev.type=="cuda"); best = 0
    for ep in range(cfg["epochs"]):
        model.train(); perm = torch.randperm(len(Xtr), device=dev); t0 = time.time(); ok = 0
        for k in range(0, len(Xtr) - cfg["bs"] + 1, cfg["bs"]):
            idx = perm[k:k+cfg["bs"]]; x = Xtr[idx].float()/255
            if cfg["aug"]: x = augment(x)
            x = (x - torch.tensor(MEAN,device=dev).view(1,3,1,1)) / torch.tensor(STD,device=dev).view(1,3,1,1)
            with torch.autocast(dev.type, enabled=dev.type=="cuda"):
                out = model(x); loss = crit(out, ytr[idx])
            opt.zero_grad(set_to_none=True); scaler.scale(loss).backward()
            scaler.unscale_(opt); nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update(); sched.step()
            ok += (out.argmax(1) == ytr[idx]).sum().item()
        va = evaluate(model, Xva, yva, dev)
        if va > best:
            best = va
            if save: torch.save({"state_dict": model.state_dict(), "config": cfg, "classes": CLASSES,
                                 "mean": MEAN, "std": STD, "val_acc": va}, save)
        if verbose: print(f"ep {ep+1:3d} train {100*ok/(len(Xtr)//cfg['bs']*cfg['bs']):.2f} val {va:.2f} best {best:.2f} ({time.time()-t0:.0f}s)", flush=True)
    return best

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="cifar10_train.pt"); p.add_argument("--out", default="vit_best.pt")
    p.add_argument("--sweep", action="store_true"); p.add_argument("--sweep_epochs", type=int, default=15)
    p.add_argument("--epochs", type=int, default=None)
    a = p.parse_args(); dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = load(a.data, dev)
    if a.sweep:
        # one-factor-at-a-time changes from your original notebook settings -> see what each buys you
        orig = dict(BASE, patch=8, dim=32, depth=4, heads=4, mlp_dim=64, drop=0., lr=1e-2, wd=0., bs=64,
                    smoothing=0., aug=False, warmup=1, epochs=a.sweep_epochs)
        steps = [("original", {}), ("+lr 1e-3 / AdamW / warmup+cosine", dict(lr=1e-3, wd=0.05)),
                 ("+patch 4", dict(patch=4)), ("+dim 192,heads 6,mlp 384", dict(dim=192, heads=6, mlp_dim=384)),
                 ("+depth 6", dict(depth=6)), ("+augmentation", dict(aug=True)),
                 ("+dropout .1, label smoothing .1", dict(drop=.1, smoothing=.1)), ("+bs 128", dict(bs=128))]
        cfg = dict(orig); res = []
        for name, ch in steps:                     # cumulative
            cfg.update(ch); acc = run(dict(cfg), data, dev, verbose=False); res.append((name, acc))
            print(f"{name:40s} best val {acc:.2f}", flush=True)
        json.dump(res, open("sweep_results.json", "w"), indent=1)
    else:
        cfg = dict(BASE); 
        if a.epochs: cfg["epochs"] = a.epochs
        print("config:", cfg); print("best val acc:", run(cfg, data, dev, save=a.out)); print("saved", a.out)
