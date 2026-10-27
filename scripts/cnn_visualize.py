"""Show how ONE image is transformed by every Conv2d layer of a trained CNN (trained weights from cnn_best.pt).
  python cnn_visualize.py --ckpt cnn_best.pt --image cat.png
  python cnn_visualize.py --ckpt cnn_best.pt --data cifar10_train.pt --index 7 --channels 24
  python cnn_visualize.py --ckpt cnn_best.pt --list_layers
In a notebook:  from cnn_visualize import visualize; visualize("cnn_best.pt", index=7, data="cifar10_train.pt", show=True)
Outputs (default folder viz_cnn/): 00_overview.png, 01_first_layer_kernels.png, layer_XX_<name>.png (one per conv layer)."""
import argparse, os, numpy as np, torch, torch.nn as nn, matplotlib.pyplot as plt
from PIL import Image
from cnn_model import load_checkpoint, prepare

def n01(a): a = a - a.min(); return a / (a.max() + 1e-8)

def collect(model, x):
    acts, hooks = [], []
    for name, m in model.named_modules():
        if isinstance(m, nn.Conv2d):
            hooks.append(m.register_forward_hook(lambda mod, i, o, n=name: acts.append((n, mod, o.detach().float().cpu()[0]))))
    with torch.no_grad(): logits = model(x)
    for h in hooks: h.remove()
    return acts, logits

def _finish(fig, path, show):
    fig.savefig(path, dpi=130, bbox_inches="tight")
    if show: plt.show()
    plt.close(fig)

def visualize(ckpt, image=None, data=None, index=0, out="viz_cnn", channels=16, order="energy", show=False, device=None):
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu"); os.makedirs(out, exist_ok=True)
    model, ck = load_checkpoint(ckpt, dev)
    if image: img = np.array(Image.open(image).convert("RGB").resize((32, 32), Image.BILINEAR))
    else: img = torch.load(data)["X"][index].permute(1, 2, 0).numpy()
    x01 = torch.tensor(img).permute(2, 0, 1).float()[None].to(dev) / 255
    acts, logits = collect(model, prepare(x01, ck["img"], ck["mean"], ck["std"]))
    p = logits.softmax(-1)[0].cpu(); top = p.topk(3)
    pred = " | ".join(f"{ck['classes'][i]} {v*100:.1f}%" for v, i in zip(top.values, top.indices)); print("Prediction:", pred)
    print(f"{len(acts)} conv layers hooked ({ck['arch']}, input {ck['img']}px)"); paths = []

    # per-layer figure: input + the most-activated channels (or the first N) of that conv's output
    for li, (name, mod, a) in enumerate(acts):
        C = a.shape[0]; n = min(channels, C)
        idx = (a.abs().mean((1, 2)).argsort(descending=True)[:n] if order == "energy" else torch.arange(n))
        cols = 8; rows = int(np.ceil((n + 1) / cols)); fig, ax = plt.subplots(rows, cols, figsize=(1.7*cols, 1.8*rows + .6))
        ax = np.atleast_2d(ax).ravel()
        for q in ax: q.axis("off")
        ax[0].imshow(img); ax[0].set_title("input", fontsize=7)
        for k, c in enumerate(idx):
            ax[k+1].imshow(n01(a[c].numpy()), cmap="viridis"); ax[k+1].set_title(f"ch {int(c)}", fontsize=7)
        fig.suptitle(f"[{li:02d}] {name}   k={mod.kernel_size[0]} s={mod.stride[0]} g={mod.groups}   out={tuple(a.shape)}   "
                     f"({n}/{C} channels, by {order})", fontsize=9)
        path = f"{out}/layer_{li:02d}_{name.replace('.', '_')}.png"; _finish(fig, path, show); paths.append(path)

    # overview: one tile per conv layer = mean |activation| over channels -> see the whole pipeline at a glance
    cols = 8; rows = int(np.ceil((len(acts) + 1) / cols)); fig, ax = plt.subplots(rows, cols, figsize=(1.9*cols, 2.1*rows))
    ax = np.atleast_2d(ax).ravel()
    for q in ax: q.axis("off")
    ax[0].imshow(img); ax[0].set_title("input", fontsize=7)
    for li, (name, mod, a) in enumerate(acts):
        ax[li+1].imshow(n01(a.abs().mean(0).numpy()), cmap="magma"); ax[li+1].set_title(f"{li}: {name[-18:]}\n{tuple(a.shape)}", fontsize=6)
    fig.suptitle("Mean |activation| after every conv layer   ->   " + pred, fontsize=10)
    _finish(fig, f"{out}/00_overview.png", show)

    # learned kernels of the first conv layer
    W = acts[0][1].weight.detach().cpu(); n = min(W.shape[0], 64); s = int(np.ceil(np.sqrt(n)))
    fig, ax = plt.subplots(s, s, figsize=(7, 7)); ax = np.atleast_2d(ax).ravel()
    for q in ax: q.axis("off")
    for k in range(n): ax[k].imshow(n01(W[k].permute(1, 2, 0).numpy()), interpolation="nearest")
    fig.suptitle(f"First-layer kernels ({acts[0][0]}, {tuple(W.shape[2:])})"); _finish(fig, f"{out}/01_first_layer_kernels.png", show)
    print(f"saved {len(paths)} layer figures + overview + kernels to {out}/"); return paths

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="cnn_best.pt"); ap.add_argument("--image"); ap.add_argument("--data"); ap.add_argument("--index", type=int, default=0)
    ap.add_argument("--out", default="viz_cnn"); ap.add_argument("--channels", type=int, default=16)
    ap.add_argument("--order", choices=["energy", "index"], default="energy"); ap.add_argument("--show", action="store_true")
    ap.add_argument("--list_layers", action="store_true"); a = ap.parse_args()
    if a.list_layers:
        m, ck = load_checkpoint(a.ckpt)
        for i, (n, mod) in enumerate([(n, m_) for n, m_ in m.named_modules() if isinstance(m_, nn.Conv2d)]):
            print(f"{i:02d} {n:40s} {mod.in_channels:4d}->{mod.out_channels:4d} k={mod.kernel_size[0]} s={mod.stride[0]} g={mod.groups}")
    else:
        assert a.image or a.data, "give --image or --data"
        visualize(a.ckpt, a.image, a.data, a.index, a.out, a.channels, a.order, a.show)
