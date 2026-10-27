"""Forward ONE image through a saved ViT and plot what every stage does.
  python visualize.py --ckpt vit_best.pt --image cat.png
  python visualize.py --ckpt vit_best.pt --data cifar10_train.pt --index 7
Outputs PNGs into --out (default viz/)."""
import argparse, os, numpy as np, torch, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from model import ViT

def norm01(a): a = a - a.min(); return a / (a.max() + 1e-8)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="vit_best.pt"); p.add_argument("--image"); p.add_argument("--data")
    p.add_argument("--index", type=int, default=0); p.add_argument("--out", default="viz")
    a = p.parse_args(); os.makedirs(a.out, exist_ok=True)
    ck = torch.load(a.ckpt, map_location="cpu"); c = ck["config"]
    m = ViT(patch=c["patch"], dim=c["dim"], depth=c["depth"], heads=c["heads"], mlp_dim=c["mlp_dim"], drop=0).eval()
    m.load_state_dict(ck["state_dict"])
    if a.image: img = np.array(Image.open(a.image).convert("RGB").resize((32, 32)))
    else:
        d = torch.load(a.data); img = d["X"][a.index].permute(1, 2, 0).numpy()
    x = torch.tensor(img).permute(2, 0, 1).float()[None] / 255
    x = (x - torch.tensor(ck["mean"]).view(1,3,1,1)) / torch.tensor(ck["std"]).view(1,3,1,1)
    P = c["patch"]; G = 32 // P; D = c["dim"]

    with torch.no_grad():
        feat = m.patch_embedding.proj(x)                      # (1,D,G,G)
        t = m.embed(x); stages = [("embed+pos", t)]; atts = []
        for i, b in enumerate(m.blocks):
            t, w = b(t, return_attn=True); stages.append((f"block {i+1}", t)); atts.append(w[0])  # (H,N+1,N+1)
        logits = m.head(m.norm(t)[:, 0]); prob = logits.softmax(-1)[0]
    top = prob.topk(3)
    title = " | ".join(f"{ck['classes'][i]} {v*100:.1f}%" for v, i in zip(top.values, top.indices))
    print("Prediction:", title)

    # 1) learned patch filters (conv kernels = what each token 'looks for' in a patch)
    W = m.patch_embedding.proj.weight.detach(); n = min(D, 64); s = int(np.ceil(np.sqrt(n)))
    fig, ax = plt.subplots(s, s, figsize=(8, 8)); ax = np.array(ax).ravel()
    for k in range(s*s):
        ax[k].axis("off")
        if k < n: ax[k].imshow(norm01(W[k].permute(1, 2, 0).numpy()), interpolation="nearest")
    fig.suptitle(f"Patch-embedding filters (first {n} of {D})"); fig.savefig(f"{a.out}/1_filters.png", dpi=150); plt.close(fig)

    # 2) feature maps after the patch conv
    n = min(D, 32); fig, ax = plt.subplots(5, 8, figsize=(14, 9)); ax = ax.ravel()
    ax[0].imshow(img); ax[0].set_title("input"); 
    for k in range(1, 40):
        ax[k].axis("off")
        if k-1 < n: ax[k].imshow(feat[0, k-1], cmap="viridis"); ax[k].set_title(f"ch {k-1}", fontsize=7)
    ax[0].axis("off"); fig.suptitle(f"Patch-conv output ({G}x{G} grid per channel)")
    fig.savefig(f"{a.out}/2_patch_feature_maps.png", dpi=150); plt.close(fig)

    # 3) token feature maps after each stage (first channels, reshaped to the patch grid) + token norm
    fig, ax = plt.subplots(len(stages), 9, figsize=(15, 1.7*len(stages)))
    for r, (name, tk) in enumerate(stages):
        pt = tk[0, 1:]                                         # drop CLS
        ax[r, 0].imshow(pt.norm(dim=-1).reshape(G, G), cmap="magma"); ax[r, 0].set_ylabel(name, fontsize=8)
        ax[r, 0].set_title("token norm" if r == 0 else "", fontsize=7)
        for k in range(8):
            ax[r, k+1].imshow(pt[:, k].reshape(G, G), cmap="viridis"); 
            if r == 0: ax[r, k+1].set_title(f"ch {k}", fontsize=7)
    for q in ax.ravel(): q.set_xticks([]); q.set_yticks([])
    fig.suptitle("Token features after every stage"); fig.savefig(f"{a.out}/3_stage_features.png", dpi=150); plt.close(fig)

    # 4) attention: CLS -> patches, every head of every block, overlaid on the image
    H = atts[0].shape[0]; fig, ax = plt.subplots(len(atts), H+1, figsize=(2*(H+1), 2*len(atts)))
    ax = np.atleast_2d(ax)
    for r, w in enumerate(atts):
        ax[r, 0].imshow(img); ax[r, 0].set_ylabel(f"block {r+1}")
        for h in range(H):
            hm = w[h, 0, 1:].reshape(G, G)[None, None]
            hm = torch.nn.functional.interpolate(hm, size=32, mode="bilinear")[0, 0]
            ax[r, h+1].imshow(img); ax[r, h+1].imshow(norm01(hm.numpy()), cmap="jet", alpha=.5)
            if r == 0: ax[r, h+1].set_title(f"head {h}", fontsize=8)
    for q in ax.ravel(): q.set_xticks([]); q.set_yticks([])
    fig.suptitle("CLS-token attention per head"); fig.savefig(f"{a.out}/4_attention_heads.png", dpi=150); plt.close(fig)

    # 5) attention rollout (what the final CLS token ultimately relies on)
    R = torch.eye(atts[0].shape[-1])
    for w in atts:
        A = w.mean(0) + torch.eye(w.shape[-1]); A = A / A.sum(-1, keepdim=True); R = A @ R
    hm = torch.nn.functional.interpolate(R[0, 1:].reshape(G, G)[None, None], size=32, mode="bilinear")[0, 0]
    fig, ax = plt.subplots(1, 2, figsize=(7, 3.5)); ax[0].imshow(img); ax[0].set_title("input")
    ax[1].imshow(img); ax[1].imshow(norm01(hm.numpy()), cmap="jet", alpha=.5); ax[1].set_title("attention rollout\n" + title, fontsize=8)
    for q in ax: q.axis("off")
    fig.savefig(f"{a.out}/5_rollout.png", dpi=150); plt.close(fig)
    print("saved figures to", a.out)

if __name__ == "__main__": main()
