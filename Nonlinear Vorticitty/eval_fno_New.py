"""
Evaluate a trained FNO checkpoint on other datasets (e.g. OOD IC families).

Usage (same folder as FNO_Vorticity.py and vortex_solver.py):
    python eval_fno.py --model fno_model.pt --data test_ellipse.pt test_pair.pt test_grf.pt
"""
import argparse
import os

import numpy as np
import torch

# FNO_Vorticity.py is your renamed train_fno.py
from FNO_Vorticity import FNO2d, Norm, evaluate, rel_l2

p = argparse.ArgumentParser()
p.add_argument("--model", default="fno_model.pt")
p.add_argument("--data", nargs="+", required=True)
p.add_argument("--out", default="ood")
a = p.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"
ck = torch.load(a.model, map_location=device)
args = ck["args"]
t_in = args["t_in"]

model = FNO2d(t_in, args["modes"], args["width"], args["layers"]).to(device)
model.load_state_dict(ck["model"])
model.eval()

# rebuild the normalization used in training (stored in the checkpoint)
norm = Norm.__new__(Norm)
norm.w_std = ck["w_std"]
norm.lo, norm.hi = ck["nu_lohi"]

results, t_axis = {}, None
for path in a.data:
    d = torch.load(path)
    name = os.path.splitext(os.path.basename(path))[0]   # label by file name (families can repeat)
    r = evaluate(model, d, norm, t_in, device)

    # persistence baseline: keep the last known frame forever
    w = d["w"]
    true, last = w[:, t_in:], w[:, t_in - 1]
    r["persistence"] = torch.stack(
        [rel_l2(last, true[:, j]) for j in range(true.shape[1])], 1).mean(0).numpy()

    results[name] = r
    t_axis = d["t"][t_in:].numpy()

print("Rollout relative L2 error (FNO vs. persistence baseline):")
for name, r in results.items():
    e, b = r["rel_l2"], r["persistence"]
    pick = [0, len(e) // 4, len(e) // 2, 3 * len(e) // 4, len(e) - 1]
    print(f"  {name}")
    print("    FNO:         " + "  ".join(f"t={t_axis[i]:.2f}: {e[i]:.3f}" for i in pick))
    print("    persistence: " + "  ".join(f"t={t_axis[i]:.2f}: {b[i]:.3f}" for i in pick))

np.savez(a.out + "_results.npz", t=t_axis,
         **{f"{k}/{m}": v for k, r in results.items() for m, v in r.items()})

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    for name, r in results.items():
        l, = ax[0].plot(t_axis, r["rel_l2"], label=name)
        ax[0].plot(t_axis, r["persistence"], "--", color=l.get_color(), alpha=0.5)
        ax[1].plot(t_axis, r["enstrophy_err"], label=name)
        ax[2].plot(t_axis, r["energy_err"], label=name)
    ax[0].set_title("relative L2 (solid FNO, dashed persistence)")
    ax[1].set_title("enstrophy rel. error")
    ax[2].set_title("energy rel. error")
    for x in ax:
        x.set_xlabel("t"); x.set_yscale("log")
    ax[0].legend()
    plt.tight_layout(); plt.savefig(a.out + "_eval.png", dpi=150)
    print("saved", a.out + "_eval.png")
except ImportError:
    print("matplotlib not available; metrics saved to", a.out + "_results.npz")
