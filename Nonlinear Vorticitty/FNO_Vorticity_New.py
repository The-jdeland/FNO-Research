"""
2D Fourier Neural Operator, autoregressive, for the vorticity data from vortex_solver.py.

Task: (last T_IN vorticity frames, nu, grid)  ->  next frame (time step = T/n_snap = 0.05).
Training is one-step (teacher forcing); evaluation is a full autoregressive rollout.

Usage:
    python train_fno.py --train train_blob.pt \
        --ood test_ellipse.pt test_pair.pt test_grf.pt --epochs 100
"""
import argparse
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as Fn

from vortex_solver import energy_enstrophy


# ----------------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------------
class SpectralConv2d(nn.Module):
    def __init__(self, cin, cout, m1, m2):
        super().__init__()
        self.cout, self.m1, self.m2 = cout, m1, m2
        scale = 1.0 / (cin * cout)
        self.w1 = nn.Parameter(scale * torch.rand(cin, cout, m1, m2, dtype=torch.cfloat))
        self.w2 = nn.Parameter(scale * torch.rand(cin, cout, m1, m2, dtype=torch.cfloat))

    def forward(self, x):
        B, _, H, W = x.shape
        xh = torch.fft.rfft2(x)
        out = torch.zeros(B, self.cout, H, W // 2 + 1, dtype=torch.cfloat, device=x.device)
        out[:, :, : self.m1, : self.m2] = torch.einsum(
            "bixy,ioxy->boxy", xh[:, :, : self.m1, : self.m2], self.w1)
        out[:, :, -self.m1:, : self.m2] = torch.einsum(
            "bixy,ioxy->boxy", xh[:, :, -self.m1:, : self.m2], self.w2)
        return torch.fft.irfft2(out, s=(H, W))


class FNO2d(nn.Module):
    def __init__(self, t_in=10, modes=16, width=64, layers=4):
        super().__init__()
        self.t_in = t_in
        self.lift = nn.Conv2d(t_in + 1 + 2, width, 1)   # frames + nu + (x, y)
        self.spec = nn.ModuleList([SpectralConv2d(width, width, modes, modes) for _ in range(layers)])
        self.skip = nn.ModuleList([nn.Conv2d(width, width, 1) for _ in range(layers)])
        self.proj1 = nn.Conv2d(width, 128, 1)
        self.proj2 = nn.Conv2d(128, 1, 1)

    def forward(self, frames, nu_n):
        """frames: (B,t_in,H,W) normalized vorticity; nu_n: (B,) normalized viscosity in [0,1]."""
        B, _, H, W = frames.shape
        gy = torch.linspace(0, 1, H, device=frames.device).view(1, 1, H, 1).expand(B, 1, H, W)
        gx = torch.linspace(0, 1, W, device=frames.device).view(1, 1, 1, W).expand(B, 1, H, W)
        nu_c = nu_n.view(B, 1, 1, 1).expand(B, 1, H, W)
        x = self.lift(torch.cat([frames, nu_c, gx, gy], dim=1))
        for s, k in zip(self.spec, self.skip):
            x = Fn.gelu(s(x) + k(x))
        delta = self.proj2(Fn.gelu(self.proj1(x)))
        return frames[:, -1:] + delta           # predict the increment over one step


# ----------------------------------------------------------------------------
# Data helpers
# ----------------------------------------------------------------------------
class Norm:
    def __init__(self, w, nu):
        self.w_std = w.std().item()
        self.lo, self.hi = nu.log().min().item(), nu.log().max().item()

    def nu(self, nu):
        return (nu.log() - self.lo) / (self.hi - self.lo)


def rel_l2(p, y):
    return (p - y).flatten(1).norm(dim=1) / y.flatten(1).norm(dim=1)


@torch.no_grad()
def rollout(model, w, nu, norm, t_in):
    """w: (B,T,H,W) true trajectory (physical units). Seeds with first t_in frames, rolls out the rest."""
    wn = w / norm.w_std
    nun = norm.nu(nu)
    hist = wn[:, :t_in].clone()
    preds = []
    for _ in range(w.shape[1] - t_in):
        nxt = model(hist, nun)
        preds.append(nxt)
        hist = torch.cat([hist[:, 1:], nxt], dim=1)
    return torch.cat(preds, dim=1) * norm.w_std      # (B, T-t_in, H, W)


def spectrum(w):
    """Isotropic energy spectrum E(k), shell-averaged. w: (B,M,M) -> (M//2,)"""
    M = w.shape[-1]
    wh = torch.fft.fft2(w) / (M * M)
    k = torch.fft.fftfreq(M, d=1.0 / M).to(w.device)
    k2 = k[None, :] ** 2 + k[:, None] ** 2
    e = 0.5 * wh.abs() ** 2 / k2.clamp(min=1)       # velocity energy = |w|^2/(2k^2)
    shell = k2.sqrt().round().long()
    out = torch.zeros(M // 2, device=w.device)
    for s in range(1, M // 2):
        out[s] = e[:, shell == s].sum(dim=-1).mean()
    return out


@torch.no_grad()
def evaluate(model, data, norm, t_in, device, bs=50):
    model.eval()
    w, nu = data["w"], data["nu"]
    errs, en_err, ens_err, spec_p, spec_t = [], [], [], [], []
    for i in range(0, w.shape[0], bs):
        wb, nb = w[i:i + bs].to(device), nu[i:i + bs].to(device)
        pred = rollout(model, wb, nb, norm, t_in)
        true = wb[:, t_in:]
        errs.append(torch.stack([rel_l2(pred[:, j], true[:, j]) for j in range(pred.shape[1])], 1).cpu())
        ep, sp = energy_enstrophy(pred)
        et, st = energy_enstrophy(true)
        en_err.append(((ep - et).abs() / et).cpu())
        ens_err.append(((sp - st).abs() / st).cpu())
        spec_p.append(spectrum(pred[:, -1]).cpu())
        spec_t.append(spectrum(true[:, -1]).cpu())
    c = lambda L: torch.cat(L).mean(0).numpy()
    return {"rel_l2": c(errs), "energy_err": c(en_err), "enstrophy_err": c(ens_err),
            "spec_pred": torch.stack(spec_p).mean(0).numpy(),
            "spec_true": torch.stack(spec_t).mean(0).numpy()}


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train", required=True)
    p.add_argument("--ood", nargs="*", default=[])
    p.add_argument("--t_in", type=int, default=10)
    p.add_argument("--modes", type=int, default=16)
    p.add_argument("--width", type=int, default=64)
    p.add_argument("--layers", type=int, default=4)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--bs", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--wd", type=float, default=1e-4)
    p.add_argument("--noise", type=float, default=0.0,
                   help="Gaussian noise (in normalized units) added to inputs in training; "
                        "try 1e-3..1e-2 if rollouts blow up")
    p.add_argument("--unroll", type=int, default=1,
                   help="during training, let the model predict this many steps in a row from "
                        "its own outputs and score all of them (1 = ordinary one-step training)")
    p.add_argument("--val_frac", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="fno")
    a = p.parse_args()

    torch.manual_seed(a.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    data = torch.load(a.train)
    w_all, nu_all = data["w"], data["nu"]
    n = w_all.shape[0]
    perm = torch.randperm(n)
    n_val = max(1, int(a.val_frac * n))
    val_idx, tr_idx = perm[:n_val], perm[n_val:]
    w_tr, nu_tr = w_all[tr_idx].to(device), nu_all[tr_idx].to(device)
    val = {"w": w_all[val_idx], "nu": nu_all[val_idx]}
    norm = Norm(w_tr, nu_tr)
    T = w_tr.shape[1]
    n_starts = T - a.t_in - (a.unroll - 1)                 # windows per trajectory
    print(f"train traj {w_tr.shape[0]}, val traj {n_val}, windows/traj {n_starts}, "
          f"w_std {norm.w_std:.3f}, device {device}")

    model = FNO2d(a.t_in, a.modes, a.width, a.layers).to(device)
    print("params:", sum(p.numel() for p in model.parameters() if p.requires_grad))
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.wd)
    n_windows = w_tr.shape[0] * n_starts
    steps_per_epoch = n_windows // a.bs
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=a.epochs * steps_per_epoch)

    wn_tr = w_tr / norm.w_std
    nun_tr = norm.nu(nu_tr)
    t0 = time.time()
    for ep in range(a.epochs):
        model.train()
        order = torch.randperm(n_windows, device=device)
        tot = 0.0
        for b in range(steps_per_epoch):
            idx = order[b * a.bs:(b + 1) * a.bs]
            traj, s = idx // n_starts, idx % n_starts
            ar = torch.arange(a.t_in, device=device)[None, :]
            x = wn_tr[traj[:, None], s[:, None] + ar]                 # (B,t_in,H,W)
            if a.noise > 0:
                x = x + a.noise * torch.randn_like(x)
            hist, loss = x, 0.0
            for k in range(a.unroll):
                y = wn_tr[traj, s + a.t_in + k].unsqueeze(1)          # true frame k steps ahead
                pred = model(hist, nun_tr[traj])
                loss = loss + rel_l2(pred, y).mean() / a.unroll
                # feed the model's own prediction back in (no gradient through earlier steps)
                hist = torch.cat([hist[:, 1:], pred.detach()], dim=1)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            tot += loss.item()
        msg = f"epoch {ep + 1}/{a.epochs}  train relL2 (avg over unroll steps) {tot / steps_per_epoch:.4e}  ({time.time() - t0:.0f}s)"
        if (ep + 1) % 10 == 0 or ep == a.epochs - 1:
            r = evaluate(model, val, norm, a.t_in, device)
            msg += f"  | val rollout relL2: mid {r['rel_l2'][len(r['rel_l2']) // 2]:.3f} end {r['rel_l2'][-1]:.3f}"
        print(msg)

    torch.save({"model": model.state_dict(), "args": vars(a),
                "w_std": norm.w_std, "nu_lohi": (norm.lo, norm.hi)}, a.out + "_model.pt")

    # ---- final evaluation: in-distribution val + OOD sets -------------------------
    results = {"val(in-dist)": evaluate(model, val, norm, a.t_in, device)}
    for path in a.ood:
        d = torch.load(path)
        results[d.get("family", path)] = evaluate(model, d, norm, a.t_in, device)

    t_axis = data["t"][a.t_in:].numpy()
    print("\nRollout relative L2 error at selected times:")
    for name, r in results.items():
        e = r["rel_l2"]
        pick = [0, len(e) // 4, len(e) // 2, 3 * len(e) // 4, len(e) - 1]
        print(f"  {name:14s} " + "  ".join(f"t={t_axis[i]:.2f}: {e[i]:.3f}" for i in pick))
    np.savez(a.out + "_results.npz", t=t_axis,
             **{f"{k}/{m}": v for k, r in results.items() for m, v in r.items()})

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 4, figsize=(18, 4))
        for name, r in results.items():
            ax[0].plot(t_axis, r["rel_l2"], label=name)
            ax[1].plot(t_axis, r["enstrophy_err"], label=name)
            ax[2].plot(t_axis, r["energy_err"], label=name)
        ks = np.arange(len(next(iter(results.values()))["spec_true"]))
        r0 = results["val(in-dist)"]
        ax[3].loglog(ks[1:], r0["spec_true"][1:], "k-", label="true (final t)")
        ax[3].loglog(ks[1:], r0["spec_pred"][1:], "r--", label="FNO (final t)")
        for a_, t in zip(ax, ["relative L2 error", "enstrophy rel. error",
                              "energy rel. error", "energy spectrum (in-dist)"]):
            a_.set_title(t)
        for a_ in ax[:3]:
            a_.set_xlabel("t"); a_.set_yscale("log")
        ax[3].set_xlabel("k")
        ax[0].legend(); ax[3].legend()
        plt.tight_layout(); plt.savefig(a.out + "_eval.png", dpi=150)
        print("saved", a.out + "_eval.png")
    except ImportError:
        print("matplotlib not available; metrics are in", a.out + "_results.npz")


if __name__ == "__main__":
    main()
