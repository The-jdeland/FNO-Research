"""
benchmark.py -- cost vs. accuracy: trained FNO vs. the pseudospectral solver at several resolutions.

Both are given the same information (the 64x64 vorticity field at the last input frame) and asked to
forecast to the end of the trajectory. Accuracy is measured against the stored true trajectories.

Usage (same folder as FNO_Vorticity.py and vortex_solver_chaotic.py):
    python benchmark.py --model fno_forced_unroll_model.pt --data test_forced.pt
"""
import argparse
import time

import numpy as np
import torch

try:
    from vortex_solver_chaotic import Vorticity2D
except ImportError:
    from vortex_solver import Vorticity2D
from FNO_Vorticity import FNO2d, Norm, rollout, rel_l2

p = argparse.ArgumentParser()
p.add_argument("--model", default="fno_forced_unroll_model.pt")
p.add_argument("--data", default="test_forced.pt")
p.add_argument("--n", type=int, default=100, help="number of test trajectories")
p.add_argument("--resolutions", type=int, nargs="+", default=[64, 128, 256])
p.add_argument("--repeats", type=int, default=3, help="timing repeats for the FNO")
p.add_argument("--solver_fp32", action="store_true", help="run the solver in float32")
p.add_argument("--out", default="benchmark")
a = p.parse_args()

dev = "cuda" if torch.cuda.is_available() else "cpu"
if dev == "cpu":
    print("WARNING: no GPU found; these timings are not representative of GPU use.")
if not hasattr(Vorticity2D, "set_forcing"):
    raise SystemExit("Need the newer solver (with set_forcing): vortex_solver_chaotic.py")


def sync():
    if dev == "cuda":
        torch.cuda.synchronize()


# ---- load model and data ------------------------------------------------------------------
ck = torch.load(a.model, map_location=dev)
targs = ck["args"]
t_in = targs["t_in"]
model = FNO2d(t_in, targs["modes"], targs["width"], targs["layers"]).to(dev)
model.load_state_dict(ck["model"])
model.eval()
norm = Norm.__new__(Norm)
norm.w_std = ck["w_std"]
norm.lo, norm.hi = ck["nu_lohi"]

d = torch.load(a.data)
dargs = d["args"]
w = d["w"][: a.n].to(dev)           # (n, frames, M, M)
nu = d["nu"][: a.n].to(dev)
t = d["t"]
dt_snap = (t[1] - t[0]).item()
n, nframes, M = w.shape[0], w.shape[1], w.shape[-1]
truth = w[:, t_in:]                  # frames the methods must predict
times = t[t_in:].numpy()
n_snap_sim = nframes - t_in          # snapshots to produce, starting from frame t_in-1
print(f"{n} trajectories, {nframes} frames, forecasting {n_snap_sim} frames "
      f"({n_snap_sim * dt_snap:.2f} time units) from t={t[t_in - 1].item():.2f}\n")


def err_curve(pred, ref):
    return torch.stack([rel_l2(pred[:, j], ref[:, j]) for j in range(ref.shape[1])], 1).mean(0).cpu().numpy()


def upsample(w64, N):
    """Spectral zero-padding of a real (B,M,M) field to (B,N,N); field values are preserved."""
    if N == w64.shape[-1]:
        return w64
    wh = torch.fft.rfft2(w64)
    out = torch.zeros(w64.shape[0], N, N // 2 + 1, dtype=wh.dtype, device=w64.device)
    out[:, : M // 2, : M // 2 + 1] = wh[:, : M // 2, : M // 2 + 1]
    out[:, -M // 2:, : M // 2 + 1] = wh[:, -M // 2:, : M // 2 + 1]
    return torch.fft.irfft2(out, s=(N, N)) * (N / M) ** 2


results = {}

# ---- FNO ---------------------------------------------------------------------------------
def time_fno(wb, nb):
    rollout(model, wb, nb, norm, t_in)          # warm-up
    sync()
    ts = []
    for _ in range(a.repeats):
        sync()
        t0 = time.perf_counter()
        pred = rollout(model, wb, nb, norm, t_in)
        sync()
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts)), pred


t1, _ = time_fno(w[:1], nu[:1])
tn, pred = time_fno(w, nu)
results["FNO (64x64)"] = dict(ms1=1e3 * t1, msn=1e3 * tn / n, err=err_curve(pred, truth))
print(f"FNO: {1e3 * t1:.1f} ms for 1 trajectory, {1e3 * tn / n:.2f} ms/trajectory with batch {n}")

# ---- Solver at several resolutions ---------------------------------------------------------
dtype = torch.float32 if a.solver_fp32 else torch.float64
for N in a.resolutions:
    S = Vorticity2D(N, device=dev, dtype=dtype)
    if dargs.get("f_amp", 0) > 0 or dargs.get("drag", 0) > 0:
        S.set_forcing(dargs["f_amp"] * torch.cos(dargs["f_n"] * S.Y), dargs["drag"])
    dtN = dargs["dt"] * dargs["N"] / N          # keep the same CFL number as the data generation

    # small warm-up, then one timed full run for each batch size
    S.solve(upsample(w[:1, t_in - 1].to(dtype), N), nu[:1], T=dt_snap, n_snap=1, dt=dtN, out_res=M)
    stats = {}
    for B in (1, n):
        w0 = upsample(w[:B, t_in - 1].to(dtype), N)
        sync()
        t0 = time.perf_counter()
        fr = S.solve(w0, nu[:B], T=n_snap_sim * dt_snap, n_snap=n_snap_sim, dt=dtN, out_res=M)
        sync()
        stats[B] = (time.perf_counter() - t0, fr)
    err = err_curve(stats[n][1][:, 1:].float(), truth)
    results[f"solver N={N}"] = dict(ms1=1e3 * stats[1][0], msn=1e3 * stats[n][0] / n, err=err)
    print(f"solver N={N}: {1e3 * stats[1][0]:.0f} ms for 1 trajectory, "
          f"{1e3 * stats[n][0] / n:.1f} ms/trajectory with batch {n}")


# ---- Summary table -------------------------------------------------------------------------
def at(err, tt):
    return err[np.abs(times - tt).argmin()]


def useful_time(err, thr=0.5):
    idx = np.where(err > thr)[0]
    return times[idx[0]] if len(idx) else float("nan")


print(f"\n{'method':16s} {'ms/traj (batch 1)':>18s} {'ms/traj (batch %d)' % n:>19s} "
      f"{'err t=5':>8s} {'err t=10':>9s} {'time err>0.5':>13s}")
for name, r in results.items():
    print(f"{name:16s} {r['ms1']:18.1f} {r['msn']:19.2f} {at(r['err'], 5.0):8.3f} "
          f"{at(r['err'], 10.0):9.3f} {useful_time(r['err']):13.2f}")

np.savez(a.out + ".npz", times=times,
         **{f"{k}/{m}": np.asarray(v) for k, r in results.items() for m, v in r.items()})

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    for name, r in results.items():
        ax[0].plot(times, r["err"], "-" if name.startswith("FNO") else "--", label=name,
                   lw=2.5 if name.startswith("FNO") else 1.5)
    ax[0].set_yscale("log"); ax[0].set_xlabel("t"); ax[0].set_ylabel("relative L2 error")
    ax[0].set_title("forecast accuracy"); ax[0].legend()
    names = list(results)
    ax[1].bar(range(len(names)), [results[k]["msn"] for k in names])
    ax[1].set_xticks(range(len(names))); ax[1].set_xticklabels(names, rotation=20)
    ax[1].set_yscale("log"); ax[1].set_ylabel(f"ms per trajectory (batch {n})")
    ax[1].set_title("cost")
    plt.tight_layout(); plt.savefig(a.out + ".png", dpi=150)
    print("saved", a.out + ".png")
except ImportError:
    print("matplotlib not available; numbers saved to", a.out + ".npz")
