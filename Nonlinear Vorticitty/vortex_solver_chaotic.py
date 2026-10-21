"""
Batched pseudospectral solver for the 2D periodic vorticity equation

    w_t + u w_x + v w_y = nu * Lap(w),    -Lap(psi) = w,   u = psi_y,  v = -psi_x

on [-pi, pi)^2.  Port of VortexFourierSpectralAB3.m with:
  * the viscous sign fixed (the .m file had backward diffusion),
  * nu as a per-sample parameter (batched),
  * 2/3-rule dealiasing,
  * integrating-factor RK4 (viscous term exact, so dt is limited by advection only).

Usage:
    python vortex_solver.py --selftest
    python vortex_solver.py --family blob --n_samples 1000 --out train_blob.pt
    python vortex_solver.py --family ellipse --n_samples 200 --out test_ellipse.pt
"""
import argparse
import math
import time

import torch

TWO_PI = 2 * math.pi


# ----------------------------------------------------------------------------
# Solver
# ----------------------------------------------------------------------------
class Vorticity2D:
    def __init__(self, N=256, device="cpu", dtype=torch.float64):
        self.N, self.device, self.dtype = N, device, dtype
        h = TWO_PI / N
        x1 = -math.pi + h * torch.arange(1, N + 1, device=device, dtype=dtype)
        # Same convention as MATLAB meshgrid: X varies along the last dim, Y along dim -2
        self.Y, self.X = torch.meshgrid(x1, x1, indexing="ij")

        ky = torch.fft.fftfreq(N, d=1.0 / N).to(device=device, dtype=dtype)[:, None]
        kx = torch.fft.rfftfreq(N, d=1.0 / N).to(device=device, dtype=dtype)[None, :]
        self.kx, self.ky = kx, ky
        self.k2 = kx ** 2 + ky ** 2
        self.inv_k2 = torch.where(self.k2 > 0, 1.0 / self.k2.clamp(min=1e-30),
                                  torch.zeros_like(self.k2))
        kmax = N / 3.0  # 2/3 rule
        self.mask = ((kx.abs() <= kmax) & (ky.abs() <= kmax)).to(dtype)
        self.force_h, self.drag = None, 0.0   # optional steady forcing and linear drag

    def set_forcing(self, f, drag=0.0):
        """Add a steady vorticity forcing f(x,y) (N,N) and linear drag: w_t = ... + f - drag*w."""
        self.force_h = self.fft(f.to(self.device, self.dtype)) * self.mask
        self.force_h[..., 0, 0] = 0
        self.drag = drag

    # -- helpers -------------------------------------------------------------
    def fft(self, w):
        return torch.fft.rfft2(w)

    def ifft(self, wh):
        return torch.fft.irfft2(wh, s=(self.N, self.N))

    def velocity(self, wh):
        psi = wh * self.inv_k2
        u = self.ifft(1j * self.ky * psi)
        v = self.ifft(-1j * self.kx * psi)
        return u, v

    def rhs_nonlinear(self, wh):
        u, v = self.velocity(wh)
        wx = self.ifft(1j * self.kx * wh)
        wy = self.ifft(1j * self.ky * wh)
        out = -self.fft(u * wx + v * wy) * self.mask
        return out if self.force_h is None else out + self.force_h

    def _step(self, wh, E, E2, dt):
        k1 = self.rhs_nonlinear(wh)
        k2 = self.rhs_nonlinear(E * (wh + 0.5 * dt * k1))
        k3 = self.rhs_nonlinear(E * wh + 0.5 * dt * k2)
        k4 = self.rhs_nonlinear(E2 * wh + dt * E * k3)
        return E2 * wh + (dt / 6.0) * (E2 * k1 + 2.0 * E * (k2 + k3) + k4)

    def downsample(self, wh, M):
        """Spectral truncation of rfft2 coefficients (B,N,N/2+1) -> physical (B,M,M)."""
        if M == self.N:
            return self.ifft(wh)
        rows = torch.cat([wh[:, : M // 2], wh[:, -M // 2:]], dim=1)
        cols = rows[:, :, : M // 2 + 1]
        return torch.fft.irfft2(cols, s=(M, M)) * (M / self.N) ** 2

    # -- main entry point ----------------------------------------------------
    @torch.no_grad()
    def solve(self, w0, nu, T=2.0, n_snap=40, dt=5e-3, out_res=None, verbose=False, t_spin=0.0):
        """
        w0: (B,N,N) initial vorticity.  nu: (B,) viscosities.
        Returns frames (B, n_snap+1, M, M) at times linspace(0, T, n_snap+1).
        """
        B = w0.shape[0]
        M = out_res or self.N
        steps_per = round(T / n_snap / dt)
        dt = T / n_snap / steps_per  # make snapshots land exactly
        nu = nu.to(self.device, self.dtype).reshape(B, 1, 1)

        wh = self.fft(w0.to(self.device, self.dtype))
        wh[:, 0, 0] = 0.0          # zero-mean vorticity
        wh = wh * self.mask

        L = -(nu * self.k2 + self.drag)
        E, E2 = torch.exp(L * dt / 2), torch.exp(L * dt)

        for _ in range(round(t_spin / dt)):      # spin-up: advance without saving
            wh = self._step(wh, E, E2, dt)

        frames = [self.downsample(wh, M)]
        worst_cfl = 0.0
        for s in range(n_snap):
            for _ in range(steps_per):
                wh = self._step(wh, E, E2, dt)
            frames.append(self.downsample(wh, M))
            u, v = self.velocity(wh)
            cfl = dt * (u.abs().amax() + v.abs().amax()).item() * self.N / 3.0
            worst_cfl = max(worst_cfl, cfl)
            if verbose:
                print(f"  snap {s + 1}/{n_snap}  cfl~{cfl:.2f}")
        if worst_cfl > 2.0:
            print(f"WARNING: advective CFL-like number {worst_cfl:.2f} > 2.0 "
                  f"(RK4 limit ~2.8); reduce dt.")
        if not torch.isfinite(wh).all():
            print("WARNING: non-finite values in solution (blow-up). Reduce dt.")
        # resolution check: share of enstrophy in the top of the retained wavenumbers
        kinf = torch.maximum(self.kx.abs(), self.ky.abs())
        pw = wh.abs() ** 2
        hk = ((pw * (kinf > 0.8 * self.N / 3).to(self.dtype)).sum((-2, -1))
              / pw.sum((-2, -1)).clamp(min=1e-300)).max().item()
        print(f"  resolution check: worst high-k enstrophy fraction {hk:.1e} "
              f"(rule of thumb: want < ~5e-3; larger means increase --N or raise --nu_min)")
        return torch.stack(frames, dim=1)


# ----------------------------------------------------------------------------
# Initial-condition families
# ----------------------------------------------------------------------------
def _wrap(d):
    return (d + math.pi) % TWO_PI - math.pi


def _U(B, lo, hi, gen):
    return (lo + (hi - lo) * torch.rand(B, 1, 1, generator=gen, dtype=torch.float64))


def sample_ic(family, B, solver, gen):
    X, Y = solver.X.cpu(), solver.Y.cpu()
    if family == "blob":
        # In-distribution family: randomized version of
        #   20 exp(-20 x^4 - 10 y^4) sin(3x) cos(5y)
        A = _U(B, 10, 30, gen) * torch.where(torch.rand(B, 1, 1, generator=gen) < 0.5, -1.0, 1.0)
        a, b = _U(B, 10, 40, gen), _U(B, 5, 20, gen)
        kx, ky = _U(B, 2, 4, gen), _U(B, 3, 6, gen)
        dx, dy = _wrap(X - _U(B, -0.3, 0.3, gen)), _wrap(Y - _U(B, -0.3, 0.3, gen))
        w = A * torch.exp(-a * dx ** 4 - b * dy ** 4) * torch.sin(kx * dx) * torch.cos(ky * dy)
    elif family == "ellipse":
        # OOD: elliptical Gaussian vortex (sheds filaments) -- cf. exp(-40x^2-5y^2)
        A = _U(B, 10, 30, gen)
        a, b = _U(B, 20, 60, gen), _U(B, 3, 10, gen)
        dx, dy = _wrap(X - _U(B, -0.3, 0.3, gen)), _wrap(Y - _U(B, -0.3, 0.3, gen))
        w = A * torch.exp(-a * dx ** 2 - b * dy ** 2)
    elif family == "pair":
        # OOD: two like-signed Gaussian vortices (merger problem, cf. Melander et al.)
        sig = _U(B, 0.25, 0.4, gen)
        sep = sig * _U(B, 2.0, 5.0, gen)
        ang = _U(B, 0, math.pi, gen)
        cx, cy = _U(B, -0.3, 0.3, gen), _U(B, -0.3, 0.3, gen)
        A = _U(B, 5, 15, gen)
        w = 0
        for sgn in (+1, -1):
            dx = _wrap(X - cx - sgn * 0.5 * sep * torch.cos(ang))
            dy = _wrap(Y - cy - sgn * 0.5 * sep * torch.sin(ang))
            w = w + A * torch.exp(-(dx ** 2 + dy ** 2) / (2 * sig ** 2))
    elif family == "grf":
        # OOD / benchmark-style: Gaussian random field, spectrum ~ (k^2 + tau^2)^(-alpha/2)
        tau, alpha, std = 7.0, 2.5, 5.0
        k2 = (solver.kx ** 2 + solver.ky ** 2).cpu()
        amp = (k2 + tau ** 2) ** (-alpha / 2)
        xi = torch.randn(B, *k2.shape, 2, generator=gen, dtype=torch.float64)
        wh = torch.complex(xi[..., 0], xi[..., 1]) * amp
        wh[:, 0, 0] = 0
        w = torch.fft.irfft2(wh, s=(solver.N, solver.N))
        w = std * w / w.std(dim=(-2, -1), keepdim=True)
    else:
        raise ValueError(f"unknown family {family}")
    w = w.expand(B, -1, -1).clone() if w.shape[0] == 1 and B > 1 else w
    return w - w.mean(dim=(-2, -1), keepdim=True)


# ----------------------------------------------------------------------------
# Diagnostics (useful later for evaluation)
# ----------------------------------------------------------------------------
def energy_enstrophy(w):
    """w: (..., M, M) vorticity -> (energy, enstrophy) per field (domain means)."""
    M = w.shape[-1]
    wh = torch.fft.rfft2(w) / (M * M)
    k = torch.fft.fftfreq(M, d=1.0 / M).to(w.device)
    kr = torch.fft.rfftfreq(M, d=1.0 / M).to(w.device)
    k2 = kr[None, :] ** 2 + k[:, None] ** 2
    wgt = torch.full_like(k2, 2.0)
    wgt[:, 0] = 1.0
    if M % 2 == 0:
        wgt[:, -1] = 1.0
    ens = 0.5 * (wgt * wh.abs() ** 2).sum(dim=(-2, -1))
    en = 0.5 * (wgt * (wh.abs() ** 2) * torch.where(k2 > 0, 1 / k2.clamp(min=1e-30), 0 * k2)).sum(dim=(-2, -1))
    return en, ens


# ----------------------------------------------------------------------------
# Self-test
# ----------------------------------------------------------------------------
def selftest(device):
    print("Self-test on", device)
    S = Vorticity2D(N=128, device=device)

    # 1) sin(x)sin(y) is an exact Euler steady state -> pure decay exp(-2 nu t)
    nu = torch.tensor([1e-2, 5e-3])
    w0 = (torch.sin(S.X) * torch.sin(S.Y))[None].repeat(2, 1, 1)
    fr = S.solve(w0, nu, T=1.0, n_snap=4, dt=1e-2)
    ratio = fr[:, -1].abs().amax(dim=(-2, -1)) / fr[:, 0].abs().amax(dim=(-2, -1))
    print("1) decay ratio  numerical:", ratio.tolist(),
          " exact:", torch.exp(-2 * nu * 1.0).tolist())

    # 2) dt convergence on a blob IC (error vs. dt/2 reference)
    gen = torch.Generator().manual_seed(0)
    w0 = sample_ic("blob", 2, S, gen).to(device)
    nu = torch.tensor([1e-3, 1e-2])
    ref = S.solve(w0, nu, T=1.0, n_snap=2, dt=1.25e-3)[:, -1]
    for dt in (1e-2, 5e-3, 2.5e-3):
        f = S.solve(w0, nu, T=1.0, n_snap=2, dt=dt)[:, -1]
        err = ((f - ref).flatten(1).norm(dim=1) / ref.flatten(1).norm(dim=1)).tolist()
        print(f"2) dt={dt:g}  rel err vs dt=1.25e-3: {err}")

    # 3) enstrophy decays monotonically with viscosity; energy too
    fr = S.solve(w0, nu, T=1.0, n_snap=5, dt=2.5e-3)
    en, ens = energy_enstrophy(fr)
    print("3) enstrophy over time (should decrease):\n", ens)
    print("   max|u|-based CFL check printed above if any warning.")


# ----------------------------------------------------------------------------
# Dataset generation
# ----------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--selftest", action="store_true")
    p.add_argument("--family", default="blob", choices=["blob", "ellipse", "pair", "grf"])
    p.add_argument("--families", nargs="+", default=None,
                   choices=["blob", "ellipse", "pair", "grf"],
                   help="mix several IC families in one file (each trajectory picks one at "
                        "random); overrides --family")
    p.add_argument("--n_samples", type=int, default=1000)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--N", type=int, default=256, help="simulation resolution")
    p.add_argument("--M", type=int, default=64, help="saved resolution (spectral crop)")
    p.add_argument("--T", type=float, default=2.0)
    p.add_argument("--n_snap", type=int, default=40)
    p.add_argument("--dt", type=float, default=5e-3)
    p.add_argument("--nu_min", type=float, default=1e-3)
    p.add_argument("--nu_max", type=float, default=1e-2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="data.pt")
    p.add_argument("--f_amp", type=float, default=0.0,
                   help="Kolmogorov forcing amplitude: f = f_amp*cos(f_n*y) (0 = unforced)")
    p.add_argument("--f_n", type=int, default=4)
    p.add_argument("--drag", type=float, default=0.0, help="linear drag coefficient")
    p.add_argument("--t_spin", type=float, default=0.0, help="spin-up time before recording")
    p.add_argument("--fp32", action="store_true", help="float32 solver (faster, less accurate)")
    a = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if a.selftest:
        return selftest(device)

    dtype = torch.float32 if a.fp32 else torch.float64
    S = Vorticity2D(a.N, device=device, dtype=dtype)
    if a.f_amp > 0 or a.drag > 0:
        S.set_forcing(a.f_amp * torch.cos(a.f_n * S.Y), a.drag)
    gen = torch.Generator().manual_seed(a.seed)

    fams = a.families or [a.family]
    W, NU, FID = [], [], []
    t0 = time.time()
    for i in range(0, a.n_samples, a.batch):
        B = min(a.batch, a.n_samples - i)
        fid = torch.randint(len(fams), (B,), generator=gen)
        w0 = torch.empty(B, a.N, a.N, dtype=torch.float64)
        for j, f in enumerate(fams):
            sel = fid == j
            if sel.any():
                w0[sel] = sample_ic(f, int(sel.sum()), S, gen)
        w0 = w0.to(device)
        FID.append(fid)
        log_nu = math.log(a.nu_min) + (math.log(a.nu_max) - math.log(a.nu_min)) * torch.rand(B, generator=gen, dtype=torch.float64)
        nu = log_nu.exp()
        fr = S.solve(w0, nu, T=a.T, n_snap=a.n_snap, dt=a.dt, out_res=a.M, t_spin=a.t_spin)
        W.append(fr.float().cpu())
        NU.append(nu.float())
        print(f"{i + B}/{a.n_samples}  ({time.time() - t0:.0f}s)")

    torch.save({"w": torch.cat(W), "nu": torch.cat(NU),
                "t": torch.linspace(0, a.T, a.n_snap + 1), "family": "+".join(fams),
                "families": fams, "family_id": torch.cat(FID), "args": vars(a)}, a.out)
    print("saved", a.out)


if __name__ == "__main__":
    main()
