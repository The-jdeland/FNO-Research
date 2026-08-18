"""
FNO1d on periodic Korteweg-de Vries (KdV) equation

A minimal, self-contained starting point for learning the solution operator
    G: u(x, 0)  |->  u(x, T)
of the periodic KdV equation

    u_t + 6 u u_x + u_xxx = 0,   x in [0, L), periodic

using a Fourier Neural Operator (Li et al., 2020).

Structure:
  1. Spectral KdV solver (data generator) -> solve_kdv()
  2. Dataset of (u0, uT) pairs -> make_dataset()
  3. FNO1d model (SpectralConv1d + FNO1d) -> class FNO1d
  4. Training loop -> main()
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt


# 1. Spectral KdV solver (Generate training data)

def solve_kdv(u0, L, T, dt, nsave=1):
    """
    Pseudo-spectral solver for u_t + 6 u u_x + u_xxx = 0 on a periodic
    domain [0, L), using an integrating-factor RK4 (ETDRK4-lite) scheme.
    The linear term u_xxx is handled exactly in Fourier space; the
    nonlinear term 6 u u_x is handled explicitly with RK4.

    Parameters:

    u0 : (N,), array initial condition on a uniform grid of size N
    L : float, domain length
    T : float, total integration time
    dt : float, time step
    nsave : int, save every `nsave` steps (1 = save every step)

    This returns:
    t_hist : (n_saved,) array of times
    u_hist : (n_saved, N) array of solutions
    """
    N = u0.shape[0]
    k = 2.0 * np.pi * np.fft.fftfreq(N, d=L / N) # wavenumbers
    ik = 1j * k
    ik3 = 1j * k**3

    v0 = np.fft.fft(u0)
    n_steps = int(round(T / dt))

    def nonlinear_rhs(v):
        # compute 6 u u_x in Fourier space via pseudo-spectral method
        u = np.real(np.fft.ifft(v))
        ux = np.real(np.fft.ifft(ik * v))
        return -np.fft.fft(6.0 * u * ux)

    # Precompute exact linear propagator for u_xxx term: v_t = ik^3 v (linear part)
    E = np.exp(ik3 * dt)
    E2 = np.exp(ik3 * dt / 2)

    v = v0.copy()
    t_hist = [0.0]
    u_hist = [u0.copy()]

    for step in range(1, n_steps + 1):
        """
        classic RK4 for the nonlinear term, with the linear term folded in
         via the integrating factor (this keeps the stiff u_xxx term stable
         at reasonable dt without needing an implicit solve)
         """
        Nv1 = nonlinear_rhs(v)
        a = E2 * (v + dt / 2 * Nv1)

        Nv2 = nonlinear_rhs(a)
        b = E2 * v + dt / 2 * Nv2

        Nv3 = nonlinear_rhs(b)
        c = E2 * a + dt / 2 * (2 * Nv3 - Nv1)

        Nv4 = nonlinear_rhs(c)
        v = E * v + (dt / 6) * (Nv1 + 2 * Nv2 + 2 * Nv3 + Nv4) * E  # rough combine

        if step % nsave == 0:
            t_hist.append(step * dt)
            u_hist.append(np.real(np.fft.ifft(v)))

    return np.array(t_hist), np.array(u_hist)


def random_initial_condition(N, L, n_modes=6, seed=None, amp_scale_range=(0.5, 3.0)):
    """
    Random smooth periodic IC: sum of a few random sinusoids, times an
    overall random amplitude scale.
    """
    rng = np.random.default_rng(seed)
    x = np.linspace(0, L, N, endpoint=False)
    u0 = np.zeros(N)
    for m in range(1, n_modes + 1):
        amp = rng.uniform(-1, 1) / m # decaying amplitude -> smooth
        phase = rng.uniform(0, 2 * np.pi)
        u0 += amp * np.cos(2 * np.pi * m * x / L + phase)
    scale = rng.uniform(*amp_scale_range)
    u0 *= scale
    return u0


# 2. Build a dataset of (u0 -> u(T)) pairs

def make_dataset(n_samples, N=256, L=20.0, T=0.5, dt=1e-4, seed=0, amp_scale_range=(0.5, 3.0)):
    rng = np.random.default_rng(seed)
    X = np.zeros((n_samples, N), dtype=np.float32)
    Y = np.zeros((n_samples, N), dtype=np.float32)
    for i in range(n_samples):
        u0 = random_initial_condition(N, L, seed=rng.integers(1e9), amp_scale_range=amp_scale_range)
        _, u_hist = solve_kdv(u0, L, T, dt, nsave=int(round(T / dt)))
        X[i] = u0
        Y[i] = u_hist[-1]
    return torch.from_numpy(X), torch.from_numpy(Y)


# 3. FNO1d model
class SpectralConv1d(nn.Module):
    """
    1D Fourier layer: FFT -> truncate to `modes` low frequencies ->
    multiply by learnable complex weights -> pad back -> inverse FFT.
    """

    def __init__(self, in_channels, out_channels, modes):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes  # number of Fourier modes to keep

        scale = 1.0 / (in_channels * out_channels)
        # weights are complex -> store as separate real/imag parameters
        self.weight = nn.Parameter(
            scale * torch.rand(in_channels, out_channels, modes, dtype=torch.cfloat)
        )

    def compl_mul1d(self, x, w):
        # x: (batch, in_channel, modes), w: (in_channel, out_channel, modes)
        return torch.einsum("bix,iox->box", x, w)

    def forward(self, x):
        # x: (batch, channels, N)
        batch_size = x.shape[0]
        x_ft = torch.fft.rfft(x) # (batch, channels, N//2+1)

        out_ft = torch.zeros(
            batch_size, self.out_channels, x_ft.shape[-1],
            dtype=torch.cfloat, device=x.device,
        )
        out_ft[:, :, : self.modes] = self.compl_mul1d(
            x_ft[:, :, : self.modes], self.weight
        )

        x_out = torch.fft.irfft(out_ft, n=x.shape[-1])
        return x_out


class FourierLayer(nn.Module):
    """Spectral conv + local (pointwise) linear skip connection + activation."""

    def __init__(self, width, modes):
        super().__init__()
        self.spectral_conv = SpectralConv1d(width, width, modes)
        self.local_linear = nn.Conv1d(width, width, kernel_size=1)

    def forward(self, x):
        return F.gelu(self.spectral_conv(x) + self.local_linear(x))


class FNO1d(nn.Module):
    """
    Full FNO1d: lift -> several Fourier layers -> project.
    Input:  (batch, N, 2)
    Output: (batch, N, 1)
    """

    def __init__(self, modes=16, width=64, n_layers=4):
        super().__init__()
        self.lift = nn.Linear(2, width)
        self.fourier_layers = nn.ModuleList(
            [FourierLayer(width, modes) for _ in range(n_layers)]
        )
        self.proj1 = nn.Linear(width, 128)
        self.proj2 = nn.Linear(128, 1)

    def forward(self, u0, grid):
        # u0, grid: (batch, N)
        x = torch.stack([u0, grid], dim=-1) # (batch, N, 2)
        x = self.lift(x) # (batch, N, width)
        x = x.permute(0, 2, 1) # (batch, width, N) for conv/FFT

        for layer in self.fourier_layers:
            x = layer(x)

        x = x.permute(0, 2, 1) # (batch, N, width)
        x = F.gelu(self.proj1(x))
        x = self.proj2(x) # (batch, N, 1)
        return x.squeeze(-1) # (batch, N)



# 4. Training loop

def relative_l2_loss(pred, target):
    num = torch.norm(pred - target, dim=1)
    den = torch.norm(target, dim=1) + 1e-8
    return torch.mean(num / den)


def plot_predictions(model, X_test, Y_test, x_mean, x_std, y_mean, y_std,
                      L, n_examples=4, savepath="kdv_predictions.png"):
    """
    Plot true vs. predicted u(x,T) for a handful of test examples, plus the
    initial condition u(x,0) for context
    """
    device = next(model.parameters()).device
    model.eval()

    N = X_test.shape[1]
    x = np.linspace(0, L, N, endpoint=False)

    n_examples = min(n_examples, X_test.shape[0])
    fig, axes = plt.subplots(n_examples, 1, figsize=(8, 3 * n_examples), squeeze=False)

    with torch.no_grad():
        X_test_n = (X_test.to(device) - x_mean) / x_std
        grid = torch.linspace(0, L, N, device=device).repeat(X_test.shape[0], 1)
        pred = model(X_test_n, grid) * y_std + y_mean
        pred = pred.cpu().numpy()

    u0_np = X_test.detach().cpu().numpy()
    true_np = Y_test.detach().cpu().numpy()

    for i in range(n_examples):
        ax = axes[i, 0]
        ax.plot(x, u0_np[i], "k--", alpha=0.5, label="u(x, 0)  [input]")
        ax.plot(x, true_np[i], "b-", linewidth=2, label="u(x, T)  [true]")
        ax.plot(x, pred[i], "r--", linewidth=2, label="u(x, T)  [FNO prediction]")
        err = np.linalg.norm(pred[i] - true_np[i]) / (np.linalg.norm(true_np[i]) + 1e-8)
        ax.set_title(f"test example {i}  |  rel-L2 error = {err:.4f}")
        ax.set_xlabel("x")
        ax.legend(loc="upper right", fontsize=8)

    fig.tight_layout()
    fig.savefig(savepath, dpi=150)
    print(f"Saved prediction plot to {savepath}")
    plt.close(fig)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    N, L, T = 256, 20.0, 0.5
    print("Generating training data (via KdV solver)...")
    X_train, Y_train = make_dataset(n_samples=800, N=N, L=L, T=T, seed=0)
    X_test, Y_test = make_dataset(n_samples=100, N=N, L=L, T=T, seed=1)

    # normalize (simple per-dataset standardization)
    x_mean, x_std = X_train.mean(), X_train.std()
    y_mean, y_std = Y_train.mean(), Y_train.std()
    X_train_n = (X_train - x_mean) / x_std
    Y_train_n = (Y_train - y_mean) / y_std
    X_test_n = (X_test - x_mean) / x_std

    grid = torch.linspace(0, L, N, device=device).repeat(len(X_train), 1)
    grid_test = torch.linspace(0, L, N, device=device).repeat(len(X_test), 1)

    model = FNO1d(modes=16, width=64, n_layers=4).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)

    X_train_n, Y_train_n = X_train_n.to(device), Y_train_n.to(device)
    X_test_n, Y_test = X_test_n.to(device), Y_test.to(device)

    batch_size = 20
    n_epochs = 200
    n_train = X_train_n.shape[0]

    for epoch in range(n_epochs):
        model.train()
        perm = torch.randperm(n_train)
        epoch_loss = 0.0
        for i in range(0, n_train, batch_size):
            idx = perm[i : i + batch_size]
            optimizer.zero_grad()
            pred = model(X_train_n[idx], grid[idx])
            loss = relative_l2_loss(pred, Y_train_n[idx])
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(idx)
        scheduler.step()

        if epoch % 20 == 0 or epoch == n_epochs - 1:
            model.eval()
            with torch.no_grad():
                pred_test = model(X_test_n, grid_test) * y_std + y_mean
                test_loss = relative_l2_loss(pred_test, Y_test)
            print(
                f"epoch {epoch:4d} | train loss {epoch_loss / n_train:.4f} "
                f"| test rel-L2 {test_loss.item():.4f}"
            )

    # visualize predictions on a few test examples
    plot_predictions(
        model, X_test, Y_test, x_mean, x_std, y_mean, y_std, L,
        n_examples=4, savepath="kdv_predictions_new.png",
    )

    # resolution-invariance sanity check
    # Evaluate the same trained model on a finer grid than it was trained on.
    print("\nTesting discretization invariance (train N=256, test N=512)...")
    N2 = 512
    X_hi, Y_hi = make_dataset(n_samples=20, N=N2, L=L, T=T, seed=2)
    X_hi_n = ((X_hi - x_mean) / x_std).to(device)
    grid_hi = torch.linspace(0, L, N2, device=device).repeat(len(X_hi), 1)
    with torch.no_grad():
        pred_hi = model(X_hi_n, grid_hi) * y_std + y_mean
        hi_loss = relative_l2_loss(pred_hi, Y_hi.to(device))
    print(f"rel-L2 error at N=512 (trained at N=256): {hi_loss.item():.4f}")


if __name__ == "__main__":
    main()
