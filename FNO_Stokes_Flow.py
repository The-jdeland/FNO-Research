import os
import time
import importlib.util
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt

torch.set_num_threads(os.cpu_count() or 1)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Load the Stokes data generator from 2d_Stokes_Full_Dataset.py
_STOKES_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "2D_Stokes_Full_Dataset.py")
_spec = importlib.util.spec_from_file_location("stokes_data_For_Training", _STOKES_PATH)
stokes2d = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(stokes2d)


# 1. FNO2d model
class SpectralConv2d(nn.Module):
    """
    2D spectral convolution: FFT -> truncate to (modes1, modes2) low-frequency
    modes -> learned complex linear map on the kept modes -> inverse FFT.
    This is the core FNO building block (Li et al. 2021) and is what makes
    the model resolution-invariant -- the learned weights live in Fourier
    space and don't reference the grid size at all.
    """

    def __init__(self, in_channels, out_channels, modes1, modes2):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2

        scale = 1.0 / (in_channels * out_channels)
        self.weights1 = nn.Parameter(
            scale * torch.rand(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat)
        )
        self.weights2 = nn.Parameter(
            scale * torch.rand(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat)
        )

    @staticmethod
    def _compl_mul2d(inp, weights):
        # (batch, in_channel, x, y) , (in_channel, out_channel, x, y) -> (batch, out_channel, x, y)
        return torch.einsum("bixy,ioxy->boxy", inp, weights)

    def forward(self, x):
        batchsize, _, Nx, Ny = x.shape
        x_ft = torch.fft.rfft2(x)

        out_ft = torch.zeros(
            batchsize, self.out_channels, Nx, Ny // 2 + 1, dtype=torch.cfloat, device=x.device
        )
        m1 = min(self.modes1, Nx)
        m2 = min(self.modes2, Ny // 2 + 1)
        out_ft[:, :, :m1, :m2] = self._compl_mul2d(x_ft[:, :, :m1, :m2], self.weights1[:, :, :m1, :m2])
        out_ft[:, :, -m1:, :m2] = self._compl_mul2d(x_ft[:, :, -m1:, :m2], self.weights2[:, :, :m1, :m2])

        x = torch.fft.irfft2(out_ft, s=(Nx, Ny))
        return x


class FNO2d(nn.Module):
    """
    Standard FNO2d: lift (input channels + 2 grid channels) -> width,
    `depth` Fourier layers (spectral conv + pointwise conv + GELU), then
    project width -> output channels. Grid coordinates are concatenated to
    the input so the model can, if useful, learn position-dependent effects
    even though the underlying PDE map here is translation-equivariant.
    """

    def __init__(self, modes1, modes2, width, depth, in_channels=2, out_channels=2):
        super().__init__()
        self.width = width
        self.depth = depth
        self.fc0 = nn.Linear(in_channels + 2, width)
        self.spectral_convs = nn.ModuleList(
            [SpectralConv2d(width, width, modes1, modes2) for _ in range(depth)]
        )
        self.ws = nn.ModuleList([nn.Conv2d(width, width, 1) for _ in range(depth)])
        self.fc1 = nn.Linear(width, 4 * width)
        self.fc2 = nn.Linear(4 * width, out_channels)

    def forward(self, x, grid):
        # x: (B, Nx, Ny, in_channels), grid: (B, Nx, Ny, 2)
        x = torch.cat([x, grid], dim=-1)
        x = self.fc0(x)
        x = x.permute(0, 3, 1, 2)  # B, C, Nx, Ny
        for i in range(self.depth):
            x1 = self.spectral_convs[i](x)
            x2 = self.ws[i](x)
            x = x1 + x2
            if i < self.depth - 1:
                x = F.gelu(x)
        x = x.permute(0, 2, 3, 1)  # B, Nx, Ny, C
        x = F.gelu(self.fc1(x))
        x = self.fc2(x)
        return x.permute(0, 3, 1, 2)  # B, out_channels, Nx, Ny

    def num_params(self):
        return sum(p.numel() for p in self.parameters())


# 2. Data utilities
class InputNormalizer:
    """Per-channel Gaussian normalizer, fit on the training split only."""

    def __init__(self, x):
        # x: (N, C, Nx, Ny)
        self.mean = x.mean(dim=(0, 2, 3), keepdim=True)
        self.std = x.std(dim=(0, 2, 3), keepdim=True) + 1e-8

    def encode(self, x):
        return (x - self.mean) / self.std

    def to(self, device):
        self.mean = self.mean.to(device)
        self.std = self.std.to(device)
        return self


def make_grid(Nx, Ny, Lx, Ly):
    """Normalized coordinate grid, shape (Nx, Ny, 2), values in [0, 1)."""
    gx = torch.linspace(0, 1, Nx + 1)[:-1]
    gy = torch.linspace(0, 1, Ny + 1)[:-1]
    GX, GY = torch.meshgrid(gx, gy, indexing="ij")
    return torch.stack([GX, GY], dim=-1)


def load_dataset(npz_path):
    """
    Load a dataset produced by 2d_Stokes_Full_Dataset.py's generate_training_dataset().
    Returns forcing (N,2,Nx,Ny), velocity (N,2,Nx,Ny), and metadata.
    """
    d = np.load(npz_path)
    f = np.stack([d["f1"], d["f2"]], axis=1).astype(np.float32)  # (N,2,Nx,Ny)
    u = np.stack([d["u"], d["v"]], axis=1).astype(np.float32)    # (N,2,Nx,Ny)
    meta = dict(Lx=float(d["Lx"]), Ly=float(d["Ly"]), nu=float(d["nu"]))
    return torch.from_numpy(f), torch.from_numpy(u), meta


def generate_dataset(n_samples, Nx, Ny, alpha=4.0, tau=5.0, nu=0.1, seed0=0,
                      save_dir="fno_data", tag=None):
    """Thin wrapper around 2d_Stokes.generate_training_dataset with a
    descriptive filename, so sweep runs cache to distinct files."""
    tag = tag or f"N{n_samples}_res{Nx}_alpha{alpha}_seed{seed0}"
    save_name = f"stokes2d_{tag}.npz"
    path = os.path.join(save_dir, save_name)
    if os.path.exists(path):
        return path
    stokes2d.generate_training_dataset(
        n_samples=n_samples, Nx=Nx, Ny=Ny,
        Lx=2 * np.pi, Ly=2 * np.pi, nu=nu,
        forcing_amplitude=1.0, alpha=alpha, tau=tau,
        seed0=seed0, save_dir=save_dir, save_name=save_name,
        check_every=max(n_samples, 1),
    )
    return path


def split_dataset(f, u, n_train, n_val, n_test, seed=0):
    """Fixed, reproducible random split into train/val/test index sets."""
    g = torch.Generator().manual_seed(seed)
    n_total = f.shape[0]
    assert n_train + n_val + n_test <= n_total, "not enough samples for requested split"
    perm = torch.randperm(n_total, generator=g)
    idx_train = perm[:n_train]
    idx_val = perm[n_train:n_train + n_val]
    idx_test = perm[n_train + n_val:n_train + n_val + n_test]
    return (f[idx_train], u[idx_train]), (f[idx_val], u[idx_val]), (f[idx_test], u[idx_test])


# 3. Loss / metrics
def relative_l2(pred, target):
    """Per-sample relative L2 error, shape (B,). Standard FNO-paper metric."""
    B = pred.shape[0]
    num = torch.norm(pred.reshape(B, -1) - target.reshape(B, -1), dim=1)
    den = torch.norm(target.reshape(B, -1), dim=1) + 1e-8
    return num / den


# 4. Training
def train_fno(f_train, u_train, f_val, u_val, Lx, Ly, modes, width, depth,
              epochs, batch_size=20, lr=1e-3, weight_decay=1e-4, verbose=False):
    """
    Trains one FNO2d on (f_train -> u_train), tracking mean relative-L2 loss
    on train and val each epoch. Returns (model, normalizer, grid, history).
    """
    Nx, Ny = f_train.shape[-2], f_train.shape[-1]
    normalizer = InputNormalizer(f_train).to(DEVICE)
    grid = make_grid(Nx, Ny, Lx, Ly).to(DEVICE)

    model = FNO2d(modes, modes, width, depth, in_channels=2, out_channels=2).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    def to_bhwc(x):
        return x.permute(0, 2, 3, 1)  # B,C,Nx,Ny -> B,Nx,Ny,C

    train_ds = TensorDataset(f_train, u_train)
    train_loader = DataLoader(train_ds, batch_size=min(batch_size, len(train_ds)), shuffle=True)

    f_val_d, u_val_d = f_val.to(DEVICE), u_val.to(DEVICE)

    history = {"train_loss": [], "val_loss": []}
    for ep in range(epochs):
        model.train()
        losses = []
        for fb, ub in train_loader:
            fb, ub = fb.to(DEVICE), ub.to(DEVICE)
            fb_n = to_bhwc(normalizer.encode(fb))
            gb = grid.unsqueeze(0).expand(fb.shape[0], -1, -1, -1)
            pred = model(fb_n, gb)
            loss = relative_l2(pred, ub).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.item())
        sched.step()

        model.eval()
        with torch.no_grad():
            fb_n = to_bhwc(normalizer.encode(f_val_d))
            gb = grid.unsqueeze(0).expand(f_val_d.shape[0], -1, -1, -1)
            pred = model(fb_n, gb)
            val_loss = relative_l2(pred, u_val_d).mean().item()

        history["train_loss"].append(float(np.mean(losses)))
        history["val_loss"].append(val_loss)
        if verbose and (ep % max(1, epochs // 5) == 0 or ep == epochs - 1):
            print(f"    epoch {ep+1:3d}/{epochs}  train {history['train_loss'][-1]:.4f}  val {val_loss:.4f}")

    return model, normalizer, grid, history


def evaluate_fno(model, normalizer, grid, f_test, u_test):
    """Returns (mean_rel_l2, per_sample_rel_l2_array, predictions)."""
    model.eval()
    with torch.no_grad():
        f_test_d = f_test.to(DEVICE)
        fb_n = normalizer.encode(f_test_d).permute(0, 2, 3, 1)
        gb = grid.unsqueeze(0).expand(f_test_d.shape[0], -1, -1, -1)
        pred = model(fb_n, gb)
        errs = relative_l2(pred, u_test.to(DEVICE)).cpu().numpy()
    return float(errs.mean()), errs, pred.cpu()


# 5. Performance figures (part A)
def plot_training_curves(history, savepath):
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.plot(history["train_loss"], label="train")
    ax.plot(history["val_loss"], label="val")
    ax.set_xlabel("epoch")
    ax.set_ylabel("mean relative $L_2$ error")
    ax.set_yscale("log")
    ax.set_title("FNO training curves (forcing $\\to$ velocity)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(savepath, dpi=150)
    plt.close(fig)
    print(f"Saved {savepath}")


def plot_prediction_examples(model, normalizer, grid, f_test, u_test, savepath, n_examples=3):
    """For a few test samples: truth vs prediction vs error, for both u and v."""
    model.eval()
    idx = np.linspace(0, f_test.shape[0] - 1, n_examples).astype(int)
    with torch.no_grad():
        f_sel = f_test[idx].to(DEVICE)
        fb_n = normalizer.encode(f_sel).permute(0, 2, 3, 1)
        gb = grid.unsqueeze(0).expand(f_sel.shape[0], -1, -1, -1)
        pred = model(fb_n, gb).cpu().numpy()
    truth = u_test[idx].numpy()

    comp_names = ["u", "v"]
    fig, axes = plt.subplots(n_examples * 2, 3, figsize=(9, 3.0 * n_examples * 2))
    for i in range(n_examples):
        for c in range(2):
            row = 2 * i + c
            t, p = truth[i, c], pred[i, c]
            err = np.abs(t - p)
            vmax = max(np.abs(t).max(), np.abs(p).max())
            im0 = axes[row, 0].imshow(t.T, origin="lower", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
            axes[row, 0].set_title(f"sample {idx[i]}: {comp_names[c]} truth")
            im1 = axes[row, 1].imshow(p.T, origin="lower", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
            axes[row, 1].set_title(f"{comp_names[c]} prediction")
            im2 = axes[row, 2].imshow(err.T, origin="lower", cmap="viridis")
            axes[row, 2].set_title(f"|error|  (rel $L_2$={relative_l2(torch.tensor(p)[None], torch.tensor(t)[None]).item():.3f})")
            for ax, im in [(axes[row, 0], im0), (axes[row, 1], im1), (axes[row, 2], im2)]:
                ax.set_xticks([]); ax.set_yticks([])
                fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(savepath, dpi=150)
    plt.close(fig)
    print(f"Saved {savepath}")


def plot_error_histogram(errs, savepath):
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.hist(errs, bins=20, color="steelblue", edgecolor="black")
    ax.axvline(errs.mean(), color="crimson", linestyle="--", label=f"mean = {errs.mean():.3f}")
    ax.set_xlabel("per-sample relative $L_2$ error")
    ax.set_ylabel("count")
    ax.set_title("Test-set error distribution")
    ax.legend()
    fig.tight_layout()
    fig.savefig(savepath, dpi=150)
    plt.close(fig)
    print(f"Saved {savepath}")


def plot_error_spectrum(model, normalizer, grid, f_test, u_test, savepath):
    """
    Radially-averaged power spectrum of the error field vs. wavenumber,
    compared to the truth's own spectrum. This shows *which* Fourier modes
    the FNO is failing to capture -- directly relevant to the "number of
    modes" question, independent of the modes sweep below.
    """
    model.eval()
    with torch.no_grad():
        f_test_d = f_test.to(DEVICE)
        fb_n = normalizer.encode(f_test_d).permute(0, 2, 3, 1)
        gb = grid.unsqueeze(0).expand(f_test_d.shape[0], -1, -1, -1)
        pred = model(fb_n, gb).cpu().numpy()
    truth = u_test.numpy()
    err = pred - truth

    Nx, Ny = truth.shape[-2], truth.shape[-1]
    kx = np.fft.fftfreq(Nx) * Nx
    ky = np.fft.fftfreq(Ny) * Ny
    KX, KY = np.meshgrid(kx, ky, indexing="ij")
    Kmag = np.sqrt(KX**2 + KY**2)
    kbins = np.arange(0, Nx // 2)

    def radial_spectrum(field):
        # field: (N, C, Nx, Ny) -> averaged power per radial wavenumber bin
        fh = np.fft.fft2(field, axes=(-2, -1))
        power = np.mean(np.abs(fh) ** 2, axis=(0, 1))  # avg over samples & components
        spec = np.zeros(len(kbins))
        for i, kb in enumerate(kbins):
            mask = (Kmag >= kb) & (Kmag < kb + 1)
            spec[i] = power[mask].mean() if mask.any() else np.nan
        return spec

    spec_truth = radial_spectrum(truth)
    spec_err = radial_spectrum(err)

    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.plot(kbins, spec_truth, label="true velocity spectrum")
    ax.plot(kbins, spec_err, label="error spectrum")
    ax.set_xlabel("wavenumber $|k|$")
    ax.set_ylabel("mean power")
    ax.set_yscale("log")
    ax.set_title("Radially-averaged spectrum: signal vs. error")
    ax.legend()
    fig.tight_layout()
    fig.savefig(savepath, dpi=150)
    plt.close(fig)
    print(f"Saved {savepath}")


# 6. Sweeps (part B)
def sweep_modes(f, u, Lx, Ly, modes_list, n_train, n_val, n_test, width, depth,
                 epochs, out_dir, seed=0):
    print("\n=== Sweep: number of Fourier modes ===")
    (f_tr, u_tr), (f_va, u_va), (f_te, u_te) = split_dataset(f, u, n_train, n_val, n_test, seed)
    results = []
    for modes in modes_list:
        t0 = time.time()
        model, norm, grid, hist = train_fno(f_tr, u_tr, f_va, u_va, Lx, Ly,
                                             modes, width, depth, epochs)
        mean_err, errs, _ = evaluate_fno(model, norm, grid, f_te, u_te)
        results.append(dict(modes=modes, test_err=mean_err, train_err=hist["train_loss"][-1],
                             n_params=model.num_params()))
        print(f"  modes={modes:3d}  test rel-L2={mean_err:.4f}  ({time.time()-t0:.1f}s)")

    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.plot([r["modes"] for r in results], [r["train_err"] for r in results], "o-", label="train")
    ax.plot([r["modes"] for r in results], [r["test_err"] for r in results], "o-", label="test")
    ax.set_xlabel("number of Fourier modes kept (per dimension)")
    ax.set_ylabel("mean relative $L_2$ error")
    ax.set_yscale("log")
    ax.set_title("Approximation error vs. number of Fourier modes")
    ax.legend()
    fig.tight_layout()
    path = os.path.join(out_dir, "sweep_modes.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved {path}")
    return results


def sweep_capacity(f, u, Lx, Ly, width_list, depth_list, n_train, n_val, n_test, modes,
                    epochs, out_dir, seed=0):
    print("\n=== Sweep: model capacity (width x depth) ===")
    (f_tr, u_tr), (f_va, u_va), (f_te, u_te) = split_dataset(f, u, n_train, n_val, n_test, seed)
    results = []
    for width in width_list:
        for depth in depth_list:
            t0 = time.time()
            model, norm, grid, hist = train_fno(f_tr, u_tr, f_va, u_va, Lx, Ly,
                                                 modes, width, depth, epochs)
            mean_err, errs, _ = evaluate_fno(model, norm, grid, f_te, u_te)
            results.append(dict(width=width, depth=depth, n_params=model.num_params(),
                                 test_err=mean_err, train_err=hist["train_loss"][-1]))
            print(f"  width={width:3d} depth={depth}  params={model.num_params():7d}  "
                  f"test rel-L2={mean_err:.4f}  ({time.time()-t0:.1f}s)")

    fig, ax = plt.subplots(figsize=(6.5, 5))
    depths = sorted(set(r["depth"] for r in results))
    cmap = plt.get_cmap("viridis")
    colors = {d: cmap(i / max(1, len(depths) - 1)) for i, d in enumerate(depths)}
    for d in depths:
        rows = sorted([r for r in results if r["depth"] == d], key=lambda r: r["n_params"])
        params = [r["n_params"] for r in rows]
        ax.plot(params, [r["train_err"] for r in rows], "o--", color=colors[d], alpha=0.6,
                label=f"depth={d} (train)")
        ax.plot(params, [r["test_err"] for r in rows], "o-", color=colors[d],
                label=f"depth={d} (test)")
    ax.set_xlabel("number of model parameters")
    ax.set_ylabel("mean relative $L_2$ error")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title("Approximation error vs. model capacity")
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = os.path.join(out_dir, "sweep_capacity.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved {path}")
    return results


def sweep_sample_size(f, u, Lx, Ly, n_train_list, n_val, n_test, modes, width, depth,
                       epochs, out_dir, seed=0, batch_size=20):
    """
    NOTE ON A REAL CONFOUND: with a fixed epoch count and fixed batch size,
    a run with fewer training samples gets proportionally *fewer gradient
    updates* per epoch (n_train=20 at batch_size=20 is 1 step/epoch;
    n_train=200 is 10 steps/epoch). Training all runs for the same number
    of *epochs* would then measure "more optimization steps" as much as
    "more data" -- not what a generalization-vs-sample-size curve is
    supposed to isolate. So instead we hold the total number of gradient
    steps roughly constant across the sweep (matching what n_train_list[-1]
    would get at `epochs` epochs), and let smaller-N runs use proportionally
    more epochs to reach that same step budget. Wall-clock cost per run
    stays roughly constant too, since a step's cost depends on batch size,
    not on how many epochs it took to accumulate.
    """
    print("\n=== Sweep: training sample size ===")
    max_n = max(n_train_list)
    steps_per_epoch_max = max(1, max_n // batch_size)
    target_steps = epochs * steps_per_epoch_max

    results = []
    for n_train in n_train_list:
        (f_tr, u_tr), (f_va, u_va), (f_te, u_te) = split_dataset(f, u, n_train, n_val, n_test, seed)
        steps_per_epoch = max(1, n_train // batch_size)
        this_epochs = max(epochs, int(np.ceil(target_steps / steps_per_epoch)))
        this_epochs = min(this_epochs, 8 * epochs)  # cap worst-case blowup for tiny n_train
        t0 = time.time()
        model, norm, grid, hist = train_fno(f_tr, u_tr, f_va, u_va, Lx, Ly,
                                             modes, width, depth, this_epochs, batch_size=batch_size)
        mean_err, errs, _ = evaluate_fno(model, norm, grid, f_te, u_te)
        results.append(dict(n_train=n_train, test_err=mean_err, train_err=hist["train_loss"][-1],
                             epochs_used=this_epochs))
        print(f"  n_train={n_train:5d}  (epochs={this_epochs:4d}, ~{target_steps} grad steps)  "
              f"test rel-L2={mean_err:.4f}  ({time.time()-t0:.1f}s)")

    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.plot([r["n_train"] for r in results], [r["train_err"] for r in results], "o-", label="train")
    ax.plot([r["n_train"] for r in results], [r["test_err"] for r in results], "o-", label="test")
    ax.set_xlabel("number of training samples")
    ax.set_ylabel("mean relative $L_2$ error")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title("Generalization error vs. training sample size")
    ax.legend()
    fig.tight_layout()
    path = os.path.join(out_dir, "sweep_sample_size.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved {path}")
    return results


def sweep_resolution(res_list, n_samples, n_train, n_val, n_test, modes_frac, width, depth,
                      epochs, out_dir, alpha=4.0, nu=0.1, seed=0, data_dir="fno_data"):
    """
    Generates a fresh dataset at each resolution (via 2d_Stokes.py), trains
    one FNO per resolution, and plots test error vs. resolution. `modes_frac`
    sets modes = max(4, int(modes_frac * Nx)) so the *fraction* of resolved
    modes stays comparable across grids as Nx grows.
    """
    print("\n=== Sweep: grid resolution ===")
    results = []
    for res in res_list:
        path = generate_dataset(n_samples, res, res, alpha=alpha, nu=nu,
                                 seed0=seed * 10_000 + res, save_dir=data_dir,
                                 tag=f"res_sweep_{res}")
        f, u, meta = load_dataset(path)
        (f_tr, u_tr), (f_va, u_va), (f_te, u_te) = split_dataset(f, u, n_train, n_val, n_test, seed)
        modes = max(4, int(modes_frac * res))
        modes = min(modes, res // 2)
        t0 = time.time()
        model, norm, grid, hist = train_fno(f_tr, u_tr, f_va, u_va, meta["Lx"], meta["Ly"],
                                             modes, width, depth, epochs)
        mean_err, errs, _ = evaluate_fno(model, norm, grid, f_te, u_te)
        results.append(dict(res=res, modes=modes, test_err=mean_err, train_err=hist["train_loss"][-1]))
        print(f"  res={res:4d}x{res:<4d} modes={modes:3d}  test rel-L2={mean_err:.4f}  ({time.time()-t0:.1f}s)")

    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.plot([r["res"] for r in results], [r["train_err"] for r in results], "o-", label="train")
    ax.plot([r["res"] for r in results], [r["test_err"] for r in results], "o-", label="test")
    ax.set_xlabel("grid resolution ($N_x = N_y$)")
    ax.set_ylabel("mean relative $L_2$ error")
    ax.set_yscale("log")
    ax.set_title("Error vs. grid resolution")
    ax.legend()
    fig.tight_layout()
    path = os.path.join(out_dir, "sweep_resolution.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved {path}")
    return results


def sweep_regularity(alpha_list, n_samples, res, n_train, n_val, n_test, modes, width, depth,
                      epochs, out_dir, tau=5.0, nu=0.1, seed=0, data_dir="fno_data"):
    """
    Generates datasets with different forcing smoothness (the spectral
    filter exponent `alpha` in gaussian_random_field_2d -- larger alpha =
    faster spectral decay = smoother/more regular forcing) and plots test
    error vs. alpha. This is the "input regularity" axis from the abstract.
    """
    print("\n=== Sweep: input regularity (forcing smoothness, alpha) ===")
    results = []
    for alpha in alpha_list:
        path = generate_dataset(n_samples, res, res, alpha=alpha, tau=tau, nu=nu,
                                 seed0=seed * 10_000 + int(alpha * 100), save_dir=data_dir,
                                 tag=f"reg_sweep_alpha{alpha}")
        f, u, meta = load_dataset(path)
        (f_tr, u_tr), (f_va, u_va), (f_te, u_te) = split_dataset(f, u, n_train, n_val, n_test, seed)
        t0 = time.time()
        model, norm, grid, hist = train_fno(f_tr, u_tr, f_va, u_va, meta["Lx"], meta["Ly"],
                                             modes, width, depth, epochs)
        mean_err, errs, _ = evaluate_fno(model, norm, grid, f_te, u_te)
        results.append(dict(alpha=alpha, test_err=mean_err, train_err=hist["train_loss"][-1]))
        print(f"  alpha={alpha:.1f} (smoother->larger)  test rel-L2={mean_err:.4f}  ({time.time()-t0:.1f}s)")

    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.plot([r["alpha"] for r in results], [r["train_err"] for r in results], "o-", label="train")
    ax.plot([r["alpha"] for r in results], [r["test_err"] for r in results], "o-", label="test")
    ax.set_xlabel(r"forcing spectral decay rate $\alpha$  (larger = smoother input)")
    ax.set_ylabel("mean relative $L_2$ error")
    ax.set_yscale("log")
    ax.set_title("Error vs. input regularity")
    ax.legend()
    fig.tight_layout()
    path = os.path.join(out_dir, "sweep_regularity.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved {path}")
    return results


# 7. Config + main

"""
# Quick, CPU-sandbox-friendly settings -- small enough to finish in a few
# minutes and sanity-check the whole pipeline. Every number here is a knob;
# see FULL_CONFIG below for what to switch to for report-quality results.
QUICK_CONFIG = dict(
    base_res=32,
    base_n_samples=300,
    n_train=200, n_val=50, n_test=50,
    base_modes=8, base_width=16, base_depth=4,
    base_epochs=20,
    modes_list=[2, 4, 8, 12],
    width_list=[8, 16, 32], depth_list=[2, 4],
    n_train_list=[20, 50, 100, 200],
    res_list=[16, 32, 48],
    res_n_samples=200, res_modes_frac=0.25,
    alpha_list=[2.0, 3.0, 4.0, 6.0, 8.0],
    reg_n_samples=200, reg_res=32,
    sweep_epochs=35,
)

"""

#Actual full size data we want for training
FULL_CONFIG = dict(
    base_res=128,
    base_n_samples=2000,
    n_train=1500, n_val=250, n_test=250,
    base_modes=16, base_width=32, base_depth=4,
    base_epochs=200,
    modes_list=[4, 8, 12, 16, 24, 32],
    width_list=[8, 16, 32, 64], depth_list=[2, 4, 6],
    n_train_list=[25, 50, 100, 250, 500, 1000, 1500],
    res_list=[32, 64, 96, 128],
    res_n_samples=1000, res_modes_frac=0.125,
    alpha_list=[1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0],
    reg_n_samples=1000, reg_res=64,
    sweep_epochs=100,
)

CONFIG = FULL_CONFIG


def main():
    cfg = CONFIG
    out_dir = "fno_results"
    data_dir = "fno_data"
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)
    print(f"Device: {DEVICE}   Config: {'FULL' if cfg is FULL_CONFIG else 'QUICK'}")

    # --- base dataset + baseline model (part A: performance figures) ---
    base_path = generate_dataset(cfg["base_n_samples"], cfg["base_res"], cfg["base_res"],
                                  alpha=4.0, save_dir=data_dir, tag="base")
    f, u, meta = load_dataset(base_path)
    (f_tr, u_tr), (f_va, u_va), (f_te, u_te) = split_dataset(
        f, u, cfg["n_train"], cfg["n_val"], cfg["n_test"], seed=0)

    print("\n=== Baseline FNO training ===")
    model, norm, grid, hist = train_fno(
        f_tr, u_tr, f_va, u_va, meta["Lx"], meta["Ly"],
        cfg["base_modes"], cfg["base_width"], cfg["base_depth"],
        cfg["base_epochs"], verbose=True)
    mean_err, errs, _ = evaluate_fno(model, norm, grid, f_te, u_te)
    print(f"Baseline test mean relative L2 error: {mean_err:.4f}")

    plot_training_curves(hist, os.path.join(out_dir, "training_curves.png"))
    plot_prediction_examples(model, norm, grid, f_te, u_te,
                              os.path.join(out_dir, "prediction_examples.png"))
    plot_error_histogram(errs, os.path.join(out_dir, "error_histogram.png"))
    plot_error_spectrum(model, norm, grid, f_te, u_te,
                         os.path.join(out_dir, "error_spectrum.png"))

    # --- sweeps (part B: approximation / generalization study) ---
    sweep_modes(f, u, meta["Lx"], meta["Ly"], cfg["modes_list"],
                cfg["n_train"], cfg["n_val"], cfg["n_test"],
                cfg["base_width"], cfg["base_depth"], cfg["sweep_epochs"], out_dir)

    sweep_capacity(f, u, meta["Lx"], meta["Ly"], cfg["width_list"], cfg["depth_list"],
                    cfg["n_train"], cfg["n_val"], cfg["n_test"],
                    cfg["base_modes"], cfg["sweep_epochs"], out_dir)

    sweep_sample_size(f, u, meta["Lx"], meta["Ly"], cfg["n_train_list"],
                       cfg["n_val"], cfg["n_test"], cfg["base_modes"],
                       cfg["base_width"], cfg["base_depth"], cfg["sweep_epochs"], out_dir)

    sweep_resolution(cfg["res_list"], cfg["res_n_samples"],
                      min(cfg["n_train"], int(0.7 * cfg["res_n_samples"])),
                      min(cfg["n_val"], int(0.15 * cfg["res_n_samples"])),
                      min(cfg["n_test"], int(0.15 * cfg["res_n_samples"])),
                      cfg["res_modes_frac"], cfg["base_width"], cfg["base_depth"],
                      cfg["sweep_epochs"], out_dir, data_dir=data_dir)

    sweep_regularity(cfg["alpha_list"], cfg["reg_n_samples"], cfg["reg_res"],
                      min(cfg["n_train"], int(0.7 * cfg["reg_n_samples"])),
                      min(cfg["n_val"], int(0.15 * cfg["reg_n_samples"])),
                      min(cfg["n_test"], int(0.15 * cfg["reg_n_samples"])),
                      cfg["base_modes"], cfg["base_width"], cfg["base_depth"],
                      cfg["sweep_epochs"], out_dir, data_dir=data_dir)

    print(f"\nAll figures saved under {out_dir}/")


if __name__ == "__main__":
    main()