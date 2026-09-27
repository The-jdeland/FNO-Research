"""
ConvLSTM baseline for 2D incompressible Stokes flow on the periodic torus.

This mirrors FNO_Stokes_Flow.py's data pipeline, training loop, evaluation
metric (mean relative L2), and figure styles, so the two models can be
compared directly and fairly.

"""

import os
import importlib.util
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt

torch.set_num_threads(os.cpu_count() or 1)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# Load the FNO script as a module so we reuse its exact data utilities
# (load_dataset, generate_dataset, split_dataset, relative_l2, InputNormalizer,
# make_grid). Reusing these -- rather than re-implementing them -- is what
# guarantees the ConvLSTM sees the identical dataset file and identical
# train/val/test split as the FNO baseline.

_HERE = os.path.dirname(os.path.abspath(__file__))
_FNO_PATH = os.path.join(_HERE, "FNO_Stokes_Flow.py")
_spec = importlib.util.spec_from_file_location("fno_stokes_flow", _FNO_PATH)
fno = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fno)


# 1. ConvLSTM model
class ConvLSTMCell(nn.Module):
    """
    Standard convolutional LSTM cell, with circular
    (periodic) padding on the spatial convolution -- matching the
    doubly-periodic torus domain the Stokes solver assumes, the same way
    the FNO's FFT-based spectral convolution is implicitly periodic.
    """

    def __init__(self, in_channels, hidden_channels, kernel_size=3):
        super().__init__()
        self.hidden_channels = hidden_channels
        padding = kernel_size // 2
        self.conv = nn.Conv2d(
            in_channels + hidden_channels,
            4 * hidden_channels,
            kernel_size,
            padding=padding,
            padding_mode="circular",
        )

    def forward(self, x, h, c):
        combined = torch.cat([x, h], dim=1)
        gates = self.conv(combined)
        i, f, o, g = gates.chunk(4, dim=1)
        i, f, o = torch.sigmoid(i), torch.sigmoid(f), torch.sigmoid(o)
        g = torch.tanh(g)
        c_next = f * c + i * g
        h_next = o * torch.tanh(c_next)
        return h_next, c_next


class ConvLSTMStokes(nn.Module):
    """
    Applies a single ConvLSTMCell for `n_steps` internal iterations on a
    *static* input (the encoded forcing field + grid channels, unchanged at
    every step), then reads out the final hidden state through a 1x1 conv
    to the two velocity components (u, v). This turns the ConvLSTM into an
    iterative local-refinement solver, which is the natural way to use a
    recurrent architecture on a steady-state (non-time-series) operator
    learning problem.
    """

    def __init__(self, in_channels, hidden_channels, out_channels=2, n_steps=16, kernel_size=3):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.n_steps = n_steps
        self.cell = ConvLSTMCell(in_channels, hidden_channels, kernel_size)
        self.readout = nn.Conv2d(hidden_channels, out_channels, 1)

    def forward(self, x):
        # x: (B, in_channels, Nx, Ny)
        B, _, Nx, Ny = x.shape
        h = torch.zeros(B, self.hidden_channels, Nx, Ny, device=x.device, dtype=x.dtype)
        c = torch.zeros_like(h)
        for _ in range(self.n_steps):
            h, c = self.cell(x, h, c)
        return self.readout(h)

    def num_params(self):
        return sum(p.numel() for p in self.parameters())


# 2. Training
def train_convlstm(f_train, u_train, f_val, u_val, Lx, Ly, hidden_channels, n_steps,
                    epochs, kernel_size=3, batch_size=20, lr=1e-3, weight_decay=1e-4,
                    verbose=False):
    """
    Trains one ConvLSTMStokes on (f_train -> u_train), tracking mean
    relative-L2 loss on train and val each epoch. Returns
    (model, normalizer, grid_chw, history) -- the same shape of return
    value as FNO_Stokes_Flow.train_fno, so downstream code (evaluate /
    plotting) is symmetric between the two models.
    """
    Nx, Ny = f_train.shape[-2], f_train.shape[-1]
    normalizer = fno.InputNormalizer(f_train).to(DEVICE)
    grid = fno.make_grid(Nx, Ny, Lx, Ly).to(DEVICE)          # (Nx, Ny, 2)
    grid_chw = grid.permute(2, 0, 1).contiguous()             # (2, Nx, Ny), channels-first

    in_channels = f_train.shape[1] + grid_chw.shape[0]        # forcing channels + grid channels
    model = ConvLSTMStokes(in_channels, hidden_channels, out_channels=u_train.shape[1],
                            n_steps=n_steps, kernel_size=kernel_size).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    train_ds = TensorDataset(f_train, u_train)
    train_loader = DataLoader(train_ds, batch_size=min(batch_size, len(train_ds)), shuffle=True)

    f_val_d, u_val_d = f_val.to(DEVICE), u_val.to(DEVICE)

    def with_grid(fb):
        gb = grid_chw.unsqueeze(0).expand(fb.shape[0], -1, -1, -1)
        return torch.cat([normalizer.encode(fb), gb], dim=1)

    history = {"train_loss": [], "val_loss": []}
    for ep in range(epochs):
        model.train()
        losses = []
        for fb, ub in train_loader:
            fb, ub = fb.to(DEVICE), ub.to(DEVICE)
            pred = model(with_grid(fb))
            loss = fno.relative_l2(pred, ub).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.item())
        sched.step()

        model.eval()
        with torch.no_grad():
            pred = model(with_grid(f_val_d))
            val_loss = fno.relative_l2(pred, u_val_d).mean().item()

        history["train_loss"].append(float(np.mean(losses)))
        history["val_loss"].append(val_loss)
        if verbose and (ep % max(1, epochs // 5) == 0 or ep == epochs - 1):
            print(f"    epoch {ep+1:3d}/{epochs}  train {history['train_loss'][-1]:.4f}  val {val_loss:.4f}")

    return model, normalizer, grid_chw, history


def evaluate_convlstm(model, normalizer, grid_chw, f_test, u_test):
    """Returns (mean_rel_l2, per_sample_rel_l2_array, predictions)."""
    model.eval()
    with torch.no_grad():
        f_test_d = f_test.to(DEVICE)
        gb = grid_chw.unsqueeze(0).expand(f_test_d.shape[0], -1, -1, -1)
        x_in = torch.cat([normalizer.encode(f_test_d), gb], dim=1)
        pred = model(x_in)
        errs = fno.relative_l2(pred, u_test.to(DEVICE)).cpu().numpy()
    return float(errs.mean()), errs, pred.cpu()


# 3. Figures
def plot_training_curves(history, savepath):
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.plot(history["train_loss"], label="train")
    ax.plot(history["val_loss"], label="val")
    ax.set_xlabel("epoch")
    ax.set_ylabel("mean relative $L_2$ error")
    ax.set_yscale("log")
    ax.set_title("ConvLSTM training curves (forcing $\\to$ velocity)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(savepath, dpi=150)
    plt.close(fig)
    print(f"Saved {savepath}")


def plot_prediction_examples(model, normalizer, grid_chw, f_test, u_test, savepath, n_examples=3):
    """For a few test samples: truth vs prediction vs |error|, for both u and v."""
    model.eval()
    idx = np.linspace(0, f_test.shape[0] - 1, n_examples).astype(int)
    with torch.no_grad():
        f_sel = f_test[idx].to(DEVICE)
        gb = grid_chw.unsqueeze(0).expand(f_sel.shape[0], -1, -1, -1)
        x_in = torch.cat([normalizer.encode(f_sel), gb], dim=1)
        pred = model(x_in).cpu().numpy()
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
            rel = fno.relative_l2(torch.tensor(p)[None], torch.tensor(t)[None]).item()
            axes[row, 2].set_title(f"|error|  (rel $L_2$={rel:.3f})")
            for ax, im in [(axes[row, 0], im0), (axes[row, 1], im1), (axes[row, 2], im2)]:
                ax.set_xticks([]); ax.set_yticks([])
                fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle("ConvLSTM: exact vs. predicted flow", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(savepath, dpi=150)
    plt.close(fig)
    print(f"Saved {savepath}")


def plot_error_histogram(errs, savepath):
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.hist(errs, bins=20, color="indianred", edgecolor="black")
    ax.axvline(errs.mean(), color="navy", linestyle="--", label=f"mean = {errs.mean():.3f}")
    ax.set_xlabel("per-sample relative $L_2$ error")
    ax.set_ylabel("count")
    ax.set_title("ConvLSTM test-set error distribution")
    ax.legend()
    fig.tight_layout()
    fig.savefig(savepath, dpi=150)
    plt.close(fig)
    print(f"Saved {savepath}")


# 4. Config + main
CONFIG = dict(
    # Points straight at the dataset produced by 2D_Stokes_Full_Dataset.py's
    # generate_training_dataset() batch call (n_samples=1000, Nx=Ny=128,
    # alpha=4.0, nu=0.1) -- if this file exists it's loaded directly and
    # base_res/base_n_samples below are ignored. Only used as a fallback (to
    # generate a fresh, differently-named dataset) if the path is missing.
    dataset_path=os.path.join("stokes_data_For_Training", "stokes2d_train_Full_Data.npz"),
    base_res=128,
    base_n_samples=2000,
    n_train=700, n_val=150, n_test=150,   # sums to 1000, matching the file above
    hidden_channels=32,   # analogous to FNO's base_width
    n_steps=16,           # unrolled ConvLSTM iterations (see receptive-field note above)
    kernel_size=3,
    base_epochs=200,      # matches FNO's base_epochs for a like-for-like epoch axis
    seed=0,
)

# Smaller, CPU-sandbox-friendly settings for a quick pipeline sanity check.
QUICK_CONFIG = dict(
    base_res=32,
    base_n_samples=300,
    n_train=200, n_val=50, n_test=50,
    hidden_channels=16,
    n_steps=8,
    kernel_size=3,
    base_epochs=20,
    seed=0,
)


def main(cfg=CONFIG, use_quick=False):
    cfg = QUICK_CONFIG if use_quick else cfg
    out_dir = "convlstm_results"
    data_dir = "fno_data"   # same directory the FNO script caches datasets in
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)
    print(f"Device: {DEVICE}   Config: {'QUICK' if use_quick else 'FULL'}")

    # Load the dataset directly if it's already on disk (e.g. the batch file
    # produced by 2D_Stokes_Full_Dataset.py's generate_training_dataset());
    # only fall back to generating a fresh FNO-style cached dataset if it's not.
    dataset_path = cfg.get("dataset_path")
    if dataset_path and os.path.exists(dataset_path):
        print(f"Loading dataset directly from {dataset_path}")
        f, u, meta = fno.load_dataset(dataset_path)
    else:
        base_path = fno.generate_dataset(cfg["base_n_samples"], cfg["base_res"], cfg["base_res"],
                                          alpha=4.0, save_dir=data_dir, tag="base")
        f, u, meta = fno.load_dataset(base_path)
    (f_tr, u_tr), (f_va, u_va), (f_te, u_te) = fno.split_dataset(
        f, u, cfg["n_train"], cfg["n_val"], cfg["n_test"], seed=cfg["seed"])

    print("\n=== ConvLSTM training ===")
    model, norm, grid_chw, hist = train_convlstm(
        f_tr, u_tr, f_va, u_va, meta["Lx"], meta["Ly"],
        cfg["hidden_channels"], cfg["n_steps"], cfg["base_epochs"],
        kernel_size=cfg["kernel_size"], verbose=True)
    print(f"ConvLSTM parameter count: {model.num_params():,}")

    mean_err, errs, _ = evaluate_convlstm(model, norm, grid_chw, f_te, u_te)
    print(f"ConvLSTM test mean relative L2 error: {mean_err:.4f}")

    plot_training_curves(hist, os.path.join(out_dir, "convlstm_training_curves.png"))
    plot_prediction_examples(model, norm, grid_chw, f_te, u_te,
                              os.path.join(out_dir, "convlstm_prediction_examples.png"))
    plot_error_histogram(errs, os.path.join(out_dir, "convlstm_error_histogram.png"))

    print(f"\nAll figures saved under {out_dir}/")
    return model, norm, grid_chw, hist, errs


if __name__ == "__main__":
    main()