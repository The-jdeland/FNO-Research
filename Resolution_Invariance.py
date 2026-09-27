"""
Demonstrates the FNO's resolution (discretization) invariance on the 2D
Stokes flow problem.

The property being tested: an FNO's spectral convolution operates on a
truncated set of Fourier modes, and its lifting/projection layers (fc0,
fc1, fc2) are pointwise linear maps -- nothing in the architecture
references the grid size Nx, Ny. So a model trained at one resolution can
be evaluated directly, with NO retraining and NO architecture change, on
inputs discretized at a completely different resolution.


The subtle part is the test data: to test invariance meaningfully, each
test resolution must represent the SAME continuous forcing function, not
an independent random resample -- otherwise "error changed" is confounded
with "the input changed." `consistent_grf_coeffs` / `evaluate_grf` below
draw a fixed, finite set of Fourier coefficients (indexed by integer
wavenumber, not grid bin) ONCE per test field, then place those same
coefficients into the correct FFT bin for whatever target grid size is
requested. By the standard band-limited-interpolation property of the
DFT, this reproduces the exact same continuous function at every
resolution (as long as the grid resolves all the kept wavenumbers without
aliasing) -- so any change in FNO error across resolutions is due to the
model, not the data.
"""

import os
import importlib.util
import numpy as np
import torch
import matplotlib.pyplot as plt

torch.set_num_threads(os.cpu_count() or 1)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Load the FNO script (model, training loop, data utilities) and the Stokes
# solver, exactly the way ConvLSTM_Stokes_Flow.py does -- so this reuses the
# identical FNO2d architecture, InputNormalizer, make_grid, relative_l2, and
# the same cached baseline dataset/training config as the rest of the project
_HERE = os.path.dirname(os.path.abspath(__file__))
_FNO_PATH = os.path.join(_HERE, "FNO_Stokes_Flow_Updated.py")
_spec_fno = importlib.util.spec_from_file_location("fno_stokes_flow", _FNO_PATH)
fno = importlib.util.module_from_spec(_spec_fno)
_spec_fno.loader.exec_module(fno)

_STOKES_PATH = os.path.join(_HERE, "2D_Stokes_Full_Dataset.py")
_spec_stokes = importlib.util.spec_from_file_location("stokes_data_for_Training", _STOKES_PATH)
stokes2d = importlib.util.module_from_spec(_spec_stokes)
_spec_stokes.loader.exec_module(stokes2d)


# 1. Resolution-consistent (band-limited) random forcing fields
def consistent_grf_coeffs(kmax, alpha, tau, seed):
    """
    Draw a FIXED set of complex Fourier coefficients for wavenumbers
    n1, n2 in [-kmax, kmax] (integers, i.e. physical wavenumber
    2*pi*n/L -- independent of any grid size), filtered by the same
    Matern-type spectral filter used in gaussian_random_field_2d.
    Returns (coeffs, n_range) where coeffs has shape
    (2*kmax+1, 2*kmax+1) and n_range = arange(-kmax, kmax+1).
    """
    rng = np.random.default_rng(seed)
    n_range = np.arange(-kmax, kmax + 1)
    N1, N2 = np.meshgrid(n_range, n_range, indexing="ij")
    # NOTE: this uses Lx = Ly = 2*pi (matching the project's domain), so
    # the physical wavenumber for integer mode n is exactly n (2*pi*n/2*pi).
    K2 = N1.astype(float) ** 2 + N2.astype(float) ** 2
    coef = (K2 + tau ** 2) ** (-alpha / 2.0)
    coef[kmax, kmax] = 0.0  # zero-mean (this is the n=(0,0) entry)
    noise = rng.normal(size=coef.shape) + 1j * rng.normal(size=coef.shape)
    return coef * noise, n_range


def evaluate_grf(coeffs, n_range, Nx, Ny, scale=1.0):
    """
    Evaluate the band-limited field defined by `coeffs` on an Nx x Ny grid.
    Placing the SAME coefficients at the (resolution-dependent) FFT bin
    that corresponds to each fixed integer wavenumber, then taking one
    ifft2, is exactly bandlimited DFT interpolation -- the returned field
    is the same continuous function sampled at Nx x Ny points, for any
    Nx, Ny large enough to avoid aliasing (Nx, Ny > 2*kmax).
    """
    kmax = n_range.max()
    if min(Nx, Ny) <= 2 * kmax:
        raise ValueError(
            f"grid ({Nx},{Ny}) too coarse for kmax={kmax}: need Nx,Ny > {2*kmax}"
        )
    field_hat = np.zeros((Nx, Ny), dtype=complex)
    idx1 = n_range % Nx
    idx2 = n_range % Ny
    field_hat[np.ix_(idx1, idx2)] = coeffs
    return np.real(np.fft.ifft2(field_hat)) * scale


def make_consistent_field_pair(kmax, alpha, tau, seed1, seed2, ref_res):
    """
    Build one (f1, f2) forcing "identity": draws coefficients once for each
    component, and calibrates a per-component amplitude scale by matching
    unit standard deviation at `ref_res` (mirroring how the project's own
    gaussian_random_field_2d normalizes each sample to unit std) -- so this
    test data's amplitude sits in the same range the FNO was trained on.
    The scale is fixed after this one calibration and reused unchanged for
    every other resolution, which is what preserves exact cross-resolution
    consistency.
    """
    coeffs1, n_range = consistent_grf_coeffs(kmax, alpha, tau, seed1)
    coeffs2, _ = consistent_grf_coeffs(kmax, alpha, tau, seed2)
    ref1 = evaluate_grf(coeffs1, n_range, ref_res, ref_res)
    ref2 = evaluate_grf(coeffs2, n_range, ref_res, ref_res)
    scale1 = 1.0 / (ref1.std() + 1e-12)
    scale2 = 1.0 / (ref2.std() + 1e-12)
    return dict(coeffs1=coeffs1, coeffs2=coeffs2, n_range=n_range,
                scale1=scale1, scale2=scale2)


def field_pair_at_resolution(field_id, Nx, Ny, forcing_amplitude=1.0):
    f1 = forcing_amplitude * evaluate_grf(field_id["coeffs1"], field_id["n_range"],
                                           Nx, Ny, scale=field_id["scale1"])
    f2 = forcing_amplitude * evaluate_grf(field_id["coeffs2"], field_id["n_range"],
                                           Nx, Ny, scale=field_id["scale2"])
    return f1, f2


# 2. Build the cross-resolution consistent test set (exact ground truth via
# the project's own spectral Stokes solver, at every resolution)
def build_invariance_test_set(res_list, n_fields, Lx, Ly, nu, kmax, alpha, tau,
                               forcing_amplitude=1.0, ref_res=128, seed0=999_000):
    """
    Returns {res: (f_tensor, u_tensor)} for res in res_list, where
    f_tensor, u_tensor have shape (n_fields, 2, res, res). Every resolution
    for a given field index uses the SAME underlying continuous forcing
    (see module docstring), with the exact velocity solved fresh at each
    resolution via solve_stokes2d (an exact spectral solver, so this is not
    an interpolation of a fixed-resolution truth -- it's the true solution
    at that resolution).
    """
    field_ids = []
    for i in range(n_fields):
        s1 = seed0 + 2 * i
        s2 = seed0 + 2 * i + 1
        field_ids.append(make_consistent_field_pair(kmax, alpha, tau, s1, s2, ref_res))

    data = {}
    for res in res_list:
        F = np.zeros((n_fields, 2, res, res), dtype=np.float32)
        U = np.zeros((n_fields, 2, res, res), dtype=np.float32)
        for i, fid in enumerate(field_ids):
            f1, f2 = field_pair_at_resolution(fid, res, res, forcing_amplitude)
            u, v, _ = stokes2d.solve_stokes2d(f1, f2, Lx, Ly, nu)
            F[i, 0], F[i, 1] = f1, f2
            U[i, 0], U[i, 1] = u, v
        data[res] = (torch.from_numpy(F), torch.from_numpy(U))
        print(f"  built consistent test set at {res}x{res}  ({n_fields} fields)")
    return data


# 3. Evaluate a trained (frozen) FNO at an arbitrary resolution
def evaluate_at_resolution(model, normalizer, Lx, Ly, f_tensor, u_tensor):
    """No retraining, no architecture change -- just a different grid size
    passed straight into the same model and the same fitted normalizer."""
    Nx, Ny = f_tensor.shape[-2], f_tensor.shape[-1]
    grid = fno.make_grid(Nx, Ny, Lx, Ly).to(DEVICE)
    model.eval()
    with torch.no_grad():
        f_d = f_tensor.to(DEVICE)
        fb_n = normalizer.encode(f_d).permute(0, 2, 3, 1)
        gb = grid.unsqueeze(0).expand(f_d.shape[0], -1, -1, -1)
        pred = model(fb_n, gb)
        errs = fno.relative_l2(pred, u_tensor.to(DEVICE)).cpu().numpy()
    return float(errs.mean()), float(errs.std()), errs, pred.cpu().numpy()


# 4. Figures
def plot_invariance_curve(res_list, means, stds, train_res, savepath):
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.errorbar(res_list, means, yerr=stds, fmt="o-", capsize=3, color="#2b6cb0")
    ax.axvline(train_res, color="crimson", linestyle="--",
               label=f"training resolution ({train_res})")
    ax.set_xlabel("test grid resolution ($N_x = N_y$)")
    ax.set_ylabel("mean relative $L_2$ error")
    ax.set_title("FNO resolution invariance: same frozen model, varying test grid")
    ax.legend()
    fig.tight_layout()
    fig.savefig(savepath, dpi=150)
    plt.close(fig)
    print(f"Saved {savepath}")


def plot_multi_resolution_fields(model, normalizer, Lx, Ly, field_id, res_show,
                                  forcing_amplitude, savepath, component=0):
    """
    Same continuous forcing field, evaluated (and predicted) at several
    different grid resolutions side by side -- the qualitative companion
    to the error-vs-resolution plot above.
    """
    comp_name = ["u", "v"][component]
    fig, axes = plt.subplots(2, len(res_show), figsize=(3.2 * len(res_show), 6.2))
    for j, res in enumerate(res_show):
        f1, f2 = field_pair_at_resolution(field_id, res, res, forcing_amplitude)
        u, v, _ = stokes2d.solve_stokes2d(f1, f2, Lx, Ly, nu=0.1)
        truth = [u, v][component]

        f_t = torch.from_numpy(np.stack([f1, f2], axis=0)).float().unsqueeze(0)
        grid = fno.make_grid(res, res, Lx, Ly).to(DEVICE)
        model.eval()
        with torch.no_grad():
            fb_n = normalizer.encode(f_t.to(DEVICE)).permute(0, 2, 3, 1)
            gb = grid.unsqueeze(0)
            pred = model(fb_n, gb).cpu().numpy()[0, component]

        vmax = max(np.abs(truth).max(), np.abs(pred).max())
        axes[0, j].imshow(truth.T, origin="lower", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        axes[0, j].set_title(f"{res}x{res}: {comp_name} truth")
        axes[1, j].imshow(pred.T, origin="lower", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        axes[1, j].set_title(f"{res}x{res}: {comp_name} prediction")
        for r in (0, 1):
            axes[r, j].set_xticks([]); axes[r, j].set_yticks([])
    fig.suptitle("Same forcing field, evaluated at different grid resolutions "
                 "-- no retraining", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(savepath, dpi=150)
    plt.close(fig)
    print(f"Saved {savepath}")


# 5. Config + main
CONFIG = dict(
    res_list=[48, 64, 96, 128, 160, 192, 256],  # includes the training resolution
    n_fields=30,
    kmax=16,          # must be < min(res_list)/2 = 24 to avoid aliasing
    alpha=4.0, tau=5.0,   # match the project's training-data smoothness
    forcing_amplitude=1.0,
    seed0=999_000,        # disjoint from all training/eval seeds used elsewhere
    res_show=[48, 128, 256],  # subset shown in the qualitative figure
)


def main():
    out_dir = "fno_invariance_results"
    data_dir = "fno_data"
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)
    print(f"Device: {DEVICE}")

    # --- train the FNO exactly once, using the project's own baseline
    #     config and cached dataset, so this is the SAME model referenced
    #     everywhere else in the project (not a separate one-off) ---
    cfg = fno.CONFIG
    base_path = fno.generate_dataset(cfg["base_n_samples"], cfg["base_res"], cfg["base_res"],
                                      alpha=4.0, save_dir=data_dir, tag="base")
    f, u, meta = fno.load_dataset(base_path)
    (f_tr, u_tr), (f_va, u_va), (f_te, u_te) = fno.split_dataset(
        f, u, cfg["n_train"], cfg["n_val"], cfg["n_test"], seed=0)
    Lx, Ly = meta["Lx"], meta["Ly"]
    train_res = cfg["base_res"]

    print("\n=== Training baseline FNO (once) ===")
    model, normalizer, grid, hist = fno.train_fno(
        f_tr, u_tr, f_va, u_va, Lx, Ly,
        cfg["base_modes"], cfg["base_width"], cfg["base_depth"],
        cfg["base_epochs"], verbose=True)
    mean_err, errs, _ = fno.evaluate_fno(model, normalizer, grid, f_te, u_te)
    print(f"Baseline test mean relative L2 (at training resolution {train_res}): {mean_err:.4f}")

    # --- build the resolution-consistent invariance test set ---
    icfg = CONFIG
    if train_res not in icfg["res_list"]:
        icfg["res_list"] = sorted(set(icfg["res_list"] + [train_res]))
    print("\n=== Building cross-resolution consistent test fields ===")
    test_data = build_invariance_test_set(
        icfg["res_list"], icfg["n_fields"], Lx, Ly, nu=meta["nu"],
        kmax=icfg["kmax"], alpha=icfg["alpha"], tau=icfg["tau"],
        forcing_amplitude=icfg["forcing_amplitude"], ref_res=train_res,
        seed0=icfg["seed0"])

    # --- evaluate the SAME frozen model at every resolution, no retraining ---
    print("\n=== Evaluating the frozen FNO at each resolution ===")
    means, stds = [], []
    for res in icfg["res_list"]:
        f_res, u_res = test_data[res]
        mean_e, std_e, _, _ = evaluate_at_resolution(model, normalizer, Lx, Ly, f_res, u_res)
        means.append(mean_e); stds.append(std_e)
        marker = "  <-- training resolution" if res == train_res else ""
        print(f"  res={res:4d}x{res:<4d}  mean rel L2 = {mean_e:.4f}  (std {std_e:.4f}){marker}")

    plot_invariance_curve(icfg["res_list"], means, stds, train_res,
                          os.path.join(out_dir, "fno_resolution_invariance_error.png"))

    # --- qualitative figure: one field, several resolutions, side by side ---
    field_id = make_consistent_field_pair(icfg["kmax"], icfg["alpha"], icfg["tau"],
                                          icfg["seed0"], icfg["seed0"] + 1, train_res)
    plot_multi_resolution_fields(model, normalizer, Lx, Ly, field_id, icfg["res_show"],
                                 icfg["forcing_amplitude"],
                                 os.path.join(out_dir, "fno_resolution_invariance_fields.png"))

    print(f"\nAll figures saved under {out_dir}/")
    return model, normalizer, icfg["res_list"], means, stds


if __name__ == "__main__":
    main()