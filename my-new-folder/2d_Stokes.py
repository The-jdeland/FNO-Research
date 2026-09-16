"""
2D incompressible Stokes flow on a doubly-periodic domain ("torus")
=====================================================================

Solves the steady, linear (inertia-free) incompressible Stokes equations

    -nu * Laplacian(u) + grad(p) = f (momentum, Stokes/creeping flow)
    div(u) = 0 (incompressibility)

on a doubly-periodic domain [0, Lx) x [0, Ly) -- the "torus" -- for a random
smooth forcing field f(x, y) = (f1, f2).

-------------------------------------------------------------------------

Structure:
  1. Random smooth forcing generator (Gaussian random field) -> gaussian_random_field_2d()
  2. Exact spectral Stokes solver -> solve_stokes2d()
  3. Diagnostics (divergence-free check, vorticity) -> compute_vorticity(), check_divergence()
  4. Save sample to disk + visualize -> generate_and_save_sample(), plot_sample()

"""

import numpy as np
import matplotlib.pyplot as plt
import os


# 1. Random smooth forcing field generator
def gaussian_random_field_2d(Nx, Ny, Lx, Ly, alpha=4.0, tau=5.0, seed=None):
    """
    Generate a smooth, mean-zero random field on a doubly-periodic grid by
    filtering white noise in Fourier space with a Matern-type spectral
    filter (this is the same style of random field FNO benchmark datasets
    use to generate forcing/initial conditions -- see Li et al., 2020).

    coef(k) = (|k|^2 + tau^2)^(-alpha/2)

    Larger `alpha` => faster spectral decay => smoother (more large-scale,
    fewer small-scale wiggles) field. `tau` sets an overall correlation
    length scale.
    """
    rng = np.random.default_rng(seed)

    kx = 2.0 * np.pi * np.fft.fftfreq(Nx, d=Lx / Nx)
    ky = 2.0 * np.pi * np.fft.fftfreq(Ny, d=Ly / Ny)
    KX, KY = np.meshgrid(kx, ky, indexing="ij")
    K2 = KX**2 + KY**2

    coef = (K2 + tau**2) ** (-alpha / 2.0)
    coef[0, 0] = 0.0  # zero out the mean (DC) mode -> mean-zero field

    noise = rng.normal(size=(Nx, Ny)) + 1j * rng.normal(size=(Nx, Ny))
    field_hat = coef * noise
    field = np.real(np.fft.ifft2(field_hat))

    # normalize to unit std so the caller controls amplitude explicitly
    field -= field.mean()
    field /= field.std() + 1e-12
    return field


# 2. Exact spectral Stokes solver
def solve_stokes2d(f1, f2, Lx, Ly, nu):
    """
    Parameters
    ----------
    f1, f2 : (Nx, Ny) arrays   forcing field components on a uniform grid
    Lx, Ly : float              domain size
    nu : float              viscosity

    Returns
    -------
    u, v : (Nx, Ny) arrays, velocity field components
    p    : (Nx, Ny) array, pressure field (defined up to an additive
                              constant, as usual for incompressible flow --
                              we fix that constant by setting mean(p) = 0)
    """
    Nx, Ny = f1.shape
    kx = 2.0 * np.pi * np.fft.fftfreq(Nx, d=Lx / Nx)
    ky = 2.0 * np.pi * np.fft.fftfreq(Ny, d=Ly / Ny)
    KX, KY = np.meshgrid(kx, ky, indexing="ij")
    K2 = KX**2 + KY**2
    K2_safe = K2.copy()
    K2_safe[0, 0] = 1.0  # avoid divide-by-zero; k=0 mode is set explicitly below

    f1_hat = np.fft.fft2(f1)
    f2_hat = np.fft.fft2(f2)

    # k . f_hat  (dot product of wavevector with forcing, in Fourier space)
    k_dot_f = KX * f1_hat + KY * f2_hat

    # pressure: p_hat = -i (k.f_hat) / |k|^2
    p_hat = -1j * k_dot_f / K2_safe
    p_hat[0, 0] = 0.0  # fix the pressure constant: mean(p) = 0

    # velocity: u_hat = (f_hat - k (k.f_hat)/|k|^2) / (nu |k|^2)
    # i.e. Leray-project f_hat, then apply the Stokes resolvent
    u1_hat = (f1_hat - KX * k_dot_f / K2_safe) / (nu * K2_safe)
    u2_hat = (f2_hat - KY * k_dot_f / K2_safe) / (nu * K2_safe)
    u1_hat[0, 0] = 0.0  # zero mean flow (k=0 mode undetermined by local forcing balance)
    u2_hat[0, 0] = 0.0

    u = np.real(np.fft.ifft2(u1_hat))
    v = np.real(np.fft.ifft2(u2_hat))
    p = np.real(np.fft.ifft2(p_hat))

    return u, v, p


# 3. Diagnostics
def compute_vorticity(u, v, Lx, Ly):
    """omega = dv/dx - du/dy, computed spectrally (exact to machine precision)."""
    Nx, Ny = u.shape
    kx = 2.0 * np.pi * np.fft.fftfreq(Nx, d=Lx / Nx)
    ky = 2.0 * np.pi * np.fft.fftfreq(Ny, d=Ly / Ny)
    KX, KY = np.meshgrid(kx, ky, indexing="ij")

    v_hat = np.fft.fft2(v)
    u_hat = np.fft.fft2(u)
    dvdx = np.real(np.fft.ifft2(1j * KX * v_hat))
    dudy = np.real(np.fft.ifft2(1j * KY * u_hat))
    return dvdx - dudy


def check_divergence(u, v, Lx, Ly):
    """Return the RMS of div(u) = du/dx + dv/dy -- should be ~machine precision."""
    Nx, Ny = u.shape
    kx = 2.0 * np.pi * np.fft.fftfreq(Nx, d=Lx / Nx)
    ky = 2.0 * np.pi * np.fft.fftfreq(Ny, d=Ly / Ny)
    KX, KY = np.meshgrid(kx, ky, indexing="ij")

    u_hat = np.fft.fft2(u)
    v_hat = np.fft.fft2(v)
    dudx = np.real(np.fft.ifft2(1j * KX * u_hat))
    dvdy = np.real(np.fft.ifft2(1j * KY * v_hat))
    div = dudx + dvdy
    return np.sqrt(np.mean(div**2))


# 4. Generate, save, and visualize one sample
def generate_and_save_sample(
    seed=0,
    Nx=128,
    Ny=128,
    Lx=2 * np.pi,
    Ly=2 * np.pi,
    nu=0.1,
    forcing_amplitude=1.0,
    alpha=4.0,
    tau=5.0,
    save_dir="stokes_data",
    save_name="stokes2d_sample.npz",
):
    """
    Generate one (forcing -> velocity) Stokes flow sample, save all fields
    to a .npz file, and return everything needed for plotting/inspection.
    """
    os.makedirs(save_dir, exist_ok=True)

    f1 = forcing_amplitude * gaussian_random_field_2d(Nx, Ny, Lx, Ly, alpha, tau, seed=seed)
    f2 = forcing_amplitude * gaussian_random_field_2d(Nx, Ny, Lx, Ly, alpha, tau, seed=seed + 1)

    u, v, p = solve_stokes2d(f1, f2, Lx, Ly, nu)
    omega = compute_vorticity(u, v, Lx, Ly)

    div_rms = check_divergence(u, v, Lx, Ly)
    rel_div = div_rms / (np.sqrt(np.mean(u**2 + v**2)) + 1e-12)
    print(f"[seed {seed}] incompressibility check: RMS(div u) = {div_rms:.3e}  "
          f"(relative to |u| scale: {rel_div:.3e} -- solver is algebraically exact, "
          f"this should stay well under 1e-4)")

    x = np.linspace(0, Lx, Nx, endpoint=False)
    y = np.linspace(0, Ly, Ny, endpoint=False)

    save_path = os.path.join(save_dir, save_name)
    np.savez(
        save_path,
        f1=f1, f2=f2,          # input: forcing field
        u=u, v=v,                # target: velocity field
        p=p, omega=omega,       # extra fields, useful for diagnostics later
        x=x, y=y,
        Lx=Lx, Ly=Ly, nu=nu, seed=seed,
    )
    print(f"Saved sample to {save_path}")

    return dict(f1=f1, f2=f2, u=u, v=v, p=p, omega=omega, x=x, y=y, Lx=Lx, Ly=Ly, nu=nu)


def plot_sample(sample, savepath="stokes2d_overview.png"):
    """
    4-panel overview figure: forcing field, velocity field (as streamlines
    over speed), vorticity, and pressure. This is the "what does the system
    look like" sanity check
    """
    f1, f2 = sample["f1"], sample["f2"]
    u, v = sample["u"], sample["v"]
    omega, p = sample["omega"], sample["p"]
    x, y = sample["x"], sample["y"]

    X, Y = np.meshgrid(x, y, indexing="ij")
    speed = np.sqrt(u**2 + v**2)
    force_mag = np.sqrt(f1**2 + f2**2)

    fig, axes = plt.subplots(2, 2, figsize=(11, 10))

    ax = axes[0, 0]
    im = ax.pcolormesh(X, Y, force_mag, shading="auto", cmap="viridis")
    ax.quiver(X[::8, ::8], Y[::8, ::8], f1[::8, ::8], f2[::8, ::8], color="white", scale=30)
    ax.set_title("Forcing field  f(x, y)  [input]")
    ax.set_xlabel("x"); ax.set_ylabel("y")
    fig.colorbar(im, ax=ax, label="|f|")

    ax = axes[0, 1]
    im = ax.pcolormesh(X, Y, speed, shading="auto", cmap="viridis")
    ax.streamplot(y, x, v.T, u.T, color="white", density=1.0, linewidth=0.7)
    ax.set_title("Velocity field  u(x, y)  [target]")
    ax.set_xlabel("x"); ax.set_ylabel("y")
    fig.colorbar(im, ax=ax, label="|u|")

    ax = axes[1, 0]
    vmax = np.abs(omega).max()
    im = ax.pcolormesh(X, Y, omega, shading="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    ax.set_title("Vorticity  omega = dv/dx - du/dy")
    ax.set_xlabel("x"); ax.set_ylabel("y")
    fig.colorbar(im, ax=ax, label="omega")

    ax = axes[1, 1]
    vmax = np.abs(p).max()
    im = ax.pcolormesh(X, Y, p, shading="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    ax.set_title("Pressure  p(x, y)")
    ax.set_xlabel("x"); ax.set_ylabel("y")
    fig.colorbar(im, ax=ax, label="p")

    fig.suptitle(f"2D Stokes flow on the periodic torus  (nu = {sample['nu']})", fontsize=13)
    fig.tight_layout()
    fig.savefig(savepath, dpi=150)
    print(f"Saved overview figure to {savepath}")
    plt.close(fig)


def main():
    sample = generate_and_save_sample(
        seed=0,
        Nx=128, Ny=128,
        Lx=2 * np.pi, Ly=2 * np.pi,
        nu=0.1,
        forcing_amplitude=1.0,
        alpha=4.0, tau=5.0,
        save_dir="stokes_data",
        save_name="stokes2d_sample.npz",
    )
    plot_sample(sample, savepath="stokes_data/stokes2d_overview.png")


if __name__ == "__main__":
    main()