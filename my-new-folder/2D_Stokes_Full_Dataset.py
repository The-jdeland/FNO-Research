"""
2D incompressible Stokes flow on a doubly-periodic domain ("torus")

Solves the steady, linear (inertia-free) incompressible Stokes equations

    -nu * Laplacian(u) + grad(p) = f (momentum, Stokes/creeping flow)
    div(u) = 0 (incompressibility)

on a doubly-periodic domain [0, Lx) x [0, Ly) -- the "torus" -- for a random
smooth forcing field f(x, y) = (f1, f2)

Structure:
  1. Random smooth forcing generator (Gaussian random field) -> gaussian_random_field_2d()
  2. Exact spectral Stokes solver -> solve_stokes2d()
  3. Diagnostics (divergence-free check, vorticity) -> compute_vorticity(), check_divergence()
  4. Tracer advection through the steady field (periodic RK4) -> advect_tracers()
  5. Single-sample generation + tracer animation -> generate_and_save_sample(), animate_sample()
  6. Batch dataset generation for FNO training -> generate_training_dataset()
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.ndimage import map_coordinates
import os
import time


# 1. Random smooth forcing field generator
def gaussian_random_field_2d(Nx, Ny, Lx, Ly, alpha=4.0, tau=5.0, seed=None):
    """
    Generate a smooth, mean-zero random field on a doubly-periodic grid by
    filtering white noise in Fourier space with a Matern-type spectral
    filter.

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
    Parameters:
    f1, f2 : (Nx, Ny) arrays forcing field components on a uniform grid
    Lx, Ly : float domain size
    nu : float viscosity

    Returns:
    u, v : (Nx, Ny) arrays, velocity field components
    p : (Nx, Ny) array, pressure field (defined up to an additive constant,
    as usual for incompressible flow we fix that constant by setting mean(p) = 0)
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
    """Return the RMS of div(u) = du/dx + dv/dy"""
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


# 4. Tracer advection through the (steady) velocity field
def _sample_periodic(field, xg, yg, Lx, Ly):
    """
    Bilinearly interpolate `field` (shape (Nx, Ny), defined on the periodic
    grid [0,Lx)x[0,Ly)) at arbitrary points xg, yg (arrays of any matching
    shape), wrapping across the domain edges. Uses scipy's grid-wrap mode
    so tracers that cross a boundary sample the field on the other side
    seamlessly, exactly as the doubly-periodic torus requires.
    """
    Nx, Ny = field.shape
    # map physical coords -> fractional grid-index coords
    ix = (xg / Lx) * Nx
    iy = (yg / Ly) * Ny
    coords = np.stack([ix, iy])
    return map_coordinates(field, coords, order=1, mode="grid-wrap")


def advect_tracers(u, v, Lx, Ly, n_frames, dt, n_particles_per_side=24, seed=0):
    """
    Advect a grid of massless tracer particles through the steady velocity
    field (u, v) using classic RK4, with periodic wraparound after every
    step. Returns an array of positions with shape (n_frames+1, N, 2),
    where N = n_particles_per_side**2.

    dt should be chosen so the total advected time n_frames*dt covers a
    couple of "domain crossings" (~ Lx / typical|u|) so the animation
    visibly shows particles wrapping around at least once or twice.
    """
    rng = np.random.default_rng(seed)
    gx = np.linspace(0, Lx, n_particles_per_side, endpoint=False)
    gy = np.linspace(0, Ly, n_particles_per_side, endpoint=False)
    X0, Y0 = np.meshgrid(gx, gy, indexing="ij")
    # small random jitter so particles don't sit exactly on a lattice
    jitter_x = (rng.random(X0.shape) - 0.5) * (Lx / n_particles_per_side)
    jitter_y = (rng.random(Y0.shape) - 0.5) * (Ly / n_particles_per_side)
    x = (X0 + jitter_x).ravel() % Lx
    y = (Y0 + jitter_y).ravel() % Ly

    positions = np.zeros((n_frames + 1, x.size, 2))
    positions[0, :, 0] = x
    positions[0, :, 1] = y

    def vel(xp, yp):
        up = _sample_periodic(u, xp, yp, Lx, Ly)
        vp = _sample_periodic(v, xp, yp, Lx, Ly)
        return up, vp

    for n in range(n_frames):
        k1x, k1y = vel(x, y)
        k2x, k2y = vel(x + 0.5 * dt * k1x, y + 0.5 * dt * k1y)
        k3x, k3y = vel(x + 0.5 * dt * k2x, y + 0.5 * dt * k2y)
        k4x, k4y = vel(x + dt * k3x, y + dt * k3y)

        x = x + (dt / 6.0) * (k1x + 2 * k2x + 2 * k3x + k4x)
        y = y + (dt / 6.0) * (k1y + 2 * k2y + 2 * k3y + k4y)
        x %= Lx
        y %= Ly

        positions[n + 1, :, 0] = x
        positions[n + 1, :, 1] = y

    return positions


# 5. Generate, save, and animate one sample
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
    save_dir="stokes_data_Still_Image",
    save_name="stokes2d_sample_Presentation.npz",
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


def plot_sample(sample, savepath="stokes2d_overview_Still_Image_Presentation.png"):
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


def animate_sample(
    sample,
    savepath="stokes_data/stokes2d_animation_Presentation.gif",
    n_frames=90,
    n_domain_crossings=2.0,
    n_particles_per_side=24,
    fps=20,
    seed=0,
):
    """
    Build a 4-panel animation: forcing / velocity / vorticity / pressure
    fields all shown as their (static) heatmaps, with tracer particles
    advected through the steady velocity field overlaid on all four panels.
    The particles wrap across the domain edges (periodic BCs), and
    `n_domain_crossings` sets the total advected distance in units of a
    full domain width, so you can directly control how many times the
    tracers loop around before the animation ends.
    """
    f1, f2 = sample["f1"], sample["f2"]
    u, v = sample["u"], sample["v"]
    omega, p = sample["omega"], sample["p"]
    x, y = sample["x"], sample["y"]
    Lx, Ly = sample["Lx"], sample["Ly"]

    X, Y = np.meshgrid(x, y, indexing="ij")
    speed = np.sqrt(u**2 + v**2)
    force_mag = np.sqrt(f1**2 + f2**2)

    # pick dt so that n_frames steps advect a typical particle roughly
    # n_domain_crossings full widths of the domain: total_time * mean|u| ~ n_domain_crossings * Lx
    mean_speed = np.mean(speed) + 1e-8
    total_time = n_domain_crossings * Lx / mean_speed
    dt = total_time / n_frames

    print(f"Advecting tracers: dt={dt:.4f}, total_time={total_time:.3f}, "
          f"n_frames={n_frames} (target ~{n_domain_crossings} domain crossings)")

    positions = advect_tracers(u, v, Lx, Ly, n_frames, dt,
                                n_particles_per_side=n_particles_per_side, seed=seed)

    fig, axes = plt.subplots(2, 2, figsize=(11, 10))
    panels = [
        (axes[0, 0], force_mag, "viridis", "Forcing field  |f(x, y)|  [input]", "|f|"),
        (axes[0, 1], speed, "viridis", "Velocity field  |u(x, y)|  [target]  -- tracers show flow", "|u|"),
        (axes[1, 0], omega, "RdBu_r", "Vorticity  omega = dv/dx - du/dy", "omega"),
        (axes[1, 1], p, "RdBu_r", "Pressure  p(x, y)", "p"),
    ]

    scatters = []
    for ax, field, cmap, title, label in panels:
        if cmap == "RdBu_r":
            vmax = np.abs(field).max()
            im = ax.pcolormesh(X, Y, field, shading="auto", cmap=cmap, vmin=-vmax, vmax=vmax)
        else:
            im = ax.pcolormesh(X, Y, field, shading="auto", cmap=cmap)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("x"); ax.set_ylabel("y")
        ax.set_xlim(0, Lx); ax.set_ylim(0, Ly)
        fig.colorbar(im, ax=ax, label=label)
        sc = ax.scatter(positions[0, :, 0], positions[0, :, 1], s=8, c="white",
                         edgecolors="black", linewidths=0.3)
        scatters.append(sc)

    title_text = fig.suptitle(
        f"2D Stokes flow on the periodic torus (nu={sample['nu']})  --  frame 0/{n_frames}",
        fontsize=13,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    def update(frame):
        for sc in scatters:
            sc.set_offsets(positions[frame])
        title_text.set_text(
            f"2D Stokes flow on the periodic torus (nu={sample['nu']})  --  frame {frame}/{n_frames}"
        )
        return scatters + [title_text]

    anim = animation.FuncAnimation(
        fig, update, frames=n_frames + 1, interval=1000 / fps, blit=False
    )

    os.makedirs(os.path.dirname(savepath) or ".", exist_ok=True)
    anim.save(savepath, writer="pillow", fps=fps)
    plt.close(fig)
    print(f"Saved tracer-advection animation to {savepath} "
          f"({n_frames+1} frames, ~{n_domain_crossings} domain crossings)")


# 6. Batch dataset generation for FNO training
def generate_training_dataset(
    n_samples=1000,
    Nx=128,
    Ny=128,
    Lx=2 * np.pi,
    Ly=2 * np.pi,
    nu=0.1,
    forcing_amplitude=1.0,
    alpha=4.0,
    tau=5.0,
    seed0=0,
    save_dir="stokes_data_For_Training",
    save_name="stokes2d_train_FullData.npz",
    check_every=100,
):
    """
    Generate `n_samples` independent (forcing -> velocity/pressure/vorticity)
    Stokes-flow samples and stack them into a single .npz file with a
    leading batch dimension -- i.e. exactly the shape an FNO training loop
    expects: f1, f2 as inputs of shape (N, Nx, Ny), and u, v, p, omega as
    the corresponding targets of shape (N, Nx, Ny).

    Each sample uses a distinct pair of seeds (seed0 + 2*i, seed0 + 2*i + 1)
    for its two forcing components, so samples are independent draws from
    the same random-field distribution. Grid/domain metadata (x, y, Lx, Ly,
    nu) is shared across all samples and stored once.
    """
    os.makedirs(save_dir, exist_ok=True)

    F1 = np.zeros((n_samples, Nx, Ny))
    F2 = np.zeros((n_samples, Nx, Ny))
    U = np.zeros((n_samples, Nx, Ny))
    V = np.zeros((n_samples, Nx, Ny))
    P = np.zeros((n_samples, Nx, Ny))
    OMEGA = np.zeros((n_samples, Nx, Ny))
    seeds_used = np.zeros(n_samples, dtype=np.int64)

    max_rel_div = 0.0
    t0 = time.time()

    for i in range(n_samples):
        s1 = seed0 + 2 * i
        s2 = seed0 + 2 * i + 1

        f1 = forcing_amplitude * gaussian_random_field_2d(Nx, Ny, Lx, Ly, alpha, tau, seed=s1)
        f2 = forcing_amplitude * gaussian_random_field_2d(Nx, Ny, Lx, Ly, alpha, tau, seed=s2)
        u, v, p = solve_stokes2d(f1, f2, Lx, Ly, nu)
        omega = compute_vorticity(u, v, Lx, Ly)

        div_rms = check_divergence(u, v, Lx, Ly)
        rel_div = div_rms / (np.sqrt(np.mean(u**2 + v**2)) + 1e-12)
        max_rel_div = max(max_rel_div, rel_div)

        F1[i], F2[i] = f1, f2
        U[i], V[i], P[i], OMEGA[i] = u, v, p, omega
        seeds_used[i] = s1

        if (i + 1) % check_every == 0 or (i + 1) == n_samples:
            elapsed = time.time() - t0
            print(f"  [{i+1}/{n_samples}] generated  "
                  f"(elapsed {elapsed:.1f}s, worst relative div so far: {max_rel_div:.2e})")

    x = np.linspace(0, Lx, Nx, endpoint=False)
    y = np.linspace(0, Ly, Ny, endpoint=False)

    save_path = os.path.join(save_dir, save_name)
    np.savez(
        save_path,
        f1=F1, f2=F2,
        u=U, v=V, p=P, omega=OMEGA,
        x=x, y=y, Lx=Lx, Ly=Ly, nu=nu, seeds=seeds_used,
    )

    size_mb = os.path.getsize(save_path) / (1024 ** 2)
    print(f"Saved training dataset with {n_samples} samples to {save_path} "
          f"({size_mb:.1f} MB, worst relative divergence: {max_rel_div:.2e})")

    return save_path


def main():
    # single sample + tracer-advection animation
    sample = generate_and_save_sample(
        seed=0,
        Nx=128, Ny=128,
        Lx=2 * np.pi, Ly=2 * np.pi,
        nu=0.1,
        forcing_amplitude=1.0,
        alpha=4.0, tau=5.0,
        save_dir="stokes_data_Still_Image",
        save_name="stokes2d_sample_Presentation.npz",
    )
    plot_sample(sample, savepath="stokes_data/stokes2d_overview_Still_Image_Presentation.png")
    animate_sample(
        sample,
        savepath="stokes_data/stokes2d_animation_Presentation.gif",
        n_frames=90,
        n_domain_crossings=2.0,
        n_particles_per_side=24,
        fps=20,
        seed=0,
    )

    # batch dataset for FNO training
    generate_training_dataset(
        n_samples=1000,
        Nx=128, Ny=128,
        Lx=2 * np.pi, Ly=2 * np.pi,
        nu=0.1,
        forcing_amplitude=1.0,
        alpha=4.0, tau=5.0,
        seed0=1000,  # disjoint from the single-sample seed above
        save_dir="stokes_data_For_Training",
        save_name="stokes2d_train_Full_Data.npz",
    )


if __name__ == "__main__":
    main()