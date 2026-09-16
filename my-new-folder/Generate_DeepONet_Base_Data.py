import os
import importlib.util
import numpy as np


# ============================================================
# Locate your partner's dataset-generator file
# ============================================================

HERE = os.path.dirname(os.path.abspath(__file__))

STOKES_FILE = os.path.join(
    HERE,
    "2D_Stokes_Full_Dataset.py"
)

print("Looking for:")
print(STOKES_FILE)


if not os.path.exists(STOKES_FILE):
    raise FileNotFoundError(
        "\nCould not find 2D_Stokes_Full_Dataset.py.\n"
        "Make sure it is on your Desktop alongside this file."
    )


# ============================================================
# Import your partner's code
# ============================================================

spec = importlib.util.spec_from_file_location(
    "stokes_data",
    STOKES_FILE
)

stokes_data = importlib.util.module_from_spec(spec)

spec.loader.exec_module(stokes_data)


print("\nSuccessfully loaded:")
print(STOKES_FILE)


# ============================================================
# Where we want the dataset saved
# ============================================================

SAVE_DIR = os.path.join(
    HERE,
    "fno_data"
)

SAVE_NAME = "stokes2d_base.npz"


# ============================================================
# Generate EXACT baseline dataset used by the FNO setup
# ============================================================

print("\nGenerating baseline Stokes dataset...")
print("This will generate 2000 samples on a 128 x 128 grid.\n")


stokes_data.generate_training_dataset(

    n_samples=2000,

    Nx=128,
    Ny=128,

    Lx=2 * np.pi,
    Ly=2 * np.pi,

    nu=0.1,

    forcing_amplitude=1.0,

    alpha=4.0,
    tau=5.0,

    seed0=0,

    save_dir=SAVE_DIR,
    save_name=SAVE_NAME,

    check_every=100
)


# ============================================================
# Confirm output
# ============================================================

SAVE_PATH = os.path.join(
    SAVE_DIR,
    SAVE_NAME
)


print("\n========================================")
print("DATASET GENERATION COMPLETE")
print("========================================")

print("\nDataset saved to:")
print(SAVE_PATH)

print(
    "\nFile size:",
    os.path.getsize(SAVE_PATH) / (1024 ** 2),
    "MB"
)