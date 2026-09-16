# ============================================================
# DEEPONET FOR 2-D PERIODIC STOKES FLOW
#
# BATCH 1:
# Dataset loading, splitting, normalization, and verification
#
# IMPORTANT:
# We do NOT generate a new dataset here.
#
# We load the exact same saved Stokes dataset used by the FNO.
# ============================================================


# ============================================================
# 0. IMPORTS
# ============================================================

import os
import numpy as np
import torch


# ============================================================
# 1. CONFIGURATION
#
# These values match the FNO FULL_CONFIG baseline experiment.
# ============================================================

SEED = 0

N_TRAIN = 1500
N_VAL = 250
N_TEST = 250


# ============================================================
# 2. DATASET PATH
#
# Change ONLY this path to wherever the existing FNO base
# dataset is located on your machine / SOL.
#
# The FNO code generates the base dataset with the tag "base",
# giving a filename like:
#
#     stokes2d_base.npz
#
# inside the fno_data directory.
# ============================================================

DATASET_PATH = os.path.join(
    "fno_data",
    "stokes2d_base.npz"
)


# ============================================================
# 3. LOAD DATASET
#
# This intentionally follows the same convention as
# load_dataset() in FNO_Stokes_Flow.py.
#
# Stored:
#
#       f1, f2  = forcing components
#       u, v    = velocity components
#
# Returned:
#
#       forcing : (N, 2, Nx, Ny)
#       velocity: (N, 2, Nx, Ny)
# ============================================================

def load_dataset(npz_path):

    if not os.path.exists(npz_path):

        raise FileNotFoundError(
            "\nCould not find the Stokes dataset:\n"
            f"{os.path.abspath(npz_path)}\n\n"
            "Set DATASET_PATH to the existing dataset generated "
            "by the FNO workflow."
        )


    data = np.load(npz_path)


    # --------------------------------------------------------
    # Make sure the fields required for DeepONet/FNO training
    # actually exist.
    # --------------------------------------------------------

    required_fields = [
        "f1",
        "f2",
        "u",
        "v",
        "Lx",
        "Ly",
        "nu"
    ]


    missing_fields = [

        field

        for field in required_fields

        if field not in data.files

    ]


    if missing_fields:

        raise KeyError(
            "Dataset is missing required fields: "
            f"{missing_fields}"
        )


    # --------------------------------------------------------
    # Input:
    #
    #       f = (f1, f2)
    #
    # Shape:
    #
    #       (samples, 2, Nx, Ny)
    # --------------------------------------------------------

    forcing = np.stack(
        [
            data["f1"],
            data["f2"]
        ],
        axis=1
    ).astype(
        np.float32
    )


    # --------------------------------------------------------
    # Target:
    #
    #       velocity = (u, v)
    #
    # Shape:
    #
    #       (samples, 2, Nx, Ny)
    # --------------------------------------------------------

    velocity = np.stack(
        [
            data["u"],
            data["v"]
        ],
        axis=1
    ).astype(
        np.float32
    )


    # --------------------------------------------------------
    # Metadata
    # --------------------------------------------------------

    metadata = {

        "Lx": float(data["Lx"]),

        "Ly": float(data["Ly"]),

        "nu": float(data["nu"])

    }


    # --------------------------------------------------------
    # Convert arrays to PyTorch tensors
    # --------------------------------------------------------

    forcing = torch.from_numpy(
        forcing
    )

    velocity = torch.from_numpy(
        velocity
    )


    return (
        forcing,
        velocity,
        metadata
    )


# ============================================================
# 4. TRAIN / VALIDATION / TEST SPLIT
#
# This deliberately reproduces split_dataset() from the FNO.
#
# Same:
#
#       torch.Generator()
#       manual_seed(seed)
#       torch.randperm(...)
#
# Therefore, assuming the same original dataset ordering and
# seed, DeepONet receives exactly the same observations in
# train / validation / test.
# ============================================================

def split_dataset(
    forcing,
    velocity,
    n_train,
    n_val,
    n_test,
    seed=0
):

    n_total = forcing.shape[0]


    if (
        n_train
        +
        n_val
        +
        n_test
        >
        n_total
    ):

        raise ValueError(
            "Requested split is larger than dataset.\n"
            f"Dataset samples: {n_total}\n"
            f"Requested: "
            f"{n_train} train + "
            f"{n_val} val + "
            f"{n_test} test"
        )


    # --------------------------------------------------------
    # Same seeded random generator as FNO code
    # --------------------------------------------------------

    generator = (
        torch.Generator()
        .manual_seed(seed)
    )


    permutation = torch.randperm(
        n_total,
        generator=generator
    )


    # --------------------------------------------------------
    # Split indices
    # --------------------------------------------------------

    idx_train = permutation[
        :n_train
    ]


    idx_val = permutation[
        n_train:
        n_train + n_val
    ]


    idx_test = permutation[
        n_train + n_val:
        n_train + n_val + n_test
    ]


    # --------------------------------------------------------
    # Extract data
    # --------------------------------------------------------

    f_train = forcing[
        idx_train
    ]

    u_train = velocity[
        idx_train
    ]


    f_val = forcing[
        idx_val
    ]

    u_val = velocity[
        idx_val
    ]


    f_test = forcing[
        idx_test
    ]

    u_test = velocity[
        idx_test
    ]


    return (

        (
            f_train,
            u_train
        ),

        (
            f_val,
            u_val
        ),

        (
            f_test,
            u_test
        ),

        (
            idx_train,
            idx_val,
            idx_test
        )

    )


# ============================================================
# 5. INPUT NORMALIZER
#
# Same convention as the FNO:
#
# Per forcing component:
#
#              f - mean
#       f* = ------------
#                 std
#
# Statistics are computed ONLY from the training set.
#
# No target normalization is performed because the FNO
# baseline does not normalize the velocity target.
# ============================================================

class InputNormalizer:

    def __init__(
        self,
        x
    ):

        # x has shape:
        #
        # (N, C, Nx, Ny)

        self.mean = x.mean(
            dim=(0, 2, 3),
            keepdim=True
        )


        self.std = (

            x.std(
                dim=(0, 2, 3),
                keepdim=True
            )

            +

            1e-8

        )


    def encode(
        self,
        x
    ):

        return (

            x
            -
            self.mean

        ) / self.std


# ============================================================
# 6. DATASET VERIFICATION
# ============================================================

def verify_dataset(

    forcing,
    velocity,
    metadata,

    f_train,
    u_train,

    f_val,
    u_val,

    f_test,
    u_test,

    normalizer,

    idx_train,
    idx_val,
    idx_test

):

    print(
        "\n"
        +
        "=" * 70
    )

    print(
        "DEEPONET STOKES DATASET VERIFICATION"
    )

    print(
        "=" * 70
    )


    # --------------------------------------------------------
    # Dataset location
    # --------------------------------------------------------

    print(
        "\nDataset:"
    )

    print(
        os.path.abspath(
            DATASET_PATH
        )
    )


    # --------------------------------------------------------
    # Metadata
    # --------------------------------------------------------

    print(
        "\n--- PDE / DOMAIN METADATA ---"
    )

    print(
        f"Lx = {metadata['Lx']}"
    )

    print(
        f"Ly = {metadata['Ly']}"
    )

    print(
        f"nu = {metadata['nu']}"
    )


    # --------------------------------------------------------
    # Full dataset
    # --------------------------------------------------------

    print(
        "\n--- FULL DATASET ---"
    )

    print(
        "Forcing shape:",
        tuple(
            forcing.shape
        )
    )

    print(
        "Velocity shape:",
        tuple(
            velocity.shape
        )
    )

    print(
        "Forcing dtype:",
        forcing.dtype
    )

    print(
        "Velocity dtype:",
        velocity.dtype
    )


    # --------------------------------------------------------
    # Resolution
    # --------------------------------------------------------

    Nx = forcing.shape[-2]
    Ny = forcing.shape[-1]


    print(
        f"Resolution: "
        f"{Nx} x {Ny}"
    )


    # --------------------------------------------------------
    # Splits
    # --------------------------------------------------------

    print(
        "\n--- SPLITS ---"
    )


    print(
        "Training forcing:",
        tuple(
            f_train.shape
        )
    )

    print(
        "Training velocity:",
        tuple(
            u_train.shape
        )
    )


    print(
        "\nValidation forcing:",
        tuple(
            f_val.shape
        )
    )

    print(
        "Validation velocity:",
        tuple(
            u_val.shape
        )
    )


    print(
        "\nTest forcing:",
        tuple(
            f_test.shape
        )
    )

    print(
        "Test velocity:",
        tuple(
            u_test.shape
        )
    )


    # --------------------------------------------------------
    # Verify no overlap in split indices
    # --------------------------------------------------------

    train_set = set(
        idx_train.tolist()
    )

    val_set = set(
        idx_val.tolist()
    )

    test_set = set(
        idx_test.tolist()
    )


    train_val_overlap = (
        train_set
        &
        val_set
    )


    train_test_overlap = (
        train_set
        &
        test_set
    )


    val_test_overlap = (
        val_set
        &
        test_set
    )


    print(
        "\n--- SPLIT OVERLAP CHECK ---"
    )

    print(
        "Train / validation overlap:",
        len(
            train_val_overlap
        )
    )

    print(
        "Train / test overlap:",
        len(
            train_test_overlap
        )
    )

    print(
        "Validation / test overlap:",
        len(
            val_test_overlap
        )
    )


    # --------------------------------------------------------
    # Normalization statistics
    # --------------------------------------------------------

    print(
        "\n--- INPUT NORMALIZATION ---"
    )


    print(
        "Training forcing mean "
        "(before normalization):"
    )

    print(
        normalizer.mean
        .flatten()
        .tolist()
    )


    print(
        "\nTraining forcing std "
        "(before normalization):"
    )

    print(
        normalizer.std
        .flatten()
        .tolist()
    )


    # --------------------------------------------------------
    # Encode training set and verify
    # --------------------------------------------------------

    f_train_normalized = (
        normalizer.encode(
            f_train
        )
    )


    normalized_mean = (
        f_train_normalized.mean(
            dim=(0, 2, 3)
        )
    )


    normalized_std = (
        f_train_normalized.std(
            dim=(0, 2, 3)
        )
    )


    print(
        "\nTraining forcing mean "
        "(after normalization):"
    )

    print(
        normalized_mean.tolist()
    )


    print(
        "\nTraining forcing std "
        "(after normalization):"
    )

    print(
        normalized_std.tolist()
    )


    # --------------------------------------------------------
    # Raw value ranges
    # --------------------------------------------------------

    print(
        "\n--- VALUE RANGES ---"
    )


    print(
        "Forcing min/max:",
        float(
            forcing.min()
        ),
        float(
            forcing.max()
        )
    )


    print(
        "Velocity min/max:",
        float(
            velocity.min()
        ),
        float(
            velocity.max()
        )
    )


    # --------------------------------------------------------
    # Numerical sanity checks
    # --------------------------------------------------------

    print(
        "\n--- NUMERICAL CHECKS ---"
    )


    print(
        "NaNs in forcing:",
        bool(
            torch.isnan(
                forcing
            ).any()
        )
    )


    print(
        "NaNs in velocity:",
        bool(
            torch.isnan(
                velocity
            ).any()
        )
    )


    print(
        "Infs in forcing:",
        bool(
            torch.isinf(
                forcing
            ).any()
        )
    )


    print(
        "Infs in velocity:",
        bool(
            torch.isinf(
                velocity
            ).any()
        )
    )


    # --------------------------------------------------------
    # First few split indices
    #
    # Useful later if we want to verify against the FNO run.
    # --------------------------------------------------------

    print(
        "\n--- FIRST 10 SPLIT INDICES ---"
    )


    print(
        "Train:",
        idx_train[
            :10
        ].tolist()
    )


    print(
        "Validation:",
        idx_val[
            :10
        ].tolist()
    )


    print(
        "Test:",
        idx_test[
            :10
        ].tolist()
    )


    print(
        "\n"
        +
        "=" * 70
    )

    print(
        "BATCH 1 COMPLETE"
    )

    print(
        "=" * 70
    )


# ============================================================
# 7. MAIN
# ============================================================

def main():

    # --------------------------------------------------------
    # Load exact existing FNO dataset
    # --------------------------------------------------------

    (
        forcing,
        velocity,
        metadata
    ) = load_dataset(
        DATASET_PATH
    )


    # --------------------------------------------------------
    # Reproduce FNO train / val / test split
    # --------------------------------------------------------

    (
        (
            f_train,
            u_train
        ),

        (
            f_val,
            u_val
        ),

        (
            f_test,
            u_test
        ),

        (
            idx_train,
            idx_val,
            idx_test
        )

    ) = split_dataset(

        forcing,
        velocity,

        N_TRAIN,
        N_VAL,
        N_TEST,

        seed=SEED

    )


    # --------------------------------------------------------
    # Fit normalizer on TRAINING forcing only
    # --------------------------------------------------------

    normalizer = InputNormalizer(
        f_train
    )


    # --------------------------------------------------------
    # Verify everything
    # --------------------------------------------------------

    verify_dataset(

        forcing,
        velocity,
        metadata,

        f_train,
        u_train,

        f_val,
        u_val,

        f_test,
        u_test,

        normalizer,

        idx_train,
        idx_val,
        idx_test

    )


# ============================================================
# 8. RUN
# ============================================================

if __name__ == "__main__":

    main()