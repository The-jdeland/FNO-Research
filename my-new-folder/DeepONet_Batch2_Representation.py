# ============================================================
# DEEPONET FOR 2-D PERIODIC STOKES FLOW
#
# BATCH 2:
# Construct and verify the DeepONet data representation.
#
# No neural network is defined or trained in this file.
#
# Exact learning problem:
#
#       f(x,y) = (f1,f2)
#
#               |
#               v
#
#       u(x,y) = (u,v)
#
# for the same 2-D periodic Stokes dataset used by the FNO.
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
# These match the FNO FULL_CONFIG baseline.
# ============================================================

SEED = 0

N_TRAIN = 1500
N_VAL = 250
N_TEST = 250


# ============================================================
# 2. DATASET PATH
#
# This assumes this Python file is on your Desktop and that
# the generated dataset is:
#
# Desktop/
#     fno_data/
#         stokes2d_base.npz
# ============================================================

HERE = os.path.dirname(
    os.path.abspath(__file__)
)

DATASET_PATH = os.path.join(
    HERE,
    "fno_data",
    "stokes2d_base.npz"
)


# ============================================================
# 3. LOAD THE EXISTING FNO DATASET
# ============================================================

def load_dataset(npz_path):

    if not os.path.exists(npz_path):

        raise FileNotFoundError(
            "\nCould not find dataset:\n"
            f"{npz_path}"
        )

    data = np.load(
        npz_path
    )

    # --------------------------------------------------------
    # Input:
    #
    #       f = (f1,f2)
    #
    # Shape:
    #
    #       N x 2 x Nx x Ny
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
    #       velocity = (u,v)
    #
    # Shape:
    #
    #       N x 2 x Nx x Ny
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

    metadata = {

        "Lx":
            float(
                data["Lx"]
            ),

        "Ly":
            float(
                data["Ly"]
            ),

        "nu":
            float(
                data["nu"]
            )

    }

    return (
        torch.from_numpy(
            forcing
        ),

        torch.from_numpy(
            velocity
        ),

        metadata
    )


# ============================================================
# 4. REPRODUCE THE FNO DATA SPLIT
#
# This is deliberately the same procedure as the FNO:
#
#       torch.Generator().manual_seed(0)
#
#       torch.randperm(...)
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

    assert (
        n_train
        +
        n_val
        +
        n_test
        <=
        n_total
    )


    generator = (
        torch.Generator()
        .manual_seed(seed)
    )


    permutation = torch.randperm(
        n_total,
        generator=generator
    )


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


    return (

        (
            forcing[idx_train],
            velocity[idx_train]
        ),

        (
            forcing[idx_val],
            velocity[idx_val]
        ),

        (
            forcing[idx_test],
            velocity[idx_test]
        ),

        (
            idx_train,
            idx_val,
            idx_test
        )

    )


# ============================================================
# 5. SAME INPUT NORMALIZATION AS FNO
#
# Per-channel mean and standard deviation are calculated using
# the TRAINING forcing fields only.
# ============================================================

class InputNormalizer:

    def __init__(
        self,
        x
    ):

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
# 6. CONSTRUCT DEEPONET REPRESENTATION
# ============================================================

def build_deeponet_representation(

    f_train,
    u_train,

    f_val,
    u_val,

    f_test,
    u_test,

    normalizer

):

    # ========================================================
    # A. NORMALIZE INPUT FORCING
    #
    # We do NOT normalize velocity targets because the FNO
    # baseline does not normalize them.
    # ========================================================

    f_train_normalized = (
        normalizer.encode(
            f_train
        )
    )

    f_val_normalized = (
        normalizer.encode(
            f_val
        )
    )

    f_test_normalized = (
        normalizer.encode(
            f_test
        )
    )


    # ========================================================
    # B. DIMENSIONS
    # ========================================================

    n_train = (
        f_train.shape[0]
    )

    n_val = (
        f_val.shape[0]
    )

    n_test = (
        f_test.shape[0]
    )

    Nx = (
        f_train.shape[-2]
    )

    Ny = (
        f_train.shape[-1]
    )


    # ========================================================
    # C. BRANCH INPUT
    #
    # Standard DeepONet branch input:
    #
    #       sampled input function
    #
    #       f = (f1,f2)
    #
    # Original shape:
    #
    #       2 x 128 x 128
    #
    # Flattened:
    #
    #       32768
    #
    # because
    #
    #       2 * 128 * 128 = 32768.
    # ========================================================

    branch_train = (
        f_train_normalized
        .reshape(
            n_train,
            -1
        )
    )

    branch_val = (
        f_val_normalized
        .reshape(
            n_val,
            -1
        )
    )

    branch_test = (
        f_test_normalized
        .reshape(
            n_test,
            -1
        )
    )


    # ========================================================
    # D. TRUNK INPUT
    #
    # DeepONet's trunk network receives spatial coordinates.
    #
    # We use the same normalized coordinate convention as the
    # FNO:
    #
    #       x,y in [0,1)
    #
    # Number of coordinates:
    #
    #       128 * 128 = 16384
    # ========================================================

    x_coordinates = torch.linspace(
        0.0,
        1.0,
        Nx + 1,
        dtype=torch.float32
    )[:-1]


    y_coordinates = torch.linspace(
        0.0,
        1.0,
        Ny + 1,
        dtype=torch.float32
    )[:-1]


    X, Y = torch.meshgrid(
        x_coordinates,
        y_coordinates,
        indexing="ij"
    )


    trunk_coordinates = torch.stack(
        [
            X.reshape(-1),
            Y.reshape(-1)
        ],
        dim=1
    )


    # ========================================================
    # E. TARGET REPRESENTATION
    #
    # Original:
    #
    #       N x 2 x Nx x Ny
    #
    # DeepONet representation:
    #
    #       N x (Nx*Ny) x 2
    #
    # Each spatial point therefore has:
    #
    #       [u(x,y), v(x,y)]
    # ========================================================

    target_train = (

        u_train
        .permute(
            0,
            2,
            3,
            1
        )
        .reshape(
            n_train,
            Nx * Ny,
            2
        )

    )


    target_val = (

        u_val
        .permute(
            0,
            2,
            3,
            1
        )
        .reshape(
            n_val,
            Nx * Ny,
            2
        )

    )


    target_test = (

        u_test
        .permute(
            0,
            2,
            3,
            1
        )
        .reshape(
            n_test,
            Nx * Ny,
            2
        )

    )


    return {

        "branch_train":
            branch_train,

        "branch_val":
            branch_val,

        "branch_test":
            branch_test,

        "trunk_coordinates":
            trunk_coordinates,

        "target_train":
            target_train,

        "target_val":
            target_val,

        "target_test":
            target_test,

        "Nx":
            Nx,

        "Ny":
            Ny

    }


# ============================================================
# 7. VERIFY THE DEEPONET REPRESENTATION
# ============================================================

def verify_representation(
    deeponet_data,
    u_train
):

    branch_train = (
        deeponet_data[
            "branch_train"
        ]
    )

    branch_val = (
        deeponet_data[
            "branch_val"
        ]
    )

    branch_test = (
        deeponet_data[
            "branch_test"
        ]
    )

    trunk = (
        deeponet_data[
            "trunk_coordinates"
        ]
    )

    target_train = (
        deeponet_data[
            "target_train"
        ]
    )

    target_val = (
        deeponet_data[
            "target_val"
        ]
    )

    target_test = (
        deeponet_data[
            "target_test"
        ]
    )

    Nx = (
        deeponet_data[
            "Nx"
        ]
    )

    Ny = (
        deeponet_data[
            "Ny"
        ]
    )


    print(
        "\n"
        +
        "=" * 70
    )

    print(
        "DEEPONET BATCH 2: DATA REPRESENTATION"
    )

    print(
        "=" * 70
    )


    # ========================================================
    # BRANCH
    # ========================================================

    print(
        "\n--- BRANCH INPUT ---"
    )


    print(
        "Training branch:",
        tuple(
            branch_train.shape
        )
    )

    print(
        "Validation branch:",
        tuple(
            branch_val.shape
        )
    )

    print(
        "Test branch:",
        tuple(
            branch_test.shape
        )
    )


    print(
        "\nBranch input dimension:",
        branch_train.shape[1]
    )


    # ========================================================
    # TRUNK
    # ========================================================

    print(
        "\n--- TRUNK INPUT ---"
    )


    print(
        "Trunk shape:",
        tuple(
            trunk.shape
        )
    )


    print(
        "Number of spatial points:",
        trunk.shape[0]
    )


    print(
        "\nFirst five coordinates:"
    )

    print(
        trunk[:5]
    )


    print(
        "\nLast five coordinates:"
    )

    print(
        trunk[-5:]
    )


    # ========================================================
    # TARGET
    # ========================================================

    print(
        "\n--- TARGET OUTPUT ---"
    )


    print(
        "Training target:",
        tuple(
            target_train.shape
        )
    )

    print(
        "Validation target:",
        tuple(
            target_val.shape
        )
    )

    print(
        "Test target:",
        tuple(
            target_test.shape
        )
    )


    # ========================================================
    # RESHAPE SANITY CHECK
    #
    # Verify that a particular velocity value has not changed
    # during our reshaping.
    # ========================================================

    sample_index = 0

    ix = 10
    iy = 20


    flattened_index = (
        ix * Ny
        +
        iy
    )


    original_u = (
        u_train[
            sample_index,
            0,
            ix,
            iy
        ].item()
    )


    original_v = (
        u_train[
            sample_index,
            1,
            ix,
            iy
        ].item()
    )


    converted_u = (
        target_train[
            sample_index,
            flattened_index,
            0
        ].item()
    )


    converted_v = (
        target_train[
            sample_index,
            flattened_index,
            1
        ].item()
    )


    print(
        "\n--- RESHAPE SANITY CHECK ---"
    )


    print(
        f"Grid index: "
        f"({ix},{iy})"
    )


    print(
        "Flattened spatial index:",
        flattened_index
    )


    print(
        "\nOriginal:"
    )

    print(
        "u =",
        original_u
    )

    print(
        "v =",
        original_v
    )


    print(
        "\nDeepONet target:"
    )

    print(
        "u =",
        converted_u
    )

    print(
        "v =",
        converted_v
    )


    matches = (

        original_u
        ==
        converted_u

        and

        original_v
        ==
        converted_v

    )


    print(
        "\nExact match:",
        matches
    )


    # ========================================================
    # MEMORY
    # ========================================================

    branch_memory = (

        branch_train.numel()

        *

        branch_train.element_size()

        /

        (1024 ** 2)

    )


    target_memory = (

        target_train.numel()

        *

        target_train.element_size()

        /

        (1024 ** 2)

    )


    print(
        "\n--- MEMORY ---"
    )


    print(
        f"Branch training tensor: "
        f"{branch_memory:.2f} MB"
    )


    print(
        f"Target training tensor: "
        f"{target_memory:.2f} MB"
    )


    # ========================================================
    # MATHEMATICAL SUMMARY
    # ========================================================

    print(
        "\n--- DEEPONET MAPPING ---"
    )


    print(
        "\nBranch network will receive:"
    )

    print(
        f"    R^{branch_train.shape[1]}"
    )


    print(
        "\nTrunk network will receive:"
    )

    print(
        "    R^2"
    )


    print(
        "\nNumber of trunk evaluation points:"
    )

    print(
        f"    {Nx * Ny}"
    )


    print(
        "\nOutput at each trunk point:"
    )

    print(
        "    R^2 = (u,v)"
    )


    print(
        "\n"
        +
        "=" * 70
    )

    print(
        "BATCH 2 COMPLETE"
    )

    print(
        "=" * 70
    )


# ============================================================
# 8. MAIN
# ============================================================

def main():

    print(
        "\nLoading:"
    )

    print(
        DATASET_PATH
    )


    # --------------------------------------------------------
    # Load dataset
    # --------------------------------------------------------

    (
        forcing,
        velocity,
        metadata
    ) = load_dataset(
        DATASET_PATH
    )


    print(
        "\nDataset loaded."
    )

    print(
        "Forcing:",
        tuple(
            forcing.shape
        )
    )

    print(
        "Velocity:",
        tuple(
            velocity.shape
        )
    )

    print(
        f"Lx={metadata['Lx']}, "
        f"Ly={metadata['Ly']}, "
        f"nu={metadata['nu']}"
    )


    # --------------------------------------------------------
    # Same split as FNO
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
    # Same normalization as FNO
    # --------------------------------------------------------

    normalizer = InputNormalizer(
        f_train
    )


    # --------------------------------------------------------
    # Build DeepONet representation
    # --------------------------------------------------------

    deeponet_data = (
        build_deeponet_representation(

            f_train,
            u_train,

            f_val,
            u_val,

            f_test,
            u_test,

            normalizer

        )
    )


    # --------------------------------------------------------
    # Verify
    # --------------------------------------------------------

    verify_representation(
        deeponet_data,
        u_train
    )


# ============================================================
# 9. RUN
# ============================================================

if __name__ == "__main__":

    main()