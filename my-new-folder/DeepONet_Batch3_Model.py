# ============================================================
# DEEPONET FOR 2-D PERIODIC STOKES FLOW
#
# BATCH 3:
# Define the DeepONet architecture and verify a forward pass.
#
# NO TRAINING IS PERFORMED.
#
# Learning problem:
#
#       f(x,y) = (f1,f2)
#
#               |
#               v
#
#       u(x,y) = (u,v)
#
# using the exact same Stokes dataset as the FNO.
# ============================================================


# ============================================================
# 0. IMPORTS
# ============================================================

import os
import numpy as np
import torch
import torch.nn as nn


# ============================================================
# 1. CONFIGURATION
# ============================================================

SEED = 0

N_TRAIN = 1500
N_VAL = 250
N_TEST = 250

# DeepONet architecture
LATENT_DIM = 128

BRANCH_WIDTH = 128
TRUNK_WIDTH = 128

# Only use a tiny batch for this architecture test.
TEST_BATCH_SIZE = 2


torch.manual_seed(SEED)
np.random.seed(SEED)


# ============================================================
# 2. DEVICE
#
# CUDA on SOL
# MPS on Apple Silicon
# CPU otherwise
# ============================================================

if torch.cuda.is_available():

    DEVICE = torch.device("cuda")

elif torch.backends.mps.is_available():

    DEVICE = torch.device("mps")

else:

    DEVICE = torch.device("cpu")


print("\n" + "=" * 70)
print("DEVICE")
print("=" * 70)

print(
    "PyTorch version:",
    torch.__version__
)

print(
    "Selected device:",
    DEVICE
)


if DEVICE.type == "cuda":

    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )


print("=" * 70)


# ============================================================
# 3. DATASET PATH
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
# 4. LOAD DATASET
# ============================================================

def load_dataset(
    npz_path
):

    if not os.path.exists(
        npz_path
    ):

        raise FileNotFoundError(
            f"\nCould not find dataset:\n{npz_path}"
        )


    data = np.load(
        npz_path
    )


    forcing = np.stack(
        [
            data["f1"],
            data["f2"]
        ],
        axis=1
    ).astype(
        np.float32
    )


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
# 5. SAME TRAIN / VALIDATION / TEST SPLIT AS FNO
# ============================================================

def split_dataset(

    forcing,
    velocity,

    n_train,
    n_val,
    n_test,

    seed=0

):

    n_total = (
        forcing.shape[0]
    )


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
        .manual_seed(
            seed
        )

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
            forcing[
                idx_train
            ],

            velocity[
                idx_train
            ]
        ),

        (
            forcing[
                idx_val
            ],

            velocity[
                idx_val
            ]
        ),

        (
            forcing[
                idx_test
            ],

            velocity[
                idx_test
            ]
        )

    )


# ============================================================
# 6. SAME INPUT NORMALIZATION AS FNO
# ============================================================

class InputNormalizer:

    def __init__(
        self,
        x
    ):

        self.mean = x.mean(

            dim=(
                0,
                2,
                3
            ),

            keepdim=True

        )


        self.std = (

            x.std(

                dim=(
                    0,
                    2,
                    3
                ),

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
# 7. BUILD DEEPONET REPRESENTATION
# ============================================================

def build_representation(

    f_train,
    u_train,

    f_val,
    u_val,

    f_test,
    u_test,

    normalizer

):

    # --------------------------------------------------------
    # Normalize forcing only.
    # --------------------------------------------------------

    f_train_n = normalizer.encode(
        f_train
    )

    f_val_n = normalizer.encode(
        f_val
    )

    f_test_n = normalizer.encode(
        f_test
    )


    Nx = (
        f_train.shape[-2]
    )

    Ny = (
        f_train.shape[-1]
    )


    # --------------------------------------------------------
    # Branch representation
    #
    # (N,2,128,128)
    #
    #       ->
    #
    # (N,32768)
    # --------------------------------------------------------

    branch_train = (
        f_train_n.reshape(
            f_train.shape[0],
            -1
        )
    )


    branch_val = (
        f_val_n.reshape(
            f_val.shape[0],
            -1
        )
    )


    branch_test = (
        f_test_n.reshape(
            f_test.shape[0],
            -1
        )
    )


    # --------------------------------------------------------
    # Trunk coordinates
    #
    # Same normalized [0,1) convention as FNO.
    # --------------------------------------------------------

    x = torch.linspace(
        0.0,
        1.0,
        Nx + 1,
        dtype=torch.float32
    )[:-1]


    y = torch.linspace(
        0.0,
        1.0,
        Ny + 1,
        dtype=torch.float32
    )[:-1]


    X, Y = torch.meshgrid(
        x,
        y,
        indexing="ij"
    )


    trunk = torch.stack(
        [
            X.reshape(-1),
            Y.reshape(-1)
        ],
        dim=1
    )


    # --------------------------------------------------------
    # Targets
    #
    # (N,2,128,128)
    #
    #       ->
    #
    # (N,16384,2)
    # --------------------------------------------------------

    target_train = (

        u_train
        .permute(
            0,
            2,
            3,
            1
        )
        .reshape(
            u_train.shape[0],
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
            u_val.shape[0],
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
            u_test.shape[0],
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

        "trunk":
            trunk,

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
# 8. BRANCH NETWORK
#
# Input:
#
#       entire sampled forcing field
#
#       R^32768
#
# Output:
#
#       two sets of latent coefficients:
#
#       b_u(f) in R^p
#       b_v(f) in R^p
#
# Thus final dimension = 2*p.
# ============================================================

class BranchNet(
    nn.Module
):

    def __init__(

        self,

        input_dim,

        width,

        latent_dim

    ):

        super().__init__()


        self.latent_dim = (
            latent_dim
        )


        self.network = nn.Sequential(

            nn.Linear(
                input_dim,
                width
            ),

            nn.GELU(),

            nn.Linear(
                width,
                width
            ),

            nn.GELU(),

            nn.Linear(
                width,
                2 * latent_dim
            )

        )


    def forward(
        self,
        x
    ):

        output = self.network(
            x
        )


        # ----------------------------------------------------
        # Convert
        #
        #       (B,2p)
        #
        # to
        #
        #       (B,2,p)
        #
        # where dimension 1 corresponds to
        #
        #       u component
        #       v component
        # ----------------------------------------------------

        output = output.reshape(

            x.shape[0],

            2,

            self.latent_dim

        )


        return output


# ============================================================
# 9. TRUNK NETWORK
#
# Input:
#
#       coordinate (x,y) in R^2
#
# Output:
#
#       latent basis vector t(x,y) in R^p
# ============================================================

class TrunkNet(
    nn.Module
):

    def __init__(

        self,

        width,

        latent_dim

    ):

        super().__init__()


        self.network = nn.Sequential(

            nn.Linear(
                2,
                width
            ),

            nn.Tanh(),

            nn.Linear(
                width,
                width
            ),

            nn.Tanh(),

            nn.Linear(
                width,
                latent_dim
            )

        )


    def forward(
        self,
        coordinates
    ):

        return self.network(
            coordinates
        )


# ============================================================
# 10. VECTOR-VALUED DEEPONET
#
# Branch:
#
#       B(f)
#
#       shape = (batch,2,p)
#
#
# Trunk:
#
#       T(x,y)
#
#       shape = (points,p)
#
#
# Combination:
#
#       u_c(x,y)
#
#           =
#
#       sum_j B_cj(f) T_j(x,y)
#
#
# for c = 1,2 corresponding to velocity components u and v.
# ============================================================

class DeepONet(
    nn.Module
):

    def __init__(

        self,

        branch_input_dim,

        branch_width,

        trunk_width,

        latent_dim

    ):

        super().__init__()


        self.branch = BranchNet(

            branch_input_dim,

            branch_width,

            latent_dim

        )


        self.trunk = TrunkNet(

            trunk_width,

            latent_dim

        )


        # ----------------------------------------------------
        # One learnable scalar bias per velocity component.
        # ----------------------------------------------------

        self.bias = nn.Parameter(
            torch.zeros(
                2
            )
        )


    def forward(

        self,

        branch_input,

        trunk_coordinates

    ):

        # ----------------------------------------------------
        # Branch:
        #
        # (B,32768)
        #
        #       ->
        #
        # (B,2,p)
        # ----------------------------------------------------

        branch_output = self.branch(
            branch_input
        )


        # ----------------------------------------------------
        # Trunk:
        #
        # (16384,2)
        #
        #       ->
        #
        # (16384,p)
        # ----------------------------------------------------

        trunk_output = self.trunk(
            trunk_coordinates
        )


        # ----------------------------------------------------
        # Inner products over latent dimension p.
        #
        # b = batch
        # c = velocity component
        # p = latent feature
        # n = spatial point
        #
        # Result:
        #
        #       (B,16384,2)
        # ----------------------------------------------------

        output = torch.einsum(

            "bcp,np->bnc",

            branch_output,

            trunk_output

        )


        # ----------------------------------------------------
        # Add one bias per output component.
        # ----------------------------------------------------

        output = (

            output

            +

            self.bias[
                None,
                None,
                :
            ]

        )


        return output


# ============================================================
# 11. PARAMETER COUNT
# ============================================================

def count_parameters(
    model
):

    return sum(

        parameter.numel()

        for parameter
        in model.parameters()

        if parameter.requires_grad

    )


# ============================================================
# 12. MAIN
# ============================================================

def main():

    # ========================================================
    # LOAD DATA
    # ========================================================

    print(
        "\nLoading dataset..."
    )


    (
        forcing,
        velocity,
        metadata
    ) = load_dataset(
        DATASET_PATH
    )


    print(
        "Dataset loaded."
    )


    # ========================================================
    # SPLIT
    # ========================================================

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
        )

    ) = split_dataset(

        forcing,
        velocity,

        N_TRAIN,
        N_VAL,
        N_TEST,

        seed=SEED

    )


    # ========================================================
    # NORMALIZER
    # ========================================================

    normalizer = InputNormalizer(
        f_train
    )


    # ========================================================
    # BUILD REPRESENTATION
    # ========================================================

    data = build_representation(

        f_train,
        u_train,

        f_val,
        u_val,

        f_test,
        u_test,

        normalizer

    )


    branch_train = (
        data[
            "branch_train"
        ]
    )


    trunk = (
        data[
            "trunk"
        ]
    )


    Nx = (
        data[
            "Nx"
        ]
    )


    Ny = (
        data[
            "Ny"
        ]
    )


    branch_input_dim = (
        branch_train.shape[1]
    )


    # ========================================================
    # CREATE MODEL
    # ========================================================

    model = DeepONet(

        branch_input_dim=
            branch_input_dim,

        branch_width=
            BRANCH_WIDTH,

        trunk_width=
            TRUNK_WIDTH,

        latent_dim=
            LATENT_DIM

    ).to(
        DEVICE
    )


    # ========================================================
    # PARAMETER COUNTS
    # ========================================================

    branch_parameters = (
        count_parameters(
            model.branch
        )
    )


    trunk_parameters = (
        count_parameters(
            model.trunk
        )
    )


    total_parameters = (
        count_parameters(
            model
        )
    )


    print(
        "\n"
        +
        "=" * 70
    )

    print(
        "DEEPONET ARCHITECTURE"
    )

    print(
        "=" * 70
    )


    print(
        f"\nBranch input dimension: "
        f"{branch_input_dim}"
    )


    print(
        f"Latent dimension p: "
        f"{LATENT_DIM}"
    )


    print(
        f"Branch width: "
        f"{BRANCH_WIDTH}"
    )


    print(
        f"Trunk width: "
        f"{TRUNK_WIDTH}"
    )


    print(
        "\nParameter counts:"
    )


    print(
        f"  Branch: "
        f"{branch_parameters:,}"
    )


    print(
        f"  Trunk:  "
        f"{trunk_parameters:,}"
    )


    print(
        f"  Total:  "
        f"{total_parameters:,}"
    )


    # ========================================================
    # PREPARE TINY TEST BATCH
    # ========================================================

    branch_batch = (

        branch_train[
            :TEST_BATCH_SIZE
        ]

        .to(
            DEVICE
        )

    )


    trunk_device = (
        trunk.to(
            DEVICE
        )
    )


    # ========================================================
    # FORWARD PASS
    # ========================================================

    print(
        "\n"
        +
        "=" * 70
    )

    print(
        "FORWARD PASS TEST"
    )

    print(
        "=" * 70
    )


    print(
        "\nBranch batch:"
    )

    print(
        tuple(
            branch_batch.shape
        )
    )


    print(
        "\nTrunk coordinates:"
    )

    print(
        tuple(
            trunk_device.shape
        )
    )


    model.eval()


    with torch.no_grad():

        prediction = model(

            branch_batch,

            trunk_device

        )


    print(
        "\nRaw DeepONet prediction:"
    )

    print(
        tuple(
            prediction.shape
        )
    )


    # ========================================================
    # EXPECTED SHAPE
    #
    #       (batch, Nx*Ny, 2)
    # ========================================================

    expected_shape = (

        TEST_BATCH_SIZE,

        Nx * Ny,

        2

    )


    print(
        "\nExpected:"
    )

    print(
        expected_shape
    )


    print(
        "\nShape correct:",
        tuple(
            prediction.shape
        )
        ==
        expected_shape
    )


    # ========================================================
    # CONVERT BACK TO FNO/PDE FIELD FORMAT
    #
    # DeepONet:
    #
    #       (B,16384,2)
    #
    # becomes
    #
    #       (B,2,128,128)
    #
    # so it can later be compared directly to the FNO output.
    # ========================================================

    prediction_field = (

        prediction
        .reshape(
            TEST_BATCH_SIZE,
            Nx,
            Ny,
            2
        )
        .permute(
            0,
            3,
            1,
            2
        )

    )


    print(
        "\nPrediction reshaped to PDE field:"
    )

    print(
        tuple(
            prediction_field.shape
        )
    )


    expected_field_shape = (

        TEST_BATCH_SIZE,

        2,

        Nx,

        Ny

    )


    print(
        "\nExpected PDE field:"
    )

    print(
        expected_field_shape
    )


    print(
        "\nField shape correct:",
        tuple(
            prediction_field.shape
        )
        ==
        expected_field_shape
    )


    # ========================================================
    # CHECK NUMERICAL OUTPUT
    # ========================================================

    print(
        "\n--- NUMERICAL SANITY CHECK ---"
    )


    print(
        "NaNs in prediction:",
        bool(
            torch.isnan(
                prediction
            ).any()
        )
    )


    print(
        "Infs in prediction:",
        bool(
            torch.isinf(
                prediction
            ).any()
        )
    )


    print(
        "Prediction min:",
        prediction.min().item()
    )


    print(
        "Prediction max:",
        prediction.max().item()
    )


    # ========================================================
    # DEVICE INFORMATION
    # ========================================================

    print(
        "\n--- DEVICE CHECK ---"
    )


    print(
        "Model device:",
        next(
            model.parameters()
        ).device
    )


    print(
        "Branch batch device:",
        branch_batch.device
    )


    print(
        "Trunk device:",
        trunk_device.device
    )


    print(
        "Prediction device:",
        prediction.device
    )


    # ========================================================
    # CUDA MEMORY INFORMATION
    # ========================================================

    if DEVICE.type == "cuda":

        allocated = (
            torch.cuda.memory_allocated()
            /
            (1024 ** 2)
        )

        reserved = (
            torch.cuda.memory_reserved()
            /
            (1024 ** 2)
        )


        print(
            "\n--- CUDA MEMORY ---"
        )


        print(
            f"Allocated: "
            f"{allocated:.2f} MB"
        )


        print(
            f"Reserved: "
            f"{reserved:.2f} MB"
        )


    # ========================================================
    # MPS MEMORY INFORMATION
    # ========================================================

    elif DEVICE.type == "mps":

        print(
            "\n--- MPS MEMORY ---"
        )


        if hasattr(
            torch.mps,
            "current_allocated_memory"
        ):

            allocated = (

                torch.mps.current_allocated_memory()

                /

                (1024 ** 2)

            )


            print(
                f"Allocated: "
                f"{allocated:.2f} MB"
            )


    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print(
        "\n"
        +
        "=" * 70
    )

    print(
        "BATCH 3 SUMMARY"
    )

    print(
        "=" * 70
    )


    print(
        "\nDeepONet mapping:"
    )


    print(
        f"\n    Branch: "
        f"R^{branch_input_dim} "
        f"-> R^(2 x {LATENT_DIM})"
    )


    print(
        "\n    Trunk:  "
        f"R^2 "
        f"-> R^{LATENT_DIM}"
    )


    print(
        "\n    Output:"
    )

    print(
        f"        "
        f"(forcing, coordinate) "
        f"-> (u,v)"
    )


    print(
        "\nTotal trainable parameters:",
        f"{total_parameters:,}"
    )


    print(
        "\nForward pass successful:",
        (
            tuple(
                prediction_field.shape
            )
            ==
            expected_field_shape
        )
    )


    print(
        "\n"
        +
        "=" * 70
    )

    print(
        "BATCH 3 COMPLETE"
    )

    print(
        "=" * 70
    )


# ============================================================
# 13. RUN
# ============================================================

if __name__ == "__main__":

    main()