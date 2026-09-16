# ============================================================
# DEEPONET FOR 2-D PERIODIC STOKES FLOW
#
# BATCH 4:
# Training sanity check
#
# Goal:
#   - use the exact same Stokes dataset as the FNO
#   - use the exact same train/val/test split
#   - use the same input normalization
#   - use the same relative L2 metric
#   - use Adam + weight decay + cosine annealing
#   - train for ONLY 5 epochs
#
# This is NOT the final SOL training run.
# ============================================================


# ============================================================
# 0. IMPORTS
# ============================================================

import os
import time
import numpy as np

import torch
import torch.nn as nn

from torch.utils.data import TensorDataset, DataLoader


# ============================================================
# 1. CONFIGURATION
# ============================================================

SEED = 0

# Same FNO baseline split
N_TRAIN = 1500
N_VAL = 250
N_TEST = 250

# DeepONet architecture from Batch 3
LATENT_DIM = 128
BRANCH_WIDTH = 128
TRUNK_WIDTH = 128

# Same default optimization settings as FNO
BATCH_SIZE = 20
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4

# IMPORTANT:
# This is only a sanity test.
EPOCHS = 5


torch.manual_seed(SEED)
np.random.seed(SEED)


# ============================================================
# 2. DEVICE
#
# SOL:
#     CUDA
#
# Apple Silicon:
#     MPS
#
# Otherwise:
#     CPU
# ============================================================

if torch.cuda.is_available():

    DEVICE = torch.device("cuda")

elif torch.backends.mps.is_available():

    DEVICE = torch.device("mps")

else:

    DEVICE = torch.device("cpu")


print("\n" + "=" * 72)
print("DEVICE")
print("=" * 72)

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


print("=" * 72)


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


    # --------------------------------------------------------
    # Input forcing:
    #
    #       (f1,f2)
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
    # Target velocity:
    #
    #       (u,v)
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
# 5. SAME SPLIT AS FNO
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
# 7. BUILD DEEPONET DATA REPRESENTATION
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
    # Normalize forcing only
    # --------------------------------------------------------

    f_train_n = (
        normalizer.encode(
            f_train
        )
    )


    f_val_n = (
        normalizer.encode(
            f_val
        )
    )


    f_test_n = (
        normalizer.encode(
            f_test
        )
    )


    Nx = (
        f_train.shape[-2]
    )

    Ny = (
        f_train.shape[-1]
    )


    # --------------------------------------------------------
    # Branch inputs
    #
    # N x 2 x 128 x 128
    #
    # ->
    #
    # N x 32768
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
    # Same normalized [0,1) convention as FNO
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
    # N x 2 x 128 x 128
    #
    # ->
    #
    # N x 16384 x 2
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


        return output.reshape(

            x.shape[0],

            2,

            self.latent_dim

        )


# ============================================================
# 9. TRUNK NETWORK
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
        # B x 2 x p
        # ----------------------------------------------------

        branch_output = (
            self.branch(
                branch_input
            )
        )


        # ----------------------------------------------------
        # spatial_points x p
        # ----------------------------------------------------

        trunk_output = (
            self.trunk(
                trunk_coordinates
            )
        )


        # ----------------------------------------------------
        # B x spatial_points x 2
        # ----------------------------------------------------

        output = torch.einsum(

            "bcp,np->bnc",

            branch_output,

            trunk_output

        )


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
# 11. SAME RELATIVE L2 METRIC AS FNO
#
# For each sample:
#
#                  ||prediction - target||_2
#       error = -------------------------------
#                         ||target||_2
#
# Returns one error per sample.
# ============================================================

def relative_l2(
    prediction,
    target
):

    batch = (
        prediction.shape[0]
    )


    numerator = torch.norm(

        prediction.reshape(
            batch,
            -1
        )

        -

        target.reshape(
            batch,
            -1
        ),

        dim=1

    )


    denominator = (

        torch.norm(

            target.reshape(
                batch,
                -1
            ),

            dim=1

        )

        +

        1e-8

    )


    return (

        numerator

        /

        denominator

    )


# ============================================================
# 12. PARAMETER COUNT
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
# 13. DEVICE SYNCHRONIZATION FOR TIMING
# ============================================================

def synchronize_device():

    if DEVICE.type == "cuda":

        torch.cuda.synchronize()

    elif DEVICE.type == "mps":

        torch.mps.synchronize()


# ============================================================
# 14. MAIN
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
    # NORMALIZATION
    # ========================================================

    normalizer = InputNormalizer(
        f_train
    )


    # ========================================================
    # REPRESENTATION
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


    branch_val = (
        data[
            "branch_val"
        ]
    )


    branch_test = (
        data[
            "branch_test"
        ]
    )


    target_train = (
        data[
            "target_train"
        ]
    )


    target_val = (
        data[
            "target_val"
        ]
    )


    target_test = (
        data[
            "target_test"
        ]
    )


    trunk = (
        data[
            "trunk"
        ]
    )


    # ========================================================
    # DATA LOADERS
    #
    # Each batch contains:
    #
    #       branch input
    #       complete velocity target
    #
    # The trunk grid is shared by all samples.
    # ========================================================

    train_dataset = TensorDataset(

        branch_train,

        target_train

    )


    train_loader = DataLoader(

        train_dataset,

        batch_size=min(
            BATCH_SIZE,
            len(
                train_dataset
            )
        ),

        shuffle=True,

        num_workers=0

    )


    # ========================================================
    # CREATE MODEL
    # ========================================================

    branch_input_dim = (
        branch_train.shape[1]
    )


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


    print(
        "\n"
        +
        "=" * 72
    )

    print(
        "DEEPONET TRAINING SANITY CHECK"
    )

    print(
        "=" * 72
    )


    print(
        "\nDevice:",
        DEVICE
    )


    print(
        "Parameters:",
        f"{count_parameters(model):,}"
    )


    print(
        "Training samples:",
        N_TRAIN
    )


    print(
        "Validation samples:",
        N_VAL
    )


    print(
        "Batch size:",
        BATCH_SIZE
    )


    print(
        "Epochs:",
        EPOCHS
    )


    print(
        "Learning rate:",
        LEARNING_RATE
    )


    print(
        "Weight decay:",
        WEIGHT_DECAY
    )


    # ========================================================
    # MOVE SHARED TRUNK TO DEVICE
    # ========================================================

    trunk_device = (
        trunk.to(
            DEVICE
        )
    )


    # ========================================================
    # MOVE VALIDATION DATA TO DEVICE
    #
    # Same style as the FNO code:
    # validation is evaluated as a complete set.
    # ========================================================

    branch_val_device = (
        branch_val.to(
            DEVICE
        )
    )


    target_val_device = (
        target_val.to(
            DEVICE
        )
    )


    # ========================================================
    # OPTIMIZER
    # ========================================================

    optimizer = torch.optim.Adam(

        model.parameters(),

        lr=LEARNING_RATE,

        weight_decay=WEIGHT_DECAY

    )


    # ========================================================
    # COSINE ANNEALING
    #
    # Same scheduler type as FNO.
    # ========================================================

    scheduler = (
        torch.optim.lr_scheduler
        .CosineAnnealingLR(

            optimizer,

            T_max=EPOCHS

        )
    )


    # ========================================================
    # TRAINING HISTORY
    # ========================================================

    history = {

        "train_loss": [],

        "val_loss": []

    }


    # ========================================================
    # TRAIN
    # ========================================================

    print(
        "\nStarting 5-epoch sanity training...\n"
    )


    total_start = time.perf_counter()


    for epoch in range(
        EPOCHS
    ):

        synchronize_device()

        epoch_start = (
            time.perf_counter()
        )


        # ====================================================
        # TRAINING PHASE
        # ====================================================

        model.train()


        batch_losses = []


        for (
            branch_batch,
            target_batch
        ) in train_loader:


            branch_batch = (
                branch_batch.to(
                    DEVICE
                )
            )


            target_batch = (
                target_batch.to(
                    DEVICE
                )
            )


            # ------------------------------------------------
            # Forward
            # ------------------------------------------------

            prediction = model(

                branch_batch,

                trunk_device

            )


            # ------------------------------------------------
            # Same loss convention as FNO:
            #
            # mean of per-sample relative L2 errors
            # ------------------------------------------------

            loss = (
                relative_l2(

                    prediction,

                    target_batch

                ).mean()
            )


            # ------------------------------------------------
            # Backpropagation
            # ------------------------------------------------

            optimizer.zero_grad()


            loss.backward()


            optimizer.step()


            batch_losses.append(
                loss.item()
            )


        # ----------------------------------------------------
        # Scheduler advances once per epoch
        # ----------------------------------------------------

        scheduler.step()


        # ====================================================
        # VALIDATION PHASE
        # ====================================================

        model.eval()


        with torch.no_grad():

            val_prediction = model(

                branch_val_device,

                trunk_device

            )


            val_loss = (

                relative_l2(

                    val_prediction,

                    target_val_device

                )

                .mean()

                .item()

            )


        # ====================================================
        # RECORD HISTORY
        # ====================================================

        train_loss = float(
            np.mean(
                batch_losses
            )
        )


        history[
            "train_loss"
        ].append(
            train_loss
        )


        history[
            "val_loss"
        ].append(
            val_loss
        )


        synchronize_device()


        epoch_time = (

            time.perf_counter()

            -

            epoch_start

        )


        current_lr = (
            optimizer
            .param_groups[0]["lr"]
        )


        print(

            f"Epoch "
            f"{epoch + 1:02d}/{EPOCHS} | "

            f"train rel-L2 = "
            f"{train_loss:.6f} | "

            f"val rel-L2 = "
            f"{val_loss:.6f} | "

            f"lr = "
            f"{current_lr:.3e} | "

            f"time = "
            f"{epoch_time:.2f}s"

        )


    # ========================================================
    # TOTAL TIME
    # ========================================================

    synchronize_device()


    total_time = (

        time.perf_counter()

        -

        total_start

    )


    # ========================================================
    # SANITY CHECKS
    # ========================================================

    first_train = (
        history[
            "train_loss"
        ][0]
    )


    final_train = (
        history[
            "train_loss"
        ][-1]
    )


    first_val = (
        history[
            "val_loss"
        ][0]
    )


    final_val = (
        history[
            "val_loss"
        ][-1]
    )


    train_decreased = (

        final_train

        <

        first_train

    )


    val_decreased = (

        final_val

        <

        first_val

    )


    finite_losses = (

        np.isfinite(
            history[
                "train_loss"
            ]
        ).all()

        and

        np.isfinite(
            history[
                "val_loss"
            ]
        ).all()

    )


    print(
        "\n"
        +
        "=" * 72
    )

    print(
        "BATCH 4 TRAINING SUMMARY"
    )

    print(
        "=" * 72
    )


    print(
        f"\nTotal training time: "
        f"{total_time:.2f}s"
    )


    print(
        f"\nInitial training error: "
        f"{first_train:.6f}"
    )


    print(
        f"Final training error:   "
        f"{final_train:.6f}"
    )


    print(
        f"\nInitial validation error: "
        f"{first_val:.6f}"
    )


    print(
        f"Final validation error:   "
        f"{final_val:.6f}"
    )


    print(
        "\nTraining loss decreased:",
        train_decreased
    )


    print(
        "Validation loss decreased:",
        val_decreased
    )


    print(
        "All losses finite:",
        finite_losses
    )


    # ========================================================
    # DEVICE MEMORY
    # ========================================================

    if DEVICE.type == "cuda":

        allocated = (

            torch.cuda
            .memory_allocated()

            /

            (1024 ** 2)

        )


        peak = (

            torch.cuda
            .max_memory_allocated()

            /

            (1024 ** 2)

        )


        print(
            f"\nCUDA currently allocated: "
            f"{allocated:.2f} MB"
        )


        print(
            f"CUDA peak allocated: "
            f"{peak:.2f} MB"
        )


    elif DEVICE.type == "mps":

        if hasattr(
            torch.mps,
            "current_allocated_memory"
        ):

            allocated = (

                torch.mps
                .current_allocated_memory()

                /

                (1024 ** 2)

            )


            print(
                f"\nMPS currently allocated: "
                f"{allocated:.2f} MB"
            )


    # ========================================================
    # IMPORTANT:
    #
    # We intentionally do NOT evaluate the test set yet.
    #
    # The test set should remain untouched while we establish
    # that the architecture and training procedure work.
    # ========================================================

    print(
        "\nTest set evaluated: False"
    )


    print(
        "\n"
        +
        "=" * 72
    )

    print(
        "BATCH 4 COMPLETE"
    )

    print(
        "=" * 72
    )


# ============================================================
# 15. RUN
# ============================================================

if __name__ == "__main__":

    main()