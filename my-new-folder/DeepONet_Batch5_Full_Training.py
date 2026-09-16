# ============================================================
# DEEPONET FOR 2-D PERIODIC STOKES FLOW
#
# BATCH 5:
# Full standalone DeepONet training experiment
#
# SAME:
#   - Stokes dataset
#   - train / validation / test split
#   - input normalization
#   - relative L2 metric
#   - optimizer conventions
#   - 200 epoch training budget
#
# as the baseline FNO experiment where applicable.
#
# The test set is evaluated ONLY after training/model selection.
# ============================================================


# ============================================================
# 0. IMPORTS
# ============================================================

import os
import time
import json
import copy

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

from torch.utils.data import TensorDataset, DataLoader


# ============================================================
# 1. CONFIGURATION
# ============================================================

SEED = 0

# ------------------------------------------------------------
# Dataset split -- same as FNO FULL_CONFIG baseline
# ------------------------------------------------------------

N_TRAIN = 1500
N_VAL = 250
N_TEST = 250


# ------------------------------------------------------------
# DeepONet architecture verified in Batch 3
# ------------------------------------------------------------

LATENT_DIM = 128
BRANCH_WIDTH = 128
TRUNK_WIDTH = 128


# ------------------------------------------------------------
# Training -- aligned with FNO conventions
# ------------------------------------------------------------

BATCH_SIZE = 20

EPOCHS = 200

LEARNING_RATE = 1e-3

WEIGHT_DECAY = 1e-4


# ------------------------------------------------------------
# Reproducibility
# ------------------------------------------------------------

torch.manual_seed(SEED)
np.random.seed(SEED)


if torch.cuda.is_available():

    torch.cuda.manual_seed_all(
        SEED
    )


# ============================================================
# 2. DEVICE
# ============================================================

if torch.cuda.is_available():

    DEVICE = torch.device(
        "cuda"
    )

elif torch.backends.mps.is_available():

    DEVICE = torch.device(
        "mps"
    )

else:

    DEVICE = torch.device(
        "cpu"
    )


print(
    "\n"
    +
    "=" * 75
)

print(
    "DEVICE"
)

print(
    "=" * 75
)


print(
    "PyTorch:",
    torch.__version__
)


print(
    "Device:",
    DEVICE
)


if DEVICE.type == "cuda":

    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )


print(
    "=" * 75
)


# ============================================================
# 3. PATHS
# ============================================================

HERE = os.path.dirname(
    os.path.abspath(__file__)
)


DATASET_PATH = os.path.join(

    HERE,

    "fno_data",

    "stokes2d_base.npz"

)


RESULTS_DIR = os.path.join(

    HERE,

    "deeponet_results"

)


os.makedirs(
    RESULTS_DIR,
    exist_ok=True
)


CHECKPOINT_PATH = os.path.join(

    RESULTS_DIR,

    "deeponet_stokes_best.pt"

)


HISTORY_PATH = os.path.join(

    RESULTS_DIR,

    "deeponet_training_history.npz"

)


CONFIG_PATH = os.path.join(

    RESULTS_DIR,

    "deeponet_config.json"

)


SUMMARY_PATH = os.path.join(

    RESULTS_DIR,

    "deeponet_summary.json"

)


# ============================================================
# 4. LOAD EXACT STOKES DATASET
# ============================================================

def load_dataset(
    npz_path
):

    if not os.path.exists(
        npz_path
    ):

        raise FileNotFoundError(

            "\nCould not find dataset:\n"

            f"{npz_path}"

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
            "Requested split exceeds dataset size."
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
        ),

        (
            idx_train,

            idx_val,

            idx_test
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
    # Normalize forcing inputs
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
    # Branch input:
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
    # Trunk coordinates:
    #
    # 16384 x 2
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
    # Targets:
    #
    # N x 2 x Nx x Ny
    #
    # ->
    #
    # N x (Nx*Ny) x 2
    # --------------------------------------------------------

    def convert_target(
        target
    ):

        return (

            target

            .permute(
                0,
                2,
                3,
                1
            )

            .reshape(
                target.shape[0],
                Nx * Ny,
                2
            )

        )


    target_train = convert_target(
        u_train
    )


    target_val = convert_target(
        u_val
    )


    target_test = convert_target(
        u_test
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

        branch_output = self.branch(
            branch_input
        )


        trunk_output = self.trunk(
            trunk_coordinates
        )


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
# 11. RELATIVE L2
#
# Same per-sample metric as FNO.
# ============================================================

def relative_l2(

    prediction,

    target

):

    batch = (
        prediction.shape[0]
    )


    prediction_flat = (
        prediction.reshape(
            batch,
            -1
        )
    )


    target_flat = (
        target.reshape(
            batch,
            -1
        )
    )


    numerator = torch.norm(

        prediction_flat
        -
        target_flat,

        dim=1

    )


    denominator = (

        torch.norm(

            target_flat,

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

        p.numel()

        for p in model.parameters()

        if p.requires_grad

    )


# ============================================================
# 13. DEVICE SYNCHRONIZATION
# ============================================================

def synchronize_device():

    if DEVICE.type == "cuda":

        torch.cuda.synchronize()


    elif DEVICE.type == "mps":

        torch.mps.synchronize()


# ============================================================
# 14. EVALUATION FUNCTION
#
# Evaluate in batches so validation/test does not require
# storing the entire prediction tensor on the GPU.
# ============================================================

def evaluate_model(

    model,

    branch_data,

    target_data,

    trunk_device,

    batch_size

):

    dataset = TensorDataset(

        branch_data,

        target_data

    )


    loader = DataLoader(

        dataset,

        batch_size=batch_size,

        shuffle=False,

        num_workers=0

    )


    model.eval()


    total_error = 0.0

    n_samples = 0


    with torch.no_grad():


        for (
            branch_batch,
            target_batch
        ) in loader:


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


            prediction = model(

                branch_batch,

                trunk_device

            )


            errors = relative_l2(

                prediction,

                target_batch

            )


            total_error += (
                errors.sum().item()
            )


            n_samples += (
                branch_batch.shape[0]
            )


    return (

        total_error

        /

        n_samples

    )


# ============================================================
# 15. SAVE TRAINING CURVE
# ============================================================

def save_training_curve(

    train_history,

    val_history,

    save_path

):

    plt.figure(
        figsize=(7, 5)
    )


    plt.plot(

        range(
            1,
            len(
                train_history
            )
            +
            1
        ),

        train_history,

        label="Training"

    )


    plt.plot(

        range(
            1,
            len(
                val_history
            )
            +
            1
        ),

        val_history,

        label="Validation"

    )


    plt.xlabel(
        "Epoch"
    )


    plt.ylabel(
        "Relative $L^2$ Error"
    )


    plt.title(
        "DeepONet Training on 2-D Stokes Flow"
    )


    plt.legend()


    plt.tight_layout()


    plt.savefig(

        save_path,

        dpi=200

    )


    plt.close()


# ============================================================
# 16. MAIN
# ============================================================

def main():

    # ========================================================
    # LOAD DATA
    # ========================================================

    print(
        "\nLoading exact Stokes dataset..."
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


    print(
        f"Lx={metadata['Lx']}, "
        f"Ly={metadata['Ly']}, "
        f"nu={metadata['nu']}"
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


    # ========================================================
    # NORMALIZATION
    # ========================================================

    normalizer = InputNormalizer(
        f_train
    )


    # ========================================================
    # DEEPONET REPRESENTATION
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


    # ========================================================
    # TRAINING DATA LOADER
    # ========================================================

    train_dataset = TensorDataset(

        branch_train,

        target_train

    )


    train_loader = DataLoader(

        train_dataset,

        batch_size=BATCH_SIZE,

        shuffle=True,

        num_workers=0

    )


    # ========================================================
    # CREATE MODEL
    # ========================================================

    model = DeepONet(

        branch_input_dim=
            branch_train.shape[1],

        branch_width=
            BRANCH_WIDTH,

        trunk_width=
            TRUNK_WIDTH,

        latent_dim=
            LATENT_DIM

    ).to(
        DEVICE
    )


    parameter_count = (
        count_parameters(
            model
        )
    )


    trunk_device = (
        trunk.to(
            DEVICE
        )
    )


    # ========================================================
    # OPTIMIZER + SCHEDULER
    # ========================================================

    optimizer = torch.optim.Adam(

        model.parameters(),

        lr=LEARNING_RATE,

        weight_decay=WEIGHT_DECAY

    )


    scheduler = (

        torch.optim.lr_scheduler

        .CosineAnnealingLR(

            optimizer,

            T_max=EPOCHS

        )

    )


    # ========================================================
    # CONFIGURATION REPORT
    # ========================================================

    print(
        "\n"
        +
        "=" * 75
    )

    print(
        "FULL DEEPONET STOKES TRAINING"
    )

    print(
        "=" * 75
    )


    print(
        "Device:",
        DEVICE
    )


    print(
        "Parameters:",
        f"{parameter_count:,}"
    )


    print(
        "Resolution:",
        f"{Nx} x {Ny}"
    )


    print(
        "Train / val / test:",
        f"{N_TRAIN} / "
        f"{N_VAL} / "
        f"{N_TEST}"
    )


    print(
        "Branch width:",
        BRANCH_WIDTH
    )


    print(
        "Trunk width:",
        TRUNK_WIDTH
    )


    print(
        "Latent dimension:",
        LATENT_DIM
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


    print(
        "=" * 75
    )


    # ========================================================
    # SAVE CONFIGURATION
    # ========================================================

    configuration = {

        "seed":
            SEED,

        "n_train":
            N_TRAIN,

        "n_val":
            N_VAL,

        "n_test":
            N_TEST,

        "resolution":
            [
                Nx,
                Ny
            ],

        "latent_dim":
            LATENT_DIM,

        "branch_width":
            BRANCH_WIDTH,

        "trunk_width":
            TRUNK_WIDTH,

        "batch_size":
            BATCH_SIZE,

        "epochs":
            EPOCHS,

        "learning_rate":
            LEARNING_RATE,

        "weight_decay":
            WEIGHT_DECAY,

        "parameters":
            parameter_count,

        "Lx":
            metadata["Lx"],

        "Ly":
            metadata["Ly"],

        "nu":
            metadata["nu"]

    }


    with open(

        CONFIG_PATH,

        "w"

    ) as file:

        json.dump(

            configuration,

            file,

            indent=4

        )


    # ========================================================
    # TRAINING STATE
    # ========================================================

    train_history = []

    val_history = []


    best_val_error = (
        float("inf")
    )


    best_epoch = (
        -1
    )


    best_model_state = (
        None
    )


    # ========================================================
    # TRAIN
    # ========================================================

    print(
        "\nBeginning training...\n"
    )


    synchronize_device()


    training_start = (
        time.perf_counter()
    )


    for epoch in range(
        1,
        EPOCHS + 1
    ):


        synchronize_device()


        epoch_start = (
            time.perf_counter()
        )


        # ====================================================
        # TRAINING PHASE
        # ====================================================

        model.train()


        total_train_error = (
            0.0
        )


        n_seen = (
            0
        )


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


            sample_errors = (
                relative_l2(

                    prediction,

                    target_batch

                )
            )


            loss = (
                sample_errors.mean()
            )


            # ------------------------------------------------
            # Backpropagation
            # ------------------------------------------------

            optimizer.zero_grad()


            loss.backward()


            optimizer.step()


            total_train_error += (
                sample_errors
                .detach()
                .sum()
                .item()
            )


            n_seen += (
                branch_batch.shape[0]
            )


        train_error = (

            total_train_error

            /

            n_seen

        )


        # ====================================================
        # VALIDATION
        # ====================================================

        val_error = evaluate_model(

            model,

            branch_val,

            target_val,

            trunk_device,

            BATCH_SIZE

        )


        # ====================================================
        # CHECKPOINT BEST VALIDATION MODEL
        # ====================================================

        if (
            val_error
            <
            best_val_error
        ):

            best_val_error = (
                val_error
            )


            best_epoch = (
                epoch
            )


            # Keep a CPU copy in memory.
            best_model_state = {

                key:
                    value
                    .detach()
                    .cpu()
                    .clone()

                for (
                    key,
                    value
                )
                in model.state_dict().items()

            }


            torch.save(

                {

                    "epoch":
                        best_epoch,

                    "model_state_dict":
                        best_model_state,

                    "validation_error":
                        best_val_error,

                    "normalizer_mean":
                        normalizer.mean,

                    "normalizer_std":
                        normalizer.std,

                    "configuration":
                        configuration

                },

                CHECKPOINT_PATH

            )


        # ====================================================
        # SCHEDULER
        # ====================================================

        scheduler.step()


        # ====================================================
        # HISTORY
        # ====================================================

        train_history.append(
            train_error
        )


        val_history.append(
            val_error
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


        # ----------------------------------------------------
        # Print first epoch and every 10 epochs.
        # ----------------------------------------------------

        if (

            epoch == 1

            or

            epoch % 10 == 0

            or

            epoch == EPOCHS

        ):

            print(

                f"Epoch "
                f"{epoch:03d}/{EPOCHS} | "

                f"train = "
                f"{train_error:.6f} | "

                f"val = "
                f"{val_error:.6f} | "

                f"best val = "
                f"{best_val_error:.6f} "
                f"(epoch {best_epoch}) | "

                f"lr = "
                f"{current_lr:.3e} | "

                f"time = "
                f"{epoch_time:.2f}s"

            )


    # ========================================================
    # TRAINING TIME
    # ========================================================

    synchronize_device()


    training_time = (

        time.perf_counter()

        -

        training_start

    )


    # ========================================================
    # SAVE HISTORY
    # ========================================================

    np.savez(

        HISTORY_PATH,

        train_error=np.array(
            train_history
        ),

        val_error=np.array(
            val_history
        )

    )


    # ========================================================
    # SAVE TRAINING CURVE
    # ========================================================

    training_curve_path = os.path.join(

        RESULTS_DIR,

        "deeponet_training_curve.png"

    )


    save_training_curve(

        train_history,

        val_history,

        training_curve_path

    )


    # ========================================================
    # RESTORE BEST VALIDATION MODEL
    # ========================================================

    print(
        "\nRestoring best validation model..."
    )


    if (
        best_model_state
        is None
    ):

        raise RuntimeError(
            "No validation checkpoint was created."
        )


    model.load_state_dict(
        best_model_state
    )


    model = model.to(
        DEVICE
    )


    # ========================================================
    # FINAL TEST EVALUATION
    #
    # This is the first and only test evaluation in this
    # training script.
    # ========================================================

    print(
        "Evaluating untouched test set..."
    )


    test_error = evaluate_model(

        model,

        branch_test,

        target_test,

        trunk_device,

        BATCH_SIZE

    )


    # ========================================================
    # SUMMARY
    # ========================================================

    summary = {

        "best_epoch":
            best_epoch,

        "best_validation_relative_l2":
            best_val_error,

        "test_relative_l2":
            test_error,

        "final_training_relative_l2":
            train_history[-1],

        "training_time_seconds":
            training_time,

        "parameters":
            parameter_count,

        "device":
            str(
                DEVICE
            )

    }


    if DEVICE.type == "cuda":

        summary[
            "gpu"
        ] = torch.cuda.get_device_name(
            0
        )


        summary[
            "peak_cuda_memory_mb"
        ] = (

            torch.cuda
            .max_memory_allocated()

            /

            (1024 ** 2)

        )


    with open(

        SUMMARY_PATH,

        "w"

    ) as file:

        json.dump(

            summary,

            file,

            indent=4

        )


    # ========================================================
    # FINAL TERMINAL REPORT
    # ========================================================

    print(
        "\n"
        +
        "=" * 75
    )

    print(
        "DEEPONET TRAINING COMPLETE"
    )

    print(
        "=" * 75
    )


    print(
        f"\nBest epoch: "
        f"{best_epoch}"
    )


    print(
        f"Best validation relative L2: "
        f"{best_val_error:.6f}"
    )


    print(
        f"Final test relative L2: "
        f"{test_error:.6f}"
    )


    print(
        f"Final epoch training relative L2: "
        f"{train_history[-1]:.6f}"
    )


    print(
        f"\nTotal training time: "
        f"{training_time:.2f} seconds"
    )


    print(
        f"Parameters: "
        f"{parameter_count:,}"
    )


    if DEVICE.type == "cuda":

        print(
            f"Peak CUDA memory: "
            f"{summary['peak_cuda_memory_mb']:.2f} MB"
        )


    print(
        "\nSaved best model:"
    )

    print(
        CHECKPOINT_PATH
    )


    print(
        "\nSaved history:"
    )

    print(
        HISTORY_PATH
    )


    print(
        "\nSaved training curve:"
    )

    print(
        training_curve_path
    )


    print(
        "\nSaved summary:"
    )

    print(
        SUMMARY_PATH
    )


    print(
        "\n"
        +
        "=" * 75
    )


# ============================================================
# 17. RUN
# ============================================================

if __name__ == "__main__":

    main()