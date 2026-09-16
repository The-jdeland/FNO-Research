# ============================================================
# DEEPONET FOR 2-D PERIODIC STOKES FLOW
#
# BATCH 6:
# Standalone diagnostics for the trained DeepONet.
#
# NO TRAINING.
#
# Loads:
#   fno_data/stokes2d_base.npz
#   deeponet_results/deeponet_stokes_best.pt
#
# Produces diagnostic plots for the held-out test set.
# ============================================================


# ============================================================
# 0. IMPORTS
# ============================================================

import os
import json
import numpy as np

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt

import torch
import torch.nn as nn

from torch.utils.data import TensorDataset, DataLoader


# ============================================================
# 1. CONFIGURATION
# ============================================================

SEED = 0

N_TRAIN = 1500
N_VAL = 250
N_TEST = 250

BATCH_SIZE = 20

# Test samples to visualize.
# These are positions WITHIN the held-out test set.
EXAMPLE_INDICES = [
    0,
    1,
    2
]


# ============================================================
# 2. DEVICE
# ============================================================

if torch.cuda.is_available():

    DEVICE = torch.device("cuda")

elif torch.backends.mps.is_available():

    DEVICE = torch.device("mps")

else:

    DEVICE = torch.device("cpu")


print("\n" + "=" * 75)
print("DEEPONET BATCH 6 DIAGNOSTICS")
print("=" * 75)

print(
    "Device:",
    DEVICE
)

if DEVICE.type == "cuda":

    print(
        "GPU:",
        torch.cuda.get_device_name(0)
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


CHECKPOINT_PATH = os.path.join(
    RESULTS_DIR,
    "deeponet_stokes_best.pt"
)


HISTORY_PATH = os.path.join(
    RESULTS_DIR,
    "deeponet_training_history.npz"
)


DIAGNOSTICS_DIR = os.path.join(
    RESULTS_DIR,
    "diagnostics"
)


os.makedirs(
    DIAGNOSTICS_DIR,
    exist_ok=True
)


# ============================================================
# 4. LOAD DATASET
# ============================================================

def load_dataset(
    npz_path
):

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
# 5. EXACT SAME SPLIT
# ============================================================

def split_dataset(

    forcing,
    velocity,

    n_train,
    n_val,
    n_test,

    seed=0

):

    generator = (
        torch.Generator()
        .manual_seed(
            seed
        )
    )


    permutation = torch.randperm(

        forcing.shape[0],

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
# 6. NORMALIZATION
# ============================================================

class InputNormalizer:

    def __init__(
        self,
        mean,
        std
    ):

        self.mean = mean
        self.std = std


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
# 7. BRANCH NETWORK
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
# 8. TRUNK NETWORK
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
# 9. DEEPONET
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
# 10. RELATIVE L2
# ============================================================

def relative_l2(
    prediction,
    target
):

    batch = prediction.shape[0]


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
# 11. BUILD TRUNK COORDINATES
# ============================================================

def build_trunk(
    Nx,
    Ny
):

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


    return (
        trunk,
        X,
        Y
    )


# ============================================================
# 12. DIVERGENCE
#
# Spectral divergence on the periodic domain.
# ============================================================

def spectral_divergence(

    velocity,

    Lx,

    Ly

):

    # velocity shape:
    #
    #       2 x Nx x Ny

    Nx = velocity.shape[-2]
    Ny = velocity.shape[-1]


    dx = (
        Lx
        /
        Nx
    )


    dy = (
        Ly
        /
        Ny
    )


    kx = (
        2
        *
        np.pi
        *
        np.fft.fftfreq(
            Nx,
            d=dx
        )
    )


    ky = (
        2
        *
        np.pi
        *
        np.fft.fftfreq(
            Ny,
            d=dy
        )
    )


    KX, KY = np.meshgrid(

        kx,

        ky,

        indexing="ij"

    )


    ux = velocity[0]
    uy = velocity[1]


    ux_hat = np.fft.fft2(
        ux
    )


    uy_hat = np.fft.fft2(
        uy
    )


    div_hat = (

        1j
        *
        KX
        *
        ux_hat

        +

        1j
        *
        KY
        *
        uy_hat

    )


    divergence = np.fft.ifft2(
        div_hat
    ).real


    return divergence


# ============================================================
# 13. TRAINING CURVE
# ============================================================

def plot_training_history():

    if not os.path.exists(
        HISTORY_PATH
    ):

        print(
            "Training history not found; skipping curve."
        )

        return


    history = np.load(
        HISTORY_PATH
    )


    train_error = history[
        "train_error"
    ]


    val_error = history[
        "val_error"
    ]


    epochs = np.arange(
        1,
        len(
            train_error
        )
        +
        1
    )


    plt.figure(
        figsize=(7, 5)
    )


    plt.plot(
        epochs,
        train_error,
        label="Training"
    )


    plt.plot(
        epochs,
        val_error,
        label="Validation"
    )


    plt.xlabel(
        "Epoch"
    )


    plt.ylabel(
        "Relative $L^2$ Error"
    )


    plt.title(
        "DeepONet Training and Validation Error"
    )


    plt.legend()


    plt.tight_layout()


    plt.savefig(

        os.path.join(
            DIAGNOSTICS_DIR,
            "training_validation_curve.png"
        ),

        dpi=200

    )


    plt.close()


# ============================================================
# 14. MAIN
# ============================================================

def main():

    # ========================================================
    # CHECK FILES
    # ========================================================

    if not os.path.exists(
        CHECKPOINT_PATH
    ):

        raise FileNotFoundError(
            f"Checkpoint not found:\n{CHECKPOINT_PATH}"
        )


    if not os.path.exists(
        DATASET_PATH
    ):

        raise FileNotFoundError(
            f"Dataset not found:\n{DATASET_PATH}"
        )


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


    Nx = (
        f_test.shape[-2]
    )

    Ny = (
        f_test.shape[-1]
    )


    print(
        "Test samples:",
        f_test.shape[0]
    )


    print(
        "Resolution:",
        f"{Nx} x {Ny}"
    )


    # ========================================================
    # LOAD CHECKPOINT
    # ========================================================

    checkpoint = torch.load(

        CHECKPOINT_PATH,

        map_location="cpu",

        weights_only=False

    )


    configuration = checkpoint[
        "configuration"
    ]


    print(
        "\nCheckpoint epoch:",
        checkpoint[
            "epoch"
        ]
    )


    print(
        "Checkpoint validation error:",
        checkpoint[
            "validation_error"
        ]
    )


    # ========================================================
    # NORMALIZER FROM TRAINING CHECKPOINT
    # ========================================================

    normalizer = InputNormalizer(

        checkpoint[
            "normalizer_mean"
        ],

        checkpoint[
            "normalizer_std"
        ]

    )


    # ========================================================
    # TEST BRANCH INPUT
    # ========================================================

    f_test_normalized = (
        normalizer.encode(
            f_test
        )
    )


    branch_test = (
        f_test_normalized.reshape(
            N_TEST,
            -1
        )
    )


    # ========================================================
    # TEST TARGET
    # ========================================================

    target_test = (

        u_test

        .permute(
            0,
            2,
            3,
            1
        )

        .reshape(
            N_TEST,
            Nx * Ny,
            2
        )

    )


    # ========================================================
    # TRUNK
    # ========================================================

    (
        trunk,
        X,
        Y
    ) = build_trunk(
        Nx,
        Ny
    )


    trunk_device = (
        trunk.to(
            DEVICE
        )
    )


    # ========================================================
    # REBUILD MODEL
    # ========================================================

    model = DeepONet(

        branch_input_dim=
            branch_test.shape[1],

        branch_width=
            configuration[
                "branch_width"
            ],

        trunk_width=
            configuration[
                "trunk_width"
            ],

        latent_dim=
            configuration[
                "latent_dim"
            ]

    ).to(
        DEVICE
    )


    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )


    model.eval()


    # ========================================================
    # RUN TEST INFERENCE
    # ========================================================

    test_dataset = TensorDataset(

        branch_test,

        target_test

    )


    test_loader = DataLoader(

        test_dataset,

        batch_size=BATCH_SIZE,

        shuffle=False,

        num_workers=0

    )


    all_predictions = []
    all_targets = []
    all_errors = []


    print(
        "\nRunning inference on held-out test set..."
    )


    with torch.no_grad():


        for (
            branch_batch,
            target_batch
        ) in test_loader:


            branch_batch = (
                branch_batch.to(
                    DEVICE
                )
            )


            target_device = (
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

                target_device

            )


            all_predictions.append(
                prediction.cpu()
            )


            all_targets.append(
                target_batch
            )


            all_errors.append(
                errors.cpu()
            )


    predictions = torch.cat(
        all_predictions,
        dim=0
    )


    targets = torch.cat(
        all_targets,
        dim=0
    )


    test_errors = torch.cat(
        all_errors,
        dim=0
    ).numpy()


    # ========================================================
    # FIELD REPRESENTATION
    #
    # N x points x 2
    #
    # ->
    #
    # N x 2 x Nx x Ny
    # ========================================================

    prediction_fields = (

        predictions

        .reshape(
            N_TEST,
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

        .numpy()

    )


    target_fields = (

        targets

        .reshape(
            N_TEST,
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

        .numpy()

    )


    # ========================================================
    # TEST ERROR STATISTICS
    # ========================================================

    mean_error = float(
        np.mean(
            test_errors
        )
    )


    median_error = float(
        np.median(
            test_errors
        )
    )


    std_error = float(
        np.std(
            test_errors
        )
    )


    min_error = float(
        np.min(
            test_errors
        )
    )


    max_error = float(
        np.max(
            test_errors
        )
    )


    print(
        "\n"
        +
        "=" * 75
    )

    print(
        "TEST ERROR STATISTICS"
    )

    print(
        "=" * 75
    )


    print(
        f"Mean:   {mean_error:.6f}"
    )


    print(
        f"Median: {median_error:.6f}"
    )


    print(
        f"Std:    {std_error:.6f}"
    )


    print(
        f"Min:    {min_error:.6f}"
    )


    print(
        f"Max:    {max_error:.6f}"
    )


    # ========================================================
    # SAVE TEST ERRORS
    # ========================================================

    np.save(

        os.path.join(
            DIAGNOSTICS_DIR,
            "test_relative_l2_errors.npy"
        ),

        test_errors

    )


    # ========================================================
    # TRAINING CURVE
    # ========================================================

    plot_training_history()


    # ========================================================
    # TEST ERROR HISTOGRAM
    # ========================================================

    plt.figure(
        figsize=(7, 5)
    )


    plt.hist(
        test_errors,
        bins=25
    )


    plt.axvline(
        mean_error,
        linestyle="--",
        label=
            f"Mean = {mean_error:.3f}"
    )


    plt.xlabel(
        "Relative $L^2$ Error"
    )


    plt.ylabel(
        "Number of Test Samples"
    )


    plt.title(
        "DeepONet Test Error Distribution"
    )


    plt.legend()


    plt.tight_layout()


    plt.savefig(

        os.path.join(
            DIAGNOSTICS_DIR,
            "test_error_histogram.png"
        ),

        dpi=200

    )


    plt.close()


    # ========================================================
    # INDIVIDUAL TEST EXAMPLES
    # ========================================================

    x_physical = np.linspace(

        0.0,

        metadata["Lx"],

        Nx,

        endpoint=False

    )


    y_physical = np.linspace(

        0.0,

        metadata["Ly"],

        Ny,

        endpoint=False

    )


    X_phys, Y_phys = np.meshgrid(

        x_physical,

        y_physical,

        indexing="ij"

    )


    # Reduce arrow density for quiver plots.
    arrow_step = 8


    for example_index in EXAMPLE_INDICES:


        truth = target_fields[
            example_index
        ]


        prediction = prediction_fields[
            example_index
        ]


        ux_true = truth[0]
        uy_true = truth[1]


        ux_pred = prediction[0]
        uy_pred = prediction[1]


        true_speed = np.sqrt(

            ux_true**2

            +

            uy_true**2

        )


        pred_speed = np.sqrt(

            ux_pred**2

            +

            uy_pred**2

        )


        pointwise_error = np.sqrt(

            (
                ux_pred
                -
                ux_true
            )**2

            +

            (
                uy_pred
                -
                uy_true
            )**2

        )


        speed_max = max(

            float(
                true_speed.max()
            ),

            float(
                pred_speed.max()
            )

        )


        # ====================================================
        # TRUE VELOCITY MAGNITUDE
        # ====================================================

        plt.figure(
            figsize=(6, 5)
        )


        plt.imshow(

            true_speed.T,

            origin="lower",

            extent=[
                0,
                metadata["Lx"],
                0,
                metadata["Ly"]
            ],

            vmin=0,

            vmax=speed_max,

            aspect="equal"

        )


        plt.colorbar(
            label=r"$|\mathbf{u}|$"
        )


        plt.xlabel(
            r"$x$"
        )


        plt.ylabel(
            r"$y$"
        )


        plt.title(
            f"Exact Stokes Velocity — Test {example_index}"
        )


        plt.tight_layout()


        plt.savefig(

            os.path.join(

                DIAGNOSTICS_DIR,

                f"test_{example_index:03d}_exact_velocity.png"

            ),

            dpi=200

        )


        plt.close()


        # ====================================================
        # PREDICTED VELOCITY MAGNITUDE
        # ====================================================

        plt.figure(
            figsize=(6, 5)
        )


        plt.imshow(

            pred_speed.T,

            origin="lower",

            extent=[
                0,
                metadata["Lx"],
                0,
                metadata["Ly"]
            ],

            vmin=0,

            vmax=speed_max,

            aspect="equal"

        )


        plt.colorbar(
            label=r"$|\mathbf{u}_{\theta}|$"
        )


        plt.xlabel(
            r"$x$"
        )


        plt.ylabel(
            r"$y$"
        )


        plt.title(
            f"DeepONet Prediction — Test {example_index}"
        )


        plt.tight_layout()


        plt.savefig(

            os.path.join(

                DIAGNOSTICS_DIR,

                f"test_{example_index:03d}_predicted_velocity.png"

            ),

            dpi=200

        )


        plt.close()


        # ====================================================
        # POINTWISE ERROR
        # ====================================================

        plt.figure(
            figsize=(6, 5)
        )


        plt.imshow(

            pointwise_error.T,

            origin="lower",

            extent=[
                0,
                metadata["Lx"],
                0,
                metadata["Ly"]
            ],

            aspect="equal"

        )


        plt.colorbar(

            label=
                r"$|\mathbf{u}_{\theta}-\mathbf{u}|$"

        )


        plt.xlabel(
            r"$x$"
        )


        plt.ylabel(
            r"$y$"
        )


        plt.title(

            f"Pointwise Velocity Error — Test {example_index}\n"

            f"Relative $L^2$ = "
            f"{test_errors[example_index]:.4f}"

        )


        plt.tight_layout()


        plt.savefig(

            os.path.join(

                DIAGNOSTICS_DIR,

                f"test_{example_index:03d}_pointwise_error.png"

            ),

            dpi=200

        )


        plt.close()


        # ====================================================
        # EXACT VECTOR FIELD
        # ====================================================

        plt.figure(
            figsize=(6, 5)
        )


        plt.quiver(

            X_phys[
                ::arrow_step,
                ::arrow_step
            ],

            Y_phys[
                ::arrow_step,
                ::arrow_step
            ],

            ux_true[
                ::arrow_step,
                ::arrow_step
            ],

            uy_true[
                ::arrow_step,
                ::arrow_step
            ]

        )


        plt.xlim(
            0,
            metadata["Lx"]
        )


        plt.ylim(
            0,
            metadata["Ly"]
        )


        plt.xlabel(
            r"$x$"
        )


        plt.ylabel(
            r"$y$"
        )


        plt.title(
            f"Exact Stokes Vector Field — Test {example_index}"
        )


        plt.tight_layout()


        plt.savefig(

            os.path.join(

                DIAGNOSTICS_DIR,

                f"test_{example_index:03d}_exact_quiver.png"

            ),

            dpi=200

        )


        plt.close()


        # ====================================================
        # PREDICTED VECTOR FIELD
        # ====================================================

        plt.figure(
            figsize=(6, 5)
        )


        plt.quiver(

            X_phys[
                ::arrow_step,
                ::arrow_step
            ],

            Y_phys[
                ::arrow_step,
                ::arrow_step
            ],

            ux_pred[
                ::arrow_step,
                ::arrow_step
            ],

            uy_pred[
                ::arrow_step,
                ::arrow_step
            ]

        )


        plt.xlim(
            0,
            metadata["Lx"]
        )


        plt.ylim(
            0,
            metadata["Ly"]
        )


        plt.xlabel(
            r"$x$"
        )


        plt.ylabel(
            r"$y$"
        )


        plt.title(
            f"DeepONet Vector Field — Test {example_index}"
        )


        plt.tight_layout()


        plt.savefig(

            os.path.join(

                DIAGNOSTICS_DIR,

                f"test_{example_index:03d}_predicted_quiver.png"

            ),

            dpi=200

        )


        plt.close()


        # ====================================================
        # DIVERGENCE
        # ====================================================

        divergence = spectral_divergence(

            prediction,

            metadata["Lx"],

            metadata["Ly"]

        )


        plt.figure(
            figsize=(6, 5)
        )


        plt.imshow(

            np.abs(
                divergence
            ).T,

            origin="lower",

            extent=[
                0,
                metadata["Lx"],
                0,
                metadata["Ly"]
            ],

            aspect="equal"

        )


        plt.colorbar(
            label=
                r"$|\nabla\cdot\mathbf{u}_{\theta}|$"
        )


        plt.xlabel(
            r"$x$"
        )


        plt.ylabel(
            r"$y$"
        )


        plt.title(
            f"DeepONet Divergence — Test {example_index}"
        )


        plt.tight_layout()


        plt.savefig(

            os.path.join(

                DIAGNOSTICS_DIR,

                f"test_{example_index:03d}_divergence.png"

            ),

            dpi=200

        )


        plt.close()


    # ========================================================
    # SAVE DIAGNOSTIC SUMMARY
    # ========================================================

    diagnostic_summary = {

        "mean_test_relative_l2":
            mean_error,

        "median_test_relative_l2":
            median_error,

        "std_test_relative_l2":
            std_error,

        "minimum_test_relative_l2":
            min_error,

        "maximum_test_relative_l2":
            max_error,

        "checkpoint_epoch":
            int(
                checkpoint[
                    "epoch"
                ]
            ),

        "checkpoint_validation_error":
            float(
                checkpoint[
                    "validation_error"
                ]
            )

    }


    with open(

        os.path.join(
            DIAGNOSTICS_DIR,
            "diagnostic_summary.json"
        ),

        "w"

    ) as file:

        json.dump(

            diagnostic_summary,

            file,

            indent=4

        )


    # ========================================================
    # DONE
    # ========================================================

    print(
        "\n"
        +
        "=" * 75
    )


    print(
        "DIAGNOSTICS COMPLETE"
    )


    print(
        "=" * 75
    )


    print(
        "\nPlots saved to:"
    )


    print(
        DIAGNOSTICS_DIR
    )


    print(
        "\nMean test relative L2:",
        f"{mean_error:.6f}"
    )


    print(
        "=" * 75
    )


# ============================================================
# 15. RUN
# ============================================================

if __name__ == "__main__":

    main()