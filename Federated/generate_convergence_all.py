import csv
from pathlib import Path

import torch


# ============================================================
# NEUROVERSE
# Reconstruct Round-wise Global Models
#
# E1 = 3D U-Net + FedAvg
# E2 = Swin UNETR + FedAvg
# E3 = Swin UNETR + Adaptive FL
#
# This script DOES NOT train or modify any experiment.
# It only reconstructs the global model for each round
# from the already-saved client checkpoints.
# ============================================================


# ============================================================
# PATHS
# ============================================================

BASE = Path.home() / "Neuroverse" / "Federated"

E1 = BASE / "FedAvg_BraTS"
E2 = BASE / "Swin_FedAvg"
E3 = BASE / "Swin_Adaptive_FL"

OUTPUT_DIR = BASE / "Convergence_Graphs"

RECON_DIR = (
    OUTPUT_DIR /
    "reconstructed_global_models"
)

NUM_ROUNDS = 10

CLIENTS = [
    "C1",
    "C2",
    "C3"
]

# Training examples used by FedAvg
CLIENT_SAMPLE_COUNTS = {
    "C1": 284,
    "C2": 283,
    "C3": 283
}


# ============================================================
# CREATE OUTPUT DIRECTORIES
# ============================================================

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

RECON_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# HEADER
# ============================================================

print()
print("=" * 75)
print("NEUROVERSE - ROUND-WISE GLOBAL MODEL RECONSTRUCTION")
print("=" * 75)
print()

print("E1 : 3D U-Net + FedAvg")
print("E2 : Swin UNETR + FedAvg")
print("E3 : Swin UNETR + Adaptive FL")
print()

print("Number of rounds:", NUM_ROUNDS)
print()

print("Output directory:")
print(RECON_DIR)
print()


# ============================================================
# LOAD CHECKPOINT
# ============================================================

def load_checkpoint(path):
    """
    Load a client checkpoint.

    E1 format:
        {
            'round': ...,
            'client': ...,
            'model_state_dict': ...,
            'train_history': ...
        }

    E2/E3 format:
        OrderedDict containing model parameters.
    """

    checkpoint = torch.load(
        path,
        map_location="cpu",
        weights_only=False
    )

    # E1 checkpoint format
    if (
        isinstance(checkpoint, dict)
        and "model_state_dict" in checkpoint
    ):
        return checkpoint["model_state_dict"]

    # E2/E3 format
    return checkpoint


# ============================================================
# VALIDATE STATE DICTIONARIES
# ============================================================

def validate_state_dicts(states, experiment, round_number):

    if len(states) != 3:
        raise ValueError(
            f"{experiment} Round {round_number}: "
            f"Expected 3 client states, got {len(states)}"
        )

    reference_keys = list(
        states[0].keys()
    )

    for index, state in enumerate(states):

        client = CLIENTS[index]

        if list(state.keys()) != reference_keys:

            raise ValueError(
                f"\nState-dict mismatch!\n"
                f"Experiment: {experiment}\n"
                f"Round: {round_number}\n"
                f"Client: {client}\n"
                f"The model parameters do not match."
            )

    return True


# ============================================================
# LOAD THREE CLIENT STATES
# ============================================================

def get_client_states(
    experiment_path,
    experiment_name,
    round_number
):

    states = []

    print(
        f"\n  Loading {experiment_name} "
        f"Round {round_number:02d}"
    )

    for client in CLIENTS:

        checkpoint_path = (
            experiment_path
            / "results"
            / "client_models"
            / client
            / f"round_{round_number:02d}.pth"
        )

        if not checkpoint_path.exists():

            raise FileNotFoundError(
                "\nMissing checkpoint:\n"
                f"{checkpoint_path}"
            )

        print(
            f"    {client}: "
            f"{checkpoint_path.name}"
        )

        state = load_checkpoint(
            checkpoint_path
        )

        states.append(state)

    validate_state_dicts(
        states,
        experiment_name,
        round_number
    )

    return states


# ============================================================
# AGGREGATE STATE DICTIONARIES
# ============================================================

def aggregate_state_dict(
    states,
    weights
):
    """
    Weighted aggregation of model parameters.

    Used for:
        E1 FedAvg
        E2 FedAvg
        E3 Adaptive FL
    """

    if len(states) != len(weights):

        raise ValueError(
            "Number of states and weights "
            "must match."
        )

    total_weight = sum(weights)

    if total_weight <= 0:

        raise ValueError(
            "Total aggregation weight must "
            "be greater than zero."
        )

    # Normalize weights so numerical rounding
    # cannot affect aggregation.
    normalized_weights = [
        weight / total_weight
        for weight in weights
    ]

    result = {}

    reference_state = states[0]

    for key in reference_state.keys():

        tensors = [
            state[key]
            for state in states
        ]

        reference_tensor = tensors[0]

        # ----------------------------------------------------
        # Floating-point tensors
        # ----------------------------------------------------

        if torch.is_floating_point(
            reference_tensor
        ):

            aggregated = torch.zeros_like(
                reference_tensor,
                dtype=torch.float32
            )

            for tensor, weight in zip(
                tensors,
                normalized_weights
            ):

                aggregated += (
                    tensor.float()
                    * weight
                )

            result[key] = aggregated.to(
                reference_tensor.dtype
            )

        # ----------------------------------------------------
        # Non-floating tensors
        #
        # Swin contains buffers such as relative position
        # indices that should not be numerically averaged.
        # ----------------------------------------------------

        else:

            result[key] = (
                reference_tensor.clone()
            )

    return result


# ============================================================
# E1 RECONSTRUCTION
# ============================================================

def reconstruct_e1(round_number):

    states = get_client_states(
        E1,
        "E1",
        round_number
    )

    weights = [
        CLIENT_SAMPLE_COUNTS["C1"],
        CLIENT_SAMPLE_COUNTS["C2"],
        CLIENT_SAMPLE_COUNTS["C3"]
    ]

    print(
        "    FedAvg weights: "
        f"C1={weights[0]}, "
        f"C2={weights[1]}, "
        f"C3={weights[2]}"
    )

    global_state = aggregate_state_dict(
        states,
        weights
    )

    return global_state


# ============================================================
# E2 RECONSTRUCTION
# ============================================================

def reconstruct_e2(round_number):

    states = get_client_states(
        E2,
        "E2",
        round_number
    )

    weights = [
        CLIENT_SAMPLE_COUNTS["C1"],
        CLIENT_SAMPLE_COUNTS["C2"],
        CLIENT_SAMPLE_COUNTS["C3"]
    ]

    print(
        "    FedAvg weights: "
        f"C1={weights[0]}, "
        f"C2={weights[1]}, "
        f"C3={weights[2]}"
    )

    global_state = aggregate_state_dict(
        states,
        weights
    )

    return global_state


# ============================================================
# LOAD E3 ADAPTIVE WEIGHTS
# ============================================================

def load_e3_weights():

    csv_path = (
        E3
        / "results"
        / "adaptive_weights.csv"
    )

    if not csv_path.exists():

        raise FileNotFoundError(
            "\nE3 adaptive weight file not found:\n"
            f"{csv_path}"
        )

    print()
    print("=" * 75)
    print("READING E3 ADAPTIVE WEIGHTS")
    print("=" * 75)
    print()

    # --------------------------------------------------------
    # IMPORTANT:
    #
    # E3 Round 1 uses equal aggregation.
    # The adaptive weights start from Round 2.
    # --------------------------------------------------------

    weights = {}

    weights[1] = {
        "C1": 1.0 / 3.0,
        "C2": 1.0 / 3.0,
        "C3": 1.0 / 3.0
    }

    print(
        "Round 01: "
        "C1=0.333333, "
        "C2=0.333333, "
        "C3=0.333333, "
        "sum=1.000000"
    )

    # --------------------------------------------------------
    # READ CSV
    # --------------------------------------------------------

    with open(
        csv_path,
        "r",
        newline=""
    ) as file:

        reader = csv.DictReader(file)

        fieldnames = reader.fieldnames

        if fieldnames is None:

            raise ValueError(
                "adaptive_weights.csv "
                "does not contain a header."
            )

        print(
            "CSV columns:"
        )

        print(
            fieldnames
        )

        print()

        # ----------------------------------------------------
        # Find columns
        # ----------------------------------------------------

        round_column = None
        client_column = None
        weight_column = None

        for name in fieldnames:

            clean_name = (
                name
                .strip()
                .lower()
            )

            if clean_name == "round":

                round_column = name

            elif clean_name == "client":

                client_column = name

            elif clean_name == "aggregation_weight":

                weight_column = name

        # ----------------------------------------------------
        # Validate columns
        # ----------------------------------------------------

        if round_column is None:

            raise ValueError(
                "Could not find 'round' "
                "column in adaptive_weights.csv."
            )

        if client_column is None:

            raise ValueError(
                "Could not find 'client' "
                "column in adaptive_weights.csv."
            )

        if weight_column is None:

            raise ValueError(
                "Could not find "
                "'aggregation_weight' "
                "column in adaptive_weights.csv."
            )

        print(
            "Detected weight column:",
            weight_column
        )

        print()

        # ----------------------------------------------------
        # Read adaptive weights
        # ----------------------------------------------------

        for row in reader:

            round_number = int(
                row[round_column]
            )

            client = (
                row[client_column]
                .strip()
            )

            weight = float(
                row[weight_column]
            )

            # Round 1 is already explicitly
            # defined as equal weighting.
            if round_number == 1:
                continue

            if round_number not in weights:

                weights[round_number] = {}

            weights[
                round_number
            ][client] = weight

    # --------------------------------------------------------
    # Verify Rounds 2-10
    # --------------------------------------------------------

    print(
        "E3 adaptive aggregation weights:"
    )

    print()

    for round_number in range(
        2,
        NUM_ROUNDS + 1
    ):

        if round_number not in weights:

            raise ValueError(
                f"E3 adaptive weights "
                f"missing Round {round_number}."
            )

        for client in CLIENTS:

            if client not in weights[
                round_number
            ]:

                raise ValueError(
                    f"E3 adaptive weight "
                    f"missing for "
                    f"Round {round_number}, "
                    f"{client}."
                )

        c1 = weights[
            round_number
        ]["C1"]

        c2 = weights[
            round_number
        ]["C2"]

        c3 = weights[
            round_number
        ]["C3"]

        total = c1 + c2 + c3

        # The adaptive weights should sum to 1.
        if abs(total - 1.0) > 1e-4:

            raise ValueError(
                f"E3 Round {round_number} "
                f"weights sum to {total}, "
                f"not 1.0."
            )

        print(
            f"Round {round_number:02d}: "
            f"C1={c1:.6f}, "
            f"C2={c2:.6f}, "
            f"C3={c3:.6f}, "
            f"sum={total:.6f}"
        )

    print()

    return weights


# ============================================================
# E3 RECONSTRUCTION
# ============================================================

def reconstruct_e3(
    round_number,
    adaptive_weights
):

    states = get_client_states(
        E3,
        "E3",
        round_number
    )

    round_weights = (
        adaptive_weights[
            round_number
        ]
    )

    weights = [
        round_weights["C1"],
        round_weights["C2"],
        round_weights["C3"]
    ]

    print(
        "    Adaptive weights: "
        f"C1={weights[0]:.6f}, "
        f"C2={weights[1]:.6f}, "
        f"C3={weights[2]:.6f}"
    )

    global_state = aggregate_state_dict(
        states,
        weights
    )

    return global_state


# ============================================================
# VERIFY ALL CLIENT CHECKPOINTS
# ============================================================

print()
print("=" * 75)
print("CHECKING ALL CLIENT CHECKPOINTS")
print("=" * 75)
print()

experiments = [
    (E1, "E1"),
    (E2, "E2"),
    (E3, "E3")
]

for experiment_path, experiment_name in experiments:

    print(
        f"Checking {experiment_name}..."
    )

    for round_number in range(
        1,
        NUM_ROUNDS + 1
    ):

        for client in CLIENTS:

            checkpoint_path = (
                experiment_path
                / "results"
                / "client_models"
                / client
                / f"round_{round_number:02d}.pth"
            )

            if not checkpoint_path.exists():

                raise FileNotFoundError(
                    "\nMissing checkpoint:\n"
                    f"{checkpoint_path}"
                )

    print(
        f"{experiment_name}: "
        "all 30 expected client checkpoints "
        "are present."
    )

print()


# ============================================================
# LOAD E3 WEIGHTS
# ============================================================

e3_adaptive_weights = (
    load_e3_weights()
)


# ============================================================
# RECONSTRUCT GLOBAL MODELS
# ============================================================

print()
print("=" * 75)
print("RECONSTRUCTING GLOBAL MODELS")
print("=" * 75)
print()

global_models = {
    "E1": {},
    "E2": {},
    "E3": {}
}


for round_number in range(
    1,
    NUM_ROUNDS + 1
):

    print()
    print("-" * 75)
    print(
        f"ROUND {round_number:02d}"
    )
    print("-" * 75)

    # --------------------------------------------------------
    # E1
    # --------------------------------------------------------

    global_models["E1"][
        round_number
    ] = reconstruct_e1(
        round_number
    )

    print(
        "    E1 global model reconstructed."
    )

    # --------------------------------------------------------
    # E2
    # --------------------------------------------------------

    global_models["E2"][
        round_number
    ] = reconstruct_e2(
        round_number
    )

    print(
        "    E2 global model reconstructed."
    )

    # --------------------------------------------------------
    # E3
    # --------------------------------------------------------

    global_models["E3"][
        round_number
    ] = reconstruct_e3(
        round_number,
        e3_adaptive_weights
    )

    print(
        "    E3 global model reconstructed."
    )


# ============================================================
# SAVE RECONSTRUCTED MODELS
# ============================================================

print()
print("=" * 75)
print("SAVING RECONSTRUCTED GLOBAL MODELS")
print("=" * 75)
print()


for experiment in [
    "E1",
    "E2",
    "E3"
]:

    experiment_output_dir = (
        RECON_DIR
        / experiment
    )

    experiment_output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    for round_number in range(
        1,
        NUM_ROUNDS + 1
    ):

        output_path = (
            experiment_output_dir
            / f"global_round_{round_number:02d}.pth"
        )

        torch.save(
            global_models[
                experiment
            ][round_number],
            output_path
        )

    print(
        f"{experiment}: "
        f"{NUM_ROUNDS} models saved."
    )


# ============================================================
# VERIFY OUTPUT FILES
# ============================================================

print()
print("=" * 75)
print("VERIFYING OUTPUT")
print("=" * 75)
print()

total_saved = 0

for experiment in [
    "E1",
    "E2",
    "E3"
]:

    experiment_dir = (
        RECON_DIR
        / experiment
    )

    files = sorted(
        experiment_dir.glob(
            "global_round_*.pth"
        )
    )

    print(
        f"{experiment}: "
        f"{len(files)} global checkpoints"
    )

    if len(files) != NUM_ROUNDS:

        raise RuntimeError(
            f"{experiment}: expected "
            f"{NUM_ROUNDS} checkpoints, "
            f"found {len(files)}."
        )

    total_saved += len(files)


# ============================================================
# FINAL MESSAGE
# ============================================================

print()
print("=" * 75)
print("RECONSTRUCTION COMPLETE")
print("=" * 75)
print()

print(
    f"Total reconstructed global models: "
    f"{total_saved}"
)

print()

print(
    "E1: 10 FedAvg global models"
)

print(
    "E2: 10 FedAvg global models"
)

print(
    "E3: 10 Adaptive-FL global models"
)

print()

print(
    "Saved at:"
)

print(
    RECON_DIR
)

print()
print(
    "E3 Round 1 used equal weights:"
)

print(
    "C1 = 0.333333"
)

print(
    "C2 = 0.333333"
)

print(
    "C3 = 0.333333"
)

print()

print(
    "E3 Rounds 2-10 used the saved "
    "aggregation_weight values."
)

print()

print(
    "No training was performed."
)

print(
    "No original E1/E2/E3 files were modified."
)

print()