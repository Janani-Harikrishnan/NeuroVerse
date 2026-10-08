import time

import torch

from flwr.app import (
    Context,
    Message,
    RecordDict,
    ArrayRecord,
    MetricRecord,
)

from flwr.clientapp import ClientApp

from pytorchexample.task import (
    get_model,
    get_client_data,
    train,
    RESULT_ROOT,
)


app = ClientApp()


# ============================================================
# CLIENT MAPPING
# ============================================================

def get_client_id(context: Context):
    partition_id = int(context.node_config["partition-id"])

    client_map = {
        0: "C1",
        1: "C2",
        2: "C3",
    }

    if partition_id not in client_map:
        raise ValueError(f"Unknown partition-id: {partition_id}")

    return client_map[partition_id]


def get_client_code(client_id):
    mapping = {"C1": 1, "C2": 2, "C3": 3}
    if client_id not in mapping:
        raise ValueError(f"Unknown client ID: {client_id}")
    return mapping[client_id]


# ============================================================
# COMPACT MODEL-UPDATE PROFILE
# ============================================================

def build_update_profile(
    local_state_dict,
    global_state_dict,
    profile_size=256,
):
    """
    Build a deterministic 256-D compact representation of the
    local model update.

    For each floating-point parameter tensor we compute:
        1. mean signed parameter delta
        2. mean absolute parameter delta

    These statistics are accumulated into fixed buckets.

    This is NOT a full gradient and NOT a reconstruction of the
    model update. It is only a compact update-profile signature.
    """

    signed_buckets = torch.zeros(
        profile_size,
        dtype=torch.float64,
    )
    abs_buckets = torch.zeros(
        profile_size,
        dtype=torch.float64,
    )
    counts = torch.zeros(
        profile_size,
        dtype=torch.float64,
    )

    bucket = 0

    for name, local_tensor in local_state_dict.items():
        if name not in global_state_dict:
            continue

        global_tensor = global_state_dict[name]

        if not torch.is_floating_point(local_tensor):
            continue

        if not torch.is_floating_point(global_tensor):
            continue

        delta = (
            local_tensor.detach().cpu().to(torch.float64)
            - global_tensor.detach().cpu().to(torch.float64)
        )

        flat = delta.reshape(-1)

        if flat.numel() == 0:
            continue

        # Each parameter tensor contributes one bucket.
        idx = bucket % profile_size
        signed_buckets[idx] += flat.mean()
        abs_buckets[idx] += flat.abs().mean()
        counts[idx] += 1.0
        bucket += 1

    counts = torch.clamp(counts, min=1.0)

    signed = signed_buckets / counts
    absolute = abs_buckets / counts

    profile = torch.empty(
        2 * profile_size,
        dtype=torch.float64,
    )
    profile[:profile_size] = signed
    profile[profile_size:] = absolute

    # Reduce 512 statistics to exactly 256 dimensions by pairing.
    profile = 0.5 * (
        profile[:profile_size] + profile[profile_size:]
    )

    norm = torch.linalg.vector_norm(profile)
    if torch.isfinite(norm) and norm > 1e-12:
        profile = profile / norm

    return profile.to(torch.float32).numpy()


# ============================================================
# TRAINING
# ============================================================

@app.train()
def train_client(msg, context):
    client_id = get_client_id(context)

    server_round = int(msg.content["config"]["server-round"])
    local_epochs = int(msg.content["config"]["local-epochs"])
    learning_rate = float(msg.content["config"]["learning-rate"])
    patches_per_epoch = int(msg.content["config"]["patches-per-epoch"])

    print("\n" + "=" * 70)
    print("NEUROVERSE E3-QK — QUANTUM-KERNEL ADAPTIVE FL")
    print("=" * 70)
    print(f"Client:              {client_id}")
    print(f"Server round:        {server_round}")
    print(f"Local epochs:        {local_epochs}")
    print(f"Learning rate:       {learning_rate}")
    print(f"Patches per epoch:   {patches_per_epoch}")

    seed = 42 + server_round
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device:              {device}")

    train_loader, _ = get_client_data(
        client_id,
        patches_per_epoch=patches_per_epoch,
    )

    num_train_examples = len(train_loader.dataset)
    num_train_patients = len(train_loader.dataset.files)

    print(f"Training patches:    {num_train_examples}")
    print(f"Training patients:   {num_train_patients}")

    model = get_model().to(device)

    global_state_dict = msg.content["arrays"].to_torch_state_dict()

    # Keep a CPU copy of the global model before local training.
    global_state_cpu = {
        key: value.detach().cpu().clone()
        for key, value in global_state_dict.items()
    }

    model.load_state_dict(global_state_dict)

    start_time = time.time()

    train_history = train(
        model=model,
        train_loader=train_loader,
        epochs=local_epochs,
        learning_rate=learning_rate,
        device=device,
    )

    training_time = time.time() - start_time

    loss_values = []

    if isinstance(train_history, list):
        for item in train_history:
            if isinstance(item, dict) and "train_loss" in item:
                loss_values.append(float(item["train_loss"]))
            elif isinstance(item, (int, float)):
                loss_values.append(float(item))
    elif isinstance(train_history, (int, float)):
        loss_values.append(float(train_history))

    if not loss_values:
        raise RuntimeError("Training returned no valid loss values.")

    mean_train_loss = sum(loss_values) / len(loss_values)
    final_train_loss = loss_values[-1]

    # --------------------------------------------------------
    # NEW: compact model-update profile for quantum kernel
    # --------------------------------------------------------

    update_profile = build_update_profile(
        local_state_dict=model.state_dict(),
        global_state_dict=global_state_cpu,
        profile_size=256,
    )

    client_dir = RESULT_ROOT / "client_models" / client_id
    client_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = client_dir / f"round_{server_round:02d}.pth"
    torch.save(model.state_dict(), checkpoint_path)

    print(f"Saved local model:   {checkpoint_path}")
    print(f"Mean training loss:  {mean_train_loss:.6f}")
    print(f"Final training loss: {final_train_loss:.6f}")
    print(f"Training time:       {training_time:.2f} seconds")
    print(f"Update profile:      {update_profile.shape}")

    metrics = MetricRecord({
        "train_loss": float(mean_train_loss),
        "final_train_loss": float(final_train_loss),
        "training_time_seconds": float(training_time),
        "num-examples": int(num_train_examples),
        "num-patients": int(num_train_patients),
        "client_code": int(get_client_code(client_id)),
    })

    arrays = ArrayRecord(model.state_dict())
    update_profile_record = ArrayRecord.from_numpy_ndarrays(
        [update_profile]
    )

    return Message(
        content=RecordDict({
            "arrays": arrays,
            "update_profile": update_profile_record,
            "metrics": metrics,
        }),
        reply_to=msg,
    )