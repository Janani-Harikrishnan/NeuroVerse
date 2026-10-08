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

    partition_id = int(
        context.node_config["partition-id"]
    )

    client_map = {
        0: "C1",
        1: "C2",
        2: "C3",
    }

    if partition_id not in client_map:
        raise ValueError(
            f"Unknown partition-id: {partition_id}"
        )

    return client_map[partition_id]


def get_client_code(client_id):

    mapping = {
        "C1": 1,
        "C2": 2,
        "C3": 3,
    }

    if client_id not in mapping:
        raise ValueError(
            f"Unknown client ID: {client_id}"
        )

    return mapping[client_id]


# ============================================================
# TRAINING
# ============================================================

@app.train()
def train_client(msg, context):

    # --------------------------------------------------------
    # Identify client
    # --------------------------------------------------------

    client_id = get_client_id(context)

    # --------------------------------------------------------
    # Read server configuration
    # --------------------------------------------------------

    server_round = int(
        msg.content["config"]["server-round"]
    )

    local_epochs = int(
        msg.content["config"]["local-epochs"]
    )

    learning_rate = float(
        msg.content["config"]["learning-rate"]
    )

    patches_per_epoch = int(
        msg.content["config"]["patches-per-epoch"]
    )

    # --------------------------------------------------------
    # Header
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("NEUROVERSE E3 — ADAPTIVE FL TRAINING")
    print("=" * 70)

    print(
        f"Client:              {client_id}"
    )

    print(
        f"Server round:        {server_round}"
    )

    print(
        f"Local epochs:        {local_epochs}"
    )

    print(
        f"Learning rate:       {learning_rate}"
    )

    print(
        f"Patches per epoch:   {patches_per_epoch}"
    )

    # --------------------------------------------------------
    # Reproducibility
    # --------------------------------------------------------

    seed = 42 + server_round

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        f"Device:              {device}"
    )

    # --------------------------------------------------------
    # Load client data
    # --------------------------------------------------------

    train_loader, _ = get_client_data(
        client_id,
        patches_per_epoch=patches_per_epoch,
    )

    num_train_examples = len(
        train_loader.dataset
    )

    print(
        f"Training patients:   {num_train_examples}"
    )

    # --------------------------------------------------------
    # Create model
    # --------------------------------------------------------

    model = get_model().to(device)

    # --------------------------------------------------------
    # Load global model received from server
    # --------------------------------------------------------

    global_state_dict = (
        msg.content["arrays"]
        .to_torch_state_dict()
    )

    model.load_state_dict(
        global_state_dict
    )

    # --------------------------------------------------------
    # Local training
    # --------------------------------------------------------

    start_time = time.time()

    train_history = train(
        model=model,
        train_loader=train_loader,
        epochs=local_epochs,
        learning_rate=learning_rate,
        device=device,
    )

    training_time = (
        time.time()
        - start_time
    )

    # --------------------------------------------------------
    # Calculate training losses
    # --------------------------------------------------------

    loss_values = []

    if isinstance(train_history, list):

        for item in train_history:

            if isinstance(item, dict):

                if "train_loss" in item:

                    loss_values.append(
                        float(
                            item["train_loss"]
                        )
                    )

            elif isinstance(
                item,
                (int, float)
            ):

                loss_values.append(
                    float(item)
                )

    elif isinstance(
        train_history,
        (int, float)
    ):

        loss_values.append(
            float(train_history)
        )

    if loss_values:

        mean_train_loss = (
            sum(loss_values)
            / len(loss_values)
        )

        final_train_loss = (
            loss_values[-1]
        )

    else:

        raise RuntimeError(
            "Training returned no valid loss values."
        )

    # --------------------------------------------------------
    # Save local checkpoint
    # --------------------------------------------------------

    client_dir = (
        RESULT_ROOT
        / "client_models"
        / client_id
    )

    client_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    checkpoint_path = (
        client_dir
        / f"round_{server_round:02d}.pth"
    )

    torch.save(
        model.state_dict(),
        checkpoint_path
    )

    print(
        f"Saved local model: "
        f"{checkpoint_path}"
    )

    print(
        f"Mean training loss: "
        f"{mean_train_loss:.6f}"
    )

    print(
        f"Final training loss: "
        f"{final_train_loss:.6f}"
    )

    print(
        f"Training time: "
        f"{training_time:.2f} seconds"
    )

    # --------------------------------------------------------
    # Return metrics to server
    # --------------------------------------------------------

    metrics = MetricRecord({

        "train_loss":
            float(mean_train_loss),

        "final_train_loss":
            float(final_train_loss),

        "training_time_seconds":
            float(training_time),

        "num-examples":
            int(num_train_examples),

        "client_code":
            int(
                get_client_code(client_id)
            ),
    })

    # --------------------------------------------------------
    # Return trained model
    # --------------------------------------------------------

    arrays = ArrayRecord(
        model.state_dict()
    )

    return Message(
        content=RecordDict({
            "arrays": arrays,
            "metrics": metrics,
        }),
        reply_to=msg,
    )