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
    evaluate_client_round,
)


app = ClientApp()


# ============================================================
# CLIENT ID
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

    return client_map[partition_id]


# ============================================================
# CLIENT TRAINING
# ============================================================

@app.train()
def train_client(
    msg: Message,
    context: Context
):

    client_id = get_client_id(context)

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

    print("\n" + "=" * 70)
    print("SWIN UNETR FEDERATED TRAINING")
    print("=" * 70)

    print(f"Client:              {client_id}")
    print(f"Server round:        {server_round}")
    print(f"Local epochs:        {local_epochs}")
    print(f"Learning rate:       {learning_rate}")
    print(f"Patches per epoch:   {patches_per_epoch}")

    # ---------------------------------------------------------
    # Reproducibility
    # ---------------------------------------------------------

    seed = 42 + server_round

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # ---------------------------------------------------------
    # Device
    # ---------------------------------------------------------

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(f"Device:              {device}")

    # ---------------------------------------------------------
    # Load client data
    # ---------------------------------------------------------

    train_loader, val_loader = get_client_data(
        client_id,
        patches_per_epoch=patches_per_epoch,
    )

    num_train_examples = len(
        train_loader.dataset
    )

    num_val_examples = len(
        val_loader.dataset
    )

    print(
        f"Training patients:   {num_train_examples}"
    )

    print(
        f"Validation patients: {num_val_examples}"
    )

    # ---------------------------------------------------------
    # Create Swin UNETR
    # ---------------------------------------------------------

    model = get_model()

    model = model.to(device)

    # ---------------------------------------------------------
    # Load global model
    # ---------------------------------------------------------

    global_state_dict = (
        msg.content["arrays"]
        .to_torch_state_dict()
    )

    model.load_state_dict(
        global_state_dict
    )

    # ---------------------------------------------------------
    # Local training
    # ---------------------------------------------------------

    start_time = time.time()

    train_loss = train(
        model=model,
        train_loader=train_loader,
        epochs=local_epochs,
        learning_rate=learning_rate,
        device=device,
    )

    training_time = (
        time.time() - start_time
    )

    # ---------------------------------------------------------
    # Convert training history to mean loss
    # ---------------------------------------------------------

    if isinstance(train_loss, list):

        loss_values = []

        for item in train_loss:

            if isinstance(item, dict):

                if "train_loss" in item:

                    loss_values.append(
                        float(item["train_loss"])
                    )

            elif isinstance(
                item,
                (int, float)
            ):

                loss_values.append(
                    float(item)
                )

        if len(loss_values) > 0:

            mean_train_loss = (
                sum(loss_values)
                /
                len(loss_values)
            )

        else:

            mean_train_loss = 0.0

    else:

        mean_train_loss = float(
            train_loss
        )

    # ---------------------------------------------------------
    # Save local checkpoint
    # ---------------------------------------------------------

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
        /
        f"round_{server_round:02d}.pth"
    )

    torch.save(
        model.state_dict(),
        checkpoint_path
    )

    print(
        f"Saved local model: {checkpoint_path}"
    )

    print(
        f"Training loss:     {mean_train_loss:.6f}"
    )

    print(
        f"Training time:     "
        f"{training_time:.2f} seconds"
    )

    # ---------------------------------------------------------
    # Return updated model + metrics
    # ---------------------------------------------------------

    arrays = ArrayRecord(
        model.state_dict()
    )

    metrics = MetricRecord({

        "train_loss":
            float(mean_train_loss),

        "training_time_seconds":
            float(training_time),

        "num-examples":
            int(num_train_examples),

    })

    return Message(

        content=RecordDict({

            "arrays": arrays,

            "metrics": metrics,

        }),

        reply_to=msg,

    )


# ============================================================
# CLIENT EVALUATION
# ============================================================

@app.evaluate()
def evaluate_client(
    msg: Message,
    context: Context
):

    client_id = get_client_id(context)

    server_round = int(
        msg.content["config"]["server-round"]
    )

    # ---------------------------------------------------------
    # Device
    # ---------------------------------------------------------

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("\n" + "=" * 70)
    print("SWIN UNETR CLIENT EVALUATION")
    print("=" * 70)

    print(f"Client:       {client_id}")
    print(f"Server round: {server_round}")
    print(f"Device:       {device}")

    # ---------------------------------------------------------
    # Create model
    # ---------------------------------------------------------

    model = get_model()

    model = model.to(device)

    # ---------------------------------------------------------
    # Load global model
    # ---------------------------------------------------------

    global_state_dict = (
        msg.content["arrays"]
        .to_torch_state_dict()
    )

    model.load_state_dict(
        global_state_dict
    )

    # ---------------------------------------------------------
    # Evaluate
    #
    # IMPORTANT:
    # evaluate_client_round() does NOT accept val_loader.
    # It loads the validation files itself.
    # ---------------------------------------------------------

    summary, _ = evaluate_client_round(
        model,
        client_id,
        server_round,
        device,
    )

    # ---------------------------------------------------------
    # Convert evaluation summary to Flower metrics
    # ---------------------------------------------------------

    metrics = {

        "num-examples":
            int(summary["num_patients"])

    }

    for key, value in summary.items():

        if key == "num_patients":
            continue

        if isinstance(
            value,
            (int, float)
        ):

            if not torch.isfinite(
                torch.tensor(float(value))
            ):

                value = 0.0

            metrics[key] = float(value)

    metric_record = MetricRecord(
        metrics
    )

    # ---------------------------------------------------------
    # Return evaluation metrics
    # ---------------------------------------------------------

    return Message(

        content=RecordDict({

            "metrics":
                metric_record

        }),

        reply_to=msg,

    )