import time
import numpy as np
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
    RESULT_ROOT,
    get_model,
    get_client_data,
    train,
    evaluate_client_round,
    set_seed,
)


app = ClientApp()


# ============================================================
# CLIENT ID
# ============================================================

def get_client_id(context):

    partition_id = int(
        context.node_config["partition-id"]
    )

    mapping = {
        0: "C1",
        1: "C2",
        2: "C3",
    }

    if partition_id not in mapping:

        raise ValueError(
            f"Unknown partition-id: "
            f"{partition_id}"
        )

    return mapping[partition_id]


# ============================================================
# TRAIN
# ============================================================

@app.train()
def train_client(
    msg: Message,
    context: Context
):

    start_time = time.time()

    client_id = get_client_id(
        context
    )

    round_number = int(
        msg.content["config"].get(
            "server-round",
            1
        )
    )

    local_epochs = int(
        context.run_config[
            "local-epochs"
        ]
    )

    learning_rate = float(
        msg.content["config"].get(
            "lr",
            context.run_config[
                "learning-rate"
            ]
        )
    )

    patches_per_epoch = int(
        context.run_config[
            "patches-per-epoch"
        ]
    )

    set_seed(
        42 + round_number
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print()
    print("=" * 70)
    print(
        f"FEDERATED TRAINING — {client_id}"
    )
    print("=" * 70)

    print(
        f"Round: {round_number}"
    )

    print(
        f"Device: {device}"
    )

    if torch.cuda.is_available():

        print(
            f"GPU: "
            f"{torch.cuda.get_device_name(0)}"
        )

    # --------------------------------------------------------
    # CLIENT DATA
    # --------------------------------------------------------

    train_loader, _ = get_client_data(
        client_id,
        patches_per_epoch
    )

    print(
        f"Training patients: "
        f"{len(train_loader.dataset.files)}"
    )

    print(
        f"Patches per epoch: "
        f"{len(train_loader.dataset)}"
    )

    # --------------------------------------------------------
    # CREATE MODEL
    # --------------------------------------------------------

    model = get_model()

    # --------------------------------------------------------
    # RECEIVE GLOBAL MODEL
    # --------------------------------------------------------

    state_dict = (
        msg.content["arrays"]
        .to_torch_state_dict()
    )

    model.load_state_dict(
        state_dict
    )

    model.to(device)

    # --------------------------------------------------------
    # LOCAL TRAINING
    # --------------------------------------------------------

    history = train(
        model,
        train_loader,
        local_epochs,
        learning_rate,
        device
    )

    final_train_loss = float(
        history[-1]["train_loss"]
    )

    # --------------------------------------------------------
    # SAVE LOCAL MODEL
    # --------------------------------------------------------

    client_checkpoint_dir = (
        RESULT_ROOT
        / "client_models"
        / client_id
    )

    client_checkpoint_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    torch.save(
        {
            "round": round_number,
            "client": client_id,
            "model_state_dict":
                model.state_dict(),
            "train_history": history,
        },
        client_checkpoint_dir
        /
        f"round_{round_number:02d}.pth"
    )

    training_time = (
        time.time()
        -
        start_time
    )

    # --------------------------------------------------------
    # RETURN LOCAL MODEL TO SERVER
    # --------------------------------------------------------

    metrics = {

        "train_loss":
            final_train_loss,

        "training_time_seconds":
            float(training_time),

        "num-examples":
            len(train_loader.dataset),
    }

    model_record = ArrayRecord(
        model.state_dict()
    )

    metric_record = MetricRecord(
        metrics
    )

    content = RecordDict({

        "arrays": model_record,

        "metrics": metric_record,
    })

    print()
    print(
        f"{client_id} local training completed."
    )

    print(
        f"Final training loss: "
        f"{final_train_loss:.6f}"
    )

    print(
        "Sending FINAL local weights "
        "to server..."
    )

    return Message(
        content=content,
        reply_to=msg
    )


# ============================================================
# EVALUATE
# ============================================================

@app.evaluate()
def evaluate_client(
    msg: Message,
    context: Context
):

    client_id = get_client_id(
        context
    )

    round_number = int(
        msg.content["config"].get(
            "server-round",
            1
        )
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    model = get_model()

    state_dict = (
        msg.content["arrays"]
        .to_torch_state_dict()
    )

    model.load_state_dict(
        state_dict
    )

    model.to(device)

    print()
    print("=" * 70)
    print(
        f"FEDERATED VALIDATION — "
        f"{client_id}"
    )
    print(
        f"Round: {round_number}"
    )
    print("=" * 70)

    summary, patient_df = (
        evaluate_client_round(
            model,
            client_id,
            round_number,
            device
        )
    )

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

            if not np.isfinite(value):

                value = 0.0

            metrics[key] = float(value)

    metric_record = MetricRecord(
        metrics
    )

    content = RecordDict({

        "metrics": metric_record
    })

    return Message(
        content=content,
        reply_to=msg
    )