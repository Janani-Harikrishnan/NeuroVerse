import torch

from flwr.app import (
    Context,
    Message,
    RecordDict,
    ArrayRecord,
    ConfigRecord,
)

from flwr.serverapp import (
    ServerApp,
    Grid,
)

from flwr.serverapp.strategy import FedAvg

from pytorchexample.task import (
    get_model,
    RESULT_ROOT,
    evaluate_global_test,
)


app = ServerApp()


# ============================================================
# CUSTOM FEDAVG
# ============================================================

class SwinFedAvgSelectiveEvaluation(
    FedAvg
):

    # Client validation will be performed
    # only on these rounds.

    EVALUATION_ROUNDS = {
        1,
        5,
        10,
    }

    def configure_evaluate(
        self,
        server_round: int,
        arrays: ArrayRecord,
        config: ConfigRecord,
        grid: Grid,
    ):

        if (
            server_round
            not in self.EVALUATION_ROUNDS
        ):

            print()

            print(
                f"[ROUND {server_round}] "
                f"Skipping client validation"
            )

            return []

        return super().configure_evaluate(
            server_round,
            arrays,
            config,
            grid,
        )


# ============================================================
# SERVER
# ============================================================

@app.main()
def main(
    grid: Grid,
    context: Context
):

    print("\n" + "=" * 70)
    print("NEUROVERSE — E2")
    print("SWIN UNETR + FEDAVG")
    print("=" * 70)

    # ---------------------------------------------------------
    # Configuration
    # ---------------------------------------------------------

    num_rounds = int(
        context.run_config[
            "num-server-rounds"
        ]
    )

    fraction_train = float(
        context.run_config[
            "fraction-train"
        ]
    )

    fraction_evaluate = float(
        context.run_config[
            "fraction-evaluate"
        ]
    )

    local_epochs = int(
        context.run_config[
            "local-epochs"
        ]
    )

    learning_rate = float(
        context.run_config[
            "learning-rate"
        ]
    )

    patches_per_epoch = int(
        context.run_config[
            "patches-per-epoch"
        ]
    )

    print(
        f"Experiment:          E2"
    )

    print(
        f"Model:               Swin UNETR"
    )

    print(
        f"Federated algorithm: FedAvg"
    )

    print(
        f"Rounds:              {num_rounds}"
    )

    print(
        f"Local epochs:        {local_epochs}"
    )

    print(
        f"Learning rate:       {learning_rate}"
    )

    print(
        f"Patches/epoch:       {patches_per_epoch}"
    )

    print(
        f"Adaptive FL:         DISABLED"
    )

    print(
        f"Differential Privacy: DISABLED"
    )

    print(
        f"Quantum optimization: DISABLED"
    )

    # ---------------------------------------------------------
    # Create initial global Swin UNETR
    # ---------------------------------------------------------

    global_model = get_model()

    print("\nGlobal model:")
    print(
        type(global_model).__name__
    )

    arrays = ArrayRecord(
        global_model.state_dict()
    )

    # ---------------------------------------------------------
    # FedAvg strategy
    # ---------------------------------------------------------

    strategy = (
        SwinFedAvgSelectiveEvaluation(

            fraction_train=
                fraction_train,

            fraction_evaluate=
                fraction_evaluate,

            min_train_nodes=3,

            min_evaluate_nodes=3,

            min_available_nodes=3,

            weighted_by_key=
                "num-examples",
        )
    )

    # ---------------------------------------------------------
    # Start federated training
    # ---------------------------------------------------------

    result = strategy.start(

        grid=grid,

        initial_arrays=arrays,

        train_config=ConfigRecord({

            "server-round":
                0,

            "local-epochs":
                local_epochs,

            "learning-rate":
                learning_rate,

            "patches-per-epoch":
                patches_per_epoch,

        }),

        evaluate_config=ConfigRecord({

            "server-round":
                0,

        }),

        num_rounds=num_rounds,

    )

    # ---------------------------------------------------------
    # Save final global model
    # ---------------------------------------------------------

    RESULT_ROOT.mkdir(
        parents=True,
        exist_ok=True
    )

    final_model_path = (
        RESULT_ROOT
        /
        "global_model_final.pth"
    )

    final_state_dict = (
        result.arrays
        .to_torch_state_dict()
    )

    torch.save(
        final_state_dict,
        final_model_path
    )

    print("\n" + "=" * 70)
    print("FEDERATED TRAINING COMPLETED")
    print("=" * 70)

    print(
        f"Final global model:"
    )

    print(
        final_model_path
    )

    # ---------------------------------------------------------
    # GlobalTest evaluation
    # ---------------------------------------------------------

    print(
        "\nEvaluating final global "
        "Swin UNETR on GlobalTest..."
    )

    final_model = get_model()

    final_model.load_state_dict(
        final_state_dict
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    final_model = final_model.to(
        device
    )

    global_metrics = evaluate_global_test(
        model=final_model
    )

    # ---------------------------------------------------------
    # Print final metrics
    # ---------------------------------------------------------

    print("\n" + "=" * 70)
    print("E2 FINAL GLOBAL TEST RESULTS")
    print("=" * 70)

    # evaluate_global_test() returns:
    #
    # summary, dataframe

    summary, _ = global_metrics

    for key, value in summary.items():

        if isinstance(
            value,
            (int, float)
        ):

            print(
                f"{key}: {value:.6f}"
            )

    print("=" * 70)