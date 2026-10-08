import torch

from flwr.app import (
    Context,
    ArrayRecord,
    ConfigRecord,
)

from flwr.serverapp import (
    ServerApp,
    Grid,
)
from collections.abc import Iterable
from flwr.serverapp.strategy import FedAvg
from flwr.app import Message

from pytorchexample.task import (
    RESULT_ROOT,
    get_model,
    evaluate_global_test,
)
# ============================================================
# FEDAVG WITH SELECTIVE VALIDATION
# ============================================================

class FedAvgSelectiveEvaluation(FedAvg):

    # Validate only on these federated rounds
    EVALUATION_ROUNDS = {1, 5, 10}

    def configure_evaluate(
        self,
        server_round: int,
        arrays: ArrayRecord,
        config: ConfigRecord,
        grid: Grid,
    ) -> Iterable[Message]:

        # Skip client validation on all other rounds
        if server_round not in self.EVALUATION_ROUNDS:

            print()
            print(
                f"[ROUND {server_round}] "
                f"Skipping client validation"
            )

            return []

        # Run normal FedAvg evaluation on rounds
        # 1, 5, and 10
        print()
        print(
            f"[ROUND {server_round}] "
            f"Running client validation"
        )

        return super().configure_evaluate(
            server_round,
            arrays,
            config,
            grid,
        )

app = ServerApp()


# ============================================================
# SERVER
# ============================================================

@app.main()
def main(
    grid: Grid,
    context: Context
):

    RESULT_ROOT.mkdir(
        parents=True,
        exist_ok=True
    )

    num_rounds = int(
        context.run_config[
            "num-server-rounds"
        ]
    )

    learning_rate = float(
        context.run_config[
            "learning-rate"
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

    # ========================================================
    # CREATE FRESH GLOBAL MODEL
    # ========================================================

    global_model = get_model()

    total_parameters = sum(
        p.numel()
        for p in global_model.parameters()
        if p.requires_grad
    )

    print()
    print("=" * 80)
    print(
        "NEUROVERSE FEDERATED LEARNING"
    )
    print("=" * 80)

    print(
        "Model: 3D U-Net"
    )

    print(
        f"Trainable parameters: "
        f"{total_parameters:,}"
    )

    print(
        f"Federated rounds: "
        f"{num_rounds}"
    )

    print(
        f"Learning rate: "
        f"{learning_rate}"
    )

    print("=" * 80)

    # ========================================================
    # INITIAL GLOBAL WEIGHTS
    # ========================================================

    arrays = ArrayRecord(
        global_model.state_dict()
    )

    # ========================================================
    # FEDAVG
    # ========================================================

    strategy = FedAvgSelectiveEvaluation(

        fraction_train=fraction_train,

        fraction_evaluate=fraction_evaluate,

        min_train_nodes=3,

        min_evaluate_nodes=3,

        min_available_nodes=3,

        weighted_by_key="num-examples",
    )

    # ========================================================
    # START FEDERATED TRAINING
    # ========================================================

    result = strategy.start(

        grid=grid,

        initial_arrays=arrays,

        train_config=ConfigRecord({

            "lr": learning_rate,

        }),

        evaluate_config=ConfigRecord({}),

        num_rounds=num_rounds,
    )

    # ========================================================
    # SAVE FINAL GLOBAL MODEL
    # ========================================================

    final_state_dict = (
        result.arrays
        .to_torch_state_dict()
    )

    final_model_path = (
        RESULT_ROOT
        /
        "global_model_final.pth"
    )

    torch.save(
        final_state_dict,
        final_model_path
    )

    print()
    print("=" * 80)
    print(
        "FINAL GLOBAL MODEL SAVED"
    )
    print("=" * 80)

    print(
        final_model_path
    )

    # ========================================================
    # PRINT FLOWER TRAINING METRICS
    # ========================================================

    print()
    print(
        "Federated training metrics:"
    )

    print(
        result.train_metrics_clientapp
    )

    print()
    print(
        "Federated validation metrics:"
    )

    print(
        result.evaluate_metrics_clientapp
    )

    # ========================================================
    # FINAL GLOBAL TEST
    #
    # GlobalTest is used ONLY after all
    # federated rounds have completed.
    # ========================================================

    final_model = get_model()

    final_model.load_state_dict(
        final_state_dict
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    final_model.to(device)

    print()
    print("=" * 80)
    print(
        "FINAL GLOBAL TEST"
    )
    print("=" * 80)

    summary, patient_df = (
        evaluate_global_test(
            final_model
        )
    )

    print()
    print(
        f"GlobalTest patients: "
        f"{len(patient_df)}"
    )

    print(
        f"Mean Dice: "
        f"{summary['mean_dice']:.6f}"
    )

    print(
        f"Mean IoU: "
        f"{summary['mean_iou']:.6f}"
    )

    print(
        f"Mean Accuracy: "
        f"{summary['mean_accuracy']:.6f}"
    )

    print(
        f"Mean Precision: "
        f"{summary['mean_precision']:.6f}"
    )

    print(
        f"Mean Sensitivity: "
        f"{summary['mean_sensitivity']:.6f}"
    )

    print(
        f"Mean Specificity: "
        f"{summary['mean_specificity']:.6f}"
    )

    print(
        f"Mean F1: "
        f"{summary['mean_f1']:.6f}"
    )

    print(
        f"Mean Balanced Accuracy: "
        f"{summary['mean_balanced_accuracy']:.6f}"
    )

    print(
        f"Mean MCC: "
        f"{summary['mean_mcc']:.6f}"
    )

    print(
        f"Mean HD95: "
        f"{summary['mean_hd95']:.6f}"
    )

    print(
        f"Mean ROC-AUC: "
        f"{summary['mean_roc_auc']:.6f}"
    )

    print(
        f"Mean PR-AUC: "
        f"{summary['mean_pr_auc']:.6f}"
    )

    print()
    print("=" * 80)
    print(
        "FEDAVG EXPERIMENT COMPLETE"
    )
    print("=" * 80)