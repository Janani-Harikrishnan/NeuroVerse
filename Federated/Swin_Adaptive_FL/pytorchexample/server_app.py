import csv
import math
import gc
from pathlib import Path

import torch

from flwr.app import (
    Context,
    Message,
    RecordDict,
    ArrayRecord,
    ConfigRecord,
    MetricRecord,
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
# ADAPTIVE FEDERATED STRATEGY
#
# E3:
# Swin UNETR + Adaptive Federated Aggregation
#
# Adaptive signals:
#   1. Training quality
#   2. Training improvement
#   3. Participation
#   4. Historical quality
#
# Intermediate full-volume validation:
#   DISABLED
#
# Final GlobalTest:
#   ENABLED
# ============================================================


class SwinAdaptiveFedAvg(FedAvg):

    # --------------------------------------------------------
    # Adaptive coefficients
    # --------------------------------------------------------

    LAMBDA_QUALITY = 0.40

    LAMBDA_IMPROVEMENT = 0.25

    LAMBDA_PARTICIPATION = 0.20

    LAMBDA_HISTORY = 0.15

    # EMA smoothing
    HISTORY_BETA = 0.70

    CLIENTS = [
        "C1",
        "C2",
        "C3",
    ]

    def __init__(
        self,
        *args,
        **kwargs
    ):

        super().__init__(
            *args,
            **kwargs
        )

        # ----------------------------------------------------
        # Previous-round training losses
        # ----------------------------------------------------

        self.previous_loss = {}

        # ----------------------------------------------------
        # Historical training quality
        # ----------------------------------------------------

        self.historical_quality = {

            "C1": 0.0,
            "C2": 0.0,
            "C3": 0.0,

        }

        # ----------------------------------------------------
        # Previous aggregation weights
        # ----------------------------------------------------

        self.previous_weights = {

            "C1": 1.0 / 3.0,
            "C2": 1.0 / 3.0,
            "C3": 1.0 / 3.0,

        }

        # ----------------------------------------------------
        # Adaptive decision history
        # ----------------------------------------------------

        self.adaptive_history = []


    # ========================================================
    # CLIENT CODE → CLIENT NAME
    # ========================================================

    @staticmethod
    def code_to_client(code):

        mapping = {

            1: "C1",
            2: "C2",
            3: "C3",

        }

        return mapping.get(
            int(code),
            None
        )


    # ========================================================
    # EXTRACT CLIENT TRAINING METRICS
    # ========================================================

    def extract_training_metrics(
        self,
        replies
    ):

        client_metrics = {}

        for reply in replies:

            if reply.has_error():
                continue

            if not reply.has_content():
                continue

            if (
                "metrics"
                not in reply.content.metric_records
            ):
                continue

            metrics = (
                reply.content
                .metric_records["metrics"]
            )

            client_code = metrics.get(
                "client_code"
            )

            if client_code is None:
                continue

            client_id = (
                self.code_to_client(
                    client_code
                )
            )

            if client_id is None:
                continue

            train_loss = float(
                metrics.get(
                    "train_loss",
                    0.0
                )
            )

            num_examples = float(
                metrics.get(
                    "num-examples",
                    1.0
                )
            )

            training_time = float(
                metrics.get(
                    "training_time_seconds",
                    0.0
                )
            )

            client_metrics[client_id] = {

                "train_loss":
                    train_loss,

                "num_examples":
                    num_examples,

                "training_time":
                    training_time,

            }

        return client_metrics


    # ========================================================
    # ADAPTIVE WEIGHT CALCULATION
    # ========================================================

    def calculate_adaptive_weights(
        self,
        server_round,
        replies
    ):

        client_metrics = (
            self.extract_training_metrics(
                replies
            )
        )

        # ----------------------------------------------------
        # Round 1
        # ----------------------------------------------------

        if server_round == 1:

            weights = {

                client:
                    1.0 / 3.0

                for client in self.CLIENTS

            }

            print(
                "\n[ADAPTIVE FL] "
                "Round 1: equal initialization"
            )

            for client in self.CLIENTS:

                metrics = (
                    client_metrics.get(
                        client,
                        {}
                    )
                )

                loss = float(
                    metrics.get(
                        "train_loss",
                        0.0
                    )
                )

                self.previous_loss[
                    client
                ] = loss

                self.historical_quality[
                    client
                ] = 0.0

            return weights


        # ----------------------------------------------------
        # Collect available losses
        # ----------------------------------------------------

        losses = {}

        train_sizes = {}

        for client in self.CLIENTS:

            metrics = (
                client_metrics.get(
                    client,
                    {}
                )
            )

            losses[client] = float(
                metrics.get(
                    "train_loss",
                    0.0
                )
            )

            train_sizes[client] = float(
                metrics.get(
                    "num_examples",
                    1.0
                )
            )


        total_train_size = sum(
            train_sizes.values()
        )


        # ----------------------------------------------------
        # Normalize training quality
        #
        # Lower loss = higher quality.
        #
        # Quality range:
        #     0 ... 1
        # ----------------------------------------------------

        valid_losses = [

            loss

            for loss in losses.values()

            if math.isfinite(loss)

            and loss >= 0.0

        ]


        if not valid_losses:

            valid_losses = [
                1.0,
                1.0,
                1.0
            ]


        min_loss = min(
            valid_losses
        )

        max_loss = max(
            valid_losses
        )

        loss_range = (
            max_loss
            - min_loss
        )


        quality = {}

        for client in self.CLIENTS:

            loss = losses[client]

            if not math.isfinite(loss):

                loss = max_loss

            if loss_range < 1e-8:

                quality[client] = 1.0

            else:

                quality[client] = (

                    max_loss - loss

                ) / (

                    loss_range + 1e-8

                )


        # ----------------------------------------------------
        # Calculate improvement
        #
        # Positive improvement means the client's
        # training loss decreased compared with its
        # previous round.
        # ----------------------------------------------------

        improvement = {}

        for client in self.CLIENTS:

            current_loss = losses[client]

            previous_loss = (
                self.previous_loss.get(
                    client,
                    current_loss
                )
            )

            if previous_loss > 1e-8:

                raw_improvement = (

                    previous_loss
                    - current_loss

                ) / previous_loss

            else:

                raw_improvement = 0.0


            # Map improvement to a stable [0,1] range.
            #
            # 0.0 = no improvement
            # 1.0 = strong improvement

            improvement[client] = max(

                0.0,

                min(

                    1.0,

                    0.5
                    + raw_improvement

                )

            )


        # ----------------------------------------------------
        # Participation
        # ----------------------------------------------------

        participation = {}

        for client in self.CLIENTS:

            if total_train_size > 0:

                participation[client] = (

                    train_sizes[client]
                    /
                    total_train_size

                )

            else:

                participation[client] = (
                    1.0 / 3.0
                )


        # ----------------------------------------------------
        # Historical quality
        #
        # EMA:
        #
        # H_t =
        # beta * H_(t-1)
        # +
        # (1-beta) * Q_t
        # ----------------------------------------------------

        for client in self.CLIENTS:

            current_quality = (
                quality[client]
            )

            previous_history = (
                self.historical_quality.get(
                    client,
                    current_quality
                )
            )

            historical_quality = (

                self.HISTORY_BETA
                * previous_history

                +

                (
                    1.0
                    - self.HISTORY_BETA
                )
                * current_quality

            )

            self.historical_quality[
                client
            ] = historical_quality


        # ----------------------------------------------------
        # Adaptive score
        #
        # Higher score = larger contribution.
        # ----------------------------------------------------

        scores = {}

        for client in self.CLIENTS:

            score = (

                self.LAMBDA_QUALITY
                * quality[client]

                +

                self.LAMBDA_IMPROVEMENT
                * improvement[client]

                +

                self.LAMBDA_PARTICIPATION
                * participation[client]

                +

                self.LAMBDA_HISTORY
                * self.historical_quality[
                    client
                ]

            )

            scores[client] = score


        # ----------------------------------------------------
        # Softmax
        # ----------------------------------------------------

        max_score = max(
            scores.values()
        )

        exp_scores = {}

        for client in self.CLIENTS:

            exp_scores[client] = math.exp(

                scores[client]
                - max_score

            )


        denominator = sum(
            exp_scores.values()
        )


        if denominator <= 0:

            weights = {

                client:
                    1.0 / 3.0

                for client in self.CLIENTS

            }

        else:

            weights = {

                client:

                    exp_scores[client]
                    /
                    denominator

                for client in self.CLIENTS

            }


        # ----------------------------------------------------
        # Logging
        # ----------------------------------------------------

        print(
            "\n"
            + "-" * 70
        )

        print(
            f"[ADAPTIVE FL] "
            f"ROUND {server_round}"
        )

        print(
            "Training-signal adaptive aggregation"
        )

        print(
            "-" * 70
        )


        for client in self.CLIENTS:

            print(

                f"{client}: "

                f"Loss="
                f"{losses[client]:.6f} | "

                f"Quality="
                f"{quality[client]:.6f} | "

                f"Improvement="
                f"{improvement[client]:.6f} | "

                f"P="
                f"{participation[client]:.6f} | "

                f"History="
                f"{self.historical_quality[client]:.6f} | "

                f"Score="
                f"{scores[client]:.6f} | "

                f"Weight="
                f"{weights[client]:.6f}"

            )


            self.adaptive_history.append({

                "round":
                    int(server_round),

                "client":
                    client,

                "train_loss":
                    float(
                        losses[client]
                    ),

                "training_quality":
                    float(
                        quality[client]
                    ),

                "improvement":
                    float(
                        improvement[client]
                    ),

                "participation":
                    float(
                        participation[client]
                    ),

                "historical_quality":
                    float(
                        self.historical_quality[
                            client
                        ]
                    ),

                "adaptive_score":
                    float(
                        scores[client]
                    ),

                "aggregation_weight":
                    float(
                        weights[client]
                    ),

            })


        print(
            "-" * 70
        )


        # ----------------------------------------------------
        # Store current losses
        # ----------------------------------------------------

        for client in self.CLIENTS:

            self.previous_loss[
                client
            ] = losses[client]


        # ----------------------------------------------------
        # Store weights
        # ----------------------------------------------------

        self.previous_weights = (
            dict(weights)
        )


        return weights


    # ========================================================
    # AGGREGATE TRAINING
    # ========================================================

    def aggregate_train(
        self,
        server_round,
        replies
    ):

        replies = list(replies)

        valid_replies = [

            reply

            for reply in replies

            if not reply.has_error()

            and reply.has_content()

        ]


        if not valid_replies:

            print(
                "[ADAPTIVE FL] "
                "No valid client replies."
            )

            return None, None


        # ----------------------------------------------------
        # Calculate adaptive weights
        # ----------------------------------------------------

        adaptive_weights = (
            self.calculate_adaptive_weights(
                server_round,
                valid_replies
            )
        )


        # ----------------------------------------------------
        # Extract model states
        # ----------------------------------------------------

        client_states = []

        client_weights = []

        client_ids = []


        for reply in valid_replies:

            if (
                "arrays"
                not in reply.content.array_records
            ):
                continue

            if (
                "metrics"
                not in reply.content.metric_records
            ):
                continue


            metrics = (
                reply.content
                .metric_records["metrics"]
            )


            client_code = metrics.get(
                "client_code"
            )

            if client_code is None:
                continue


            client_id = (
                self.code_to_client(
                    client_code
                )
            )

            if client_id is None:
                continue


            state_dict = (

                reply.content
                .array_records["arrays"]
                .to_torch_state_dict()

            )


            client_states.append(
                state_dict
            )

            client_weights.append(
                float(
                    adaptive_weights[
                        client_id
                    ]
                )
            )

            client_ids.append(
                client_id
            )


        if not client_states:

            return None, None


        # ----------------------------------------------------
        # Normalize weights
        # ----------------------------------------------------

        total_weight = sum(
            client_weights
        )


        if total_weight <= 0:

            client_weights = [

                1.0 / len(client_states)

                for _ in client_states

            ]

        else:

            client_weights = [

                weight / total_weight

                for weight
                in client_weights

            ]


        # ----------------------------------------------------
        # Weighted aggregation
        # ----------------------------------------------------

        aggregated_state = {}

        first_state = (
            client_states[0]
        )


        for key in first_state:

            first_tensor = (
                first_state[key]
            )


            if torch.is_floating_point(
                first_tensor
            ):

                aggregated_tensor = (
                    torch.zeros_like(
                        first_tensor
                    )
                )


                for state_dict, weight in zip(
                    client_states,
                    client_weights
                ):

                    aggregated_tensor += (

                        state_dict[key]
                        * weight

                    )


                aggregated_state[key] = (
                    aggregated_tensor
                )

            else:

                aggregated_state[key] = (
                    first_tensor.clone()
                )


        aggregated_arrays = (
            ArrayRecord(
                aggregated_state
            )
        )


        # ----------------------------------------------------
        # Aggregate training metrics
        # ----------------------------------------------------

        train_losses = []

        training_times = []

        total_examples = 0


        for reply in valid_replies:

            if (
                "metrics"
                not in reply.content.metric_records
            ):
                continue


            metrics = (
                reply.content
                .metric_records["metrics"]
            )


            if "train_loss" in metrics:

                train_losses.append(
                    float(
                        metrics["train_loss"]
                    )
                )


            if "training_time_seconds" in metrics:

                training_times.append(
                    float(
                        metrics[
                            "training_time_seconds"
                        ]
                    )
                )


            if "num-examples" in metrics:

                total_examples += int(
                    metrics["num-examples"]
                )


        aggregated_metrics = {

            "train_loss":

                (
                    sum(train_losses)
                    /
                    len(train_losses)
                )
                if train_losses
                else 0.0,

            "training_time_seconds":

                (
                    sum(training_times)
                    /
                    len(training_times)
                )
                if training_times
                else 0.0,

            "num-examples":
                int(total_examples),

        }


        # ----------------------------------------------------
        # Print aggregation result
        # ----------------------------------------------------

        print(
            "\n"
            + "=" * 70
        )

        print(
            f"[ADAPTIVE FL] "
            f"Round {server_round} "
            f"aggregation complete"
        )

        print(
            "Aggregation weights:"
        )


        for client in self.CLIENTS:

            print(

                f"  {client}: "
                f"{adaptive_weights[client]:.6f}"

            )


        print(
            "=" * 70
        )


        return (

            aggregated_arrays,

            MetricRecord(
                aggregated_metrics
            )

        )


# ============================================================
# SAVE ADAPTIVE HISTORY
# ============================================================


def save_adaptive_history(
    history
):

    if not history:

        return


    RESULT_ROOT.mkdir(
        parents=True,
        exist_ok=True
    )


    output_path = (
        RESULT_ROOT
        / "adaptive_weights.csv"
    )


    fieldnames = [

        "round",
        "client",
        "train_loss",
        "training_quality",
        "improvement",
        "participation",
        "historical_quality",
        "adaptive_score",
        "aggregation_weight",

    ]


    with open(
        output_path,
        "w",
        newline=""
    ) as csv_file:

        writer = csv.DictWriter(
            csv_file,
            fieldnames=fieldnames
        )

        writer.writeheader()


        for row in history:

            writer.writerow(row)


    print(
        f"\nSaved adaptive history: "
        f"{output_path}"
    )


# ============================================================
# SERVER APP
# ============================================================


@app.main()
def main(
    grid: Grid,
    context: Context
):

    # --------------------------------------------------------
    # Read configuration
    # --------------------------------------------------------

    num_rounds = int(
        context.run_config[
            "num-server-rounds"
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


    fraction_train = float(
        context.run_config.get(
            "fraction-train",
            1.0
        )
    )


    # --------------------------------------------------------
    # Header
    # --------------------------------------------------------

    print("\n")
    print("=" * 70)

    print(
        "NEUROVERSE — E3"
    )

    print(
        "SWIN UNETR + "
        "TRAINING-SIGNAL ADAPTIVE FL"
    )

    print("=" * 70)


    print(
        "Experiment:          E3"
    )


    print(
        "Model:               Swin UNETR"
    )


    print(
        "Federated algorithm: "
        "Adaptive FL"
    )


    print(
        f"Rounds:              "
        f"{num_rounds}"
    )


    print(
        f"Local epochs:        "
        f"{local_epochs}"
    )


    print(
        f"Learning rate:       "
        f"{learning_rate}"
    )


    print(
        f"Patches/epoch:       "
        f"{patches_per_epoch}"
    )


    print(
        "Adaptive FL:         ENABLED"
    )


    print(
        "Intermediate validation: DISABLED"
    )


    print(
        "Final GlobalTest:    ENABLED"
    )


    print(
        "Differential Privacy: DISABLED"
    )


    print(
        "Quantum optimization: DISABLED"
    )


    print(
        "\nAdaptive coefficients:"
    )


    print(
        "λ1 Training quality = "
        "0.40"
    )


    print(
        "λ2 Training improvement = "
        "0.25"
    )


    print(
        "λ3 Participation = "
        "0.20"
    )


    print(
        "λ4 Historical quality = "
        "0.15"
    )


    print(
        f"Historical smoothing β = "
        f"{SwinAdaptiveFedAvg.HISTORY_BETA:.2f}"
    )


    # --------------------------------------------------------
    # Create global model
    # --------------------------------------------------------

    model = get_model()


    print(
        "\nGlobal model:"
    )

    print(
        "SwinUNETR"
    )


    # --------------------------------------------------------
    # Strategy
    # --------------------------------------------------------

    strategy = SwinAdaptiveFedAvg(

        fraction_train=
            fraction_train,

        fraction_evaluate=0.0,

        min_train_nodes=3,

        min_evaluate_nodes=0,

    )


    # --------------------------------------------------------
    # Start federated learning
    # --------------------------------------------------------

    result = strategy.start(

        grid=grid,

        initial_arrays=
            ArrayRecord(
                model.state_dict()
            ),

        # IMPORTANT:
        # Use the configured number of rounds.
        num_rounds=num_rounds,

        train_config=
            ConfigRecord({

                "server-round":
                    0,

                "local-epochs":
                    local_epochs,

                "learning-rate":
                    learning_rate,

                "patches-per-epoch":
                    patches_per_epoch,

            }),

        evaluate_config=
            ConfigRecord({}),

    )


    # --------------------------------------------------------
    # Release server-side GPU memory
    # --------------------------------------------------------

    print(
        "\nReleasing server-side GPU memory..."
    )


    gc.collect()


    if torch.cuda.is_available():

        torch.cuda.empty_cache()

        torch.cuda.synchronize()


    print(
        "Server-side GPU cleanup completed."
    )


    # --------------------------------------------------------
    # Save adaptive history
    # --------------------------------------------------------

    save_adaptive_history(
        strategy.adaptive_history
    )


    # --------------------------------------------------------
    # Final global model
    # --------------------------------------------------------

    final_model = get_model()


    final_state_dict = (
        result.arrays
        .to_torch_state_dict()
    )


    final_model.load_state_dict(
        final_state_dict
    )


    final_model_path = (
        RESULT_ROOT
        / "global_model_final.pth"
    )


    RESULT_ROOT.mkdir(
        parents=True,
        exist_ok=True
    )


    torch.save(
        final_model.state_dict(),
        final_model_path
    )


    print(
        "\nSaved final global model:"
    )


    print(
        final_model_path
    )


    # --------------------------------------------------------
    # FINAL GLOBAL TEST
    #
    # This is the ONLY full-volume evaluation.
    # --------------------------------------------------------

    print("\n")
    print("=" * 70)

    print(
        "NEUROVERSE E3 — FINAL GLOBAL TEST"
    )

    print("=" * 70)


    print(
        "Running GlobalTest on "
        "188 held-out patients..."
    )


    final_summary = (
        evaluate_global_test(
            model=final_model
        )
    )


    print(
        "\nFinal GlobalTest completed."
    )


    print(
        "=" * 70
    )