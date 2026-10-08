import torch
from pathlib import Path

from pytorchexample.task import (
    UNet3D,
    evaluate_patient,
    BASE_CHANNELS,
    NUM_CLASSES,
)


# --------------------------------------------------
# CONFIG
# --------------------------------------------------

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_PATH = Path(
    "/home/janani/Neuroverse/Federated/FedAvg_BraTS/"
    "results/global_model_final.pth"
)

PATIENT_FILE = Path(
    "/home/janani/Neuroverse/IID_Clients/C1/val/"
    "BraTS2021_00454.npz"
)


# --------------------------------------------------
# GPU INFO
# --------------------------------------------------

print("=" * 70)
print("ONE-PATIENT VALIDATION TEST")
print("=" * 70)

print(f"Device: {DEVICE}")

if DEVICE.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(
        f"GPU memory allocated before loading: "
        f"{torch.cuda.memory_allocated() / 1024**3:.2f} GB"
    )
    print(
        f"GPU memory reserved before loading: "
        f"{torch.cuda.memory_reserved() / 1024**3:.2f} GB"
    )


# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------

print("\nLoading model...")

model = UNet3D(
    in_channels=4,
    num_classes=NUM_CLASSES,
    base=BASE_CHANNELS,
)

checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE,
    weights_only=False,
)

# Handle possible checkpoint formats
if isinstance(checkpoint, dict):

    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]

    elif "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]

    else:
        state_dict = checkpoint

else:
    state_dict = checkpoint


model.load_state_dict(state_dict)

model.to(DEVICE)
model.eval()

print("Model loaded successfully.")


# --------------------------------------------------
# PARAMETER COUNT
# --------------------------------------------------

num_params = sum(
    p.numel()
    for p in model.parameters()
    if p.requires_grad
)

print(f"Trainable parameters: {num_params:,}")


# --------------------------------------------------
# PATIENT VALIDATION
# --------------------------------------------------

print("\nEvaluating:")
print(PATIENT_FILE)

if not PATIENT_FILE.exists():
    raise FileNotFoundError(
        f"Patient file not found: {PATIENT_FILE}"
    )

try:

    result = evaluate_patient(
        model=model,
        file=PATIENT_FILE,
        device=DEVICE,
    )

    print("\n" + "=" * 70)
    print("VALIDATION SUCCESSFUL")
    print("=" * 70)

    print(f"Patient: {result['patient_id']}")

    # Print available metrics
    for key, value in result.items():

        if key == "patient_id":
            continue

        if isinstance(value, (int, float)):

            if "dice" in key.lower():
                print(f"{key}: {value:.4f}")

            elif "hd95" in key.lower():
                print(f"{key}: {value:.4f}")

            else:
                print(f"{key}: {value:.4f}")

    if DEVICE.type == "cuda":

        print("\nGPU memory after evaluation:")

        print(
            f"Allocated: "
            f"{torch.cuda.memory_allocated() / 1024**3:.2f} GB"
        )

        print(
            f"Reserved: "
            f"{torch.cuda.memory_reserved() / 1024**3:.2f} GB"
        )

except torch.cuda.OutOfMemoryError:

    print("\n" + "=" * 70)
    print("CUDA OUT OF MEMORY")
    print("=" * 70)

    if DEVICE.type == "cuda":
        print(
            f"Allocated: "
            f"{torch.cuda.memory_allocated() / 1024**3:.2f} GB"
        )

        print(
            f"Reserved: "
            f"{torch.cuda.memory_reserved() / 1024**3:.2f} GB"
        )

    raise

finally:

    del model

    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()

print("\nTest finished.")
