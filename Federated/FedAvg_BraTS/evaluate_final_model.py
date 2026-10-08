import torch
from pathlib import Path

from pytorchexample.task import get_model, evaluate_global_test, get_device


# Paths
MODEL_PATH = Path.home() / "Neuroverse" / "Federated" / "FedAvg_BraTS" / "results" / "global_model_final.pth"

print("=" * 70)
print("FINAL GLOBAL MODEL EVALUATION")
print("=" * 70)

# Device
device = get_device()
print(f"Device: {device}")

# Create the same 3D U-Net architecture
model = get_model()

# Load the trained FedAvg global model
print(f"\nLoading model:")
print(MODEL_PATH)

state_dict = torch.load(
    MODEL_PATH,
    map_location="cpu"
)

model.load_state_dict(state_dict)

# Move model to GPU
model.to(device)
model.eval()

print("Model loaded successfully.")

# Evaluate ONLY on GlobalTest
print("\nStarting GlobalTest evaluation...")
print("Number of patients: 188")
print("=" * 70)

summary, patient_df = evaluate_global_test(model)

print("\n" + "=" * 70)
print("FINAL GLOBALTEST RESULTS")
print("=" * 70)

print("\nSummary:")
print(summary)

print("\nPatient-level results:")
print(patient_df)

print("\nEvaluation completed successfully.")
