import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import torch


# =========================================================
# PATHS
# =========================================================

ROOT = Path.home() / "Neuroverse"

GLOBAL_TEST = (
    ROOT / "IID_Clients" / "GlobalTest" / "test"
)

MODEL_PATH = (
    ROOT / "Federated"
    / "FedAvg_BraTS"
    / "results"
    / "global_model_final.pth"
)

OUTPUT_DIR = (
    ROOT / "Federated"
    / "FedAvg_BraTS"
    / "results"
    / "visualizations"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# =========================================================
# IMPORT E1 MODEL
# =========================================================

from pytorchexample.task import get_model


# =========================================================
# SELECT PATIENT
# =========================================================

files = sorted(
    GLOBAL_TEST.glob("*.npz")
)

if len(files) == 0:
    raise RuntimeError(
        f"No patients found in {GLOBAL_TEST}"
    )

print(
    f"Found {len(files)} GlobalTest patients."
)

# Example:
# python visualize_patient.py 0
# python visualize_patient.py 10

patient_index = (
    int(sys.argv[1])
    if len(sys.argv) > 1
    else 0
)

if patient_index < 0 or patient_index >= len(files):
    raise ValueError(
        f"Patient index must be between "
        f"0 and {len(files)-1}"
    )

patient_file = files[patient_index]

print()
print("Selected patient:")
print(patient_file.name)


# =========================================================
# LOAD PATIENT
# =========================================================

data = np.load(patient_file)

print()
print("NPZ contents:")

for key in data.files:
    print(
        f"  {key}: "
        f"{data[key].shape} "
        f"{data[key].dtype}"
    )


# =========================================================
# GET IMAGE + MASK
# =========================================================

if "image" in data:
    image = data["image"]

elif "images" in data:
    image = data["images"]

else:
    raise KeyError(
        "Could not find image/images in NPZ"
    )


if "mask" in data:
    mask = data["mask"]

elif "label" in data:
    mask = data["label"]

elif "seg" in data:
    mask = data["seg"]

else:
    raise KeyError(
        "Could not find mask/label/seg in NPZ"
    )


print()
print("Image shape:", image.shape)
print("Mask shape :", mask.shape)


# =========================================================
# MODEL
# =========================================================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print()
print("Device:", device)

model = get_model()

checkpoint = torch.load(
    MODEL_PATH,
    map_location=device
)

if isinstance(checkpoint, dict):

    if "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]

model.load_state_dict(
    checkpoint
)

model.to(device)
model.eval()


# =========================================================
# SLIDING-WINDOW INFERENCE
# =========================================================

PATCH_SIZE = (
    96,
    96,
    96
)

OVERLAP = 0.5

print()
print("=" * 60)
print("RUNNING SLIDING-WINDOW PREDICTION")
print("=" * 60)


# ---------------------------------------------------------
# Original dimensions
# ---------------------------------------------------------

channels, H, W, D = image.shape


# ---------------------------------------------------------
# Padding
# ---------------------------------------------------------

pad_h = max(
    0,
    PATCH_SIZE[0] - H
)

pad_w = max(
    0,
    PATCH_SIZE[1] - W
)

pad_d = max(
    0,
    PATCH_SIZE[2] - D
)

image_padded = np.pad(
    image,
    (
        (0, 0),
        (0, pad_h),
        (0, pad_w),
        (0, pad_d)
    ),
    mode="constant"
)

_, PH, PW, PD = image_padded.shape

print(
    "Original volume:",
    image.shape
)

print(
    "Padded volume:",
    image_padded.shape
)


# ---------------------------------------------------------
# Calculate stride
# ---------------------------------------------------------

stride_h = int(
    PATCH_SIZE[0] * (1 - OVERLAP)
)

stride_w = int(
    PATCH_SIZE[1] * (1 - OVERLAP)
)

stride_d = int(
    PATCH_SIZE[2] * (1 - OVERLAP)
)


# ---------------------------------------------------------
# Generate patch positions
# ---------------------------------------------------------

def get_starts(
    size,
    patch,
    stride
):

    if size <= patch:
        return [0]

    starts = list(
        range(
            0,
            size - patch + 1,
            stride
        )
    )

    last_start = size - patch

    if starts[-1] != last_start:
        starts.append(last_start)

    return starts


starts_h = get_starts(
    PH,
    PATCH_SIZE[0],
    stride_h
)

starts_w = get_starts(
    PW,
    PATCH_SIZE[1],
    stride_w
)

starts_d = get_starts(
    PD,
    PATCH_SIZE[2],
    stride_d
)


total_patches = (
    len(starts_h)
    * len(starts_w)
    * len(starts_d)
)

print(
    "Stride:",
    (
        stride_h,
        stride_w,
        stride_d
    )
)

print(
    "Total patches:",
    total_patches
)


# =========================================================
# ACCUMULATION ARRAYS
# =========================================================

NUM_CLASSES = 4

prob_sum = np.zeros(
    (
        NUM_CLASSES,
        PH,
        PW,
        PD
    ),
    dtype=np.float32
)

count_map = np.zeros(
    (
        PH,
        PW,
        PD
    ),
    dtype=np.float32
)


# =========================================================
# RUN MODEL ON EACH PATCH
# =========================================================

patch_number = 0

with torch.no_grad():

    for h in starts_h:

        for w in starts_w:

            for d in starts_d:

                patch_number += 1

                patch = image_padded[
                    :,
                    h:h + PATCH_SIZE[0],
                    w:w + PATCH_SIZE[1],
                    d:d + PATCH_SIZE[2]
                ]

                patch_tensor = torch.from_numpy(
                    patch
                ).float()

                patch_tensor = (
                    patch_tensor
                    .unsqueeze(0)
                    .to(device)
                )

                # Model prediction
                output = model(
                    patch_tensor
                )

                # Convert logits to probabilities
                probabilities = torch.softmax(
                    output,
                    dim=1
                )

                probabilities = (
                    probabilities
                    .squeeze(0)
                    .cpu()
                    .numpy()
                )

                # Accumulate probabilities
                prob_sum[
                    :,
                    h:h + PATCH_SIZE[0],
                    w:w + PATCH_SIZE[1],
                    d:d + PATCH_SIZE[2]
                ] += probabilities

                # Count overlapping predictions
                count_map[
                    h:h + PATCH_SIZE[0],
                    w:w + PATCH_SIZE[1],
                    d:d + PATCH_SIZE[2]
                ] += 1

                print(
                    f"\rProcessing patch "
                    f"{patch_number}/"
                    f"{total_patches}",
                    end=""
                )

print()


# =========================================================
# AVERAGE OVERLAPPING PREDICTIONS
# =========================================================

prob_sum /= np.maximum(
    count_map[None, ...],
    1e-8
)


# =========================================================
# FINAL SEGMENTATION
# =========================================================

prediction = np.argmax(
    prob_sum,
    axis=0
)


# Remove padding
prediction = prediction[
    :H,
    :W,
    :D
]

print()
print(
    "Prediction shape:",
    prediction.shape
)


# =========================================================
# FIND BEST TUMOR SLICE
# =========================================================

tumor_pixels = np.sum(
    mask > 0,
    axis=(0, 1)
)

slice_index = int(
    np.argmax(tumor_pixels)
)

print()
print(
    "Selected slice:",
    slice_index
)

print(
    "Ground-truth tumor pixels:",
    tumor_pixels[slice_index]
)


# =========================================================
# MRI DISPLAY
# =========================================================

# Channels:
# 0 = T1
# 1 = T1ce
# 2 = T2
# 3 = FLAIR

if image.shape[0] == 4:

    flair = image[3]

else:

    flair = image[0]


mri_slice = flair[
    :, :, slice_index
]

gt_slice = mask[
    :, :, slice_index
]

pred_slice = prediction[
    :, :, slice_index
]


# =========================================================
# NORMALIZE MRI
# =========================================================

mri_min = mri_slice.min()
mri_max = mri_slice.max()

mri_slice = (
    mri_slice - mri_min
) / (
    mri_max - mri_min + 1e-8
)


# =========================================================
# VISUALIZATION
# =========================================================

fig, axes = plt.subplots(
    1,
    4,
    figsize=(18, 5)
)


# ---------------------------------------------------------
# 1. MRI
# ---------------------------------------------------------

axes[0].imshow(
    mri_slice.T,
    cmap="gray",
    origin="lower"
)

axes[0].set_title(
    "MRI (FLAIR)",
    fontsize=13
)


# ---------------------------------------------------------
# 2. GROUND TRUTH
# ---------------------------------------------------------

axes[1].imshow(
    mri_slice.T,
    cmap="gray",
    origin="lower"
)

axes[1].imshow(
    np.ma.masked_where(
        gt_slice.T == 0,
        gt_slice.T
    ),
    alpha=0.6,
    origin="lower"
)

axes[1].set_title(
    "Ground Truth",
    fontsize=13
)


# ---------------------------------------------------------
# 3. E1 PREDICTION OVERLAY
# ---------------------------------------------------------

axes[2].imshow(
    mri_slice.T,
    cmap="gray",
    origin="lower"
)

axes[2].imshow(
    np.ma.masked_where(
        pred_slice.T == 0,
        pred_slice.T
    ),
    alpha=0.6,
    origin="lower"
)

axes[2].set_title(
    "E1 Prediction",
    fontsize=13
)


# ---------------------------------------------------------
# 4. PREDICTED SEGMENTATION
# ---------------------------------------------------------

axes[3].imshow(
    pred_slice.T,
    origin="lower"
)

axes[3].set_title(
    "Predicted Segmentation",
    fontsize=13
)


# ---------------------------------------------------------
# Remove axes
# ---------------------------------------------------------

for ax in axes:
    ax.axis("off")


# =========================================================
# TITLE
# =========================================================

fig.suptitle(
    f"E1 — 3D U-Net + FedAvg | "
    f"{patient_file.stem} | "
    f"Slice {slice_index}",
    fontsize=15
)

plt.tight_layout()


# =========================================================
# SAVE
# =========================================================

output_file = (
    OUTPUT_DIR
    / f"{patient_file.stem}_E1.png"
)

plt.savefig(
    output_file,
    dpi=300,
    bbox_inches="tight"
)

plt.show()


print()
print("=" * 60)
print("VISUALIZATION COMPLETE")
print("=" * 60)

print(
    "Patient:",
    patient_file.stem
)

print(
    "Slice:",
    slice_index
)

print(
    "Saved:",
    output_file
)