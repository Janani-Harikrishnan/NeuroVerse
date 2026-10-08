# ============================================================
# NEUROVERSE
# FINAL E1-E4 SINGLE PATIENT VISUALIZATION
#
# Patient:
#     BraTS2021_00032
#
# Outputs:
#     Individual visualization for E1
#     Individual visualization for E2
#     Individual visualization for E3
#     Individual visualization for E4
#     One combined E1-E4 comparison
#     Raw prediction masks (.npy)
#
# IMPORTANT:
# This script performs INFERENCE ONLY.
# It does NOT train anything.
# ============================================================

import os
import sys
import importlib
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt

from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from monai.inferers import sliding_window_inference


# ============================================================
# 1. ROOT
# ============================================================

ROOT = Path.home() / "Neuroverse"

PATIENT_ID = "BraTS2021_00032"

PATIENT_FILE = (
    ROOT
    / "IID_Clients"
    / "GlobalTest"
    / "test"
    / f"{PATIENT_ID}.npz"
)

OUTPUT_DIR = ROOT / "Visualization_Results"

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# 2. FINAL MODEL CHECKPOINTS
# ============================================================

CHECKPOINTS = {

    "E1": (
        ROOT
        / "Federated"
        / "Full_results"
        / "E1_3D_UNet_FedAvg"
        / "raw_results"
        / "global_model_final.pth"
    ),

    "E2": (
        ROOT
        / "Federated"
        / "Full_results"
        / "E2_Swin_FedAvg"
        / "raw_results"
        / "global_model_final.pth"
    ),

    "E3": (
        ROOT
        / "Federated"
        / "Full_results"
        / "E3_Swin_Adaptive"
        / "raw_results"
        / "global_model_final.pth"
    ),

    "E4": (
        ROOT
        / "Federated"
        / "Full_results"
        / "E4_Swin_QuantumKernel"
        / "raw_results"
        / "global_model_final.pth"
    ),
}


# ============================================================
# 3. DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)

print("=" * 80)
print("NEUROVERSE FINAL VISUALIZATION")
print("=" * 80)

print("Patient :", PATIENT_ID)
print("Device  :", DEVICE)

if torch.cuda.is_available():
    print(
        "GPU     :",
        torch.cuda.get_device_name(0)
    )

print()


# ============================================================
# 4. VERIFY FILES
# ============================================================

if not PATIENT_FILE.exists():

    raise FileNotFoundError(
        f"\nPatient file does not exist:\n{PATIENT_FILE}"
    )

for name, path in CHECKPOINTS.items():

    if not path.exists():

        raise FileNotFoundError(
            f"\n{name} checkpoint does not exist:\n{path}"
        )

print("All files found.")
print()


# ============================================================
# 5. LOAD PATIENT
# ============================================================

data = np.load(
    PATIENT_FILE
)

print("NPZ keys:", list(data.keys()))

image = data["image"]

if "mask" in data:
    ground_truth = data["mask"]
else:
    ground_truth = None

print(
    "Original image shape:",
    image.shape
)

if ground_truth is not None:

    print(
        "Ground truth shape:",
        ground_truth.shape
    )


# ============================================================
# 6. STANDARDIZE IMAGE AXES
# ============================================================

if image.ndim != 4:

    raise ValueError(
        f"Expected 4D image. Got {image.shape}"
    )


# Expected:
#
# [4, H, W, D]
#

if image.shape[0] == 4:

    pass

elif image.shape[-1] == 4:

    image = np.moveaxis(
        image,
        -1,
        0
    )

else:

    raise ValueError(
        f"Could not identify 4 MRI channels "
        f"in shape {image.shape}"
    )


# ============================================================
# 7. STANDARDIZE MASK
# ============================================================

if ground_truth is not None:

    ground_truth = np.squeeze(
        ground_truth
    )

    if ground_truth.ndim != 3:

        raise ValueError(
            f"Ground truth must be 3D. "
            f"Got {ground_truth.shape}"
        )

    # Some BraTS datasets use label 4.
    # Convert 4 -> 3.
    ground_truth = ground_truth.copy()

    ground_truth[
        ground_truth == 4
    ] = 3


# ============================================================
# 8. IMAGE TYPE
# ============================================================

image = image.astype(
    np.float32
)


# ============================================================
# 9. NORMALIZATION
# ============================================================
#
# IMPORTANT:
# We only normalize non-zero brain voxels.
#
# This prevents the zero background from affecting
# the intensity normalization.
#
# ============================================================

def normalize_modalities(img):

    img = img.copy()

    for c in range(
        img.shape[0]
    ):

        channel = img[c]

        mask = (
            np.isfinite(channel)
            & (channel != 0)
        )

        if not np.any(mask):
            continue

        values = channel[mask]

        mean = values.mean()
        std = values.std()

        if std > 1e-8:

            channel[mask] = (
                values - mean
            ) / std

        img[c] = channel

    return img


image = normalize_modalities(
    image
)


# ============================================================
# 10. TENSOR
# ============================================================

image_tensor = torch.from_numpy(
    image
).float()

# [C,H,W,D]
#
# -> [1,C,H,W,D]

image_tensor = image_tensor.unsqueeze(
    0
)

image_tensor = image_tensor.to(
    DEVICE
)

print(
    "Model input:",
    tuple(image_tensor.shape)
)


# ============================================================
# 11. MODEL CHECKPOINT UTILITIES
# ============================================================

def extract_state_dict(
    checkpoint
):

    if not isinstance(
        checkpoint,
        dict
    ):

        raise RuntimeError(
            "Checkpoint is not a dictionary."
        )

    possible_keys = [

        "state_dict",

        "model_state_dict",

        "model",

        "weights",

        "parameters",

    ]

    for key in possible_keys:

        if key in checkpoint:

            value = checkpoint[key]

            if isinstance(
                value,
                dict
            ):

                return value

    # Check whether checkpoint itself
    # is a state dictionary.

    if all(
        isinstance(
            v,
            torch.Tensor
        )
        for v in checkpoint.values()
    ):

        return checkpoint

    raise RuntimeError(
        "Could not locate state_dict."
    )


def clean_state_dict(
    state_dict
):

    cleaned = {}

    prefixes = [

        "module.",

        "model.",

        "network.",

        "net.",

    ]

    for key, value in state_dict.items():

        new_key = key

        changed = True

        while changed:

            changed = False

            for prefix in prefixes:

                if new_key.startswith(
                    prefix
                ):

                    new_key = new_key[
                        len(prefix):
                    ]

                    changed = True

        cleaned[new_key] = value

    return cleaned


# ============================================================
# 12. E1 MODEL
# ============================================================

def load_e1():

    print(
        "\nLoading E1..."
    )

    project = (
        ROOT
        / "Federated"
        / "FedAvg_BraTS"
    )

    sys.path.insert(
        0,
        str(project)
    )

    try:

        from pytorchexample.task import (
            UNet3D
        )

        model = UNet3D(
            in_channels=4,
            num_classes=4,
            base=16
        )

    except Exception as e:

        raise RuntimeError(
            "\nCould not construct E1 UNet3D.\n"
            f"{e}"
        )

    checkpoint = torch.load(
        CHECKPOINTS["E1"],
        map_location="cpu",
        weights_only=False
    )

    state_dict = extract_state_dict(
        checkpoint
    )

    state_dict = clean_state_dict(
        state_dict
    )

    model.load_state_dict(
        state_dict,
        strict=True
    )

    model = model.to(
        DEVICE
    )

    model.eval()

    print(
        "E1 loaded successfully."
    )

    return model


# ============================================================
# 13. LOAD SWIN MODEL FROM PROJECT
# ============================================================

def load_swin(
    experiment,
    folder
):

    print(
        f"\nLoading {experiment}..."
    )

    project = (
        ROOT
        / "Federated"
        / folder
    )

    # --------------------------------------------------------
    # Remove previously imported pytorchexample modules
    # --------------------------------------------------------

    for key in list(
        sys.modules.keys()
    ):

        if (
            key == "pytorchexample"
            or
            key.startswith(
                "pytorchexample."
            )
        ):

            del sys.modules[key]

    # --------------------------------------------------------
    # Project path first
    # --------------------------------------------------------

    sys.path.insert(
        0,
        str(project)
    )

    try:

        module = importlib.import_module(
            "pytorchexample.swin_unetr"
        )

    except Exception as e:

        raise RuntimeError(
            f"\nCould not import {experiment} "
            f"Swin architecture.\n\n"
            f"Project:\n{project}\n\n"
            f"Error:\n{e}"
        )

    model = None

    # --------------------------------------------------------
    # create_model()
    # --------------------------------------------------------

    if hasattr(
        module,
        "create_model"
    ):

        try:

            model = module.create_model()

            print(
                "Used project create_model()."
            )

        except Exception as e:

            print(
                "create_model() failed:"
            )

            print(e)

    # --------------------------------------------------------
    # SwinUNETR
    # --------------------------------------------------------

    if (
        model is None
        and
        hasattr(
            module,
            "SwinUNETR"
        )
    ):

        SwinUNETR = (
            module.SwinUNETR
        )

        try:

            model = SwinUNETR(
                in_channels=4,
                out_channels=4,
                feature_size=24,
                spatial_dims=3,
                use_checkpoint=False
            )

        except TypeError:

            model = SwinUNETR(
                img_size=(
                    96,
                    96,
                    96
                ),
                in_channels=4,
                out_channels=4,
                feature_size=24,
                spatial_dims=3,
                use_checkpoint=False
            )

    if model is None:

        raise RuntimeError(
            f"\nCould not create {experiment} model."
        )

    # --------------------------------------------------------
    # Load checkpoint
    # --------------------------------------------------------

    checkpoint = torch.load(
        CHECKPOINTS[experiment],
        map_location="cpu",
        weights_only=False
    )

    state_dict = extract_state_dict(
        checkpoint
    )

    state_dict = clean_state_dict(
        state_dict
    )

    try:

        model.load_state_dict(
            state_dict,
            strict=True
        )

    except RuntimeError as e:

        raise RuntimeError(
            f"\n{experiment} checkpoint "
            f"does NOT match the model "
            f"architecture being constructed.\n\n"
            f"This is important — do NOT use "
            f"this prediction.\n\n"
            f"{e}"
        )

    model = model.to(
        DEVICE
    )

    model.eval()

    print(
        f"{experiment} loaded successfully."
    )

    return model


# ============================================================
# 14. INFERENCE
# ============================================================

ROI_SIZE = (
    96,
    96,
    96
)

OVERLAP = 0.50


def predict(
    model,
    name
):

    print(
        f"\nRunning inference: {name}"
    )

    with torch.inference_mode():

        output = sliding_window_inference(

            inputs=image_tensor,

            roi_size=ROI_SIZE,

            sw_batch_size=1,

            predictor=model,

            overlap=OVERLAP,

            mode="gaussian"

        )

    # Handle tuple

    if isinstance(
        output,
        (tuple, list)
    ):

        output = output[0]

    # Handle dictionary

    elif isinstance(
        output,
        dict
    ):

        for key in [
            "out",
            "logits",
            "output",
            "pred"
        ]:

            if key in output:

                output = output[key]

                break

    print(
        "Output shape:",
        tuple(output.shape)
    )

    # --------------------------------------------------------
    # IMPORTANT:
    #
    # segmentation = argmax over classes
    # --------------------------------------------------------

    prediction = torch.argmax(
        output,
        dim=1
    )

    prediction = (
        prediction[0]
        .detach()
        .cpu()
        .numpy()
        .astype(np.uint8)
    )

    # Convert label 4 if it somehow occurs
    prediction[
        prediction == 4
    ] = 3

    labels, counts = np.unique(
        prediction,
        return_counts=True
    )

    total = prediction.size

    print(
        f"\n{name} prediction labels:"
    )

    for label, count in zip(
        labels,
        counts
    ):

        print(
            f"  Label {label}: "
            f"{count:,} voxels "
            f"({100*count/total:.2f}%)"
        )

    # --------------------------------------------------------
    # Tumor percentage
    # --------------------------------------------------------

    tumor_voxels = np.sum(
        prediction > 0
    )

    print(
        f"  Tumor voxels: "
        f"{tumor_voxels:,}"
    )

    print(
        f"  Tumor percentage: "
        f"{100*tumor_voxels/total:.2f}%"
    )

    return prediction


# ============================================================
# 15. LOAD MODELS
# ============================================================

models = {}

models["E1"] = load_e1()

models["E2"] = load_swin(
    "E2",
    "Swin_FedAvg"
)

models["E3"] = load_swin(
    "E3",
    "Swin_Adaptive_FL"
)

models["E4"] = load_swin(
    "E4",
    "Quantum_Kernel_E3"
)


# ============================================================
# 16. PREDICT
# ============================================================

predictions = {}

for name in [
    "E1",
    "E2",
    "E3",
    "E4"
]:

    predictions[name] = predict(
        models[name],
        name
    )


# ============================================================
# 17. SAVE RAW MASKS
# ============================================================

for name, pred in predictions.items():

    np.save(
        OUTPUT_DIR
        / f"{PATIENT_ID}_{name}_prediction.npy",
        pred
    )


# ============================================================
# 18. CHOOSE BEST TUMOR SLICE
# ============================================================
#
# IMPORTANT:
# Use GROUND TRUTH to choose the same slice
# for every model.
#
# ============================================================

if ground_truth is not None:

    tumor_area = np.sum(
        ground_truth > 0,
        axis=(0, 1)
    )

    slice_index = int(
        np.argmax(tumor_area)
    )

else:

    # Fall back to E1

    tumor_area = np.sum(
        predictions["E1"] > 0,
        axis=(0, 1)
    )

    slice_index = int(
        np.argmax(tumor_area)
    )


print(
    "\nSelected slice:",
    slice_index
)


# ============================================================
# 19. GET SLICES
# ============================================================

# BraTS:
#
# 0 T1
# 1 T1ce
# 2 T2
# 3 FLAIR

flair = image[3]

flair_slice = flair[
    :, :,
    slice_index
]

if ground_truth is not None:

    gt_slice = ground_truth[
        :, :,
        slice_index
    ]

else:

    gt_slice = None


pred_slices = {}

for name, pred in predictions.items():

    pred_slices[name] = pred[
        :, :,
        slice_index
    ]


# ============================================================
# 20. SAME ORIENTATION
# ============================================================

flair_slice = np.rot90(
    flair_slice
)

if gt_slice is not None:

    gt_slice = np.rot90(
        gt_slice
    )

for name in pred_slices:

    pred_slices[name] = np.rot90(
        pred_slices[name]
    )


# ============================================================
# 21. COMMON DISPLAY RANGE
# ============================================================

valid = flair_slice[
    np.isfinite(flair_slice)
]

vmin = np.percentile(
    valid,
    1
)

vmax = np.percentile(
    valid,
    99
)


# ============================================================
# 22. SEGMENTATION COLORS
# ============================================================
#
# 0 = background
# 1 = NCR/NET
# 2 = edema
# 3 = enhancing tumor
#
# ============================================================

colors = [

    (0, 0, 0, 0),

    (1.0, 0.15, 0.15, 0.60),

    (0.20, 1.0, 0.25, 0.60),

    (0.20, 0.45, 1.0, 0.70),

]

SEG_CMAP = ListedColormap(
    colors
)


# ============================================================
# 23. DRAW FUNCTION
# ============================================================

def draw_overlay(
    ax,
    background,
    mask,
    title
):

    ax.imshow(
        background,
        cmap="gray",
        vmin=vmin,
        vmax=vmax,
        interpolation="nearest"
    )

    masked = np.ma.masked_where(
        mask == 0,
        mask
    )

    ax.imshow(
        masked,
        cmap=SEG_CMAP,
        vmin=0,
        vmax=3,
        interpolation="nearest"
    )

    ax.set_title(
        title,
        fontsize=14,
        fontweight="bold"
    )

    ax.axis("off")

    ax.set_aspect(
        "equal"
    )


# ============================================================
# 24. DRAW PURE SEGMENTATION
# ============================================================

def draw_pure_segmentation(
    ax,
    mask,
    title
):

    # Black/transparent background
    # represented with a dark neutral background.

    ax.imshow(
        np.zeros_like(mask),
        cmap="gray",
        vmin=0,
        vmax=1,
        interpolation="nearest"
    )

    masked = np.ma.masked_where(
        mask == 0,
        mask
    )

    ax.imshow(
        masked,
        cmap=SEG_CMAP,
        vmin=0,
        vmax=3,
        interpolation="nearest"
    )

    ax.set_title(
        title,
        fontsize=14,
        fontweight="bold"
    )

    ax.axis("off")

    ax.set_aspect(
        "equal"
    )


# ============================================================
# 25. CREATE INDIVIDUAL IMAGES
# ============================================================

for name in [
    "E1",
    "E2",
    "E3",
    "E4"
]:

    print(
        f"\nCreating {name} visualization..."
    )

    fig, axes = plt.subplots(
        1,
        4,
        figsize=(16, 5),
        dpi=300
    )

    # --------------------------------------------------------
    # MRI
    # --------------------------------------------------------

    axes[0].imshow(
        flair_slice,
        cmap="gray",
        vmin=vmin,
        vmax=vmax,
        interpolation="nearest"
    )

    axes[0].set_title(
        "MRI (FLAIR)",
        fontsize=14
    )

    axes[0].axis("off")

    # --------------------------------------------------------
    # Ground Truth
    # --------------------------------------------------------

    if gt_slice is not None:

        draw_overlay(
            axes[1],
            flair_slice,
            gt_slice,
            "Ground Truth"
        )

    else:

        axes[1].imshow(
            flair_slice,
            cmap="gray",
            vmin=vmin,
            vmax=vmax
        )

        axes[1].set_title(
            "Ground Truth"
        )

        axes[1].axis("off")

    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    draw_overlay(
        axes[2],
        flair_slice,
        pred_slices[name],
        f"{name} Prediction"
    )

    # --------------------------------------------------------
    # Pure segmentation
    # --------------------------------------------------------

    draw_pure_segmentation(
        axes[3],
        pred_slices[name],
        "Predicted Segmentation"
    )

    fig.suptitle(
        f"BraTS2021 {PATIENT_ID} — {name}",
        fontsize=17,
        fontweight="bold"
    )

    # --------------------------------------------------------
    # Legend
    # --------------------------------------------------------

    legend = [

        Patch(
            facecolor=(1.0, 0.15, 0.15),
            label="NCR/NET"
        ),

        Patch(
            facecolor=(0.20, 1.0, 0.25),
            label="Edema"
        ),

        Patch(
            facecolor=(0.20, 0.45, 1.0),
            label="Enhancing Tumor"
        ),

    ]

    fig.legend(
        handles=legend,
        loc="lower center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, -0.02)
    )

    plt.tight_layout(
        rect=[
            0,
            0.06,
            1,
            0.94
        ]
    )

    output = (
        OUTPUT_DIR
        / f"{PATIENT_ID}_{name}_visualization.png"
    )

    plt.savefig(
        output,
        dpi=300,
        bbox_inches="tight",
        facecolor="white"
    )

    plt.close()

    print(
        "Saved:",
        output
    )


# ============================================================
# 26. COMBINED E1-E4 IMAGE
# ============================================================
#
# One clean image:
#
# E1 | E2 | E3 | E4
#
# Same MRI
# Same slice
# Same orientation
# Same scale
# Same colors
#
# ============================================================

fig, axes = plt.subplots(
    1,
    4,
    figsize=(16, 4.5),
    dpi=300
)

for ax, name in zip(
    axes,
    ["E1", "E2", "E3", "E4"]
):

    draw_overlay(
        ax,
        flair_slice,
        pred_slices[name],
        name
    )


fig.suptitle(
    f"BraTS2021 {PATIENT_ID} — E1 to E4 Segmentation Comparison",
    fontsize=17,
    fontweight="bold"
)


legend = [

    Patch(
        facecolor=(1.0, 0.15, 0.15),
        label="NCR/NET"
    ),

    Patch(
        facecolor=(0.20, 1.0, 0.25),
        label="Edema"
    ),

    Patch(
        facecolor=(0.20, 0.45, 1.0),
        label="Enhancing Tumor"
    ),

]


fig.legend(
    handles=legend,
    loc="lower center",
    ncol=3,
    frameon=False,
    bbox_to_anchor=(0.5, -0.03)
)


plt.tight_layout(
    rect=[
        0,
        0.08,
        1,
        0.94
    ]
)


combined_output = (
    OUTPUT_DIR
    / f"{PATIENT_ID}_E1_E2_E3_E4_COMBINED.png"
)

plt.savefig(
    combined_output,
    dpi=300,
    bbox_inches="tight",
    facecolor="white"
)

plt.close()


# ============================================================
# 27. FINISHED
# ============================================================

print("\n")
print("=" * 80)
print("VISUALIZATION COMPLETE")
print("=" * 80)

print(
    "\nOutput directory:"
)

print(
    OUTPUT_DIR
)

print("\nIndividual images:")

for name in [
    "E1",
    "E2",
    "E3",
    "E4"
]:

    print(
        OUTPUT_DIR
        / f"{PATIENT_ID}_{name}_visualization.png"
    )

print("\nCombined image:")

print(
    combined_output
)

print("\nRaw prediction masks:")

for name in [
    "E1",
    "E2",
    "E3",
    "E4"
]:

    print(
        OUTPUT_DIR
        / f"{PATIENT_ID}_{name}_prediction.npy"
    )

print("\nDone.")
