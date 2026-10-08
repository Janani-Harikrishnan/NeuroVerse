# ============================================================
# NEUROVERSE
# E1 vs E2 vs E3 vs E4
# SINGLE-PATIENT SEGMENTATION VISUALIZATION
#
# Patient: BraTS2021_00032
#
# Output:
# ~/Neuroverse/Visualization_Results/
#     BraTS2021_00032_E1_E2_E3_E4_segmentation.png
#
# IMPORTANT:
# - Uses SAVED models only
# - Does NOT train anything
# - Same patient for every model
# - Same axial slice for every model
# - Same FLAIR background
# - Same normalization
# - Same segmentation colors
# - Same image dimensions
# ============================================================

import os
import sys
import importlib
import importlib.util
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from monai.inferers import sliding_window_inference


# ============================================================
# 1. PATHS
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
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = (
    OUTPUT_DIR
    / f"{PATIENT_ID}_E1_E2_E3_E4_segmentation.png"
)


# ============================================================
# SAVED E1-E4 MODEL CHECKPOINTS
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
# 2. DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 70)
print("NEUROVERSE SEGMENTATION VISUALIZATION")
print("=" * 70)

print(f"Device: {DEVICE}")

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

print(f"Patient: {PATIENT_ID}")
print(f"Patient file: {PATIENT_FILE}")
print()


# ============================================================
# 3. CHECK FILES
# ============================================================

if not PATIENT_FILE.exists():
    raise FileNotFoundError(
        f"\nPatient file not found:\n{PATIENT_FILE}\n"
    )

for name, path in CHECKPOINTS.items():
    if not path.exists():
        raise FileNotFoundError(
            f"\n{name} checkpoint not found:\n{path}\n"
        )

print("All required files found.")
print()


# ============================================================
# 4. LOAD PATIENT
# ============================================================

print("Loading patient...")

data = np.load(PATIENT_FILE)

print("NPZ keys:", list(data.keys()))

if "image" not in data:
    raise KeyError(
        "The NPZ file does not contain an 'image' array."
    )

image = data["image"]

if "mask" in data:
    ground_truth = data["mask"]
else:
    ground_truth = None


# ============================================================
# 5. FIX IMAGE AXIS ORDER
# ============================================================
#
# Expected:
#
#     [4, H, W, D]
#
# Usually:
#
#     [4, 240, 240, 155]
#
# But this also handles:
#
#     [H, W, D, 4]
#
# ============================================================

print("Original image shape:", image.shape)

if image.ndim != 4:
    raise ValueError(
        f"Expected 4D image, got {image.shape}"
    )

if image.shape[0] == 4:
    # Already [C,H,W,D]
    pass

elif image.shape[-1] == 4:
    # Convert [H,W,D,C] -> [C,H,W,D]
    image = np.moveaxis(image, -1, 0)

else:
    raise ValueError(
        f"Cannot identify the 4 MRI channels in shape {image.shape}"
    )

print("Image after axis correction:", image.shape)


# ============================================================
# 6. FIX GROUND-TRUTH AXIS
# ============================================================

if ground_truth is not None:

    print("Original mask shape:", ground_truth.shape)

    # Remove unnecessary dimensions
    ground_truth = np.squeeze(ground_truth)

    # Expected mask [H,W,D]
    if ground_truth.ndim != 3:
        raise ValueError(
            f"Expected 3D mask, got {ground_truth.shape}"
        )

    print("Ground truth shape:", ground_truth.shape)


# ============================================================
# 7. CONVERT TO FLOAT32
# ============================================================

image = image.astype(np.float32)

if ground_truth is not None:
    ground_truth = ground_truth.astype(np.int64)


# ============================================================
# 8. MRI NORMALIZATION
# ============================================================
#
# Normalize EACH modality independently.
#
# This keeps the visualization consistent across models.
#
# ============================================================

print("\nNormalizing MRI channels...")

for c in range(image.shape[0]):

    channel = image[c]

    nonzero = channel[np.isfinite(channel) & (channel != 0)]

    if len(nonzero) == 0:
        continue

    p1 = np.percentile(nonzero, 1)
    p99 = np.percentile(nonzero, 99)

    if p99 > p1:
        channel = np.clip(channel, p1, p99)
        channel = (channel - p1) / (p99 - p1)

    image[c] = channel

print("Normalization complete.")


# ============================================================
# 9. TENSOR
# ============================================================

# [C,H,W,D]
image_tensor = torch.from_numpy(image).float()

# [1,C,H,W,D]
image_tensor = image_tensor.unsqueeze(0)

image_tensor = image_tensor.to(DEVICE)

print("\nModel input shape:", tuple(image_tensor.shape))


# ============================================================
# 10. SLIDING WINDOW SETTINGS
# ============================================================

ROI_SIZE = (96, 96, 96)

OVERLAP = 0.50

SW_BATCH_SIZE = 1


# ============================================================
# 11. EXTRACT STATE DICT
# ============================================================

def extract_state_dict(checkpoint):

    if isinstance(checkpoint, dict):

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

                if isinstance(value, dict):
                    return value

        # Sometimes the checkpoint itself is state_dict
        if all(
            isinstance(v, torch.Tensor)
            for v in checkpoint.values()
        ):
            return checkpoint

    raise RuntimeError(
        "Could not find a model state_dict inside checkpoint."
    )


# ============================================================
# 12. CLEAN STATE DICT KEYS
# ============================================================

def clean_state_dict(state_dict):

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

                if new_key.startswith(prefix):

                    new_key = new_key[len(prefix):]

                    changed = True

        cleaned[new_key] = value

    return cleaned


# ============================================================
# 13. LOAD E1
# ============================================================

def load_e1():

    print("\n" + "=" * 70)
    print("LOADING E1 - 3D U-NET")
    print("=" * 70)

    project_dir = ROOT / "Federated" / "FedAvg_BraTS"

    sys.path.insert(0, str(project_dir))

    try:

        from pytorchexample.task import UNet3D

        model = UNet3D(
            in_channels=4,
            num_classes=4,
            base=16
        )

    except Exception as e:

        raise RuntimeError(
            "\nCould not load E1 UNet3D.\n"
            "Expected architecture:\n"
            "UNet3D(in_channels=4, num_classes=4, base=16)\n\n"
            f"Original error:\n{e}"
        )

    checkpoint = torch.load(
        CHECKPOINTS["E1"],
        map_location="cpu",
        weights_only=False
    )

    state_dict = extract_state_dict(checkpoint)

    state_dict = clean_state_dict(state_dict)

    model.load_state_dict(
        state_dict,
        strict=True
    )

    model.to(DEVICE)
    model.eval()

    print("E1 loaded successfully.")

    return model


# ============================================================
# 14. LOAD SWIN MODEL
# ============================================================

def load_swin_model(
    model_name,
    project_folder
):

    print("\n" + "=" * 70)
    print(f"LOADING {model_name} - SWIN UNETR")
    print("=" * 70)

    project_dir = ROOT / project_folder

    swin_file = (
        project_dir
        / "pytorchexample"
        / "swin_unetr.py"
    )

    if not swin_file.exists():

        raise FileNotFoundError(
            f"\nSwin architecture file not found:\n{swin_file}"
        )

    # --------------------------------------------------------
    # Remove previously loaded pytorchexample modules.
    # This is VERY IMPORTANT because E2/E3/E4 have separate
    # project folders and may have their own swin_unetr.py.
    # --------------------------------------------------------

    modules_to_remove = [
        key
        for key in list(sys.modules.keys())
        if key == "pytorchexample"
        or key.startswith("pytorchexample.")
    ]

    for key in modules_to_remove:
        del sys.modules[key]

    # --------------------------------------------------------
    # Put this project's folder first.
    # --------------------------------------------------------

    old_sys_path = list(sys.path)

    sys.path.insert(
        0,
        str(project_dir)
    )

    try:

        # Import the project's own module
        module = importlib.import_module(
            "pytorchexample.swin_unetr"
        )

    except Exception as e:

        sys.path[:] = old_sys_path

        raise RuntimeError(
            f"\nCould not import {swin_file}\n\n"
            f"Original error:\n{e}"
        )

    # --------------------------------------------------------
    # Create model
    # --------------------------------------------------------

    model = None

    # First preference: project create_model()
    if hasattr(module, "create_model"):

        try:
            model = module.create_model()
            print("Created using create_model().")

        except TypeError:

            try:

                model = module.create_model(
                    in_channels=4,
                    out_channels=4,
                    feature_size=24,
                    spatial_dims=3,
                    use_checkpoint=False
                )

                print(
                    "Created using create_model("
                    "in_channels=4, out_channels=4, "
                    "feature_size=24)."
                )

            except Exception:
                model = None

        except Exception:
            model = None

    # --------------------------------------------------------
    # Second preference: SwinUNETR class
    # --------------------------------------------------------

    if model is None and hasattr(module, "SwinUNETR"):

        SwinUNETR = module.SwinUNETR

        try:

            model = SwinUNETR(
                in_channels=4,
                out_channels=4,
                feature_size=24,
                spatial_dims=3,
                use_checkpoint=False
            )

            print(
                "Created using SwinUNETR("
                "in_channels=4, out_channels=4, "
                "feature_size=24)."
            )

        except Exception as e:

            raise RuntimeError(
                f"\nCould not construct {model_name} SwinUNETR.\n"
                f"Error:\n{e}"
            )

    if model is None:

        raise RuntimeError(
            f"\nNo create_model() or SwinUNETR class "
            f"was found in:\n{swin_file}"
        )

    # --------------------------------------------------------
    # Load checkpoint
    # --------------------------------------------------------

    checkpoint = torch.load(
        CHECKPOINTS[model_name],
        map_location="cpu",
        weights_only=False
    )

    state_dict = extract_state_dict(checkpoint)

    state_dict = clean_state_dict(state_dict)

    # --------------------------------------------------------
    # Load weights
    # --------------------------------------------------------

    try:

        model.load_state_dict(
            state_dict,
            strict=True
        )

    except RuntimeError as strict_error:

        print(
            f"\nStrict loading failed for {model_name}."
        )

        # Try non-strict only to provide useful diagnostics.
        missing, unexpected = model.load_state_dict(
            state_dict,
            strict=False
        )

        if len(missing) > 0 or len(unexpected) > 0:

            raise RuntimeError(
                f"\n{model_name} checkpoint does not match "
                f"the architecture being loaded.\n\n"
                f"Missing keys: {len(missing)}\n"
                f"Unexpected keys: {len(unexpected)}\n\n"
                f"Original strict-loading error:\n"
                f"{strict_error}"
            )

    model.to(DEVICE)
    model.eval()

    print(f"{model_name} loaded successfully.")

    return model


# ============================================================
# 15. MODEL INFERENCE
# ============================================================

def run_segmentation(model, model_name):

    print("\n" + "-" * 70)
    print(f"Running inference: {model_name}")
    print("-" * 70)

    with torch.inference_mode():

        output = sliding_window_inference(
            inputs=image_tensor,
            roi_size=ROI_SIZE,
            sw_batch_size=SW_BATCH_SIZE,
            predictor=model,
            overlap=OVERLAP,
            mode="gaussian"
        )

    # --------------------------------------------------------
    # Handle models returning tuple/list/dict
    # --------------------------------------------------------

    if isinstance(output, (tuple, list)):

        output = output[0]

    elif isinstance(output, dict):

        possible_keys = [
            "out",
            "logits",
            "output",
            "pred"
        ]

        found = None

        for key in possible_keys:

            if key in output:

                found = output[key]
                break

        if found is None:

            raise RuntimeError(
                f"{model_name}: Could not identify "
                "prediction tensor in dictionary output."
            )

        output = found

    print(
        f"{model_name} raw output shape: "
        f"{tuple(output.shape)}"
    )

    # --------------------------------------------------------
    # Convert logits -> segmentation labels
    # --------------------------------------------------------

    prediction = torch.argmax(
        output,
        dim=1
    )

    prediction = prediction[0].detach().cpu().numpy()

    prediction = prediction.astype(np.uint8)

    print(
        f"{model_name} prediction shape: "
        f"{prediction.shape}"
    )

    print(
        f"{model_name} labels present: "
        f"{np.unique(prediction)}"
    )

    return prediction


# ============================================================
# 16. DETERMINE SAME VISUALIZATION SLICE
# ============================================================

def choose_best_slice(mask):

    if mask is None:

        # Center slice
        return image.shape[3] // 2

    tumor = mask > 0

    # Our expected mask shape is H,W,D
    tumor_per_slice = tumor.sum(
        axis=(0, 1)
    )

    slice_index = int(
        np.argmax(tumor_per_slice)
    )

    return slice_index


# ============================================================
# 17. FLAIR DISPLAY
# ============================================================
#
# BraTS channel convention:
#
# 0 = T1
# 1 = T1ce
# 2 = T2
# 3 = FLAIR
#
# ============================================================

FLAIR_CHANNEL = 3

flair = image[FLAIR_CHANNEL]


# ============================================================
# 18. LOAD ALL MODELS
# ============================================================

models = {}

models["E1"] = load_e1()

models["E2"] = load_swin_model(
    "E2",
    "Federated/Swin_FedAvg"
)

models["E3"] = load_swin_model(
    "E3",
    "Federated/Swin_Adaptive_FL"
)

models["E4"] = load_swin_model(
    "E4",
    "Federated/Quantum_Kernel_E3"
)


# ============================================================
# 19. RUN ALL FOUR MODELS
# ============================================================

predictions = {}

for name, model in models.items():

    predictions[name] = run_segmentation(
        model,
        name
    )


# ============================================================
# 20. DELETE MODELS FROM GPU
# ============================================================

for name in list(models.keys()):

    del models[name]

if torch.cuda.is_available():

    torch.cuda.empty_cache()


# ============================================================
# 21. SAME SLICE FOR EVERY MODEL
# ============================================================

slice_index = choose_best_slice(
    ground_truth
)

print("\n" + "=" * 70)
print(
    f"Visualization slice: {slice_index}"
)
print("=" * 70)


# ============================================================
# 22. FIX SLICE ORIENTATION
# ============================================================
#
# Everything is transposed in EXACTLY the same way.
# This guarantees all panels have identical orientation.
#
# ============================================================

flair_slice = flair[:, :, slice_index]

if ground_truth is not None:

    gt_slice = ground_truth[:, :, slice_index]

else:

    gt_slice = None


prediction_slices = {}

for name, prediction in predictions.items():

    prediction_slices[name] = (
        prediction[:, :, slice_index]
    )


# ============================================================
# 23. ROTATE EVERYTHING THE SAME WAY
# ============================================================

flair_slice = np.rot90(
    flair_slice
)

if gt_slice is not None:

    gt_slice = np.rot90(
        gt_slice
    )

for name in prediction_slices:

    prediction_slices[name] = np.rot90(
        prediction_slices[name]
    )


# ============================================================
# 24. SEGMENTATION COLORS
# ============================================================
#
# BraTS:
#
# 0 = Background
# 1 = NCR/NET
# 2 = ED
# 3 = ET
#
# We use:
#
# 1 = red
# 2 = green
# 3 = blue
#
# The exact same colors are used for GT and E1-E4.
#
# ============================================================

SEGMENT_COLORS = [
    (0, 0, 0, 0),       # background transparent
    (1, 0, 0, 0.55),    # label 1 - red
    (0, 1, 0, 0.55),    # label 2 - green
    (0, 0.4, 1, 0.65),  # label 3 - blue
]

SEG_CMAP = ListedColormap(
    SEGMENT_COLORS
)


# ============================================================
# 25. CREATE UNIFORM FIGURE
# ============================================================

# 5 panels:
#
# GT | E1 | E2 | E3 | E4
#
# All exactly same dimensions.
#

fig, axes = plt.subplots(
    1,
    5,
    figsize=(20, 4.5),
    dpi=300
)


# ============================================================
# 26. COMMON BACKGROUND NORMALIZATION
# ============================================================

# SAME grayscale limits for every panel.

valid_flair = flair_slice[
    np.isfinite(flair_slice)
]

vmin = np.percentile(
    valid_flair,
    1
)

vmax = np.percentile(
    valid_flair,
    99
)


# ============================================================
# 27. PANEL DRAWING FUNCTION
# ============================================================

def draw_segmentation_panel(
    ax,
    background,
    segmentation,
    title
):

    # -----------------------------------------------
    # Background
    # -----------------------------------------------

    ax.imshow(
        background,
        cmap="gray",
        vmin=vmin,
        vmax=vmax,
        interpolation="nearest"
    )

    # -----------------------------------------------
    # Segmentation
    # -----------------------------------------------

    if segmentation is not None:

        masked_seg = np.ma.masked_where(
            segmentation == 0,
            segmentation
        )

        ax.imshow(
            masked_seg,
            cmap=SEG_CMAP,
            vmin=0,
            vmax=3,
            interpolation="nearest"
        )

    # -----------------------------------------------
    # Uniform formatting
    # -----------------------------------------------

    ax.set_title(
        title,
        fontsize=13,
        fontweight="bold",
        pad=8
    )

    ax.axis("off")

    ax.set_aspect("equal")


# ============================================================
# 28. DRAW GROUND TRUTH
# ============================================================

if gt_slice is not None:

    draw_segmentation_panel(
        axes[0],
        flair_slice,
        gt_slice,
        "Ground Truth"
    )

else:

    axes[0].imshow(
        flair_slice,
        cmap="gray",
        vmin=vmin,
        vmax=vmax
    )

    axes[0].set_title(
        "Ground Truth",
        fontsize=13,
        fontweight="bold"
    )

    axes[0].axis("off")


# ============================================================
# 29. DRAW E1-E4
# ============================================================

for i, name in enumerate(
    ["E1", "E2", "E3", "E4"],
    start=1
):

    draw_segmentation_panel(
        axes[i],
        flair_slice,
        prediction_slices[name],
        name
    )


# ============================================================
# 30. MAIN TITLE
# ============================================================

fig.suptitle(
    f"BraTS2021 {PATIENT_ID} — Segmentation Comparison",
    fontsize=17,
    fontweight="bold",
    y=1.02
)


# ============================================================
# 31. LEGEND
# ============================================================

from matplotlib.patches import Patch

legend_elements = [

    Patch(
        facecolor=(1, 0, 0, 0.55),
        edgecolor="none",
        label="NCR/NET"
    ),

    Patch(
        facecolor=(0, 1, 0, 0.55),
        edgecolor="none",
        label="Edema"
    ),

    Patch(
        facecolor=(0, 0.4, 1, 0.65),
        edgecolor="none",
        label="Enhancing Tumor"
    ),
]


fig.legend(
    handles=legend_elements,
    loc="lower center",
    ncol=3,
    frameon=False,
    fontsize=10,
    bbox_to_anchor=(0.5, -0.04)
)


# ============================================================
# 32. FINAL LAYOUT
# ============================================================

plt.tight_layout(
    rect=[0, 0.07, 1, 0.96]
)


# ============================================================
# 33. SAVE
# ============================================================

plt.savefig(
    OUTPUT_FILE,
    dpi=300,
    bbox_inches="tight",
    facecolor="white"
)

plt.close(fig)


# ============================================================
# 34. DONE
# ============================================================

print("\n" + "=" * 70)
print("VISUALIZATION COMPLETE")
print("=" * 70)

print(f"\nSaved image:\n{OUTPUT_FILE}")

print(
    f"\nImage contains:"
    f"\n  Ground Truth"
    f"\n  E1 - 3D U-Net FedAvg"
    f"\n  E2 - Swin FedAvg"
    f"\n  E3 - Swin Adaptive FL"
    f"\n  E4 - Swin Quantum Kernel"
)

print(
    f"\nAll models use the SAME:"
    f"\n  Patient       : {PATIENT_ID}"
    f"\n  Slice         : {slice_index}"
    f"\n  FLAIR channel : {FLAIR_CHANNEL}"
    f"\n  ROI           : {ROI_SIZE}"
    f"\n  Overlap       : {OVERLAP}"
    f"\n  Colors        : identical"
    f"\n  Figure size   : identical"
)

print("\nDone.")
