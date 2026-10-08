from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path.home() / "Neuroverse"

MODEL_PATH = (
    ROOT
    / "Visualization_Models"
    / "3D_UNet"
    / "C3_best_model.pth"
)

GLOBAL_TEST_DIR = (
    ROOT
    / "IID_Clients"
    / "GlobalTest"
    / "test"
)

OUTPUT_DIR = (
    ROOT
    / "Visualization_Results"
    / "3D_UNet"
)

PATIENT_ID = "BraTS2021_00032"

IN_CHANNELS = 4
NUM_CLASSES = 4

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# 3D U-NET
# EXACT ARCHITECTURE OF C3 CHECKPOINT
# ============================================================

class DoubleConv(nn.Module):

    def __init__(
        self,
        in_channels,
        out_channels
    ):

        super().__init__()

        self.block = nn.Sequential(

            nn.Conv3d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False
            ),

            nn.InstanceNorm3d(
                out_channels
            ),

            nn.ReLU(
                inplace=True
            ),

            nn.Conv3d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False
            ),

            nn.InstanceNorm3d(
                out_channels
            ),

            nn.ReLU(
                inplace=True
            )
        )

    def forward(self, x):

        return self.block(x)


class UNet3D(nn.Module):

    def __init__(
        self,
        in_channels=4,
        num_classes=4,
        base=16
    ):

        super().__init__()

        # ====================================================
        # ENCODER
        # ====================================================

        self.enc1 = DoubleConv(
            in_channels,
            base
        )

        self.enc2 = DoubleConv(
            base,
            base * 2
        )

        self.enc3 = DoubleConv(
            base * 2,
            base * 4
        )

        # ====================================================
        # POOLING
        # ====================================================

        self.pool = nn.MaxPool3d(2)

        # ====================================================
        # BOTTLENECK
        # 64 -> 128
        # ====================================================

        self.bottleneck = DoubleConv(
            base * 4,
            base * 8
        )

        # ====================================================
        # DECODER
        # ====================================================

        self.up3 = nn.ConvTranspose3d(
            base * 8,
            base * 4,
            2,
            stride=2
        )

        self.dec3 = DoubleConv(
            base * 8,
            base * 4
        )

        self.up2 = nn.ConvTranspose3d(
            base * 4,
            base * 2,
            2,
            stride=2
        )

        self.dec2 = DoubleConv(
            base * 4,
            base * 2
        )

        self.up1 = nn.ConvTranspose3d(
            base * 2,
            base,
            2,
            stride=2
        )

        self.dec1 = DoubleConv(
            base * 2,
            base
        )

        # ====================================================
        # FINAL SEGMENTATION LAYER
        # ====================================================

        self.final = nn.Conv3d(
            base,
            num_classes,
            kernel_size=1
        )

    def forward(self, x):

        # ====================================================
        # ENCODER
        # ====================================================

        e1 = self.enc1(x)

        e2 = self.enc2(
            self.pool(e1)
        )

        e3 = self.enc3(
            self.pool(e2)
        )

        # ====================================================
        # BOTTLENECK
        # ====================================================

        b = self.bottleneck(
            self.pool(e3)
        )

        # ====================================================
        # DECODER
        # ====================================================

        d3 = self.up3(b)

        d3 = torch.cat(
            [d3, e3],
            dim=1
        )

        d3 = self.dec3(d3)

        d2 = self.up2(d3)

        d2 = torch.cat(
            [d2, e2],
            dim=1
        )

        d2 = self.dec2(d2)

        d1 = self.up1(d2)

        d1 = torch.cat(
            [d1, e1],
            dim=1
        )

        d1 = self.dec1(d1)

        # ====================================================
        # OUTPUT
        # ====================================================

        return self.final(d1)


# ============================================================
# LOAD MODEL
# ============================================================

def load_model():

    print()
    print("=" * 70)
    print("LOADING LOCAL 3D U-NET")
    print("=" * 70)

    print(f"Model: {MODEL_PATH}")
    print(f"Device: {DEVICE}")

    if not MODEL_PATH.exists():

        raise FileNotFoundError(
            f"Checkpoint not found:\n{MODEL_PATH}"
        )

    # --------------------------------------------------------
    # Load trusted checkpoint
    # PyTorch 2.6 requires weights_only=False for this
    # checkpoint format.
    # --------------------------------------------------------

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE,
        weights_only=False
    )

    print(
        f"Checkpoint type: {type(checkpoint)}"
    )

    # --------------------------------------------------------
    # Extract state dictionary
    # --------------------------------------------------------

    if isinstance(checkpoint, dict):

        if "model_state_dict" in checkpoint:

            print(
                "Found model_state_dict."
            )

            state_dict = (
                checkpoint["model_state_dict"]
            )

        elif "state_dict" in checkpoint:

            print(
                "Found state_dict."
            )

            state_dict = (
                checkpoint["state_dict"]
            )

        else:

            print(
                "Using checkpoint directly "
                "as state_dict."
            )

            state_dict = checkpoint

    else:

        raise TypeError(
            "Unexpected checkpoint format."
        )

    # --------------------------------------------------------
    # Create EXACT C3 model
    # --------------------------------------------------------

    model = UNet3D(
        in_channels=IN_CHANNELS,
        num_classes=NUM_CLASSES,
        base=16
    )

    # --------------------------------------------------------
    # Load weights
    # --------------------------------------------------------

    model.load_state_dict(
        state_dict,
        strict=True
    )

    model = model.to(
        DEVICE
    )

    model.eval()

    # --------------------------------------------------------
    # Parameter count
    # --------------------------------------------------------

    total_parameters = sum(
        p.numel()
        for p in model.parameters()
    )

    trainable_parameters = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print()
    print(
        f"Total parameters: "
        f"{total_parameters:,}"
    )

    print(
        f"Trainable parameters: "
        f"{trainable_parameters:,}"
    )

    print()
    print(
        "✓ Model loaded successfully."
    )

    return model


# ============================================================
# FIND PATIENT
# ============================================================

def find_patient():

    print()
    print("=" * 70)
    print("FINDING PATIENT")
    print("=" * 70)

    if not GLOBAL_TEST_DIR.exists():

        raise FileNotFoundError(
            f"GlobalTest directory not found:\n"
            f"{GLOBAL_TEST_DIR}"
        )

    # --------------------------------------------------------
    # Possible direct paths
    # --------------------------------------------------------

    possible_paths = [

        GLOBAL_TEST_DIR
        / f"{PATIENT_ID}.npz",

        GLOBAL_TEST_DIR
        / PATIENT_ID
        / f"{PATIENT_ID}.npz",

        GLOBAL_TEST_DIR
        / f"{PATIENT_ID}.npz"
    ]

    for path in possible_paths:

        if path.exists():

            print(
                f"Found: {path}"
            )

            return path

    # --------------------------------------------------------
    # Recursive search
    # --------------------------------------------------------

    matches = list(
        GLOBAL_TEST_DIR.rglob(
            f"{PATIENT_ID}.npz"
        )
    )

    if matches:

        print(
            f"Found: {matches[0]}"
        )

        return matches[0]

    # --------------------------------------------------------
    # Patient not found
    # --------------------------------------------------------

    available = list(
        GLOBAL_TEST_DIR.rglob(
            "*.npz"
        )
    )

    print(
        f"Patient {PATIENT_ID} not found."
    )

    if available:

        print()
        print(
            "Available patient files:"
        )

        for p in available[:20]:

            print(
                "  ",
                p
            )

    raise FileNotFoundError(
        f"{PATIENT_ID}.npz not found."
    )


# ============================================================
# LOAD PATIENT
# ============================================================

def load_patient(
    patient_path
):

    print()
    print("=" * 70)
    print("LOADING PATIENT")
    print("=" * 70)

    print(
        f"File: {patient_path}"
    )

    data = np.load(
        patient_path
    )

    print(
        "Available arrays:",
        list(data.keys())
    )

    if "image" not in data:

        raise KeyError(
            "Expected 'image' in NPZ file."
        )

    if "mask" not in data:

        raise KeyError(
            "Expected 'mask' in NPZ file."
        )

    image = data["image"]
    mask = data["mask"]

    print(
        f"Image shape: {image.shape}"
    )

    print(
        f"Mask shape: {mask.shape}"
    )

    print(
        f"Image dtype: {image.dtype}"
    )

    print(
        f"Mask dtype: {mask.dtype}"
    )

    print(
        "Ground-truth labels:",
        np.unique(mask)
    )

    return image, mask


# ============================================================
# NORMALIZE IMAGE
# ============================================================

def normalize_image(
    image
):

    image = image.astype(
        np.float32
    )

    normalized = np.zeros_like(
        image,
        dtype=np.float32
    )

    # --------------------------------------------------------
    # Normalize each MRI modality independently
    # --------------------------------------------------------

    for c in range(
        image.shape[0]
    ):

        channel = image[c]

        nonzero = (
            channel != 0
        )

        if np.any(nonzero):

            values = channel[
                nonzero
            ]

            mean = values.mean()
            std = values.std()

            if std > 0:

                normalized[c] = (
                    channel - mean
                ) / std

            else:

                normalized[c] = (
                    channel - mean
                )

        else:

            normalized[c] = channel

    return normalized


# ============================================================
# PAD VOLUME
# ============================================================

def pad_to_multiple(
    image,
    multiple=16
):

    _, x, y, z = image.shape

    pad_x = (
        multiple - x % multiple
    ) % multiple

    pad_y = (
        multiple - y % multiple
    ) % multiple

    pad_z = (
        multiple - z % multiple
    ) % multiple

    # --------------------------------------------------------
    # No padding needed
    # --------------------------------------------------------

    if (
        pad_x == 0
        and pad_y == 0
        and pad_z == 0
    ):

        return image, (
            x,
            y,
            z
        )

    # --------------------------------------------------------
    # Pad only at the end of each dimension
    # --------------------------------------------------------

    padded = np.pad(
        image,
        (
            (0, 0),
            (0, pad_x),
            (0, pad_y),
            (0, pad_z)
        ),
        mode="constant",
        constant_values=0
    )

    return padded, (
        x,
        y,
        z
    )


# ============================================================
# FULL-VOLUME PREDICTION
# ============================================================

@torch.no_grad()
def predict_full_volume(
    model,
    image
):

    print()
    print("=" * 70)
    print("RUNNING FULL-VOLUME PREDICTION")
    print("=" * 70)

    print(
        f"Original image shape: "
        f"{image.shape}"
    )

    # --------------------------------------------------------
    # Normalize
    # --------------------------------------------------------

    image = normalize_image(
        image
    )

    # --------------------------------------------------------
    # Pad
    # --------------------------------------------------------

    padded_image, spatial_shape = (
        pad_to_multiple(
            image,
            multiple=16
        )
    )

    print(
        f"Padded image shape: "
        f"{padded_image.shape}"
    )

    # --------------------------------------------------------
    # Convert to tensor
    # --------------------------------------------------------

    tensor = torch.from_numpy(
        padded_image
    ).unsqueeze(0)

    tensor = tensor.to(
        DEVICE,
        non_blocking=True
    )

    print(
        f"Model input shape: "
        f"{tuple(tensor.shape)}"
    )

    # --------------------------------------------------------
    # Forward pass
    # --------------------------------------------------------

    logits = model(
        tensor
    )

    print(
        f"Logits shape: "
        f"{tuple(logits.shape)}"
    )

    # --------------------------------------------------------
    # Convert logits to class probabilities
    # --------------------------------------------------------

    probabilities = torch.softmax(
        logits,
        dim=1
    )

    # --------------------------------------------------------
    # Crop padding
    # --------------------------------------------------------

    x, y, z = spatial_shape

    probabilities = probabilities[
        :,
        :,
        :x,
        :y,
        :z
    ]

    # --------------------------------------------------------
    # Select most probable class
    # --------------------------------------------------------

    prediction = torch.argmax(
        probabilities,
        dim=1
    )

    prediction = (
        prediction
        .squeeze(0)
        .cpu()
        .numpy()
        .astype(np.uint8)
    )

    print(
        f"Prediction shape: "
        f"{prediction.shape}"
    )

    print(
        "Predicted labels:",
        np.unique(prediction)
    )

    return prediction


# ============================================================
# SELECT LARGEST TUMOR SLICE
# ============================================================

def select_largest_tumor_slice(
    mask
):

    # --------------------------------------------------------
    # Whole Tumor = every non-background label
    # --------------------------------------------------------

    tumor_mask = (
        mask > 0
    )

    # Number of tumor pixels in every axial slice
    areas = tumor_mask.sum(
        axis=(0, 1)
    )

    slice_index = int(
        np.argmax(areas)
    )

    largest_area = int(
        areas[slice_index]
    )

    print()
    print("=" * 70)
    print("SELECTING VISUALIZATION SLICE")
    print("=" * 70)

    print(
        f"Selected axial slice: "
        f"{slice_index}"
    )

    print(
        f"Tumor pixels on slice: "
        f"{largest_area}"
    )

    return slice_index


# ============================================================
# PREPARE FLAIR DISPLAY
# ============================================================

def prepare_display_image(
    image,
    slice_index
):

    # --------------------------------------------------------
    # Channel 3 = FLAIR
    # --------------------------------------------------------

    flair = image[
        3,
        :,
        :,
        slice_index
    ]

    flair = flair.astype(
        np.float32
    )

    nonzero = (
        flair != 0
    )

    if np.any(nonzero):

        values = flair[
            nonzero
        ]

        low = np.percentile(
            values,
            1
        )

        high = np.percentile(
            values,
            99
        )

        flair = np.clip(
            flair,
            low,
            high
        )

        if high > low:

            flair = (
                flair - low
            ) / (
                high - low
            )

    return flair


# ============================================================
# CREATE COLORED SEGMENTATION MAP
# ============================================================

def create_segmentation_map(
    mask
):

    """
    Convert class labels into an RGB segmentation image.

    Label 0 = Background
    Label 1 = NCR/NET       -> Yellow
    Label 2 = Edema         -> Green
    Label 3 = Enhancing Tumor -> Red
    """

    h, w = mask.shape

    segmentation = np.zeros(
        (h, w, 3),
        dtype=np.float32
    )

    # --------------------------------------------------------
    # Background
    # --------------------------------------------------------

    background = (
        mask == 0
    )

    segmentation[
        background
    ] = [
        0.0,
        0.0,
        0.0
    ]

    # --------------------------------------------------------
    # Label 1
    # NCR / NET
    # Yellow
    # --------------------------------------------------------

    label1 = (
        mask == 1
    )

    segmentation[
        label1
    ] = [
        1.0,
        1.0,
        0.0
    ]

    # --------------------------------------------------------
    # Label 2
    # Edema
    # Green
    # --------------------------------------------------------

    label2 = (
        mask == 2
    )

    segmentation[
        label2
    ] = [
        0.0,
        1.0,
        0.0
    ]

    # --------------------------------------------------------
    # Label 3
    # Enhancing Tumor
    # Red
    # --------------------------------------------------------

    label3 = (
        mask == 3
    )

    segmentation[
        label3
    ] = [
        1.0,
        0.0,
        0.0
    ]

    return segmentation


# ============================================================
# CREATE SEGMENTATION OVERLAY
# ============================================================

def create_overlay(
    flair,
    mask,
    alpha=0.50
):

    # --------------------------------------------------------
    # Start with grayscale MRI
    # --------------------------------------------------------

    rgb_mri = np.stack(
        [
            flair,
            flair,
            flair
        ],
        axis=-1
    )

    # --------------------------------------------------------
    # Create segmentation RGB
    # --------------------------------------------------------

    segmentation = (
        create_segmentation_map(
            mask
        )
    )

    # --------------------------------------------------------
    # Tumor mask
    # --------------------------------------------------------

    tumor = (
        mask > 0
    )

    # --------------------------------------------------------
    # Blend only where tumor exists
    # --------------------------------------------------------

    overlay = rgb_mri.copy()

    overlay[tumor] = (
        (1.0 - alpha)
        * rgb_mri[tumor]
        +
        alpha
        * segmentation[tumor]
    )

    return overlay


# ============================================================
# CREATE VISUALIZATION
# ============================================================

def create_visualization(
    image,
    ground_truth,
    prediction,
    slice_index,
    output_path
):

    # ========================================================
    # EXTRACT SLICE
    # ========================================================

    flair = prepare_display_image(
        image,
        slice_index
    )

    gt_slice = ground_truth[
        :,
        :,
        slice_index
    ]

    pred_slice = prediction[
        :,
        :,
        slice_index
    ]

    # ========================================================
    # CREATE COLORED SEGMENTATION
    # ========================================================

    gt_segmentation = (
        create_segmentation_map(
            gt_slice
        )
    )

    pred_segmentation = (
        create_segmentation_map(
            pred_slice
        )
    )

    # ========================================================
    # CREATE OVERLAYS
    # ========================================================

    gt_overlay = create_overlay(
        flair,
        gt_slice,
        alpha=0.50
    )

    pred_overlay = create_overlay(
        flair,
        pred_slice,
        alpha=0.50
    )

    # ========================================================
    # CREATE FIGURE
    # ========================================================

    fig, axes = plt.subplots(
        1,
        4,
        figsize=(20, 5)
    )

    # ========================================================
    # PANEL 1
    # MRI FLAIR
    # ========================================================

    axes[0].imshow(
        flair.T,
        cmap="gray",
        origin="lower"
    )

    axes[0].set_title(
        "MRI (FLAIR)",
        fontsize=12,
        fontweight="bold"
    )

    axes[0].axis("off")

    # ========================================================
    # PANEL 2
    # GROUND TRUTH
    # ========================================================

    axes[1].imshow(
        gt_overlay.transpose(
            1,
            0,
            2
        ),
        origin="lower"
    )

    axes[1].set_title(
        "Ground Truth",
        fontsize=12,
        fontweight="bold"
    )

    axes[1].axis("off")

    # ========================================================
    # PANEL 3
    # E3 PREDICTION
    # ========================================================

    axes[2].imshow(
        pred_overlay.transpose(
            1,
            0,
            2
        ),
        origin="lower"
    )

    axes[2].set_title(
        "E3 Prediction",
        fontsize=12,
        fontweight="bold"
    )

    axes[2].axis("off")

    # ========================================================
    # PANEL 4
    # STANDALONE PREDICTED SEGMENTATION
    # ========================================================

    axes[3].imshow(
        pred_segmentation.transpose(
            1,
            0,
            2
        ),
        origin="lower"
    )

    axes[3].set_title(
        "Predicted Segmentation",
        fontsize=12,
        fontweight="bold"
    )

    axes[3].axis("off")

    # ========================================================
    # OVERALL TITLE
    # ========================================================

    fig.suptitle(
        (
            f"{PATIENT_ID} | "
            f"C3 Local 3D U-Net | "
            f"Axial Slice {slice_index}"
        ),
        fontsize=14,
        fontweight="bold"
    )

    # ========================================================
    # SAVE
    # ========================================================

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    plt.tight_layout(
        rect=[
            0,
            0,
            1,
            0.93
        ]
    )

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print()
    print(
        "✓ Visualization saved:"
    )

    print(
        output_path
    )


# ============================================================
# SAVE METADATA
# ============================================================

def save_metadata(
    output_path,
    patient_path,
    slice_index,
    image,
    ground_truth,
    prediction
):

    metadata_path = (
        output_path.parent
        / (
            output_path.stem
            + "_metadata.txt"
        )
    )

    gt_labels = np.unique(
        ground_truth
    )

    pred_labels = np.unique(
        prediction
    )

    with open(
        metadata_path,
        "w"
    ) as f:

        f.write(
            "LOCAL 3D U-NET VISUALIZATION\n"
        )

        f.write(
            "========================================\n\n"
        )

        f.write(
            f"Patient: {PATIENT_ID}\n"
        )

        f.write(
            "Client: C3\n"
        )

        f.write(
            "Model: 3D U-Net\n"
        )

        f.write(
            f"Checkpoint: {MODEL_PATH}\n"
        )

        f.write(
            f"Patient file: {patient_path}\n"
        )

        f.write(
            f"Device: {DEVICE}\n"
        )

        f.write(
            f"Image shape: {image.shape}\n"
        )

        f.write(
            f"Ground truth shape: "
            f"{ground_truth.shape}\n"
        )

        f.write(
            f"Prediction shape: "
            f"{prediction.shape}\n"
        )

        f.write(
            f"Visualization slice: "
            f"{slice_index}\n"
        )

        f.write(
            f"Ground truth labels: "
            f"{gt_labels.tolist()}\n"
        )

        f.write(
            f"Prediction labels: "
            f"{pred_labels.tolist()}\n"
        )

        # ----------------------------------------------------
        # Architecture
        # ----------------------------------------------------

        f.write(
            "\nArchitecture:\n"
        )

        f.write(
            "Input channels: 4\n"
        )

        f.write(
            "Output classes: 4\n"
        )

        f.write(
            "Base channels: 16\n"
        )

        f.write(
            "Encoder channels: 16, 32, 64\n"
        )

        f.write(
            "Bottleneck channels: 128\n"
        )

        f.write(
            "Decoder channels: 64, 32, 16\n"
        )

        f.write(
            "DoubleConv: Conv3d + "
            "InstanceNorm3d + ReLU x 2\n"
        )

        f.write(
            "Conv3d bias in DoubleConv: False\n"
        )

        f.write(
            "Final layer: 1x1x1 Conv3d\n"
        )

        # ----------------------------------------------------
        # Segmentation colors
        # ----------------------------------------------------

        f.write(
            "\nSegmentation visualization:\n"
        )

        f.write(
            "Label 0: Background\n"
        )

        f.write(
            "Label 1: NCR/NET - Yellow\n"
        )

        f.write(
            "Label 2: Edema - Green\n"
        )

        f.write(
            "Label 3: Enhancing Tumor - Red\n"
        )

    print(
        "✓ Metadata saved:"
    )

    print(
        metadata_path
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 70)
    print("LOCAL 3D U-NET PREDICTION VISUALIZATION")
    print("=" * 70)

    # ========================================================
    # 1. LOAD MODEL
    # ========================================================

    model = load_model()

    # ========================================================
    # 2. FIND PATIENT
    # ========================================================

    patient_path = find_patient()

    # ========================================================
    # 3. LOAD PATIENT
    # ========================================================

    image, ground_truth = load_patient(
        patient_path
    )

    # ========================================================
    # 4. VALIDATE INPUT
    # ========================================================

    if image.ndim != 4:

        raise ValueError(
            f"Expected image with shape "
            f"[C, X, Y, Z], "
            f"got {image.shape}"
        )

    if image.shape[0] != 4:

        raise ValueError(
            f"Expected 4 input channels, "
            f"got {image.shape[0]}"
        )

    if ground_truth.ndim != 3:

        raise ValueError(
            f"Expected 3D mask, "
            f"got {ground_truth.shape}"
        )

    # ========================================================
    # 5. RUN PREDICTION
    # ========================================================

    prediction = predict_full_volume(
        model,
        image
    )

    # ========================================================
    # 6. VALIDATE OUTPUT
    # ========================================================

    if prediction.shape != ground_truth.shape:

        raise ValueError(
            "Prediction and ground truth "
            "shape mismatch:\n"
            f"Prediction: {prediction.shape}\n"
            f"Ground truth: {ground_truth.shape}"
        )

    # ========================================================
    # 7. SELECT SLICE
    # ========================================================

    slice_index = (
        select_largest_tumor_slice(
            ground_truth
        )
    )

    # ========================================================
    # 8. CREATE OUTPUT DIRECTORY
    # ========================================================

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # ========================================================
    # 9. OUTPUT FILE
    # ========================================================

    output_path = (
        OUTPUT_DIR
        / (
            f"{PATIENT_ID}"
            f"_C3_3DUNet_prediction.png"
        )
    )

    # ========================================================
    # 10. CREATE VISUALIZATION
    # ========================================================

    create_visualization(
        image,
        ground_truth,
        prediction,
        slice_index,
        output_path
    )

    # ========================================================
    # 11. SAVE METADATA
    # ========================================================

    save_metadata(
        output_path,
        patient_path,
        slice_index,
        image,
        ground_truth,
        prediction
    )

    # ========================================================
    # COMPLETE
    # ========================================================

    print()
    print("=" * 70)
    print("VISUALIZATION COMPLETE")
    print("=" * 70)

    print(
        f"Patient: {PATIENT_ID}"
    )

    print(
        "Client: C3"
    )

    print(
        "Model: Local 3D U-Net"
    )

    print(
        f"Slice: {slice_index}"
    )

    print(
        "Ground truth labels:",
        np.unique(
            ground_truth
        ).tolist()
    )

    print(
        "Prediction labels:",
        np.unique(
            prediction
        ).tolist()
    )

    print()
    print(
        f"Output: {output_path}"
    )

    print()


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    main()