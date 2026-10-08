import os
import csv
import json
import time
import random
from pathlib import Path
from pytorchexample.swin_unetr import create_model
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from scipy.ndimage import distance_transform_edt
from sklearn.metrics import roc_auc_score, average_precision_score, matthews_corrcoef


# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path.home() / "Neuroverse"

DATA_ROOT = ROOT / "IID_Clients"

GLOBAL_TEST_ROOT = ROOT / "IID_Clients" / "GlobalTest" / "test"

RESULT_ROOT = ROOT / "Federated" / "Swin_FedAvg" / "results"

PATCH_SIZE = (96, 96, 96)

BATCH_SIZE = 1

BASE_CHANNELS = 16

NUM_CLASSES = 4

NUM_WORKERS = 0

SEED = 42


# ============================================================
# REPRODUCIBILITY
# ============================================================

def set_seed(seed=SEED):

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():

        torch.cuda.manual_seed_all(seed)


# ============================================================
# DEVICE
# ============================================================

def get_device():

    return torch.device(
        "cuda" if torch.cuda.is_available()
        else "cpu"
    )


# ============================================================
# DATASET
# ============================================================

class PatientDataset(Dataset):

    def __init__(
        self,
        directory,
        patch_size=PATCH_SIZE,
        training=True,
        patches_per_epoch=100
    ):

        self.directory = Path(directory)

        self.files = sorted(
            self.directory.glob("*.npz")
        )

        self.patch_size = patch_size

        self.training = training

        self.patches_per_epoch = patches_per_epoch

        if len(self.files) == 0:

            raise RuntimeError(
                f"No NPZ files found in {self.directory}"
            )

    def __len__(self):

        if self.training:

            return self.patches_per_epoch

        return len(self.files)

    def _load_patient(self, index):

        file = self.files[index]

        data = np.load(file)

        image = data["image"].astype(
            np.float32
        )

        mask = data["mask"].astype(
            np.uint8
        )

        return image, mask, file.name

    def _pad_if_needed(
        self,
        image,
        mask
    ):

        _, x, y, z = image.shape

        px = max(
            0,
            self.patch_size[0] - x
        )

        py = max(
            0,
            self.patch_size[1] - y
        )

        pz = max(
            0,
            self.patch_size[2] - z
        )

        if px > 0 or py > 0 or pz > 0:

            image = np.pad(
                image,
                (
                    (0, 0),
                    (
                        px // 2,
                        px - px // 2
                    ),
                    (
                        py // 2,
                        py - py // 2
                    ),
                    (
                        pz // 2,
                        pz - pz // 2
                    )
                ),
                mode="constant"
            )

            mask = np.pad(
                mask,
                (
                    (
                        px // 2,
                        px - px // 2
                    ),
                    (
                        py // 2,
                        py - py // 2
                    ),
                    (
                        pz // 2,
                        pz - pz // 2
                    )
                ),
                mode="constant"
            )

        return image, mask

    def _random_tumor_patch(
        self,
        image,
        mask
    ):

        image, mask = self._pad_if_needed(
            image,
            mask
        )

        _, X, Y, Z = image.shape

        tumor = np.argwhere(
            mask > 0
        )

        if len(tumor) > 0:

            center = tumor[
                np.random.randint(
                    len(tumor)
                )
            ]

            cx, cy, cz = center

        else:

            cx = np.random.randint(X)

            cy = np.random.randint(Y)

            cz = np.random.randint(Z)

        px, py, pz = self.patch_size

        x0 = int(
            cx - px // 2
        )

        y0 = int(
            cy - py // 2
        )

        z0 = int(
            cz - pz // 2
        )

        x0 = max(
            0,
            min(x0, X - px)
        )

        y0 = max(
            0,
            min(y0, Y - py)
        )

        z0 = max(
            0,
            min(z0, Z - pz)
        )

        image_patch = image[
            :,
            x0:x0 + px,
            y0:y0 + py,
            z0:z0 + pz
        ]

        mask_patch = mask[
            x0:x0 + px,
            y0:y0 + py,
            z0:z0 + pz
        ]

        return image_patch, mask_patch

    def __getitem__(self, index):

        if self.training:

            patient_index = np.random.randint(
                len(self.files)
            )

            image, mask, name = (
                self._load_patient(
                    patient_index
                )
            )

            image, mask = (
                self._random_tumor_patch(
                    image,
                    mask
                )
            )

        else:

            image, mask, name = (
                self._load_patient(
                    index
                )
            )

        return (
            torch.from_numpy(image),
            torch.from_numpy(mask).long(),
            name
        )


# ============================================================
# CLIENT DATA
# ============================================================

def get_client_data(
    client_id,
    patches_per_epoch=100
):

    train_dir = (
        DATA_ROOT
        / client_id
        / "train"
    )

    val_dir = (
        DATA_ROOT
        / client_id
        / "val"
    )

    train_dataset = PatientDataset(
        train_dir,
        training=True,
        patches_per_epoch=patches_per_epoch
    )

    val_dataset = PatientDataset(
        val_dir,
        training=False
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available()
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available()
    )

    return train_loader, val_loader


# ============================================================
# 3D U-NET
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

        self.enc4 = DoubleConv(
            base * 4,
            base * 8
        )

        self.pool = nn.MaxPool3d(2)

        self.bottleneck = DoubleConv(
            base * 8,
            base * 16
        )

        self.up4 = nn.ConvTranspose3d(
            base * 16,
            base * 8,
            2,
            stride=2
        )

        self.dec4 = DoubleConv(
            base * 16,
            base * 8
        )

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

        self.out = nn.Conv3d(
            base,
            num_classes,
            kernel_size=1
        )

    def forward(self, x):

        e1 = self.enc1(x)

        e2 = self.enc2(
            self.pool(e1)
        )

        e3 = self.enc3(
            self.pool(e2)
        )

        e4 = self.enc4(
            self.pool(e3)
        )

        b = self.bottleneck(
            self.pool(e4)
        )

        d4 = self.up4(b)

        d4 = torch.cat(
            [d4, e4],
            dim=1
        )

        d4 = self.dec4(d4)

        d3 = self.up3(d4)

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

        return self.out(d1)


def get_model():
    return create_model()


# ============================================================
# LOSS
# ============================================================

class DiceLoss(nn.Module):

    def __init__(
        self,
        num_classes=4
    ):

        super().__init__()

        self.num_classes = num_classes

    def forward(
        self,
        logits,
        target
    ):

        probabilities = torch.softmax(
            logits,
            dim=1
        )

        target_onehot = F.one_hot(
            target,
            num_classes=self.num_classes
        )

        target_onehot = target_onehot.permute(
            0,
            4,
            1,
            2,
            3
        ).float()

        dice_scores = []

        for c in range(
            1,
            self.num_classes
        ):

            pred = probabilities[:, c]

            true = target_onehot[:, c]

            intersection = (
                pred * true
            ).sum()

            denominator = (
                pred.sum()
                +
                true.sum()
            )

            dice = (
                2 * intersection + 1e-6
            ) / (
                denominator + 1e-6
            )

            dice_scores.append(dice)

        return 1 - torch.stack(
            dice_scores
        ).mean()


class CombinedLoss(nn.Module):

    def __init__(self):

        super().__init__()

        self.dice = DiceLoss()

    def forward(
        self,
        logits,
        target
    ):

        ce = F.cross_entropy(
            logits,
            target
        )

        dice = self.dice(
            logits,
            target
        )

        return ce + dice


# ============================================================
# LOCAL TRAINING
# ============================================================

def train(
    model,
    train_loader,
    epochs,
    learning_rate,
    device
):

    model.train()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=1e-5
    )

    criterion = CombinedLoss()

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=device.type == "cuda"
    )

    history = []

    for epoch in range(
        1,
        epochs + 1
    ):

        epoch_start = time.time()

        model.train()

        running_loss = 0.0

        for images, masks, _ in train_loader:

            images = images.to(
                device,
                non_blocking=True
            )

            masks = masks.to(
                device,
                non_blocking=True
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=device.type == "cuda"
            ):

                logits = model(images)

                loss = criterion(
                    logits,
                    masks
                )

            scaler.scale(
                loss
            ).backward()

            scaler.step(
                optimizer
            )

            scaler.update()

            running_loss += loss.item()

        epoch_loss = (
            running_loss
            /
            len(train_loader)
        )

        epoch_time = (
            time.time()
            -
            epoch_start
        )

        history.append({
            "epoch": epoch,
            "train_loss": epoch_loss,
            "epoch_time_seconds": epoch_time,
            "learning_rate":
                optimizer.param_groups[0]["lr"]
        })

        print(
            f"Epoch {epoch}/{epochs} | "
            f"Loss: {epoch_loss:.4f}"
        )

    return history


# ============================================================
# FULL-VOLUME PREDICTION
# ============================================================

def pad_to_multiple(
    image,
    multiple=16
):

    _, x, y, z = image.shape

    px = (
        multiple - x % multiple
    ) % multiple

    py = (
        multiple - y % multiple
    ) % multiple

    pz = (
        multiple - z % multiple
    ) % multiple

    padded = np.pad(
        image,
        (
            (0, 0),
            (0, px),
            (0, py),
            (0, pz)
        ),
        mode="constant"
    )

    return padded, (x, y, z)


# ============================================================
# SLIDING-WINDOW FULL-VOLUME PREDICTION
# ============================================================

@torch.no_grad()
def predict_full_volume(
    model,
    image,
    device
):
    """
    Memory-efficient full-volume inference.

    The MRI volume stays in CPU RAM.
    The volume is divided into 96x96x96 patches, matching
    the patch size used during training.

    Each patch is:
        CPU -> GPU -> U-Net -> CPU

    Predictions from overlapping patches are averaged.

    Returns
    -------
    probabilities : np.ndarray
        Shape: (4, X, Y, Z)

    prediction : np.ndarray
        Shape: (X, Y, Z)
    """

    # --------------------------------------------------------
    # Configuration
    # --------------------------------------------------------

    patch_size = PATCH_SIZE

    # 50% overlap.
    # Training uses 96^3 patches, so 48 gives good overlap
    # while avoiding seams between patches.
    stride = (
        patch_size[0] // 2,
        patch_size[1] // 2,
        patch_size[2] // 2
    )

    num_classes = NUM_CLASSES

    # --------------------------------------------------------
    # Original volume dimensions
    # --------------------------------------------------------

    _, X, Y, Z = image.shape

    px, py, pz = patch_size
    sx, sy, sz = stride

    # --------------------------------------------------------
    # Compute padded dimensions
    #
    # We need enough space so that every voxel is covered by
    # at least one complete patch.
    # --------------------------------------------------------

    padded_X = max(X, px)
    padded_Y = max(Y, py)
    padded_Z = max(Z, pz)

    if padded_X > px:
        padded_X = (
            ((padded_X - px + sx - 1) // sx) * sx
            + px
        )

    if padded_Y > py:
        padded_Y = (
            ((padded_Y - py + sy - 1) // sy) * sy
            + py
        )

    if padded_Z > pz:
        padded_Z = (
            ((padded_Z - pz + sz - 1) // sz) * sz
            + pz
        )

    # --------------------------------------------------------
    # Pad image on CPU
    # --------------------------------------------------------

    pad_x = padded_X - X
    pad_y = padded_Y - Y
    pad_z = padded_Z - Z

    padded = np.pad(
        image,
        (
            (0, 0),
            (0, pad_x),
            (0, pad_y),
            (0, pad_z)
        ),
        mode="constant"
    )

    # --------------------------------------------------------
    # CPU accumulation buffers
    #
    # IMPORTANT:
    # These stay in CPU RAM.
    #
    # probabilities_sum:
    #     accumulated class probabilities
    #
    # count_map:
    #     number of patches contributing to each voxel
    # --------------------------------------------------------

    probabilities_sum = np.zeros(
        (
            num_classes,
            padded_X,
            padded_Y,
            padded_Z
        ),
        dtype=np.float32
    )

    count_map = np.zeros(
        (
            padded_X,
            padded_Y,
            padded_Z
        ),
        dtype=np.float32
    )

    # --------------------------------------------------------
    # Generate patch starting positions
    # --------------------------------------------------------

    def get_starts(length, patch, stride_value):

        if length <= patch:
            return [0]

        starts = list(
            range(
                0,
                length - patch + 1,
                stride_value
            )
        )

        last_start = length - patch

        if starts[-1] != last_start:
            starts.append(last_start)

        return starts

    x_starts = get_starts(
        padded_X,
        px,
        sx
    )

    y_starts = get_starts(
        padded_Y,
        py,
        sy
    )

    z_starts = get_starts(
        padded_Z,
        pz,
        sz
    )

    total_patches = (
        len(x_starts)
        * len(y_starts)
        * len(z_starts)
    )

    patch_number = 0

    # --------------------------------------------------------
    # Model evaluation mode
    # --------------------------------------------------------

    was_training = model.training

    model.eval()

    # --------------------------------------------------------
    # Sliding-window inference
    # --------------------------------------------------------

    for x0 in x_starts:

        for y0 in y_starts:

            for z0 in z_starts:

                patch_number += 1

                # --------------------------------------------
                # Extract one 96x96x96 patch
                # --------------------------------------------

                image_patch = padded[
                    :,
                    x0:x0 + px,
                    y0:y0 + py,
                    z0:z0 + pz
                ]

                # --------------------------------------------
                # CPU -> GPU
                # --------------------------------------------

                tensor = torch.from_numpy(
                    image_patch
                ).unsqueeze(0).to(
                    device,
                    non_blocking=True
                )

                # --------------------------------------------
                # Forward pass
                #
                # AMP reduces inference GPU memory usage.
                # --------------------------------------------

                with torch.autocast(
                    device_type=device.type,
                    dtype=torch.float16,
                    enabled=device.type == "cuda"
                ):

                    logits = model(tensor)

                    probabilities = torch.softmax(
                        logits,
                        dim=1
                    )

                # --------------------------------------------
                # GPU -> CPU
                #
                # Immediately detach/delete GPU tensors after
                # every patch so GPU memory is reused.
                # --------------------------------------------

                probabilities_cpu = (
                    probabilities[0]
                    .float()
                    .cpu()
                    .numpy()
                )

                # --------------------------------------------
                # Accumulate probabilities on CPU
                # --------------------------------------------

                probabilities_sum[
                    :,
                    x0:x0 + px,
                    y0:y0 + py,
                    z0:z0 + pz
                ] += probabilities_cpu

                count_map[
                    x0:x0 + px,
                    y0:y0 + py,
                    z0:z0 + pz
                ] += 1.0

                # --------------------------------------------
                # Explicitly release GPU tensors
                # --------------------------------------------

                del tensor
                del logits
                del probabilities
                del probabilities_cpu

                # --------------------------------------------
                # Periodically release cached CUDA memory.
                # This is intentionally not done after every
                # patch because that would slow inference.
                # --------------------------------------------

                if (
                    device.type == "cuda"
                    and patch_number % 10 == 0
                ):
                    torch.cuda.empty_cache()

    # --------------------------------------------------------
    # Restore original model training state
    # --------------------------------------------------------

    if was_training:
        model.train()

    # --------------------------------------------------------
    # Average overlapping patch predictions
    # --------------------------------------------------------

    count_map = np.maximum(
        count_map,
        1.0
    )

    probabilities_sum /= count_map[
        np.newaxis,
        ...
    ]

    # --------------------------------------------------------
    # Crop back to original MRI dimensions
    # --------------------------------------------------------

    probabilities = probabilities_sum[
        :,
        :X,
        :Y,
        :Z
    ]

    # --------------------------------------------------------
    # Final segmentation
    # --------------------------------------------------------

    prediction = np.argmax(
        probabilities,
        axis=0
    ).astype(
        np.uint8
    )

    # --------------------------------------------------------
    # Free CPU accumulation buffers
    # --------------------------------------------------------

    del probabilities_sum
    del count_map

    # --------------------------------------------------------
    # Make absolutely sure CUDA cache is released
    # before the next patient.
    # --------------------------------------------------------

    if device.type == "cuda":
        torch.cuda.empty_cache()

    return (
        probabilities,
        prediction
    )

# ============================================================
# REGIONS
# ============================================================

def make_regions(mask):

    return {

        "WT": mask > 0,

        "TC": (
            (mask == 1)
            |
            (mask == 3)
        ),

        "ET": mask == 3
    }


# ============================================================
# BINARY METRICS
# ============================================================

def binary_metrics(
    prediction,
    target
):

    prediction = prediction.astype(bool)

    target = target.astype(bool)

    tp = np.logical_and(
        prediction,
        target
    ).sum()

    tn = np.logical_and(
        ~prediction,
        ~target
    ).sum()

    fp = np.logical_and(
        prediction,
        ~target
    ).sum()

    fn = np.logical_and(
        ~prediction,
        target
    ).sum()

    total = (
        tp + tn + fp + fn
    )

    accuracy = (
        (tp + tn) / total
        if total > 0 else 0.0
    )

    precision = (
        tp / (tp + fp)
        if tp + fp > 0 else 0.0
    )

    sensitivity = (
        tp / (tp + fn)
        if tp + fn > 0 else 0.0
    )

    specificity = (
        tn / (tn + fp)
        if tn + fp > 0 else 0.0
    )

    f1 = (
        2 * precision * sensitivity
        /
        (precision + sensitivity)
        if precision + sensitivity > 0
        else 0.0
    )

    dice = f1

    iou = (
        tp / (tp + fp + fn)
        if tp + fp + fn > 0
        else 0.0
    )

    balanced_accuracy = (
        sensitivity + specificity
    ) / 2

    try:

        mcc = matthews_corrcoef(
            target.ravel(),
            prediction.ravel()
        )

    except Exception:

        mcc = 0.0

    return {

        "accuracy": float(accuracy),

        "precision": float(precision),

        "sensitivity": float(sensitivity),

        "specificity": float(specificity),

        "f1": float(f1),

        "dice": float(dice),

        "iou": float(iou),

        "balanced_accuracy":
            float(balanced_accuracy),

        "mcc": float(mcc),

        "tp": int(tp),

        "tn": int(tn),

        "fp": int(fp),

        "fn": int(fn)
    }


# ============================================================
# HD95
# ============================================================

def hd95(
    pred,
    target
):

    pred = pred.astype(bool)

    target = target.astype(bool)

    if not pred.any() and not target.any():

        return 0.0

    if not pred.any() or not target.any():

        return np.nan

    pred_surface = (
        pred
        ^
        (
            distance_transform_edt(pred)
            <= 1
        )
    )

    target_surface = (
        target
        ^
        (
            distance_transform_edt(target)
            <= 1
        )
    )

    pred_distance = distance_transform_edt(
        ~pred
    )

    target_distance = distance_transform_edt(
        ~target
    )

    distances_1 = target_distance[
        pred_surface
    ]

    distances_2 = pred_distance[
        target_surface
    ]

    if (
        len(distances_1) == 0
        or
        len(distances_2) == 0
    ):

        return np.nan

    distances = np.concatenate([
        distances_1,
        distances_2
    ])

    return float(
        np.percentile(
            distances,
            95
        )
    )


# ============================================================
# ROC / PR AUC
# ============================================================

def auc_metrics(
    probability,
    target
):

    target = target.astype(
        np.uint8
    )

    probability = probability.astype(
        np.float32
    )

    flat_target = target.ravel()

    flat_probability = (
        probability.ravel()
    )

    max_voxels = 500000

    if len(flat_target) > max_voxels:

        indices = np.random.choice(
            len(flat_target),
            max_voxels,
            replace=False
        )

        flat_target = (
            flat_target[indices]
        )

        flat_probability = (
            flat_probability[indices]
        )

    if len(
        np.unique(flat_target)
    ) < 2:

        return np.nan, np.nan

    try:

        roc_auc = roc_auc_score(
            flat_target,
            flat_probability
        )

    except Exception:

        roc_auc = np.nan

    try:

        pr_auc = average_precision_score(
            flat_target,
            flat_probability
        )

    except Exception:

        pr_auc = np.nan

    return (
        float(roc_auc)
        if not np.isnan(roc_auc)
        else np.nan,
        float(pr_auc)
        if not np.isnan(pr_auc)
        else np.nan
    )


# ============================================================
# PATIENT EVALUATION
# ============================================================

@torch.no_grad()
def evaluate_patient(
    model,
    file,
    device
):

    start = time.time()

    data = np.load(file)

    image = data["image"].astype(
        np.float32
    )

    mask = data["mask"].astype(
        np.uint8
    )

    probabilities, prediction = (
        predict_full_volume(
            model,
            image,
            device
        )
    )

    true_regions = make_regions(mask)

    pred_regions = make_regions(
        prediction
    )

    result = {

        "patient_id": file.stem
    }

    for region in [
        "WT",
        "TC",
        "ET"
    ]:

        true_region = (
            true_regions[region]
        )

        pred_region = (
            pred_regions[region]
        )

        metrics = binary_metrics(
            pred_region,
            true_region
        )

        hd = hd95(
            pred_region,
            true_region
        )

        if region == "WT":

            probability = (
                probabilities[1]
                +
                probabilities[2]
                +
                probabilities[3]
            )

        elif region == "TC":

            probability = (
                probabilities[1]
                +
                probabilities[3]
            )

        else:

            probability = probabilities[3]

        roc_auc, pr_auc = auc_metrics(
            probability,
            true_region
        )

        for key in [
            "accuracy",
            "precision",
            "sensitivity",
            "specificity",
            "f1",
            "dice",
            "iou",
            "balanced_accuracy",
            "mcc",
            "tp",
            "tn",
            "fp",
            "fn"
        ]:

            result[
                f"{region}_{key}"
            ] = metrics[key]

        result[
            f"{region}_hd95"
        ] = hd

        result[
            f"{region}_roc_auc"
        ] = roc_auc

        result[
            f"{region}_pr_auc"
        ] = pr_auc

    metric_names = [
        "accuracy",
        "precision",
        "sensitivity",
        "specificity",
        "f1",
        "dice",
        "iou",
        "balanced_accuracy",
        "mcc",
        "hd95",
        "roc_auc",
        "pr_auc"
    ]

    for metric in metric_names:

        values = [
            result[
                f"{region}_{metric}"
            ]
            for region in [
                "WT",
                "TC",
                "ET"
            ]
        ]

        result[
            f"mean_{metric}"
        ] = float(
            np.nanmean(values)
        )

    result[
        "inference_time_seconds"
    ] = (
        time.time() - start
    )

    return result


# ============================================================
# EVALUATE A DATASET
# ============================================================

def evaluate_dataset(
    model,
    files,
    device,
    output_dir,
    prefix
):

    output_dir = Path(
        output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    results = []

    start = time.time()

    model.eval()

    for index, file in enumerate(
        files,
        start=1
    ):

        result = evaluate_patient(
            model,
            file,
            device
        )

        results.append(result)

        print(
            f"[{index}/{len(files)}] "
            f"{result['patient_id']} | "
            f"WT Dice="
            f"{result['WT_dice']:.4f} | "
            f"TC Dice="
            f"{result['TC_dice']:.4f} | "
            f"ET Dice="
            f"{result['ET_dice']:.4f}"
        )

    df = pd.DataFrame(
        results
    )

    patient_csv = (
        output_dir
        /
        f"{prefix}_patient_metrics.csv"
    )

    df.to_csv(
        patient_csv,
        index=False
    )

    summary = {}

    for column in df.columns:

        if column != "patient_id":

            summary[column] = (
                df[column]
                .mean()
            )

    summary["num_patients"] = len(df)

    summary[
        "total_evaluation_time_seconds"
    ] = time.time() - start

    summary_csv = (
        output_dir
        /
        f"{prefix}_summary.csv"
    )

    pd.DataFrame(
        [summary]
    ).to_csv(
        summary_csv,
        index=False
    )

    with open(
        output_dir
        /
        f"{prefix}_summary.json",
        "w"
    ) as f:

        json.dump(
            summary,
            f,
            indent=2,
            default=str
        )

    return summary, df


# ============================================================
# CLIENT ROUND EVALUATION
# ============================================================

def evaluate_client_round(
    model,
    client_id,
    round_number,
    device
):

    val_dir = (
        DATA_ROOT
        / client_id
        / "val"
    )

    files = sorted(
        val_dir.glob("*.npz")
    )

    output_dir = (
        RESULT_ROOT
        / "client_metrics"
        / client_id
        / f"round_{round_number:02d}"
    )

    return evaluate_dataset(
        model,
        files,
        device,
        output_dir,
        "validation"
    )


# ============================================================
# GLOBAL TEST EVALUATION
# ============================================================

def evaluate_global_test(
    model
):

    device = get_device()

    files = sorted(
        GLOBAL_TEST_ROOT.glob("*.npz")
    )

    if len(files) == 0:

        raise RuntimeError(
            f"No GlobalTest NPZ files found in "
            f"{GLOBAL_TEST_ROOT}"
        )

    output_dir = (
        RESULT_ROOT
        / "GlobalTest"
    )

    print()
    print("=" * 80)
    print(
        f"FINAL GLOBAL TEST — "
        f"{len(files)} PATIENTS"
    )
    print("=" * 80)

    summary, df = evaluate_dataset(
        model,
        files,
        device,
        output_dir,
        "global_test"
    )

    return summary, df


# ============================================================
# SAVE ROUND HISTORY
# ============================================================

def save_round_histories(
    train_metrics,
    evaluate_metrics
):

    rows = []

    for round_number, record in (
        train_metrics.items()
    ):

        row = {
            "round": round_number
        }

        row.update(
            dict(record)
        )

        row["phase"] = "train"

        rows.append(row)

    for round_number, record in (
        evaluate_metrics.items()
    ):

        row = {
            "round": round_number
        }

        row.update(
            dict(record)
        )

        row["phase"] = "validation"

        rows.append(row)

    if rows:

        df = pd.DataFrame(
            rows
        )

        df.to_csv(
            RESULT_ROOT
            /
            "federated_round_metrics.csv",
            index=False
        )

        with open(
            RESULT_ROOT
            /
            "federated_round_metrics.json",
            "w"
        ) as f:

            json.dump(
                rows,
                f,
                indent=2,
                default=str
            )

        return df

    return pd.DataFrame()


# ============================================================
# PLOT FEDERATED METRICS
# ============================================================

def plot_metric(
    df,
    metric,
    output_path,
    title
):

    if metric not in df.columns:

        return

    plt.figure(
        figsize=(9, 5)
    )

    plt.plot(
        df["round"],
        df[metric],
        marker="o"
    )

    plt.xlabel(
        "Federated Round"
    )

    plt.ylabel(metric)

    plt.title(title)

    plt.grid(True)

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=200
    )

    plt.close()


def create_federated_plots(
    train_metrics,
    eval_metrics
):

    plot_dir = (
        RESULT_ROOT
        / "plots"
    )

    plot_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    if train_metrics:

        train_df = pd.DataFrame([
            {
                "round": r,
                **dict(v)
            }
            for r, v
            in train_metrics.items()
        ])

        train_df.to_csv(
            RESULT_ROOT
            /
            "federated_training_metrics.csv",
            index=False
        )

        plot_metric(
            train_df,
            "train_loss",
            plot_dir
            /
            "federated_train_loss.png",
            "Federated Training Loss"
        )

    if eval_metrics:

        eval_df = pd.DataFrame([
            {
                "round": r,
                **dict(v)
            }
            for r, v
            in eval_metrics.items()
        ])

        eval_df.to_csv(
            RESULT_ROOT
            /
            "federated_validation_metrics.csv",
            index=False
        )

        metrics = [

            "mean_dice",

            "mean_iou",

            "mean_accuracy",

            "mean_precision",

            "mean_sensitivity",

            "mean_specificity",

            "mean_f1",

            "mean_balanced_accuracy",

            "mean_mcc",

            "mean_hd95",

            "mean_roc_auc",

            "mean_pr_auc"
        ]

        for metric in metrics:

            plot_metric(
                eval_df,
                metric,
                plot_dir
                /
                f"federated_{metric}.png",
                f"Federated {metric}"
            )


# ============================================================
# SAVE FINAL GLOBAL RESULTS
# ============================================================

def save_final_global_results(
    summary,
    df
):

    output_dir = (
        RESULT_ROOT
        / "GlobalTest"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        output_dir
        /
        "FINAL_GLOBAL_TEST_SUMMARY.json",
        "w"
    ) as f:

        json.dump(
            summary,
            f,
            indent=2,
            default=str
        )

    pd.DataFrame(
        [summary]
    ).to_csv(
        output_dir
        /
        "FINAL_GLOBAL_TEST_SUMMARY.csv",
        index=False
    )

    # Region-level final metric graph

    regions = [
        "WT",
        "TC",
        "ET"
    ]

    metric_names = [
        "dice",
        "iou",
        "accuracy",
        "precision",
        "sensitivity",
        "specificity",
        "f1",
        "balanced_accuracy",
        "mcc",
        "hd95",
        "roc_auc",
        "pr_auc"
    ]

    plot_dir = (
        output_dir
        / "plots"
    )

    plot_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    for metric in metric_names:

        values = [
            summary.get(
                f"{region}_{metric}",
                np.nan
            )
            for region in regions
        ]

        if all(
            pd.isna(v)
            for v in values
        ):

            continue

        plt.figure(
            figsize=(8, 5)
        )

        plt.bar(
            regions,
            values
        )

        plt.xlabel(
            "Tumor Region"
        )

        plt.ylabel(metric)

        plt.title(
            f"Final GlobalTest {metric}"
        )

        plt.grid(
            axis="y"
        )

        plt.tight_layout()

        plt.savefig(
            plot_dir
            /
            f"global_test_{metric}.png",
            dpi=200
        )

        plt.close()