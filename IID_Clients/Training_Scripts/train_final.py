import os
import csv
import time
import random
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from scipy.ndimage import distance_transform_edt

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    matthews_corrcoef
)


# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path.home() / "Neuroverse"

DATA_ROOT = ROOT / "IID_Clients"

RESULT_ROOT = DATA_ROOT / "Training_Results_Final"

PATCH_SIZE = (96, 96, 96)

BATCH_SIZE = 1

EPOCHS = 10

LEARNING_RATE = 1e-4

BASE_CHANNELS = 16

PATCHES_PER_EPOCH = 100

NUM_WORKERS = 0

SEED = 42

NUM_CLASSES = 4


# ============================================================
# REPRODUCIBILITY
# ============================================================

def set_seed(seed=42):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
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

        image = data["image"].astype(np.float32)

        mask = data["mask"].astype(np.uint8)

        return image, mask, file.name

    def _pad_if_needed(self, image, mask):

        _, x, y, z = image.shape

        px = max(0, self.patch_size[0] - x)
        py = max(0, self.patch_size[1] - y)
        pz = max(0, self.patch_size[2] - z)

        if px > 0 or py > 0 or pz > 0:

            image = np.pad(
                image,
                (
                    (0, 0),
                    (px // 2, px - px // 2),
                    (py // 2, py - py // 2),
                    (pz // 2, pz - pz // 2)
                ),
                mode="constant"
            )

            mask = np.pad(
                mask,
                (
                    (px // 2, px - px // 2),
                    (py // 2, py - py // 2),
                    (pz // 2, pz - pz // 2)
                ),
                mode="constant"
            )

        return image, mask

    def _random_tumor_patch(self, image, mask):

        image, mask = self._pad_if_needed(
            image, mask
        )

        _, X, Y, Z = image.shape

        tumor = np.argwhere(mask > 0)

        if len(tumor) > 0:

            center = tumor[
                np.random.randint(len(tumor))
            ]

            cx, cy, cz = center

        else:

            cx = np.random.randint(X)
            cy = np.random.randint(Y)
            cz = np.random.randint(Z)

        px, py, pz = self.patch_size

        x0 = int(cx - px // 2)
        y0 = int(cy - py // 2)
        z0 = int(cz - pz // 2)

        x0 = max(0, min(x0, X - px))
        y0 = max(0, min(y0, Y - py))
        z0 = max(0, min(z0, Z - pz))

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

            image, mask, name = self._load_patient(
                patient_index
            )

            image, mask = self._random_tumor_patch(
                image, mask
            )

        else:

            image, mask, name = self._load_patient(
                index
            )

        return (
            torch.from_numpy(image),
            torch.from_numpy(mask).long(),
            name
        )


# ============================================================
# 3D U-NET
# ============================================================

class DoubleConv(nn.Module):

    def __init__(self, in_channels, out_channels):

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

            nn.ReLU(inplace=True),

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

            nn.ReLU(inplace=True)
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


# ============================================================
# LOSS
# ============================================================

class DiceLoss(nn.Module):

    def __init__(self, num_classes=4):

        super().__init__()

        self.num_classes = num_classes

    def forward(self, logits, target):

        probabilities = torch.softmax(
            logits,
            dim=1
        )

        target_onehot = F.one_hot(
            target,
            num_classes=self.num_classes
        )

        target_onehot = target_onehot.permute(
            0, 4, 1, 2, 3
        ).float()

        dice_scores = []

        for c in range(1, self.num_classes):

            pred = probabilities[:, c]

            true = target_onehot[:, c]

            intersection = (
                pred * true
            ).sum()

            denominator = (
                pred.sum() +
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

    def forward(self, logits, target):

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
# PADDING FOR FULL VOLUME
# ============================================================

def pad_to_multiple(image, multiple=16):

    _, x, y, z = image.shape

    px = (multiple - x % multiple) % multiple
    py = (multiple - y % multiple) % multiple
    pz = (multiple - z % multiple) % multiple

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
# FULL VOLUME PREDICTION
# ============================================================

@torch.no_grad()
def predict_full_volume(model, image):

    padded, original_shape = pad_to_multiple(
        image
    )

    tensor = torch.from_numpy(
        padded
    ).unsqueeze(0).to(device)

    logits = model(tensor)

    probabilities = torch.softmax(
        logits,
        dim=1
    )

    probabilities = probabilities[
        0,
        :,
        :original_shape[0],
        :original_shape[1],
        :original_shape[2]
    ]

    prediction = torch.argmax(
        probabilities,
        dim=0
    )

    return (
        probabilities.cpu().numpy(),
        prediction.cpu().numpy()
    )


# ============================================================
# REGION DEFINITIONS
# ============================================================

def make_regions(mask):

    regions = {}

    # Whole tumor
    regions["WT"] = mask > 0

    # Tumor core
    regions["TC"] = (
        (mask == 1) |
        (mask == 3)
    )

    # Enhancing tumor
    regions["ET"] = mask == 3

    return regions


# ============================================================
# CONFUSION METRICS
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

    total = tp + tn + fp + fn

    accuracy = (
        (tp + tn) / total
        if total > 0 else 0
    )

    precision = (
        tp / (tp + fp)
        if tp + fp > 0 else 0
    )

    sensitivity = (
        tp / (tp + fn)
        if tp + fn > 0 else 0
    )

    specificity = (
        tn / (tn + fp)
        if tn + fp > 0 else 0
    )

    f1 = (
        2 * precision * sensitivity /
        (precision + sensitivity)
        if precision + sensitivity > 0
        else 0
    )

    dice = f1

    iou = (
        tp / (tp + fp + fn)
        if tp + fp + fn > 0 else 0
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
        "accuracy": accuracy,
        "precision": precision,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "f1": f1,
        "dice": dice,
        "iou": iou,
        "balanced_accuracy": balanced_accuracy,
        "mcc": mcc,
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn)
    }


# ============================================================
# HD95
# ============================================================

def hd95(pred, target):

    pred = pred.astype(bool)

    target = target.astype(bool)

    if not pred.any() and not target.any():
        return 0.0

    if not pred.any() or not target.any():
        return np.nan

    pred_surface = (
        pred ^
        (
            distance_transform_edt(pred) <= 1
        )
    )

    target_surface = (
        target ^
        (
            distance_transform_edt(target) <= 1
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

    if len(distances_1) == 0 or len(distances_2) == 0:
        return np.nan

    distances = np.concatenate(
        [distances_1, distances_2]
    )

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

    target = target.astype(np.uint8)

    probability = probability.astype(
        np.float32
    )

    flat_target = target.ravel()

    flat_probability = probability.ravel()

    # Avoid unnecessary memory usage while
    # preserving a representative voxel sample.
    max_voxels = 500000

    if len(flat_target) > max_voxels:

        indices = np.random.choice(
            len(flat_target),
            max_voxels,
            replace=False
        )

        flat_target = flat_target[
            indices
        ]

        flat_probability = flat_probability[
            indices
        ]

    if len(np.unique(flat_target)) < 2:

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

    return roc_auc, pr_auc


# ============================================================
# VALIDATE ONE PATIENT
# ============================================================

@torch.no_grad()
def validate_patient(
    model,
    file
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
            image
        )
    )

    regions_true = make_regions(
        mask
    )

    regions_pred = make_regions(
        prediction
    )

    result = {
        "patient_id": file.stem
    }

    for region in ["WT", "TC", "ET"]:

        true_region = regions_true[
            region
        ]

        pred_region = regions_pred[
            region
        ]

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
                probabilities[1] +
                probabilities[2] +
                probabilities[3]
            )

        elif region == "TC":

            probability = (
                probabilities[1] +
                probabilities[3]
            )

        else:

            probability = probabilities[3]

        roc_auc, pr_auc = auc_metrics(
            probability,
            true_region
        )

        result[f"{region}_accuracy"] = metrics[
            "accuracy"
        ]

        result[f"{region}_precision"] = metrics[
            "precision"
        ]

        result[f"{region}_sensitivity"] = metrics[
            "sensitivity"
        ]

        result[f"{region}_specificity"] = metrics[
            "specificity"
        ]

        result[f"{region}_f1"] = metrics[
            "f1"
        ]

        result[f"{region}_dice"] = metrics[
            "dice"
        ]

        result[f"{region}_iou"] = metrics[
            "iou"
        ]

        result[f"{region}_balanced_accuracy"] = metrics[
            "balanced_accuracy"
        ]

        result[f"{region}_mcc"] = metrics[
            "mcc"
        ]

        result[f"{region}_hd95"] = hd

        result[f"{region}_roc_auc"] = roc_auc

        result[f"{region}_pr_auc"] = pr_auc

        result[f"{region}_tp"] = metrics["tp"]

        result[f"{region}_tn"] = metrics["tn"]

        result[f"{region}_fp"] = metrics["fp"]

        result[f"{region}_fn"] = metrics["fn"]

    result["mean_dice"] = np.nanmean([
        result["WT_dice"],
        result["TC_dice"],
        result["ET_dice"]
    ])

    result["mean_iou"] = np.nanmean([
        result["WT_iou"],
        result["TC_iou"],
        result["ET_iou"]
    ])

    result["mean_accuracy"] = np.nanmean([
        result["WT_accuracy"],
        result["TC_accuracy"],
        result["ET_accuracy"]
    ])

    result["mean_precision"] = np.nanmean([
        result["WT_precision"],
        result["TC_precision"],
        result["ET_precision"]
    ])

    result["mean_sensitivity"] = np.nanmean([
        result["WT_sensitivity"],
        result["TC_sensitivity"],
        result["ET_sensitivity"]
    ])

    result["mean_specificity"] = np.nanmean([
        result["WT_specificity"],
        result["TC_specificity"],
        result["ET_specificity"]
    ])

    result["mean_f1"] = np.nanmean([
        result["WT_f1"],
        result["TC_f1"],
        result["ET_f1"]
    ])

    result["mean_balanced_accuracy"] = np.nanmean([
        result["WT_balanced_accuracy"],
        result["TC_balanced_accuracy"],
        result["ET_balanced_accuracy"]
    ])

    result["mean_mcc"] = np.nanmean([
        result["WT_mcc"],
        result["TC_mcc"],
        result["ET_mcc"]
    ])

    result["mean_hd95"] = np.nanmean([
        result["WT_hd95"],
        result["TC_hd95"],
        result["ET_hd95"]
    ])

    result["mean_roc_auc"] = np.nanmean([
        result["WT_roc_auc"],
        result["TC_roc_auc"],
        result["ET_roc_auc"]
    ])

    result["mean_pr_auc"] = np.nanmean([
        result["WT_pr_auc"],
        result["TC_pr_auc"],
        result["ET_pr_auc"]
    ])

    result["inference_time_seconds"] = (
        time.time() - start
    )

    return result


# ============================================================
# FULL VALIDATION
# ============================================================

def validate_model(
    model,
    validation_files,
    output_dir,
    epoch
):

    model.eval()

    all_results = []

    start = time.time()

    print()
    print("=" * 80)
    print(
        f"FULL VALIDATION — "
        f"{len(validation_files)} PATIENTS"
    )
    print("=" * 80)

    for i, file in enumerate(
        validation_files,
        start=1
    ):

        result = validate_patient(
            model,
            file
        )

        all_results.append(result)

        print(
            f"[{i:3d}/{len(validation_files)}] "
            f"{result['patient_id']} | "
            f"WT={result['WT_dice']:.4f} "
            f"TC={result['TC_dice']:.4f} "
            f"ET={result['ET_dice']:.4f} | "
            f"{result['inference_time_seconds']:.2f}s"
        )

    df = pd.DataFrame(
        all_results
    )

    patient_file = (
        output_dir /
        f"validation_patients_epoch_{epoch:02d}.csv"
    )

    df.to_csv(
        patient_file,
        index=False
    )

    summary = {}

    numeric_columns = [
        c for c in df.columns
        if c != "patient_id"
    ]

    for column in numeric_columns:

        summary[column] = df[
            column
        ].mean()

    summary["num_patients"] = len(df)

    validation_time = (
        time.time() - start
    )

    summary[
        "validation_time_seconds"
    ] = validation_time

    return summary, df


# ============================================================
# PLOT TRAINING HISTORY
# ============================================================

def create_plots(
    history,
    output_dir
):

    df = pd.DataFrame(history)

    # -------------------------
    # Loss
    # -------------------------

    plt.figure(figsize=(8, 5))

    plt.plot(
        df["epoch"],
        df["train_loss"],
        marker="o"
    )

    plt.xlabel("Epoch")
    plt.ylabel("Training Loss")

    plt.title(
        "Training Loss"
    )

    plt.grid(True)

    plt.tight_layout()

    plt.savefig(
        output_dir / "training_loss.png",
        dpi=200
    )

    plt.close()

    # -------------------------
    # Dice
    # -------------------------

    plt.figure(figsize=(8, 5))

    plt.plot(
        df["epoch"],
        df["WT_dice"],
        marker="o",
        label="WT"
    )

    plt.plot(
        df["epoch"],
        df["TC_dice"],
        marker="o",
        label="TC"
    )

    plt.plot(
        df["epoch"],
        df["ET_dice"],
        marker="o",
        label="ET"
    )

    plt.plot(
        df["epoch"],
        df["mean_dice"],
        marker="o",
        label="Mean Dice"
    )

    plt.xlabel("Epoch")

    plt.ylabel("Dice")

    plt.title(
        "Validation Dice"
    )

    plt.legend()

    plt.grid(True)

    plt.tight_layout()

    plt.savefig(
        output_dir / "dice_curve.png",
        dpi=200
    )

    plt.close()

    # -------------------------
    # IoU
    # -------------------------

    plt.figure(figsize=(8, 5))

    plt.plot(
        df["epoch"],
        df["WT_iou"],
        marker="o",
        label="WT"
    )

    plt.plot(
        df["epoch"],
        df["TC_iou"],
        marker="o",
        label="TC"
    )

    plt.plot(
        df["epoch"],
        df["ET_iou"],
        marker="o",
        label="ET"
    )

    plt.plot(
        df["epoch"],
        df["mean_iou"],
        marker="o",
        label="Mean IoU"
    )

    plt.xlabel("Epoch")

    plt.ylabel("IoU")

    plt.title(
        "Validation IoU"
    )

    plt.legend()

    plt.grid(True)

    plt.tight_layout()

    plt.savefig(
        output_dir / "iou_curve.png",
        dpi=200
    )

    plt.close()

    # -------------------------
    # Precision
    # -------------------------

    plt.figure(figsize=(8, 5))

    plt.plot(
        df["epoch"],
        df["WT_precision"],
        marker="o",
        label="WT"
    )

    plt.plot(
        df["epoch"],
        df["TC_precision"],
        marker="o",
        label="TC"
    )

    plt.plot(
        df["epoch"],
        df["ET_precision"],
        marker="o",
        label="ET"
    )

    plt.xlabel("Epoch")

    plt.ylabel("Precision")

    plt.title(
        "Validation Precision"
    )

    plt.legend()

    plt.grid(True)

    plt.tight_layout()

    plt.savefig(
        output_dir / "precision_curve.png",
        dpi=200
    )

    plt.close()

    # -------------------------
    # Sensitivity
    # -------------------------

    plt.figure(figsize=(8, 5))

    plt.plot(
        df["epoch"],
        df["WT_sensitivity"],
        marker="o",
        label="WT"
    )

    plt.plot(
        df["epoch"],
        df["TC_sensitivity"],
        marker="o",
        label="TC"
    )

    plt.plot(
        df["epoch"],
        df["ET_sensitivity"],
        marker="o",
        label="ET"
    )

    plt.xlabel("Epoch")

    plt.ylabel("Sensitivity")

    plt.title(
        "Validation Sensitivity"
    )

    plt.legend()

    plt.grid(True)

    plt.tight_layout()

    plt.savefig(
        output_dir / "sensitivity_curve.png",
        dpi=200
    )

    plt.close()

    # -------------------------
    # Specificity
    # -------------------------

    plt.figure(figsize=(8, 5))

    plt.plot(
        df["epoch"],
        df["WT_specificity"],
        marker="o",
        label="WT"
    )

    plt.plot(
        df["epoch"],
        df["TC_specificity"],
        marker="o",
        label="TC"
    )

    plt.plot(
        df["epoch"],
        df["ET_specificity"],
        marker="o",
        label="ET"
    )

    plt.xlabel("Epoch")

    plt.ylabel("Specificity")

    plt.title(
        "Validation Specificity"
    )

    plt.legend()

    plt.grid(True)

    plt.tight_layout()

    plt.savefig(
        output_dir / "specificity_curve.png",
        dpi=200
    )

    plt.close()

    # -------------------------
    # ROC-AUC
    # -------------------------

    plt.figure(figsize=(8, 5))

    plt.plot(
        df["epoch"],
        df["WT_roc_auc"],
        marker="o",
        label="WT"
    )

    plt.plot(
        df["epoch"],
        df["TC_roc_auc"],
        marker="o",
        label="TC"
    )

    plt.plot(
        df["epoch"],
        df["ET_roc_auc"],
        marker="o",
        label="ET"
    )

    plt.xlabel("Epoch")

    plt.ylabel("ROC-AUC")

    plt.title(
        "Validation ROC-AUC"
    )

    plt.legend()

    plt.grid(True)

    plt.tight_layout()

    plt.savefig(
        output_dir / "roc_auc_curve.png",
        dpi=200
    )

    plt.close()

    # -------------------------
    # PR-AUC
    # -------------------------

    plt.figure(figsize=(8, 5))

    plt.plot(
        df["epoch"],
        df["WT_pr_auc"],
        marker="o",
        label="WT"
    )

    plt.plot(
        df["epoch"],
        df["TC_pr_auc"],
        marker="o",
        label="TC"
    )

    plt.plot(
        df["epoch"],
        df["ET_pr_auc"],
        marker="o",
        label="ET"
    )

    plt.xlabel("Epoch")

    plt.ylabel("PR-AUC")

    plt.title(
        "Validation PR-AUC"
    )

    plt.legend()

    plt.grid(True)

    plt.tight_layout()

    plt.savefig(
        output_dir / "pr_auc_curve.png",
        dpi=200
    )

    plt.close()

    # -------------------------
    # HD95
    # -------------------------

    plt.figure(figsize=(8, 5))

    plt.plot(
        df["epoch"],
        df["WT_hd95"],
        marker="o",
        label="WT"
    )

    plt.plot(
        df["epoch"],
        df["TC_hd95"],
        marker="o",
        label="TC"
    )

    plt.plot(
        df["epoch"],
        df["ET_hd95"],
        marker="o",
        label="ET"
    )

    plt.xlabel("Epoch")

    plt.ylabel("HD95")

    plt.title(
        "Validation HD95"
    )

    plt.legend()

    plt.grid(True)

    plt.tight_layout()

    plt.savefig(
        output_dir / "hd95_curve.png",
        dpi=200
    )

    plt.close()


# ============================================================
# TRAIN ONE CLIENT
# ============================================================

def train_client(client):

    set_seed(SEED)

    train_dir = (
        DATA_ROOT /
        client /
        "train"
    )

    val_dir = (
        DATA_ROOT /
        client /
        "val"
    )

    output_dir = (
        RESULT_ROOT /
        client
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    train_dataset = PatientDataset(
        train_dir,
        training=True,
        patches_per_epoch=PATCHES_PER_EPOCH
    )

    val_dataset = PatientDataset(
        val_dir,
        training=False
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )

    train_files = train_dataset.files

    val_files = val_dataset.files

    print("=" * 70)

    print(
        f"NEUROVERSE — FINAL LOCAL TRAINING"
    )

    print("=" * 70)

    print(
        f"Client: {client}"
    )

    print(
        f"Device: {device}"
    )

    if torch.cuda.is_available():

        print(
            f"GPU: "
            f"{torch.cuda.get_device_name(0)}"
        )

        print(
            f"GPU memory: "
            f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB"
        )

    print()

    print(
        f"Training patients: "
        f"{len(train_files)}"
    )

    print(
        f"Validation patients: "
        f"{len(val_files)}"
    )

    print(
        f"Patches per epoch: "
        f"{PATCHES_PER_EPOCH}"
    )

    print(
        f"Patch size: "
        f"{PATCH_SIZE}"
    )

    print()

    model = UNet3D(
        in_channels=4,
        num_classes=NUM_CLASSES,
        base=BASE_CHANNELS
    ).to(device)

    total_parameters = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print(
        f"Trainable parameters: "
        f"{total_parameters:,}"
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=1e-5
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=2
    )

    criterion = CombinedLoss()

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=torch.cuda.is_available()
    )

    history = []

    best_dice = -1

    best_epoch = -1

    total_start = time.time()

    # ========================================================
    # EPOCHS
    # ========================================================

    for epoch in range(
        1,
        EPOCHS + 1
    ):

        print()
        print("=" * 70)

        print(
            f"EPOCH {epoch}/{EPOCHS}"
        )

        print("=" * 70)

        model.train()

        epoch_start = time.time()

        running_loss = 0.0

        for step, batch in enumerate(
            train_loader,
            start=1
        ):

            images, masks, _ = batch

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

            if step % 10 == 0:

                print(
                    f"Step {step:3d}/"
                    f"{len(train_loader):3d} | "
                    f"Loss: {loss.item():.4f}"
                )

        train_loss = (
            running_loss /
            len(train_loader)
        )

        # ====================================================
        # FULL VALIDATION
        # ====================================================

        summary, patient_df = validate_model(
            model,
            val_files,
            output_dir,
            epoch
        )

        summary["epoch"] = epoch

        summary["train_loss"] = train_loss

        summary["epoch_time_seconds"] = (
            time.time() -
            epoch_start
        )

        summary["learning_rate"] = (
            optimizer.param_groups[0]["lr"]
        )

        history.append(
            summary
        )

        # ====================================================
        # PRINT SUMMARY
        # ====================================================

        print()
        print("=" * 70)
        print("FULL VALIDATION RESULTS")
        print("=" * 70)

        print(
            f"Train loss       : "
            f"{train_loss:.4f}"
        )

        print(
            f"Mean Accuracy    : "
            f"{summary['mean_accuracy']:.4f}"
        )

        print(
            f"Mean Precision   : "
            f"{summary['mean_precision']:.4f}"
        )

        print(
            f"Mean Sensitivity : "
            f"{summary['mean_sensitivity']:.4f}"
        )

        print(
            f"Mean Specificity : "
            f"{summary['mean_specificity']:.4f}"
        )

        print(
            f"Mean F1          : "
            f"{summary['mean_f1']:.4f}"
        )

        print(
            f"Mean Dice        : "
            f"{summary['mean_dice']:.4f}"
        )

        print(
            f"Mean IoU         : "
            f"{summary['mean_iou']:.4f}"
        )

        print(
            f"Mean Balanced Acc: "
            f"{summary['mean_balanced_accuracy']:.4f}"
        )

        print(
            f"Mean MCC         : "
            f"{summary['mean_mcc']:.4f}"
        )

        print(
            f"Mean HD95        : "
            f"{summary['mean_hd95']:.4f}"
        )

        print(
            f"Mean ROC-AUC     : "
            f"{summary['mean_roc_auc']:.4f}"
        )

        print(
            f"Mean PR-AUC      : "
            f"{summary['mean_pr_auc']:.4f}"
        )

        print(
            f"Validation time  : "
            f"{summary['validation_time_seconds']:.2f}s"
        )

        # ====================================================
        # SAVE BEST MODEL
        # ====================================================

        current_dice = summary[
            "mean_dice"
        ]

        if current_dice > best_dice:

            best_dice = current_dice

            best_epoch = epoch

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict":
                        model.state_dict(),
                    "optimizer_state_dict":
                        optimizer.state_dict(),
                    "mean_dice":
                        best_dice
                },
                output_dir /
                "best_model.pth"
            )

            print()
            print(
                "✓ NEW BEST MODEL SAVED"
            )

        scheduler.step(
            current_dice
        )

    # ========================================================
    # SAVE FINAL MODEL
    # ========================================================

    torch.save(
        {
            "epoch": EPOCHS,
            "model_state_dict":
                model.state_dict(),
            "optimizer_state_dict":
                optimizer.state_dict()
        },
        output_dir /
        "final_model.pth"
    )

    # ========================================================
    # SAVE HISTORY
    # ========================================================

    history_df = pd.DataFrame(
        history
    )

    history_df.to_csv(
        output_dir /
        "training_history.csv",
        index=False
    )

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    best_row = history_df.loc[
        history_df["mean_dice"].idxmax()
    ]

    final_summary = {
        "client": client,
        "best_epoch": int(
            best_row["epoch"]
        ),
        "best_mean_dice": float(
            best_row["mean_dice"]
        ),
        "best_mean_iou": float(
            best_row["mean_iou"]
        ),
        "best_mean_accuracy": float(
            best_row["mean_accuracy"]
        ),
        "best_mean_precision": float(
            best_row["mean_precision"]
        ),
        "best_mean_sensitivity": float(
            best_row["mean_sensitivity"]
        ),
        "best_mean_specificity": float(
            best_row["mean_specificity"]
        ),
        "best_mean_f1": float(
            best_row["mean_f1"]
        ),
        "best_mean_balanced_accuracy": float(
            best_row["mean_balanced_accuracy"]
        ),
        "best_mean_mcc": float(
            best_row["mean_mcc"]
        ),
        "best_mean_hd95": float(
            best_row["mean_hd95"]
        ),
        "best_mean_roc_auc": float(
            best_row["mean_roc_auc"]
        ),
        "best_mean_pr_auc": float(
            best_row["mean_pr_auc"]
        ),
        "total_training_time_seconds":
            time.time() - total_start
    }

    pd.DataFrame(
        [final_summary]
    ).to_csv(
        output_dir /
        "final_summary.csv",
        index=False
    )

    # ========================================================
    # GRAPHS
    # ========================================================

    create_plots(
        history,
        output_dir
    )

    # ========================================================
    # DONE
    # ========================================================

    print()
    print("=" * 70)

    print(
        "TRAINING COMPLETE"
    )

    print("=" * 70)

    print(
        f"Client: {client}"
    )

    print(
        f"Best epoch: {best_epoch}"
    )

    print(
        f"Best Mean Dice: "
        f"{best_dice:.4f}"
    )

    print()

    print(
        f"Results directory:"
    )

    print(
        output_dir
    )

    print()

    print(
        "Saved:"
    )

    print(
        "  best_model.pth"
    )

    print(
        "  final_model.pth"
    )

    print(
        "  training_history.csv"
    )

    print(
        "  final_summary.csv"
    )

    print(
        "  validation_patients_epoch_XX.csv"
    )

    print(
        "  training_loss.png"
    )

    print(
        "  dice_curve.png"
    )

    print(
        "  iou_curve.png"
    )

    print(
        "  precision_curve.png"
    )

    print(
        "  sensitivity_curve.png"
    )

    print(
        "  specificity_curve.png"
    )

    print(
        "  roc_auc_curve.png"
    )

    print(
        "  pr_auc_curve.png"
    )

    print(
        "  hd95_curve.png"
    )

    print("=" * 70)


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--client",
        type=str,
        required=True,
        choices=["C1", "C2", "C3"]
    )

    args = parser.parse_args()

    train_client(
        args.client
    )
