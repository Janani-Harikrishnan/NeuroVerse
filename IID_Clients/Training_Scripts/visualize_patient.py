import os
import argparse
import tarfile
import tempfile
import shutil

import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_DIR = "/home/janani/Neuroverse"

TAR_FILE = os.path.join(
    PROJECT_DIR,
    "BraTS2021_Training_Data.tar"
)

PROCESSED_DIR = os.path.join(
    PROJECT_DIR,
    "IID_Clients",
    "GlobalTest",
    "test"
)

OUTPUT_DIR = os.path.join(
    PROJECT_DIR,
    "IID_Clients",
    "Visualization"
)


# ============================================================
# MODALITIES
# ============================================================

MODALITIES = [
    "t1",
    "t1ce",
    "t2",
    "flair"
]


# ============================================================
# NORMALIZATION FOR DISPLAY ONLY
# ============================================================

def normalize_for_display(image):

    image = image.astype(np.float32)

    nonzero = image[image != 0]

    if len(nonzero) == 0:
        return np.zeros_like(image)

    low = np.percentile(nonzero, 1)
    high = np.percentile(nonzero, 99)

    image = np.clip(
        image,
        low,
        high
    )

    image = (
        image - low
    ) / (
        high - low + 1e-8
    )

    return image


# ============================================================
# LOAD PROCESSED NPZ
# ============================================================

def load_processed_patient(patient_id):

    file = os.path.join(
        PROCESSED_DIR,
        f"{patient_id}.npz"
    )

    if not os.path.exists(file):

        raise FileNotFoundError(
            f"""
Processed patient not found:

{file}
"""
        )

    print()
    print("=" * 80)
    print("LOADING PROCESSED DATA")
    print("=" * 80)

    print(f"File: {file}")

    data = np.load(file)

    print(
        f"Keys: {data.files}"
    )

    image = data["image"]

    mask = data["mask"]

    bbox = data["bbox"]

    return (
        image,
        mask,
        bbox,
        file
    )


# ============================================================
# EXTRACT ORIGINAL PATIENT FROM TAR
# ============================================================

def extract_original_patient(
    patient_id,
    temp_dir
):

    patient_prefix = (
        f"./{patient_id}/"
    )

    print()
    print("=" * 80)
    print("EXTRACTING ORIGINAL DATA FROM TAR")
    print("=" * 80)

    print(
        f"Patient: {patient_id}"
    )

    if not os.path.exists(TAR_FILE):

        raise FileNotFoundError(
            f"TAR file not found:\n{TAR_FILE}"
        )

    extracted_files = {}

    with tarfile.open(
        TAR_FILE,
        "r"
    ) as tar:

        members = []

        for member in tar.getmembers():

            if (
                member.name.startswith(
                    patient_prefix
                )
                and member.name.endswith(
                    ".nii.gz"
                )
            ):

                members.append(member)

        if not members:

            raise FileNotFoundError(
                f"No NIfTI files found "
                f"for {patient_id}"
            )

        for member in members:

            filename = os.path.basename(
                member.name
            )

            output_path = os.path.join(
                temp_dir,
                filename
            )

            source = tar.extractfile(
                member
            )

            if source is None:
                continue

            with open(
                output_path,
                "wb"
            ) as f:

                shutil.copyfileobj(
                    source,
                    f
                )

            lower = filename.lower()

            if "_flair" in lower:
                extracted_files["flair"] = (
                    output_path
                )

            elif "_t1ce" in lower:
                extracted_files["t1ce"] = (
                    output_path
                )

            elif "_t1." in lower:
                extracted_files["t1"] = (
                    output_path
                )

            elif "_t2." in lower:
                extracted_files["t2"] = (
                    output_path
                )

            elif "_seg." in lower:
                extracted_files["seg"] = (
                    output_path
                )

    print()
    print("Original files found:")

    for key, path in extracted_files.items():

        print(
            f"  {key:<8}: "
            f"{os.path.basename(path)}"
        )

    return extracted_files


# ============================================================
# LOAD ORIGINAL NIFTI
# ============================================================

def load_original_patient(
    patient_id,
    temp_dir
):

    files = extract_original_patient(
        patient_id,
        temp_dir
    )

    images = {}

    for modality in MODALITIES:

        if modality not in files:

            raise FileNotFoundError(
                f"Missing original modality: "
                f"{modality}"
            )

        nii = nib.load(
            files[modality]
        )

        images[modality] = (
            nii.get_fdata()
        )

    if "seg" not in files:

        raise FileNotFoundError(
            "Original segmentation "
            "was not found."
        )

    seg_nii = nib.load(
        files["seg"]
    )

    segmentation = np.rint(
        seg_nii.get_fdata()
    ).astype(np.uint8)

    return (
        images,
        segmentation
    )


# ============================================================
# FIND BEST PROCESSED SLICE
# ============================================================

def find_best_processed_slice(
    mask
):

    tumor = mask > 0

    tumor_per_slice = tumor.sum(
        axis=(0, 1)
    )

    slice_idx = int(
        np.argmax(
            tumor_per_slice
        )
    )

    return slice_idx


# ============================================================
# MAP PROCESSED SLICE TO ORIGINAL
# ============================================================

def map_processed_slice_to_original(
    processed_z,
    bbox,
    original_shape
):

    bbox = np.asarray(
        bbox
    ).astype(int)

    # Expected:
    #
    # [x_min, x_max,
    #  y_min, y_max,
    #  z_min, z_max]
    #

    z_min = bbox[4]

    z_max = bbox[5]

    processed_depth = (
        z_max - z_min + 1
    )

    if processed_depth <= 0:

        raise ValueError(
            f"Invalid bbox: {bbox}"
        )

    # In most cases the processed
    # volume corresponds directly
    # to the bounding-box region.

    original_z = (
        z_min + processed_z
    )

    original_z = max(
        0,
        min(
            original_z,
            original_shape[2] - 1
        )
    )

    return original_z


# ============================================================
# CREATE SEGMENTATION RGB
# ============================================================

def create_segmentation_rgb(
    mask
):

    rgb = np.zeros(
        (*mask.shape, 3),
        dtype=np.float32
    )

    # Label 1 → red
    rgb[mask == 1] = [
        1,
        0,
        0
    ]

    # Label 2 → green
    rgb[mask == 2] = [
        0,
        1,
        0
    ]

    # Label 3 → blue
    rgb[mask == 3] = [
        0,
        0,
        1
    ]

    return rgb


# ============================================================
# PRINT INFORMATION
# ============================================================

def print_information(
    patient_id,
    original_images,
    original_mask,
    processed_image,
    processed_mask,
    bbox,
    processed_z,
    original_z
):

    print()
    print("=" * 80)
    print("PATIENT / PREPROCESSING INFORMATION")
    print("=" * 80)

    print(
        f"Patient: {patient_id}"
    )

    # --------------------------------------------------------
    # Original
    # --------------------------------------------------------

    original_shape = (
        original_images["flair"].shape
    )

    print()
    print("ORIGINAL DATA")
    print("-" * 80)

    print(
        f"Volume shape: "
        f"{original_shape}"
    )

    print(
        f"X = {original_shape[0]}"
    )

    print(
        f"Y = {original_shape[1]}"
    )

    print(
        f"Z = {original_shape[2]}"
    )

    print(
        "Segmentation labels:",
        np.unique(
            original_mask
        )
    )

    # --------------------------------------------------------
    # Processed
    # --------------------------------------------------------

    processed_shape = (
        processed_image.shape
    )

    print()
    print("PROCESSED DATA")
    print("-" * 80)

    print(
        f"Image shape: "
        f"{processed_shape}"
    )

    print(
        f"Mask shape: "
        f"{processed_mask.shape}"
    )

    print(
        "Segmentation labels:",
        np.unique(
            processed_mask
        )
    )

    # --------------------------------------------------------
    # BBOX
    # --------------------------------------------------------

    print()
    print("BRAIN / TUMOR CROP")
    print("-" * 80)

    print(
        "Bounding box:"
    )

    print(
        bbox
    )

    # --------------------------------------------------------
    # Slice correspondence
    # --------------------------------------------------------

    print()
    print("SLICE CORRESPONDENCE")
    print("-" * 80)

    print(
        f"Processed Z slice : "
        f"{processed_z}"
    )

    print(
        f"Original Z slice  : "
        f"{original_z}"
    )

    # --------------------------------------------------------
    # Tumor voxels
    # --------------------------------------------------------

    print()
    print("PROCESSED TUMOR VOXELS")
    print("-" * 80)

    total = 0

    for label, name in [
        (1, "NCR / Necrotic"),
        (2, "Edema"),
        (3, "Enhancing Tumor")
    ]:

        count = int(
            np.sum(
                processed_mask == label
            )
        )

        total += count

        print(
            f"{name:<22}: "
            f"{count:,}"
        )

    print(
        f"{'Total tumor':<22}: "
        f"{total:,}"
    )

    print("=" * 80)


# ============================================================
# VISUALIZATION
# ============================================================

def visualize_patient(
    patient_id
):

    # --------------------------------------------------------
    # Temporary extraction directory
    # --------------------------------------------------------

    temp_dir = tempfile.mkdtemp(
        prefix="brats_visualize_"
    )

    try:

        # ----------------------------------------------------
        # Load processed
        # ----------------------------------------------------

        (
            processed_image,
            processed_mask,
            bbox,
            processed_file
        ) = load_processed_patient(
            patient_id
        )

        # ----------------------------------------------------
        # Validate processed
        # ----------------------------------------------------

        if processed_image.ndim != 4:

            raise ValueError(
                "Processed image must "
                "have shape (4,X,Y,Z)"
            )

        if processed_image.shape[0] != 4:

            raise ValueError(
                "Processed image must "
                "contain 4 modalities"
            )

        if (
            processed_mask.shape
            != processed_image.shape[1:]
        ):

            raise ValueError(
                "Processed image/mask "
                "shape mismatch"
            )

        # ----------------------------------------------------
        # Load original
        # ----------------------------------------------------

        (
            original_images,
            original_mask
        ) = load_original_patient(
            patient_id,
            temp_dir
        )

        original_shape = (
            original_images["flair"].shape
        )

        # ----------------------------------------------------
        # Best tumor slice
        # ----------------------------------------------------

        processed_z = (
            find_best_processed_slice(
                processed_mask
            )
        )

        # ----------------------------------------------------
        # Map to original
        # ----------------------------------------------------

        original_z = (
            map_processed_slice_to_original(
                processed_z,
                bbox,
                original_shape
            )
        )

        # ----------------------------------------------------
        # Print information
        # ----------------------------------------------------

        print_information(
            patient_id,
            original_images,
            original_mask,
            processed_image,
            processed_mask,
            bbox,
            processed_z,
            original_z
        )

        # ----------------------------------------------------
        # ORIGINAL SLICES
        # ----------------------------------------------------

        original_t1 = (
            original_images["t1"][
                :,
                :,
                original_z
            ]
        )

        original_t1ce = (
            original_images["t1ce"][
                :,
                :,
                original_z
            ]
        )

        original_t2 = (
            original_images["t2"][
                :,
                :,
                original_z
            ]
        )

        original_flair = (
            original_images["flair"][
                :,
                :,
                original_z
            ]
        )

        original_seg = (
            original_mask[
                :,
                :,
                original_z
            ]
        )

        # ----------------------------------------------------
        # PROCESSED SLICES
        # ----------------------------------------------------

        processed_t1 = (
            processed_image[
                0,
                :,
                :,
                processed_z
            ]
        )

        processed_t1ce = (
            processed_image[
                1,
                :,
                :,
                processed_z
            ]
        )

        processed_t2 = (
            processed_image[
                2,
                :,
                :,
                processed_z
            ]
        )

        processed_flair = (
            processed_image[
                3,
                :,
                :,
                processed_z
            ]
        )

        processed_seg = (
            processed_mask[
                :,
                :,
                processed_z
            ]
        )

        # ----------------------------------------------------
        # Normalize for display
        #
        # NOTE:
        # This is ONLY for visualization.
        # It does not modify the saved data.
        # ----------------------------------------------------

        original_t1 = normalize_for_display(
            original_t1
        )

        original_t1ce = normalize_for_display(
            original_t1ce
        )

        original_t2 = normalize_for_display(
            original_t2
        )

        original_flair = normalize_for_display(
            original_flair
        )

        processed_t1 = normalize_for_display(
            processed_t1
        )

        processed_t1ce = normalize_for_display(
            processed_t1ce
        )

        processed_t2 = normalize_for_display(
            processed_t2
        )

        processed_flair = normalize_for_display(
            processed_flair
        )

        # ----------------------------------------------------
        # CREATE SEGMENTATION RGB
        # ----------------------------------------------------

        original_rgb = (
            create_segmentation_rgb(
                original_seg
            )
        )

        processed_rgb = (
            create_segmentation_rgb(
                processed_seg
            )
        )

        # ----------------------------------------------------
        # CREATE OVERLAYS
        # ----------------------------------------------------

        original_base = np.stack(
            [
                original_flair,
                original_flair,
                original_flair
            ],
            axis=-1
        )

        processed_base = np.stack(
            [
                processed_flair,
                processed_flair,
                processed_flair
            ],
            axis=-1
        )

        alpha = 0.45

        original_overlay = (
            (1 - alpha) * original_base
            + alpha * original_rgb
        )

        processed_overlay = (
            (1 - alpha) * processed_base
            + alpha * processed_rgb
        )

        # ----------------------------------------------------
        # FIGURE
        # ----------------------------------------------------

        fig, axes = plt.subplots(
            4,
            4,
            figsize=(18, 18)
        )

        fig.suptitle(
            f"BraTS2021 — {patient_id}\n"
            f"Original vs Preprocessed",
            fontsize=20,
            fontweight="bold"
        )

        # ====================================================
        # ROW 1 — T1
        # ====================================================

        axes[0, 0].imshow(
            np.rot90(original_t1),
            cmap="gray"
        )

        axes[0, 0].set_title(
            "ORIGINAL — T1"
        )

        axes[0, 1].imshow(
            np.rot90(processed_t1),
            cmap="gray"
        )

        axes[0, 1].set_title(
            "PROCESSED — T1"
        )

        axes[0, 2].axis("off")

        axes[0, 3].axis("off")

        # ====================================================
        # ROW 2 — T1ce / T2
        # ====================================================

        axes[1, 0].imshow(
            np.rot90(original_t1ce),
            cmap="gray"
        )

        axes[1, 0].set_title(
            "ORIGINAL — T1ce"
        )

        axes[1, 1].imshow(
            np.rot90(processed_t1ce),
            cmap="gray"
        )

        axes[1, 1].set_title(
            "PROCESSED — T1ce"
        )

        axes[1, 2].imshow(
            np.rot90(original_t2),
            cmap="gray"
        )

        axes[1, 2].set_title(
            "ORIGINAL — T2"
        )

        axes[1, 3].imshow(
            np.rot90(processed_t2),
            cmap="gray"
        )

        axes[1, 3].set_title(
            "PROCESSED — T2"
        )

        # ====================================================
        # ROW 3 — FLAIR
        # ====================================================

        axes[2, 0].imshow(
            np.rot90(original_flair),
            cmap="gray"
        )

        axes[2, 0].set_title(
            "ORIGINAL — FLAIR"
        )

        axes[2, 1].imshow(
            np.rot90(processed_flair),
            cmap="gray"
        )

        axes[2, 1].set_title(
            "PROCESSED — FLAIR"
        )

        axes[2, 2].imshow(
            np.rot90(original_seg),
            cmap="viridis",
            vmin=0,
            vmax=3
        )

        axes[2, 2].set_title(
            "ORIGINAL — Ground Truth"
        )

        axes[2, 3].imshow(
            np.rot90(processed_seg),
            cmap="viridis",
            vmin=0,
            vmax=3
        )

        axes[2, 3].set_title(
            "PROCESSED — Ground Truth"
        )

        # ====================================================
        # ROW 4 — OVERLAYS
        # ====================================================

        axes[3, 0].imshow(
            np.rot90(original_overlay)
        )

        axes[3, 0].set_title(
            "ORIGINAL — Tumor Overlay"
        )

        axes[3, 1].imshow(
            np.rot90(processed_overlay)
        )

        axes[3, 1].set_title(
            "PROCESSED — Tumor Overlay"
        )

        # ----------------------------------------------------
        # INFORMATION PANEL
        # ----------------------------------------------------

        axes[3, 2].axis("off")

        axes[3, 2].text(
            0.05,
            0.90,
            "ORIGINAL",
            fontsize=14,
            fontweight="bold"
        )

        axes[3, 2].text(
            0.05,
            0.78,
            f"Shape:\n"
            f"{original_shape}\n\n"
            f"Z slice: {original_z}",
            fontsize=11
        )

        axes[3, 3].axis("off")

        axes[3, 3].text(
            0.05,
            0.90,
            "PROCESSED",
            fontsize=14,
            fontweight="bold"
        )

        axes[3, 3].text(
            0.05,
            0.78,
            f"Shape:\n"
            f"{processed_image.shape[1:]}\n\n"
            f"Z slice: {processed_z}\n\n"
            f"BBox:\n"
            f"{bbox}",
            fontsize=11
        )

        # ----------------------------------------------------
        # REMOVE AXES
        # ----------------------------------------------------

        for ax in axes.flat:

            ax.set_xticks([])
            ax.set_yticks([])

        plt.tight_layout(
            rect=[
                0,
                0,
                1,
                0.95
            ]
        )

        # ----------------------------------------------------
        # SAVE
        # ----------------------------------------------------

        os.makedirs(
            OUTPUT_DIR,
            exist_ok=True
        )

        output_file = os.path.join(
            OUTPUT_DIR,
            f"{patient_id}_original_vs_processed.png"
        )

        plt.savefig(
            output_file,
            dpi=300,
            bbox_inches="tight"
        )

        print()
        print("=" * 80)
        print("VISUALIZATION COMPLETE")
        print("=" * 80)

        print(
            f"Saved to:\n{output_file}"
        )

        print("=" * 80)

        plt.show()

    finally:

        # ----------------------------------------------------
        # Remove temporary extracted NIfTI files
        # ----------------------------------------------------

        shutil.rmtree(
            temp_dir,
            ignore_errors=True
        )


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Compare original BraTS NIfTI data "
            "with preprocessed NPZ data."
        )
    )

    parser.add_argument(
        "--patient",
        required=True,
        help=(
            "Patient ID, e.g. "
            "BraTS2021_00032"
        )
    )

    args = parser.parse_args()

    visualize_patient(
        args.patient
    )


if __name__ == "__main__":

    main()