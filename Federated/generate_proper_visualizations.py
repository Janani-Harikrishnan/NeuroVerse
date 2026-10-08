# ============================================================
# NEUROVERSE
# PROPER ONE-PATIENT VISUALIZATION
# ============================================================
#
# Patient:
#     BraTS2021_00032
#
# Experiments:
#
#     E1 - 3D U-Net + FedAvg
#     E2 - Swin UNETR + FedAvg
#     E3 - Swin UNETR + Adaptive FL
#     E4 - Swin UNETR + Quantum Kernel
#
# IMPORTANT:
#
# We DO NOT implement our own Swin inference here.
#
# We use each experiment's ORIGINAL project visualization
# pipeline so preprocessing / padding / sliding-window inference
# exactly matches the experiment that produced the reported
# metrics.
#
# E4's visualization script is known to contain E3 hardcoded
# paths/text. Therefore E4 is executed through a corrected
# temporary copy.
#
# Finally, all four images are normalized to the SAME visual
# format.
#
# ============================================================


import os
import sys
import runpy
import shutil
import tempfile
import subprocess
from pathlib import Path

import numpy as np

from PIL import Image


# ============================================================
# PATHS
# ============================================================

ROOT = Path.home() / "Neuroverse"

FED_ROOT = ROOT / "Federated"

PATIENT_ID = "BraTS2021_00032"


# ============================================================
# PROJECT INFORMATION
# ============================================================

PROJECTS = {

    "E1": {

        "name":
            "E1 — 3D U-Net + FedAvg",

        "project":
            FED_ROOT / "FedAvg_BraTS",

        "visualizer":
            FED_ROOT
            / "FedAvg_BraTS"
            / "visualize_patient.py",

        "source_output":
            FED_ROOT
            / "FedAvg_BraTS"
            / "results"
            / "visualizations"
            / f"{PATIENT_ID}_E1.png",

    },


    "E2": {

        "name":
            "E2 — Swin UNETR + FedAvg",

        "project":
            FED_ROOT / "Swin_FedAvg",

        "visualizer":
            FED_ROOT
            / "Swin_FedAvg"
            / "visualize_patient.py",

        "source_output":
            FED_ROOT
            / "Swin_FedAvg"
            / "results"
            / "visualizations"
            / f"{PATIENT_ID}_E2.png",

    },


    "E3": {

        "name":
            "E3 — Swin UNETR + Adaptive FL",

        "project":
            FED_ROOT / "Swin_Adaptive_FL",

        "visualizer":
            FED_ROOT
            / "Swin_Adaptive_FL"
            / "visualize_patient.py",

        "source_output":
            FED_ROOT
            / "Swin_Adaptive_FL"
            / "results"
            / "visualizations"
            / f"{PATIENT_ID}_E3.png",

    },


    "E4": {

        "name":
            "E4 — Swin UNETR + Quantum Kernel",

        "project":
            FED_ROOT / "Quantum_Kernel_E3",

        "visualizer":
            FED_ROOT
            / "Quantum_Kernel_E3"
            / "visualize_patient.py",

        "source_output":
            FED_ROOT
            / "Quantum_Kernel_E3"
            / "results"
            / "visualizations"
            / f"{PATIENT_ID}_E4.png",

    },

}


# ============================================================
# FINAL OUTPUT
# ============================================================

OUTPUT_ROOT = (
    FED_ROOT
    / "Full_results"
    / "Visualization"
    / "FINAL_ONE_PATIENT"
)


OUTPUT_ROOT.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# FINAL FILES
# ============================================================

FINAL_FILES = {

    "E1":
        OUTPUT_ROOT
        / f"{PATIENT_ID}_E1_FINAL.png",

    "E2":
        OUTPUT_ROOT
        / f"{PATIENT_ID}_E2_FINAL.png",

    "E3":
        OUTPUT_ROOT
        / f"{PATIENT_ID}_E3_FINAL.png",

    "E4":
        OUTPUT_ROOT
        / f"{PATIENT_ID}_E4_FINAL.png",

}


# ============================================================
# CHECK FILE
# ============================================================

def check_file(
    path,
    description
):

    if not path.exists():

        raise FileNotFoundError(

            f"\n{description} not found:\n"
            f"{path}\n"

        )


# ============================================================
# RUN ORIGINAL VISUALIZER
# ============================================================

def run_original_visualizer(
    key
):

    info = PROJECTS[key]

    project = info[
        "project"
    ]

    visualizer = info[
        "visualizer"
    ]

    output = info[
        "source_output"
    ]


    print()
    print()
    print(
        "=" * 80
    )

    print(
        f"RUNNING {key}"
    )

    print(
        info["name"]
    )

    print(
        "=" * 80
    )


    check_file(
        project,
        f"{key} project directory"
    )

    check_file(
        visualizer,
        f"{key} visualization script"
    )


    # --------------------------------------------------------
    # Delete previous generated image.
    # --------------------------------------------------------

    if output.exists():

        print(
            "Removing previous visualization:"
        )

        print(
            output
        )

        output.unlink()


    # --------------------------------------------------------
    # Python executable.
    # --------------------------------------------------------

    python_executable = sys.executable


    print()
    print(
        "Python:"
    )

    print(
        python_executable
    )


    print()
    print(
        "Project:"
    )

    print(
        project
    )


    print()
    print(
        "Visualizer:"
    )

    print(
        visualizer
    )


    # --------------------------------------------------------
    # IMPORTANT:
    #
    # Run the ORIGINAL project script.
    #
    # This means its own task.py, model definition,
    # preprocessing and sliding-window implementation are used.
    # --------------------------------------------------------

    env = os.environ.copy()


    # Make project root available for imports.

    old_pythonpath = env.get(
        "PYTHONPATH",
        ""
    )


    if old_pythonpath:

        env[
            "PYTHONPATH"
        ] = (
            str(project)
            + os.pathsep
            + old_pythonpath
        )

    else:

        env[
            "PYTHONPATH"
        ] = str(project)


    print()
    print(
        "Running original project visualization..."
    )


    result = subprocess.run(

        [
            python_executable,
            str(visualizer)
        ],

        cwd=str(project),

        env=env,

        text=True

    )


    if result.returncode != 0:

        raise RuntimeError(

            f"\n{key} visualization failed.\n"
            f"Exit code: "
            f"{result.returncode}\n"

        )


    # --------------------------------------------------------
    # Verify output.
    # --------------------------------------------------------

    if not output.exists():

        # Search in the project's visualization folder
        # because some scripts may save under a different
        # hardcoded name.

        visualization_dir = (
            project
            / "results"
            / "visualizations"
        )


        candidates = list(
            visualization_dir.glob(
                f"{PATIENT_ID}*.png"
            )
        )


        if candidates:

            print()
            print(
                f"{key}: Found generated image:"
            )

            for candidate in candidates:

                print(
                    "   ",
                    candidate
                )


            # Prefer exact expected file.

            exact = [
                x for x in candidates
                if x.name ==
                f"{PATIENT_ID}_{key}.png"
            ]


            if exact:

                output = exact[0]

            else:

                output = candidates[0]


        else:

            raise RuntimeError(

                f"\n{key}: visualization script "
                f"finished but produced no PNG.\n"
                f"Expected:\n{output}"

            )


    print()
    print(
        f"{key}: Source visualization:"
    )

    print(
        output
    )


    return output


# ============================================================
# CREATE CORRECTED E4 VISUALIZER
# ============================================================

def create_corrected_e4_visualizer():

    e4 = PROJECTS["E4"]

    source = e4[
        "visualizer"
    ]

    project = e4[
        "project"
    ]


    check_file(
        source,
        "E4 visualization script"
    )


    print()
    print(
        "=" * 80
    )

    print(
        "CREATING CORRECTED E4 VISUALIZER"
    )

    print(
        "=" * 80
    )


    original_text = source.read_text(
        encoding="utf-8"
    )


    # --------------------------------------------------------
    # The existing E4 script was evidently copied from E3.
    #
    # Replace hardcoded E3 project references.
    # --------------------------------------------------------

    corrected = original_text


    # Project path references.

    corrected = corrected.replace(

        "Swin_Adaptive_FL",

        "Quantum_Kernel_E3"

    )


    # Experiment labels.

    corrected = corrected.replace(

        "E3 Prediction",

        "E4 Prediction"

    )


    corrected = corrected.replace(

        "E3 VISUALIZATION",

        "E4 VISUALIZATION"

    )


    corrected = corrected.replace(

        "E3 visualization",

        "E4 visualization"

    )


    corrected = corrected.replace(

        "E3 Visualization",

        "E4 Visualization"

    )


    corrected = corrected.replace(

        "E3 PREDICTION",

        "E4 PREDICTION"

    )


    corrected = corrected.replace(

        "ADAPTIVE SWIN UNETR PREDICTION",

        "QUANTUM KERNEL SWIN UNETR PREDICTION"

    )


    corrected = corrected.replace(

        "ADAPTIVE SWIN UNETR",

        "QUANTUM KERNEL SWIN UNETR"

    )


    # Hardcoded experiment identifier.

    corrected = corrected.replace(

        "_E3.png",

        "_E4.png"

    )


    corrected = corrected.replace(

        "'E3'",

        "'E4'"

    )


    corrected = corrected.replace(

        '"E3"',

        '"E4"'

    )


    # --------------------------------------------------------
    # Save temporary corrected script inside E4 project.
    # --------------------------------------------------------

    temporary_script = (
        project
        / "_temporary_corrected_e4_visualizer.py"
    )


    temporary_script.write_text(
        corrected,
        encoding="utf-8"
    )


    print()
    print(
        "Temporary corrected E4 script:"
    )

    print(
        temporary_script
    )


    return temporary_script


# ============================================================
# RUN E4
# ============================================================

def run_e4():

    info = PROJECTS["E4"]

    project = info[
        "project"
    ]

    expected = info[
        "source_output"
    ]


    print()
    print()
    print(
        "=" * 80
    )

    print(
        "RUNNING E4"
    )

    print(
        "E4 — Swin UNETR + Quantum Kernel"
    )

    print(
        "=" * 80
    )


    if expected.exists():

        expected.unlink()


    temporary_script = (
        create_corrected_e4_visualizer()
    )


    env = os.environ.copy()


    old_pythonpath = env.get(
        "PYTHONPATH",
        ""
    )


    if old_pythonpath:

        env[
            "PYTHONPATH"
        ] = (
            str(project)
            + os.pathsep
            + old_pythonpath
        )

    else:

        env[
            "PYTHONPATH"
        ] = str(project)


    try:

        result = subprocess.run(

            [
                sys.executable,
                str(temporary_script)
            ],

            cwd=str(project),

            env=env,

            text=True

        )


        if result.returncode != 0:

            raise RuntimeError(

                "\nE4 corrected visualization failed.\n"
                f"Exit code: {result.returncode}\n"

            )


    finally:

        if temporary_script.exists():

            temporary_script.unlink()


    # --------------------------------------------------------
    # Check generated files.
    # --------------------------------------------------------

    if expected.exists():

        return expected


    visualization_dir = (
        project
        / "results"
        / "visualizations"
    )


    candidates = list(
        visualization_dir.glob(
            f"{PATIENT_ID}*.png"
        )
    )


    # Ignore E3 if it exists.

    candidates = [

        x for x in candidates

        if "E3" not in x.name

    ]


    if not candidates:

        raise RuntimeError(

            "E4 finished but no E4 visualization "
            "was found."

        )


    # Prefer E4.

    e4_candidates = [

        x for x in candidates

        if "E4" in x.name

    ]


    if e4_candidates:

        return e4_candidates[0]


    return candidates[0]


# ============================================================
# LOAD IMAGE
# ============================================================

def load_png(
    path
):

    image = Image.open(
        path
    ).convert(
        "RGB"
    )


    print()
    print(
        "Reading:"
    )

    print(
        path
    )

    print(
        "Size:",
        image.size
    )


    return image


# ============================================================
# FIND THE FOUR-PANEL AREA
# ============================================================
#
# The original project visualizations already have the correct
# segmentation colors.
#
# We do NOT reinterpret their pixels.
#
# Instead, we crop the actual four-panel visualization area
# and place it into a standard canvas.
#
# This preserves the model's actual prediction visualization.
#
# ============================================================

def find_content_bbox(
    image
):

    arr = np.asarray(
        image
    )


    # Non-white pixels.

    nonwhite = (
        np.any(
            arr < 245,
            axis=2
        )
    )


    ys, xs = np.where(
        nonwhite
    )


    if len(xs) == 0:

        return (
            0,
            0,
            image.width,
            image.height
        )


    left = int(
        xs.min()
    )

    right = int(
        xs.max()
    )

    top = int(
        ys.min()
    )

    bottom = int(
        ys.max()
    )


    return (
        left,
        top,
        right + 1,
        bottom + 1
    )


# ============================================================
# STANDARDIZE ORIGINAL IMAGE
# ============================================================

def standardize_image(
    key,
    source_path,
    output_path
):

    image = load_png(
        source_path
    )


    # --------------------------------------------------------
    # IMPORTANT:
    #
    # We want the original project visualization to remain
    # visually intact.
    #
    # Only remove excessive empty borders.
    # --------------------------------------------------------

    bbox = find_content_bbox(
        image
    )


    print(
        f"{key}: content bbox =",
        bbox
    )


    cropped = image.crop(
        bbox
    )


    # --------------------------------------------------------
    # Keep a consistent aspect ratio.
    #
    # All source project images are approximately 4993x1485.
    #
    # We use a fixed 5:1 canvas rather than changing the
    # medical image geometry.
    # --------------------------------------------------------

    TARGET_WIDTH = 3000

    TARGET_HEIGHT = 900


    background = Image.new(

        "RGB",

        (
            TARGET_WIDTH,
            TARGET_HEIGHT
        ),

        "white"
    )


    # --------------------------------------------------------
    # Fit the complete original visualization into the same
    # canvas for every experiment.
    # --------------------------------------------------------

    margin_x = 35

    margin_y = 25


    max_width = (
        TARGET_WIDTH
        - 2 * margin_x
    )


    max_height = (
        TARGET_HEIGHT
        - 2 * margin_y
    )


    scale = min(

        max_width / cropped.width,

        max_height / cropped.height

    )


    new_width = max(
        1,
        int(
            cropped.width
            * scale
        )
    )


    new_height = max(
        1,
        int(
            cropped.height
            * scale
        )
    )


    resized = cropped.resize(

        (
            new_width,
            new_height
        ),

        Image.Resampling.LANCZOS

    )


    x = (
        TARGET_WIDTH
        - new_width
    ) // 2


    y = (
        TARGET_HEIGHT
        - new_height
    ) // 2


    background.paste(
        resized,
        (
            x,
            y
        )
    )


    background.save(

        output_path,

        "PNG",

        optimize=True

    )


    print()
    print(
        f"{key}: FINAL SAVED"
    )

    print(
        output_path
    )

    print(
        "Final size:",
        background.size
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print(
        "=" * 80
    )

    print(
        "NEUROVERSE FINAL ONE-PATIENT VISUALIZATIONS"
    )

    print(
        "=" * 80
    )


    print()
    print(
        "Patient:",
        PATIENT_ID
    )


    print()
    print(
        "IMPORTANT:"
    )

    print(
        "Using original experiment inference pipelines."
    )

    print(
        "No new model implementation."
    )

    print(
        "No training."
    )

    print(
        "No 188-patient evaluation."
    )


    # ========================================================
    # E1
    # ========================================================

    e1_source = (
        run_original_visualizer(
            "E1"
        )
    )


    # ========================================================
    # E2
    # ========================================================

    e2_source = (
        run_original_visualizer(
            "E2"
        )
    )


    # ========================================================
    # E3
    # ========================================================

    e3_source = (
        run_original_visualizer(
            "E3"
        )
    )


    # ========================================================
    # E4
    # ========================================================

    e4_source = run_e4()


    # ========================================================
    # STANDARDIZE ALL FOUR
    # ========================================================

    print()
    print()
    print(
        "=" * 80
    )

    print(
        "STANDARDIZING FINAL IMAGES"
    )

    print(
        "=" * 80
    )


    sources = {

        "E1": e1_source,

        "E2": e2_source,

        "E3": e3_source,

        "E4": e4_source,

    }


    for key in [

        "E1",
        "E2",
        "E3",
        "E4"

    ]:

        standardize_image(

            key,

            sources[key],

            FINAL_FILES[key]

        )


    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print()
    print()
    print(
        "=" * 80
    )

    print(
        "COMPLETE"
    )

    print(
        "=" * 80
    )


    print()

    for key in [

        "E1",
        "E2",
        "E3",
        "E4"

    ]:

        print(
            f"{key}:"
        )

        print(
            f"    {FINAL_FILES[key]}"
        )


    print()
    print(
        "Output directory:"
    )

    print(
        OUTPUT_ROOT
    )


    print()
    print(
        "=" * 80
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    main()
