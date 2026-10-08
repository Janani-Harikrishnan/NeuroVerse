#!/bin/bash

# ============================================================
# NEUROVERSE
# VISUALIZE ONE PATIENT USING ALL FOUR TRAINED MODELS
# ============================================================
#
# Patient:
#     BraTS2021_00032
#
# Models:
#     E1 - 3D U-Net + FedAvg
#     E2 - Swin UNETR + FedAvg
#     E3 - Swin UNETR + Adaptive FL
#     E4 - Swin UNETR + Quantum Kernel
#
# IMPORTANT:
#     This runs inference on ONLY ONE PATIENT.
#
# It does NOT:
#     - train anything
#     - evaluate 188 patients
#     - calculate GlobalTest metrics
#     - modify the models
#
# It simply loads each trained global model and generates
# its segmentation visualization.
#
# ============================================================


set -e


# ============================================================
# CONFIG
# ============================================================

ROOT="$HOME/Neuroverse/Federated"

PATIENT_ID="BraTS2021_00032"

PATIENT_INDEX=0


# ============================================================
# PATIENT FILE
# ============================================================

PATIENT_FILE="$HOME/Neuroverse/IID_Clients/GlobalTest/test/${PATIENT_ID}.npz"


echo ""
echo "============================================================"
echo "NEUROVERSE - ONE PATIENT VISUALIZATION"
echo "============================================================"
echo ""

echo "Patient:"
echo "    $PATIENT_ID"

echo ""

echo "Patient file:"
echo "    $PATIENT_FILE"

echo ""


if [ ! -f "$PATIENT_FILE" ]; then

    echo "ERROR:"
    echo "Patient file does not exist:"
    echo "$PATIENT_FILE"

    exit 1

fi


echo "Patient file found."
echo ""


# ============================================================
# E1
# ============================================================

echo "============================================================"
echo "E1 - 3D U-NET + FEDAVG"
echo "============================================================"
echo ""

cd "$ROOT/FedAvg_BraTS"

echo "Working directory:"
pwd

echo ""

echo "Running:"
echo "python visualize_patient.py $PATIENT_INDEX"

echo ""

python visualize_patient.py "$PATIENT_INDEX"

echo ""

echo "E1 visualization completed."

echo ""


# ============================================================
# E2
# ============================================================

echo "============================================================"
echo "E2 - SWIN UNETR + FEDAVG"
echo "============================================================"
echo ""

cd "$ROOT/Swin_FedAvg"

echo "Working directory:"
pwd

echo ""

echo "Running:"
echo "python visualize_patient.py $PATIENT_INDEX"

echo ""

python visualize_patient.py "$PATIENT_INDEX"

echo ""

echo "E2 visualization completed."

echo ""


# ============================================================
# E3
# ============================================================

echo "============================================================"
echo "E3 - SWIN UNETR + ADAPTIVE FL"
echo "============================================================"
echo ""

cd "$ROOT/Swin_Adaptive_FL"

echo "Working directory:"
pwd

echo ""

echo "Running:"
echo "python visualize_patient.py $PATIENT_INDEX"

echo ""

python visualize_patient.py "$PATIENT_INDEX"

echo ""

echo "E3 visualization completed."

echo ""


# ============================================================
# E4
# ============================================================

echo "============================================================"
echo "E4 - SWIN UNETR + QUANTUM KERNEL"
echo "============================================================"
echo ""

cd "$ROOT/Quantum_Kernel_E3"

echo "Working directory:"
pwd

echo ""

echo "Running:"
echo "python visualize_patient.py $PATIENT_INDEX"

echo ""

python visualize_patient.py "$PATIENT_INDEX"

echo ""

echo "E4 visualization completed."

echo ""


# ============================================================
# CREATE FINAL VISUALIZATION DIRECTORY
# ============================================================

FINAL_ROOT="$ROOT/Full_results/Visualization/One_Patient"

mkdir -p "$FINAL_ROOT"

mkdir -p "$FINAL_ROOT/E1_3D_UNet_FedAvg"
mkdir -p "$FINAL_ROOT/E2_Swin_FedAvg"
mkdir -p "$FINAL_ROOT/E3_Swin_Adaptive"
mkdir -p "$FINAL_ROOT/E4_Swin_QuantumKernel"


# ============================================================
# COPY E1 RESULT
# ============================================================

echo "============================================================"
echo "COLLECTING E1 RESULT"
echo "============================================================"
echo ""

E1_SOURCE="$ROOT/FedAvg_BraTS/results/visualizations/${PATIENT_ID}_E1.png"

if [ -f "$E1_SOURCE" ]; then

    cp "$E1_SOURCE" \
       "$FINAL_ROOT/E1_3D_UNet_FedAvg/"

    echo "Copied:"
    echo "$E1_SOURCE"

else

    echo "WARNING:"
    echo "E1 visualization not found at:"
    echo "$E1_SOURCE"

fi

echo ""


# ============================================================
# COPY E2 RESULT
# ============================================================

echo "============================================================"
echo "COLLECTING E2 RESULT"
echo "============================================================"
echo ""

E2_SOURCE="$ROOT/Swin_FedAvg/results/visualizations/${PATIENT_ID}_E2.png"

if [ -f "$E2_SOURCE" ]; then

    cp "$E2_SOURCE" \
       "$FINAL_ROOT/E2_Swin_FedAvg/"

    echo "Copied:"
    echo "$E2_SOURCE"

else

    echo "WARNING:"
    echo "E2 visualization not found at:"
    echo "$E2_SOURCE"

fi

echo ""


# ============================================================
# COPY E3 RESULT
# ============================================================

echo "============================================================"
echo "COLLECTING E3 RESULT"
echo "============================================================"
echo ""

E3_SOURCE="$ROOT/Swin_Adaptive_FL/results/visualizations/${PATIENT_ID}_E3.png"

if [ -f "$E3_SOURCE" ]; then

    cp "$E3_SOURCE" \
       "$FINAL_ROOT/E3_Swin_Adaptive/"

    echo "Copied:"
    echo "$E3_SOURCE"

else

    echo "WARNING:"
    echo "E3 visualization not found at:"
    echo "$E3_SOURCE"

fi

echo ""


# ============================================================
# COPY E4 RESULT
# ============================================================

echo "============================================================"
echo "COLLECTING E4 RESULT"
echo "============================================================"
echo ""

E4_SOURCE="$ROOT/Quantum_Kernel_E3/results/visualizations/${PATIENT_ID}_E4.png"

if [ -f "$E4_SOURCE" ]; then

    cp "$E4_SOURCE" \
       "$FINAL_ROOT/E4_Swin_QuantumKernel/"

    echo "Copied:"
    echo "$E4_SOURCE"

else

    echo "WARNING:"
    echo "E4 visualization not found at:"
    echo "$E4_SOURCE"

fi

echo ""


# ============================================================
# FINAL SUMMARY
# ============================================================

echo ""
echo "============================================================"
echo "VISUALIZATION COMPLETE"
echo "============================================================"
echo ""

echo "Patient:"
echo "    $PATIENT_ID"

echo ""

echo "Final visualization directory:"
echo "    $FINAL_ROOT"

echo ""

echo "Files currently available:"
echo ""

find "$FINAL_ROOT" \
    -type f \
    -name "*.png" \
    -print | sort

echo ""

echo "============================================================"
echo "DONE"
echo "============================================================"
echo ""
