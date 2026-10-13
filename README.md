# Neuroverse — BraTS 2021 Dataset

## 1. Project Overview

Neuroverse is a federated learning project for 3D brain tumor segmentation using the BraTS 2021 MRI dataset. It explores 3D U-Net, Swin UNETR, adaptive federated aggregation, and quantum-kernel-guided aggregation.

## 2. Dataset Source

**Official BraTS 2021 website:**  
https://www.med.upenn.edu/cbica/brats2021/

**TCIA dataset page:**  
https://www.cancerimagingarchive.net/analysis-result/rsna-asnr-miccai-brats-2021/

The dataset is subject to registration and access requirements. Follow the instructions on the official BraTS website to request the segmentation training data.

## 3. MRI Modalities

The dataset includes four MRI modalities:

- **T1:** T1-weighted MRI
- **T1ce:** Contrast-enhanced T1-weighted MRI
- **T2:** T2-weighted MRI
- **FLAIR:** Fluid-attenuated inversion recovery MRI

Segmentation masks provide labels for tumor regions.

## 4. File Format

The MRI scans and segmentation masks are commonly supplied as compressed NIfTI files (`.nii.gz`).

Example subject directory:

```text
BraTS2021_00000/
├── BraTS2021_00000_flair.nii.gz
├── BraTS2021_00000_t1.nii.gz
├── BraTS2021_00000_t1ce.nii.gz
├── BraTS2021_00000_t2.nii.gz
└── BraTS2021_00000_seg.nii.gz
```

The exact directory and filename conventions may vary depending on the download source.

## 5. Dataset Preparation

1. Obtain the dataset through an authorized access route.
2. Verify the MRI modalities and segmentation masks.
3. Apply the preprocessing required by the model.
4. Prepare training, validation, and evaluation splits.
5. Partition training data among the simulated federated clients.
6. Record the split configuration and random seed for reproducibility.

Do not assume that test labels are available or that every distribution contains identical files.

## 6. Neuroverse Experiments

The project compares the following configurations:

- **E1:** Standalone 3D U-Net
- **E2:** Federated 3D U-Net with FedAvg
- **E3:** Swin UNETR with FedAvg
- **E4:** Swin UNETR with adaptive aggregation
- **E5:** Swin UNETR with quantum-kernel-guided adaptive aggregation

The quantum kernel is evaluated using classical simulation. The experiment should not be described as demonstrating quantum advantage.

## 7. Data Availability

The BraTS 2021 dataset can be requested through the official access process:

https://www.med.upenn.edu/cbica/brats2021/

Access is subject to registration requirements and the dataset's terms of use. The availability of project code, processed data, client partitions, and model weights depends on what the project authors have actually shared.

## 8. Dataset Citation

Please cite the following publications as requested by the official BraTS 2021 website:

1. Baid, U. et al. (2021). *The RSNA-ASNR-MICCAI BraTS 2021 Benchmark on Brain Tumor Segmentation and Radiogenomic Classification*. arXiv:2107.02314.
2. Menze, B. H. et al. (2015). *The Multimodal Brain Tumor Image Segmentation Benchmark (BRATS).* IEEE Transactions on Medical Imaging, 34(10), 1993–2024. https://doi.org/10.1109/TMI.2014.2377694
3. Bakas, S. et al. (2017). *Advancing The Cancer Genome Atlas glioma MRI collections with expert segmentation labels and radiomic features.* Scientific Data, 4, 170117. https://doi.org/10.1038/sdata.2017.117

## 9. Responsible Data Use

- Follow the official dataset access conditions and terms of use.
- Cite the dataset in academic reports and publications.
- Do not upload restricted MRI scans or segmentation labels to a public repository.
- Do not claim that raw-data locality alone guarantees privacy.
- Report only the dataset sizes and evaluation splits actually used in the experiments.
