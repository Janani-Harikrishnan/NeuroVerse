import nibabel as nib
import numpy as np
import os

patient_dir = "BraTS2021_00000"
patient_id = "BraTS2021_00000"

modalities = ["flair", "t1", "t1ce", "t2", "seg"]

for modality in modalities:

    filename = f"{patient_id}_{modality}.nii.gz"
    path = os.path.join(patient_dir, filename)

    image = nib.load(path)
    data = image.get_fdata()

    print("\n" + "=" * 50)
    print("Modality:", modality)
    print("=" * 50)

    print("File:", filename)
    print("Shape:", data.shape)
    print("Data type:", data.dtype)
    print("Minimum:", data.min())
    print("Maximum:", data.max())
    print("Mean:", data.mean())
    print("Standard deviation:", data.std())

    # Number of non-zero voxels
    nonzero = np.count_nonzero(data)
    total = data.size

    print("Non-zero voxels:", nonzero)
    print("Total voxels:", total)
    print("Non-zero percentage:", (nonzero / total) * 100)

    if modality == "seg":
        print("Unique segmentation labels:", np.unique(data))
