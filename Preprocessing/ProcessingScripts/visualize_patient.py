import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt

patient = "BraTS2021_00000"

flair_path = f"{patient}/{patient}_flair.nii.gz"
seg_path = f"{patient}/{patient}_seg.nii.gz"

flair = nib.load(flair_path).get_fdata()
seg = nib.load(seg_path).get_fdata()

# Find slices containing the most tumor
tumor_per_slice = np.sum(seg > 0, axis=(0, 1))

slice_idx = np.argmax(tumor_per_slice)

print("Selected slice:", slice_idx)
print("Tumor voxels in slice:", tumor_per_slice[slice_idx])

# MRI slice
mri_slice = flair[:, :, slice_idx].T

# Segmentation slice
seg_slice = seg[:, :, slice_idx].T

plt.figure(figsize=(15, 5))

# FLAIR
plt.subplot(1, 3, 1)
plt.imshow(mri_slice, cmap="gray")
plt.title("FLAIR")
plt.axis("off")

# Segmentation
plt.subplot(1, 3, 2)
plt.imshow(seg_slice)
plt.title("Tumor Segmentation")
plt.axis("off")

# Overlay
plt.subplot(1, 3, 3)
plt.imshow(mri_slice, cmap="gray")
plt.imshow(
    np.ma.masked_where(seg_slice == 0, seg_slice),
    alpha=0.5
)
plt.title("FLAIR + Segmentation")
plt.axis("off")

plt.tight_layout()
plt.show()
