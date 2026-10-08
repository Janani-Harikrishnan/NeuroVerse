import nibabel as nib
import numpy as np

path = "BraTS2021_00000/BraTS2021_00000_seg.nii.gz"

img = nib.load(path)
seg = img.get_fdata()

print("Shape:", seg.shape)

# Physical voxel dimensions
voxel_spacing = img.header.get_zooms()[:3]

print("Voxel spacing (mm):", voxel_spacing)

# Volume of one voxel
voxel_volume = np.prod(voxel_spacing)

print("Volume of one voxel (mm³):", voxel_volume)

print("\nSegmentation statistics")
print("-" * 40)

for label in [0, 1, 2, 4]:

    count = np.sum(seg == label)

    volume_mm3 = count * voxel_volume
    volume_cm3 = volume_mm3 / 1000

    print(f"Label {label}")
    print(f"Voxel count : {count}")
    print(f"Volume      : {volume_mm3:.2f} mm³")
    print(f"Volume      : {volume_cm3:.2f} cm³")
    print()
