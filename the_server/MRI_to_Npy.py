import os
import nibabel as nib
import numpy as np
import random

#  Define base and output paths
base_path = r"C:\Users\olaol\OneDrive - Ministere de l'Enseignement Superieur et de la Recherche Scientifique\Bureau\run SCT\model_finale\Mri_to_Npy"
test_dir = os.path.join(base_path, 'npy_mri_test')
os.makedirs(test_dir, exist_ok=True)

#  Collect only MRI files
mri_files = []
for root, _, files in os.walk(base_path):
    for file in files:
        if file.endswith('_mri.nii.gz'):
            subj_id = file.replace('_mri.nii.gz', '')
            mri_path = os.path.join(root, file)
            mri_files.append((subj_id, mri_path))

#  Shuffle and split
random.shuffle(mri_files)
split_idx = int(len(mri_files) * 0.7)
test_files = mri_files[split_idx:]

#  Compute minimum shape across all
def get_min_shape(pairs):
    min_shape = None
    for _, mri_path in pairs:
        data = nib.load(mri_path).get_fdata()
        if min_shape is None:
            min_shape = data.shape
        else:
            min_shape = tuple(min(a, b) for a, b in zip(min_shape, data.shape))
    return min_shape

min_shape = get_min_shape(mri_files)

#  Center crop to uniform shape
def crop_center(volume, target_shape):
    start = [(s - t) // 2 for s, t in zip(volume.shape, target_shape)]
    slices = tuple(slice(s, s + t) for s, t in zip(start, target_shape))
    return volume[slices]

#  Save MRI to .npy
def save_mri(subj_id, mri_path, out_dir):
    mri = nib.load(mri_path).get_fdata()
    mri = crop_center(mri, min_shape)
    np.save(os.path.join(out_dir, f'{subj_id}_mri.npy'), mri.astype(np.float32))

#  Save only test samples
print(" Saving test MRI samples...")
for subj_id, mri_path in test_files:
    save_mri(subj_id, mri_path, test_dir)

print(" Done!")
print(f"Saved {len(test_files)} test MRI samples to: {test_dir}")


# --- SERVER FUNCTION FOR SINGLE FILE PROCESSING ---

def process_single_mri(mri_path, output_dir):
    """
    Process a single MRI file: crop and save as .npy using precomputed min_shape.
    Returns the path to the saved .npy file.
    """
    os.makedirs(output_dir, exist_ok=True)
    subj_id = os.path.basename(mri_path).replace('_mri.nii.gz', '')
    mri = nib.load(mri_path).get_fdata()
    mri_cropped = crop_center(mri, min_shape)
    npy_path = os.path.join(output_dir, f'{subj_id}_mri.npy')
    np.save(npy_path, mri_cropped.astype(np.float32))
    return npy_path
