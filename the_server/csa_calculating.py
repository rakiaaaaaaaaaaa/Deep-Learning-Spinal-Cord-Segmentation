import os
import subprocess
import pandas as pd
import random

def ensure_directories(*dirs):
    for d in dirs:
        os.makedirs(d, exist_ok=True)

def run_sct_label_vertebrae(anat_nii, seg_nii, output_dir):
    cmd = [
        "sct_label_vertebrae",
        "-i", anat_nii,
        "-s", seg_nii,
        "-c", "t2",
        "-ofolder", output_dir
    ]
    subprocess.run(cmd, check=True)

def run_sct_process_segmentation(seg_nii, vertfile, output_tsv):
    cmd = [
        "sct_process_segmentation",
        "-i", seg_nii,
        "-vert", "2:3",
        "-vertfile", vertfile,
        "-o", output_tsv
    ]
    subprocess.run(cmd, check=True)

def normalize_csa(csa_meas, sex, manufacturer):
    return csa_meas + 4.3317 * (0.5316 - sex) - 1.5694 * (0.5306 - manufacturer)

def compute_csa_from_nii(subject_id, anat_nii_path, mask_nii_path, label_dir, sex, manufacturer):
    ensure_directories(label_dir)

    run_sct_label_vertebrae(anat_nii_path, mask_nii_path, label_dir)
    labeled_mask_path = os.path.join(label_dir, f"{subject_id}_T2w_sc_seg_labeled.nii.gz")

    if not os.path.exists(labeled_mask_path):
        raise FileNotFoundError(f"Labeled mask file not found: {labeled_mask_path}")

    csa_tsv_path = os.path.join(label_dir, f"{subject_id}_csa.tsv")
    run_sct_process_segmentation(mask_nii_path, labeled_mask_path, csa_tsv_path)

    if not os.path.exists(csa_tsv_path):
        raise FileNotFoundError(f"CSA TSV not found: {csa_tsv_path}")

    try:
        df = pd.read_csv(csa_tsv_path, sep=',', header=0, quotechar='"')
    except Exception as e:
        raise RuntimeError(f"Failed to read CSA TSV: {e}")

    if 'Timestamp' not in df.columns or 'MEAN(area)' not in df.columns:
        raise ValueError("Expected columns 'Timestamp' and 'MEAN(area)' not found in CSA TSV")

    csa_values = []
    normalized_values = []

    for _, row in df.iterrows():
        try:
            timestamp = row['Timestamp']
            csa_meas = float(row['MEAN(area)'])

          
            # Normalize
            csa_norm = normalize_csa(csa_meas, sex, manufacturer)


            # Always store values
            csa_values.append((timestamp, csa_meas))
            normalized_values.append((timestamp, csa_norm))

        except Exception:
            continue

    
    return {
        "csa_values": csa_values,
        "normalized_csa": normalized_values
    }
