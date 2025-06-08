from flask import Flask, request, jsonify
import os, traceback, shutil, subprocess
import nibabel as nib
import numpy as np
import pandas as pd
import csv
from model_pred import predict_single_npy, UNet3D, load_model
import base64
import torch

from MRI_to_Npy import process_single_mri
from csa_calculating import compute_csa_from_nii

app = Flask(__name__)

BASE_PATH = r"C:\Users\olaol\OneDrive - Ministere de l'Enseignement Superieur et de la Recherche Scientifique\Bureau\run SCT\model_finale"
NPY_TEST_DIR = os.path.join(BASE_PATH, "Mri_to_Npy", "npy_mri_test")
PREDICTED_FOLDER = os.path.join(BASE_PATH, "mask_predicted")
SCT_LABEL_DIR = os.path.join(BASE_PATH, "sct_labels")
SCT_NII_DIR = os.path.join(BASE_PATH, "nii_transformed")
DEBUG_DIR = os.path.join(BASE_PATH, "debug")

checkpoint_path = r"C:\Users\olaol\OneDrive - Ministere de l'Enseignement Superieur et de la Recherche Scientifique\Bureau\run SCT\model_finale\best_model.pth"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
checkpoint_path = os.path.join(BASE_PATH, "best_model.pth")
model = load_model(UNet3D(), checkpoint_path, device)

for d in [NPY_TEST_DIR, PREDICTED_FOLDER, SCT_LABEL_DIR, SCT_NII_DIR, DEBUG_DIR]:
    os.makedirs(d, exist_ok=True)

@app.route("/process_and_predict", methods=["POST"])
def process_and_predict():
    try:
        # Validate uploaded file
        if 'mri' not in request.files:
            return jsonify({"error": "No MRI file uploaded. Please provide a .nii.gz file."}), 400
        mri_file = request.files['mri']
        if not mri_file.filename.endswith('.nii.gz'):
            return jsonify({"error": "Invalid file type. Only .nii.gz files are supported."}), 400

        # Validate form data
        try:
            sex = float(request.form.get("sex"))
            weight = float(request.form.get("weight"))
            assert sex in (0, 1)
        except:
            return jsonify({
                "error": "Invalid form data.",
                "details": "Expected 'sex' as 0 or 1 and 'weight' as a number.",
                "received": {
                    "sex": request.form.get("sex"),
                    "weight": request.form.get("weight")
                }
            }), 400

        uploaded_path = os.path.join(DEBUG_DIR, mri_file.filename)
        mri_file.save(uploaded_path)

        # Convert MRI to NPY
        try:
            npy_path = process_single_mri(uploaded_path, NPY_TEST_DIR)
            _, png_path = predict_single_npy(npy_path, model, device, PREDICTED_FOLDER)
        except Exception as e:
            return jsonify({"error": "Failed during MRI preprocessing.", "details": str(e)}), 500

        subject_id = os.path.basename(npy_path).replace('_mri.npy', '')
        final_anat_path = os.path.join(SCT_NII_DIR, f"{subject_id}_T2w.nii.gz")
        final_seg_path = os.path.join(SCT_NII_DIR, f"{subject_id}_T2w_sc_seg.nii.gz")
        shutil.copy2(uploaded_path, final_anat_path)

        # Run SCT segmentation
        result = subprocess.run([
            "sct_deepseg_sc", "-i", final_anat_path, "-c", "t2", "-o", final_seg_path
        ], capture_output=True, text=True)

        print("SCT segmentation stdout:\n", result.stdout)
        print("SCT segmentation stderr:\n", result.stderr)

        if result.returncode != 0:
            return jsonify({
                "error": "Spinal cord segmentation failed.",
                "stdout": result.stdout,
                "stderr": result.stderr
            }), 500

        if not os.path.exists(final_seg_path):
            return jsonify({"error": "Segmentation file not found.", "path": final_seg_path}), 500

        seg_img = nib.load(final_seg_path)
        seg_data = seg_img.get_fdata()
        if np.count_nonzero(seg_data) == 0:
            return jsonify({
                "error": "Segmentation mask is empty.",
                "details": "sct_deepseg_sc created a file, but no spinal cord was detected."
            }), 500

        # Compute CSA
        try:
            csa_results = compute_csa_from_nii(
                subject_id=subject_id,
                anat_nii_path=final_anat_path,
                mask_nii_path=final_seg_path,
                label_dir=SCT_LABEL_DIR,
                sex=sex,
                manufacturer=0
            )
        except Exception as e:
            return jsonify({"error": "CSA computation failed.", "details": str(e)}), 500

        # Read CSA TSV for additional info
        csa_tsv_path = os.path.join(SCT_LABEL_DIR, f"{subject_id}_csa.tsv")
        mean_area_avg = None
        timestamps = []
        mean_area_values = []
        if os.path.exists(csa_tsv_path):
            try:
                df = pd.read_csv(csa_tsv_path, sep=',', header=0, quotechar='"', quoting=csv.QUOTE_MINIMAL)
                mean_area_avg = round(df['MEAN(area)'].mean(), 2)
                timestamps = df['Timestamp'].tolist()
                mean_area_values = df['MEAN(area)'].tolist()
            except Exception as e:
                print(f"Failed reading CSA TSV: {e}")

        if not csa_measured_values:
            return jsonify({
                "error": "No valid CSA values in accepted range (69–79 mm²)."
            }), 500

        mean_measured = round(sum(csa_measured_values) / len(csa_measured_values), 2)
        mean_normalized = round(sum(csa_normalized_values) / len(csa_normalized_values), 2)
        # Base64-encode the prediction PNG
        encoded_png = None
        if os.path.exists(png_path):
            with open(png_path, "rb") as f:
                encoded_png = base64.b64encode(f.read()).decode('utf-8')
        return jsonify({
            "subject_id": subject_id,
            "mean_measured_csa": mean_measured,
            "mean_normalized_csa": mean_normalized,
            "count": len(csa_measured_values),
            "units": "mm^2",
            "status": "success",
            "timestamps_from_tsv": timestamps,
            
            "prediction_plot_png_base64": encoded_png  #  Include PNG

       
        }), 200

    except Exception as e:
        traceback.print_exc()
        return jsonify({
            "error": "Unexpected server error.",
            "exception": str(e),
            "trace": traceback.format_exc()
        }), 500

if __name__ == "__main__":
    print("🚀 Starting Flask server on port 5000...")
    app.run(host="0.0.0.0", port=5000, debug=True)
