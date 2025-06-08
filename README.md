🧠 Spinal Cord Segmentation & CSA Prediction using Deep Learning and Demographic Modeling
🎓 End-of-Year Project – Higher Institute of Computer Sciences and Multimedia, University of Gabès
Contributors: Tayma Hrizi & Rakia Souei
Academic Year: 2024/2025
Supervisor: Mrs. Tayssir Bousshila

📌 Project Overview
This project presents a complete pipeline for automated segmentation of the spinal cord in 3D MRI using deep learning, followed by cross-sectional area (CSA) estimation and demographic-based normalization. The aim is to streamline and standardize the analysis of spinal cord morphology, particularly useful for studying neurodegenerative diseases such as Neuromyelitis Optica (NMO) and Multiple Sclerosis (MS).

We propose an integrated solution based on U-Net convolutional neural networks for segmentation and multiple linear regression for adjusting CSA values based on demographic variables such as age, sex, height, weight, and scanner type.

🏗️ Key Components
1. Dataset
Spine Generic dataset with 208 T2-weighted 3D MRI scans from healthy individuals.

Images acquired across 43 imaging centers with consistent protocols and isotropic resolution (0.8 mm).

Segmentation masks generated using Spinal Cord Toolbox (SCT) tools, used as ground truth.

2. Spinal Cord Segmentation
Deep learning model based on the U-Net architecture.

Training performed on an 80/10/10 split of the dataset.

Loss function: Dice Loss

Achieved performance:

Dice coefficient: 0.8553

Intersection over Union (IoU): 0.8742

Accuracy: 0.9992

Loss: 0.0747
![11](https://github.com/user-attachments/assets/fb483d13-0761-4c5b-8fdf-e18d3bde7bf7)
![22](https://github.com/user-attachments/assets/a8d61563-206e-4171-a3b4-8702c76e81e0)

3. CSA Measurement
CSA computed at the C2–C3 vertebral level using SCT commands:

sct_label_vertebrae

sct_process_segmentation

Values reflect the anatomical integrity of the spinal cord.

4. Demographic Normalization
Linear regression formula adapted from Boushila et al. (2024) used to normalize CSA values:


5. Mobile Application (Flutter)
Login, Register, and Profile interfaces.

Users can:

Input demographic data.

Upload MRI scans.

Receive automated CSA analysis results via backend integration.


📊 Insights
Deep learning significantly improves segmentation accuracy compared to classical methods.

Demographic variables, particularly sex and height, strongly influence CSA variability.

The pipeline provides a reliable, reproducible foundation for personalized neuroimaging analysis.

⚠️ Limitations
Dataset limited to healthy individuals; not yet validated on pathological cases.

Use of linear regression might not capture complex relationships in CSA variation.

Limited diversity in MRI scanner models and geographic sampling.

🔮 Future Work
Extend analysis to clinical populations (e.g., MS or spinal cord injury).

Use non-linear models (e.g., Random Forests, Neural Networks) for CSA prediction.

Increase dataset diversity to improve generalization.

Integrate with clinical decision support systems for personalized medicine.

📚 References
A full list of references is provided in the PDF report and includes key works by De Leener, Cohen-Adad, Boushila, and others in the field of spinal cord imaging.

📦 Technologies Used
Python,U-NET, PyTorch – Deep learning

SCT (Spinal Cord Toolbox) – Medical image preprocessing

Flutter – Mobile application

Jupyter, Pandas, Matplotlib – Analysis and visualization
![1](https://github.com/user-attachments/assets/e9960778-45ad-47ab-90f7-508cf2179d29)
![2](https://github.com/user-attachments/assets/50599563-3160-464b-b84b-4f8367e08486)
![3](https://github.com/user-attachments/assets/81befe3e-17b8-417f-b1fa-90e6f2b88c5f)
![4](https://github.com/user-attachments/assets/e469730b-09d6-4d39-9183-a8bc325752e9)
![5](https://github.com/user-attachments/assets/1ae7dc7b-a8d5-45a3-9b5c-6ac117f0e34c)
![1237](https://github.com/user-attachments/assets/811e6c10-31e1-4963-badb-0b352e8e6719)
