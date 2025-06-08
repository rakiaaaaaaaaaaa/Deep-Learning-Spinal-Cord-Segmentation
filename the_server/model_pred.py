import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import nibabel as nib
import torch.nn as nn
from tqdm import tqdm


class MRIDataset(Dataset):
    def __init__(self, mri_dir):
        self.mri_dir = mri_dir
        self.subject_ids = sorted([f.replace('_mri.npy', '') for f in os.listdir(mri_dir) if f.endswith('_mri.npy')])

    def __len__(self):
        return len(self.subject_ids)

    def __getitem__(self, idx):
        subject_id = self.subject_ids[idx]
        mri_path = os.path.join(self.mri_dir, f'{subject_id}_mri.npy')
        mri = np.load(mri_path).astype(np.float32)
        mri = np.expand_dims(mri, axis=0)  # [1, D, H, W]
        return torch.tensor(mri), subject_id


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


class UNet3D(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, init_features=32):
        super().__init__()
        features = init_features
        self.encoder1 = DoubleConv(in_channels, features)
        self.pool1 = nn.MaxPool3d(2)
        self.encoder2 = DoubleConv(features, features * 2)
        self.pool2 = nn.MaxPool3d(2)
        self.encoder3 = DoubleConv(features * 2, features * 4)
        self.pool3 = nn.MaxPool3d(2)
        self.bottleneck = DoubleConv(features * 4, features * 8)
        self.up3 = nn.ConvTranspose3d(features * 8, features * 4, kernel_size=2, stride=2)
        self.decoder3 = DoubleConv(features * 8, features * 4)
        self.up2 = nn.ConvTranspose3d(features * 4, features * 2, kernel_size=2, stride=2)
        self.decoder2 = DoubleConv(features * 4, features * 2)
        self.up1 = nn.ConvTranspose3d(features * 2, features, kernel_size=2, stride=2)
        self.decoder1 = DoubleConv(features * 2, features)
        self.final = nn.Conv3d(features, out_channels, kernel_size=1)

    def forward(self, x):
        e1 = self.encoder1(x)
        e2 = self.encoder2(self.pool1(e1))
        e3 = self.encoder3(self.pool2(e2))
        b = self.bottleneck(self.pool3(e3))
        d3 = self.up3(b)
        d3 = torch.cat((d3, e3), dim=1)
        d3 = self.decoder3(d3)
        d2 = self.up2(d3)
        d2 = torch.cat((d2, e2), dim=1)
        d2 = self.decoder2(d2)
        d1 = self.up1(d2)
        d1 = torch.cat((d1, e1), dim=1)
        d1 = self.decoder1(d1)
        return torch.sigmoid(self.final(d1))


def load_model(model, checkpoint_path, device):
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()
    return model


def plot_mri_and_prediction(mri, pred, subject_id, save_dir, save_only=True):
    mri = mri.squeeze().numpy()
    pred = pred.squeeze().numpy()
    mid_slice = mri.shape[0] // 2

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(mri[mid_slice], cmap='gray')
    plt.title(f"{subject_id} - MRI")

    plt.subplot(1, 2, 2)
    plt.imshow(pred[mid_slice] > 0.5, cmap='gray')
    plt.title(f"{subject_id} - Predicted Mask")

    plt.tight_layout()
    plot_path = os.path.join(save_dir, f"{subject_id}_plot.png")
    plt.savefig(plot_path)
    if not save_only:
        plt.show()
    plt.close()


def save_prediction_as_nifti(pred, save_path):
    pred_np = pred.squeeze().cpu().numpy()
    pred_binary = (pred_np > 0.5).astype(np.uint8)
    nifti_img = nib.Nifti1Image(pred_binary, affine=np.eye(4))
    nib.save(nifti_img, save_path)


if __name__ == '__main__':
    mri_dir = r"C:\Users\olaol\OneDrive - Ministere de l'Enseignement Superieur et de la Recherche Scientifique\Bureau\run SCT\model_finale\Mri_to_Npy\npy_mri_test"
    checkpoint_path = r"C:\Users\olaol\OneDrive - Ministere de l'Enseignement Superieur et de la Recherche Scientifique\Bureau\run SCT\model_finale\best_model.pth"
    output_dir = r"C:\Users\olaol\OneDrive - Ministere de l'Enseignement Superieur et de la Recherche Scientifique\Bureau\run SCT\model_finale\mask_predicted"
    os.makedirs(output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet3D()
    model = load_model(model, checkpoint_path, device)

    dataset = MRIDataset(mri_dir)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    for mri, subject_id in tqdm(dataloader, desc="Predicting"):
        mri = mri.to(device)  # Shape: [1, 1, D, H, W]
        subject_id = subject_id[0]  # Get the string from list
        with torch.no_grad():
            pred = model(mri)

        save_path = os.path.join(output_dir, f'{subject_id}_prediction.nii.gz')
        save_prediction_as_nifti(pred, save_path)
        plot_mri_and_prediction(mri.cpu(), pred.cpu(), subject_id, output_dir, save_only=True)

        print(f" Saved mask and plot for {subject_id}")


# --- SERVER FUNCTION FOR SINGLE MRI NPY PREDICTION ---

def predict_single_npy(npy_path, model, device, output_dir):
    """
    Given a single npy MRI file path, run prediction and save mask + plot.
    Returns paths of saved nifti and plot.
    """
    os.makedirs(output_dir, exist_ok=True)
    subject_id = os.path.basename(npy_path).replace('_mri.npy', '')
    mri = np.load(npy_path).astype(np.float32)
    mri_tensor = torch.tensor(mri).unsqueeze(0).unsqueeze(0).to(device)  # [1,1,D,H,W]

    model.eval()
    with torch.no_grad():
        pred = model(mri_tensor)

    nifti_path = os.path.join(output_dir, f'{subject_id}_prediction.nii.gz')
    plot_path = os.path.join(output_dir, f'{subject_id}_plot.png')

    save_prediction_as_nifti(pred, nifti_path)
    plot_mri_and_prediction(mri_tensor.cpu(), pred.cpu(), subject_id, output_dir, save_only=True)

    return nifti_path, plot_path
