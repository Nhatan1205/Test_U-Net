import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from unet import UNet  # Đảm bảo bạn có file unet.py

class BloodCellDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_files = sorted([f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png'))])
        self.mask_files = sorted([f for f in os.listdir(mask_dir) if f.endswith(('.png'))])
        self.transform = transform
        print(f"Number of image files: {len(self.image_files)}")
        print(f"Number of mask files: {len(self.mask_files)}")

    def __len__(self):
        return min(len(self.image_files), len(self.mask_files))

    def __getitem__(self, idx):
        image_path = os.path.join(self.image_dir, self.image_files[idx])
        mask_path = os.path.join(self.mask_dir, self.mask_files[idx])

        image = Image.open(image_path).convert('RGB')
        mask = Image.open(mask_path).convert('L') # 'L' cho ảnh grayscale

        if self.transform:
            transformed = self.transform(image=np.array(image), mask=np.array(mask))
            image = transformed['image']
            mask = transformed['mask']

        return image, mask

# Kiểm tra xem CUDA có sẵn không
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Sử dụng thiết bị: {device}")

# Đường dẫn đến thư mục chứa ảnh huấn luyện (JPEGImages)
train_image_dir = r'D:\NCKH\BCCD_Dataset\BCCD\JPEGImages'
train_mask_dir = r'D:\NCKH\BCCD_Dataset\masks\train'
val_image_dir = r'D:\NCKH\BCCD_Dataset\BCCD\JPEGImages'
val_mask_dir = r'D:\NCKH\BCCD_Dataset\masks\val'

# Định nghĩa các biến đổi dữ liệu
transform = A.Compose([
    A.Resize(256, 256),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
], additional_targets={'mask': 'mask'})

# Tạo dataset và dataloader
train_dataset = BloodCellDataset(train_image_dir, train_mask_dir, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
val_dataset = BloodCellDataset(val_image_dir, val_mask_dir, transform=transform)
val_loader = DataLoader(val_dataset, batch_size=4)

# Khởi tạo mô hình U-Net
n_channels = 3
n_classes = 1
model = UNet(n_channels, n_classes).to(device)

# Định nghĩa loss function và optimizer
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Vòng lặp huấn luyện
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    for images, masks in train_loader:
        images = images.to(device)
        masks = masks.to(device).unsqueeze(1).float()

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * images.size(0)
    train_loss /= len(train_loader.dataset)

    print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}')

print('Finished Training')

# Sau khi vòng lặp huấn luyện kết thúc
torch.save(model.state_dict(), 'unet_blood_cell_segmentation.pth')
print('Mô hình đã được lưu.')