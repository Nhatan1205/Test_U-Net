import torch
import torch.nn.functional as F
import cv2
import numpy as np
import matplotlib.pyplot as plt
from unet import UNet  # Giả sử bạn đã lưu code U-Net vào file unet.py

# 1. Chuẩn bị Ảnh Đầu vào
image_path = 'path/to/your/blood_cell_image.jpg'
original_image = cv2.imread(image_path)
original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB) # Chuyển sang RGB

# Tiền xử lý (điều chỉnh kích thước và chuẩn hóa theo mô hình của bạn)
input_size = (256, 256) # Kích thước mà mô hình của bạn mong đợi
resized_image = cv2.resize(original_image, input_size)
normalized_image = resized_image / 255.0
input_tensor = torch.from_numpy(normalized_image).float().permute(2, 0, 1).unsqueeze(0) # [1, C, H, W]

# 2. Load Mô hình U-Net đã Huấn luyện
n_channels = 3
n_classes = 1
model = UNet(n_channels, n_classes)
model.load_state_dict(torch.load('path/to/your/trained_unet_weights.pth')) # Đường dẫn đến weights đã huấn luyện
model.eval()

# 3. Đưa Ảnh Qua Mô hình và Nhận Dự đoán
with torch.no_grad():
    output = model(input_tensor)
    if n_classes == 1:
        mask_pred_probs = torch.sigmoid(output).squeeze().cpu().numpy()
        mask_pred = (mask_pred_probs > 0.5).astype(np.uint8) * 255
    else:
        # Xử lý cho multi-class segmentation nếu cần
        mask_pred = torch.argmax(output, dim=1).squeeze().cpu().numpy()

# 4. Hậu Xử lý và Visualization Kết quả
resized_original_image = cv2.resize(original_image, input_size)

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.imshow(resized_original_image)
plt.title('Ảnh Gốc')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(mask_pred, cmap='gray') # Hoặc colormap khác
plt.title('Mask Dự đoán')
plt.axis('off')

plt.show()

# Hoặc chồng mask lên ảnh gốc
alpha = 0.5
overlay = resized_original_image.copy()
overlay[mask_pred > 0] = [255, 0, 0] # Màu đỏ cho vùng tế bào máu
plt.figure(figsize=(8, 8))
plt.imshow(cv2.addWeighted(resized_original_image, 1 - alpha, overlay, alpha, 0))
plt.title('Ảnh Gốc với Mask Chồng Lên')
plt.axis('off')
plt.show()