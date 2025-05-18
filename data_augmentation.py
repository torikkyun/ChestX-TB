# %%
import os
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
from matplotlib import pyplot as plt

# %%
AUGMENT_TIMES = 4
original_dir = "data/TB_Chest_Radiography_Database/tuberculosis"
augmented_dir = "data/tuberculosis"
os.makedirs(augmented_dir, exist_ok=True)

# %%
augmentation_transform = transforms.Compose(
    [
        transforms.RandomHorizontalFlip(p=1.0),
        transforms.RandomRotation(10),
        transforms.RandomAffine(
            degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=10
        ),
        transforms.ColorJitter(
            brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1
        ),
    ]
)

augmented_count = 0

for filename in tqdm(os.listdir(original_dir)):
    if filename.endswith((".jpg", ".jpeg", ".png")):
        img_path = os.path.join(original_dir, filename)
        image = Image.open(img_path).convert("RGB")

        for i in range(AUGMENT_TIMES):
            transformed = augmentation_transform(image)
            save_path = os.path.join(
                augmented_dir, f"{filename[:-4]}_aug{i}.jpg"
            )
            transformed.save(save_path)
            augmented_count += 1

print(f"Tổng số ảnh lao phổi ban đầu: {len(os.listdir(original_dir))}")
print(f"Đã tạo thêm: {augmented_count} ảnh tăng cường")
print(
    f"Tổng cộng ảnh lao phổi sau tăng cường: "
    f"{augmented_count + len(os.listdir(original_dir))}"
)

#  %%
original_count = len(os.listdir(original_dir))
total_after_augmentation = augmented_count + original_count

labels = ["Trước tăng cường", "Sau tăng cường"]
counts = [original_count, total_after_augmentation]

plt.figure(figsize=(6, 4))
plt.bar(labels, counts, color=["skyblue", "lightgreen"])
plt.ylabel("Số lượng ảnh")
plt.title("Số lượng ảnh lao phổi trước và sau tăng cường dữ liệu")
plt.show()
