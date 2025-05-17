# %%
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
)
import timm

# %%
# Định nghĩa transform, tạo dataset và dataloader cho cả train, val và test
train_transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        ),
    ]
)
val_test_transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        ),
    ]
)

data_dir = "data/TB_Chest_Radiography_Database"
full_dataset = datasets.ImageFolder(data_dir, transform=val_test_transform)

train_size = int(0.7 * len(full_dataset))
val_size = int(0.2 * len(full_dataset))
test_size = len(full_dataset) - train_size - val_size
train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
    full_dataset, [train_size, val_size, test_size]
)

train_loader = DataLoader(
    train_dataset, batch_size=8, shuffle=True, num_workers=4
)
val_loader = DataLoader(
    val_dataset, batch_size=8, shuffle=False, num_workers=4
)
test_loader = DataLoader(
    test_dataset, batch_size=8, shuffle=False, num_workers=4
)


# %%
# Tạo model resnetv2_50
model = timm.create_model("resnetv2_50x1_bit", pretrained=True)
for param in model.parameters():
    param.requires_grad = False
model.head = nn.Sequential(
    nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(2048, 1)
)

# %%
# Thiết lập device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Định nghĩa loss function, optimizer và scheduler
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=10, eta_min=1e-6
)


# Hàm training một epoch
def train_one_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    predictions = []
    true_labels = []

    for inputs, labels in train_loader:
        inputs = inputs.to(device)
        labels = labels.float().to(device)

        optimizer.zero_grad()

        outputs = model(inputs).squeeze()
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()
        scheduler.step()

        running_loss += loss.item()

        predictions.extend((torch.sigmoid(outputs) > 0.5).cpu().numpy())
        true_labels.extend(labels.cpu().numpy())

    epoch_loss = running_loss / len(train_loader)
    accuracy = accuracy_score(true_labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        true_labels, predictions, average="binary"
    )
    cm = confusion_matrix(true_labels, predictions)

    return (
        epoch_loss,
        accuracy,
        precision,
        recall,
        f1,
        cm,
        scheduler.get_last_lr()[0],
    )


# Hàm validation
def validate(model, val_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    predictions = []
    true_labels = []

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            labels = labels.float().to(device)

            outputs = model(inputs).squeeze()
            loss = criterion(outputs, labels)

            running_loss += loss.item()

            predictions.extend((torch.sigmoid(outputs) > 0.5).cpu().numpy())
            true_labels.extend(labels.cpu().numpy())

    val_loss = running_loss / len(val_loader)
    accuracy = accuracy_score(true_labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        true_labels, predictions, average="binary"
    )
    cm = confusion_matrix(true_labels, predictions)

    return val_loss, accuracy, precision, recall, f1, cm


# %%
num_epochs = 10
best_val_f1 = 0.0
train_losses = []
val_losses = []

for epoch in range(num_epochs):
    # Training phase
    (
        train_loss,
        train_acc,
        train_prec,
        train_recall,
        train_f1,
        train_cm,
        train_lr,
    ) = train_one_epoch(model, train_loader, criterion, optimizer, device)

    # Validation phase
    val_loss, val_acc, val_prec, val_recall, val_f1, val_cm = validate(
        model, val_loader, criterion, device
    )

    train_losses.append(train_loss)
    val_losses.append(val_loss)

    print(f"Epoch [{epoch+1}/{num_epochs}]")
    print(
        f"Train Loss: {train_loss:.4f}, "
        f"Accuracy: {train_acc:.4f}, "
        f"Precision: {train_prec:.4f}, "
        f"Recall: {train_recall:.4f}, "
        f"F1-score: {train_f1:.4f}, "
        f"Learning Rate: {train_lr:.4f}, "
        f"Confusion Matrix: {train_cm:.4f}"
    )
    print(
        f"Train Loss: {val_loss:.4f}, "
        f"Accuracy: {val_acc:.4f}, "
        f"Precision: {val_prec:.4f}, "
        f"Recall: {val_recall:.4f}, "
        f"F1-score: {val_f1:.4f}, "
        f"Confusion Matrix: {val_cm:.4f}"
    )

    # Lưu model tốt nhất dựa trên F1-score
    if val_f1 > best_val_f1:
        best_val_f1 = val_f1
        torch.save(model.state_dict(), "best_model.pth")

# Vẽ đồ thị loss
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses, label="Validation Loss")
plt.title("Training and Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.show()

# %%
# Test phase
model.load_state_dict(torch.load("best_model.pth"))
test_loss, test_acc, test_prec, test_recall, test_f1, test_cm = validate(
    model, test_loader, criterion, device
)

print("\nTest Results:")
print(f"Loss: {test_loss:.4f}")
print(f"Accuracy: {test_acc:.4f}")
print(f"Precision: {test_prec:.4f}")
print(f"Recall: {test_recall:.4f}")
print(f"F1-score: {test_f1:.4f}")
print(f"Confusion Matrix: {test_cm:.4f}")
