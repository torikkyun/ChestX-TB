# %%
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
)
from collections import Counter


# %%
# Định nghĩa transform, tạo dataset và dataloader
transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        ),
    ]
)

data_dir = "../../data/TB_Chest_Radiography_Database"
full_dataset = datasets.ImageFolder(data_dir, transform=transform)

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
dataset_labels = ["Train", "Validation", "Test"]
dataset_counts = [len(train_dataset), len(val_dataset), len(test_dataset)]

plt.figure(figsize=(7, 5))
plt.bar(
    dataset_labels, dataset_counts, color=["skyblue", "lightgreen", "salmon"]
)
plt.ylabel("Số lượng mẫu")
plt.title("Số lượng mẫu trong các tập Train, Validation, Test")
for i, count in enumerate(dataset_counts):
    plt.text(i, count + 10, str(count), ha="center")
plt.show()


#  %%
def count_classes(dataset):
    labels = [label for _, label in dataset]
    return Counter(labels)


train_class_counts = count_classes(train_dataset)
val_class_counts = count_classes(val_dataset)
test_class_counts = count_classes(test_dataset)

class_names = ["Normal", "TB"]

labels = class_names
train_counts = [train_class_counts.get(i, 0) for i in range(len(class_names))]
val_counts = [val_class_counts.get(i, 0) for i in range(len(class_names))]
test_counts = [test_class_counts.get(i, 0) for i in range(len(class_names))]

x = range(len(labels))
width = 0.25

fig, ax = plt.subplots(figsize=(10, 6))

rects1 = ax.bar(
    [i - width for i in x], train_counts, width, label="Train", color="skyblue"
)
rects2 = ax.bar(x, val_counts, width, label="Validation", color="lightgreen")
rects3 = ax.bar(
    [i + width for i in x], test_counts, width, label="Test", color="salmon"
)

ax.set_ylabel("Số lượng mẫu")
ax.set_title("Phân bố lớp trong các tập Train, Validation, Test")
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()


def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(
            "{}".format(height),
            xy=(rect.get_x() + rect.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
        )


autolabel(rects1)
autolabel(rects2)
autolabel(rects3)

fig.tight_layout()
plt.show()

# %%
# Tạo model resnet18
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
for param in model.parameters():
    param.requires_grad = False
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 1)
for param in model.fc.parameters():
    param.requires_grad = True

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
    lr = scheduler.get_last_lr()[0]

    return (
        epoch_loss,
        accuracy,
        precision,
        recall,
        f1,
        cm,
        lr,
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
        f"Train Loss: {float(train_loss):.4f}, "
        f"Accuracy: {float(train_acc):.4f}, "
        f"Precision: {float(train_prec):.4f}, "
        f"Recall: {float(train_recall):.4f}, "
        f"F1-score: {float(train_f1):.4f}, "
        f"Confusion Matrix:\n{train_cm}, "
        f"Learning Rate: {float(train_lr)}"
    )
    print(
        f"Validation Loss: {float(val_loss):.4f}, "
        f"Accuracy: {float(val_acc):.4f}, "
        f"Precision: {float(val_prec):.4f}, "
        f"Recall: {float(val_recall):.4f}, "
        f"F1-score: {float(val_f1):.4f}, "
        f"Confusion Matrix:\n{val_cm}"
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
model.load_state_dict(
    torch.load("best_model.pth", map_location=torch.device("cpu"))
    # torch.load("best_model.pth")
)
test_loss, test_acc, test_prec, test_recall, test_f1, test_cm = validate(
    model, test_loader, criterion, device
)

plt.figure(figsize=(15, 5))
plt.subplots_adjust(top=0.9)

plt.subplot(1, 2, 1)
sns.heatmap(test_cm, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix", pad=20)
plt.xlabel("Predicted")
plt.ylabel("Actual")

metrics = ["Accuracy", "Precision", "Recall", "F1-score"]
values = [test_acc, test_prec, test_recall, test_f1]

plt.subplot(1, 2, 2)
bars = plt.bar(metrics, values)
plt.title("Model Performance Metrics", pad=20),
plt.ylim(0, 1.1)

for bar in bars:
    height = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width() / 2.0,
        height + 0.02,
        f"{height:.4f}",
        ha="center",
        va="bottom",
    )

plt.tight_layout()

print("\nTest Results:")
print(f"Loss: {test_loss:.4f}")
print(f"Accuracy: {test_acc:.4f}")
print(f"Precision: {test_prec:.4f}")
print(f"Recall: {test_recall:.4f}")
print(f"F1-score: {test_f1:.4f}")
print(f"Confusion Matrix:\n{test_cm}")

plt.savefig("test_results.png", bbox_inches="tight", dpi=300)
plt.close()
