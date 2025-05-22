import timm
import torch
import torch.nn as nn
from torchvision import transforms, models
import matplotlib.pyplot as plt
from PIL import Image


def preprocess_image(image_path):
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),
        ]
    )
    image = Image.open(image_path).convert("RGB")
    image = transform(image)
    return image.unsqueeze(0)  # Thêm batch dimension


def predict_with_model(model_type, model_path, image_path, device):
    if model_type == "vgg11":
        model = models.vgg11(weights=models.VGG11_Weights.DEFAULT)
        num_ftrs = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(num_ftrs, 1)
    elif model_type == "resnet18":
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 1)
    elif model_type == "resnetv2_50":
        model = timm.create_model("resnetv2_50x1_bit", pretrained=True)
        model.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(2048, 1),
        )
    else:
        raise ValueError("Unsupported model type")

    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()

    image_tensor = preprocess_image(image_path).to(device)

    with torch.no_grad():
        output = model(image_tensor).squeeze()
        probability = torch.sigmoid(output).item()
        prediction = 1 if probability > 0.5 else 0

    return prediction, probability


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
image_path = r"data\test\image.png"
model_paths = {
    "vgg11": r"src\vgg11\best_model.pth",
    "resnet18": r"src\resnet18\best_model.pth",
    "resnetv2_50": r"src\resnetv2\best_model.pth",
}

label_map = {0: "Normal", 1: "TB"}
results = {}
for model_type in ["vgg11", "resnet18", "resnetv2_50"]:
    pred_class, confidence = predict_with_model(
        model_type, model_paths[model_type], image_path, device
    )
    results[model_type] = (pred_class, confidence)
    print(
        f"{model_type.upper()} - Predicted class: {label_map[pred_class]}"
        f"(Confidence: {confidence:.4f})"
    )

img = Image.open(image_path)

plt.imshow(img)
plt.axis("off")
plt.title(
    f"VGG11: {label_map[results['vgg11'][0]]}"
    f"({results['vgg11'][1]:.2f})\n"
    f"ResNet18: {label_map[results['resnet18'][0]]}"
    f"({results['resnet18'][1]:.2f})\n"
    f"ResNetV2_50: {label_map[results['resnetv2_50'][0]]}"
    f"({results['resnetv2_50'][1]:.2f})"
)
plt.show()
