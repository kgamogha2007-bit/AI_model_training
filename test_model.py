import os
import torch
import torch.nn as nn
from torchvision import datasets, models, transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

test_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

test_dataset = datasets.ImageFolder('data/test', test_transform)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=8, shuffle=False)
class_names = test_dataset.classes

model = models.resnet18(weights=None)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, len(class_names))
model.load_state_dict(torch.load('best_model.pt', weights_only=True))
model = model.to(device)
model.eval()

correct = 0
total = 0
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        for i in range(len(labels)):
            actual = class_names[labels[i]]
            predicted = class_names[preds[i]]
            status = "✓" if actual == predicted else "✗"
            print(f"{status} Actual: {actual}, Predicted: {predicted}")

print(f"\nTest Accuracy: {correct}/{total} = {100*correct/total:.1f}%")