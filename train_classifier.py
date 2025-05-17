import os
import torch
import torch.nn as nn
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm


def train_icon_classifier(data_dir="data", output_model='models/resnet_icon.pt', num_epochs=20, batch_size=16):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    best_acc = 0.0

    # Transformming 
    train_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomResizedCrop(224, scale=(0.6, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(25),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
    ])

    # For validation only:
    val_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                            [0.229, 0.224, 0.225])
    ])



    dataset_train = datasets.ImageFolder(os.path.join(data_dir, 'train'), transform=train_transform)
    dataset_val = datasets.ImageFolder(os.path.join(data_dir, 'val'), transform=val_transform)

    loader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
    loader_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=False)

    class_names = dataset_train.classes
    num_classes = len(class_names)

    # Load ResNet50
    model = models.resnet50(weights='IMAGENET1K_V1')
    model.fc = nn.Linear(model.fc.in_features, num_classes)  # Replace final layer
    model = model.to(device)

    # Loss & optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        correct = 0

        for images, labels in tqdm(loader_train, desc=f"Epoch {epoch+1}/{num_epochs} [Training]"):
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()

        train_acc = correct / len(dataset_train)

        # === Validation Phase ===
        model.eval()  # Evaluates the modal?
        val_correct = 0

        with torch.no_grad():
            for val_images, val_labels in tqdm(loader_val, desc=f"Epoch {epoch+1}/{num_epochs} [Validation]"):
                val_images, val_labels = val_images.to(device), val_labels.to(device)
                outputs = model(val_images)
                _, preds = torch.max(outputs, 1)
                val_correct += (preds == val_labels).sum().item()

        val_acc = val_correct / len(dataset_val)

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), output_model)
            print(f"📦 New best model saved! Val Acc: {val_acc:.2%}")


        # === Epoch Summary ===
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss:.4f}, "
            f"Train Acc: {train_acc:.2%}, Val Acc: {val_acc:.2%}")


    os.makedirs(os.path.dirname(output_model), exist_ok=True)
    torch.save(model.state_dict(), output_model)
    print(f"Model saved to: {output_model}")


    return model, class_names