import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import numpy as np
import cv2
import matplotlib.pyplot as plt
from torchvision.datasets import ImageFolder
from model.facialemotion import EmotionCNN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
def preprocess():
    # Data transformations
    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((48, 48)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # Load dataset (Use FER-2013 or a custom dataset folder)
    dataset_path = "./dataset"  # Replace with actual dataset path
    train_dataset = ImageFolder(root=dataset_path + "/train", transform=transform)
    test_dataset = ImageFolder(root=dataset_path + "/test", transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    return train_loader, test_loader

def train(model, train_loader):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    num_epochs = 20
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}")
    return model 

def test(model, test_loader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f"Test Accuracy: {accuracy:.2f}%")

def main():
    #Preprocess Data 
    train_loader, test_loader = preprocess()

    #Model Definition 
    model = EmotionCNN(num_classes=7).to(device)

    facial_model = train(model, train_loader)

    test(facial_model, test_loader)

    torch.save(facial_model.state_dict(), "facialemotionmodel.pth")
if __name__ == "__main__":
    main()


    

