
import torch
from torch import nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# 1. Download handwritten digit data
train_data = datasets.MNIST(
    root="./data",
    train=True,
    download=True,
    transform=transforms.ToTensor()
)

test_data = datasets.MNIST(
    root="./data",
    train=False,
    download=True,
    transform=transforms.ToTensor()
)

train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
test_loader = DataLoader(test_data, batch_size=64)

# 2. Define the MLP
model = nn.Sequential(
    nn.Flatten(),
    nn.Linear(784, 128),
    nn.ReLU(),
    nn.Linear(128, 64),
    nn.ReLU(),
    nn.Linear(64, 10)
)

# 3. Configure learning
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 4. Train the model
for epoch in range(5):
    model.train()

    for images, labels in train_loader:
        predictions = model(images)
        loss = loss_fn(predictions, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Evaluate accuracy
    model.eval()
    correct = 0

    with torch.no_grad():
        for images, labels in test_loader:
            predictions = model(images)
            correct += (predictions.argmax(1) == labels).sum().item()

    accuracy = 100 * correct / len(test_data)
    print(f"Epoch {epoch+1}: Accuracy = {accuracy:.2f}%")

# 5. Save the trained network
torch.save(model.state_dict(), "digit_mlp.pth")
print("Model saved!")
