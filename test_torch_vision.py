"""
test_torch_vision.py
author: roger erismann

PyTorch + TorchVision Functional Test Script with CUDA Verification

This script verifies that PyTorch and TorchVision are properly installed and functional,
including checking for CUDA availability. It uses the MNIST dataset and a simple custom
CNN (SmallCNN) to perform a minimal training + evaluation run.

Key features:
- Uses MNIST as a lightweight dataset for fast testing
- Implements a small CNN suitable for MNIST classification
- Verifies GPU support by running on CUDA if available
- Logs training batch loss to a file and summarizes each epoch in the terminal
- Stores logs in ./logs/torch_vision and MNIST data in ./data

Use this as a basic sanity check for PyTorch environments or Docker container validation.
"""
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import os
import logging
import time
from datetime import datetime

# Setup logging

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_DIR, "data")
LOG_DIR = os.path.join(PROJECT_DIR, "logs", "torch_vision")

os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

log_filename = os.path.join(LOG_DIR, f"torchvision_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

# Custom filter to suppress batch logs in terminal output
class TerminalFilter(logging.Filter):
    def filter(self, record):
        return "[Epoch" not in record.getMessage() or "Batch" not in record.getMessage()

file_handler = logging.FileHandler(log_filename)
console_handler = logging.StreamHandler()
console_handler.addFilter(TerminalFilter())

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[file_handler, console_handler]
)


# Decorator for timing
def timed_step(name):
    def decorator(func):
        def wrapper(*args, **kwargs):
            logging.info(f"⏳ Starting: {name}...")
            start = time.time()
            result = func(*args, **kwargs)
            elapsed = time.time() - start
            logging.info(f"✅ Finished: {name} (Elapsed time: {elapsed:.2f}s)")
            return result
        return wrapper
    return decorator

# Version Check
@timed_step("Version Check")
def check_versions():
    logging.info(f"PyTorch version: {torch.__version__}")
    logging.info(f"TorchVision version: {torchvision.__version__}")

# CUDA Check
@timed_step("CUDA Check")
def check_cuda():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logging.info(f"CUDA is available. GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        logging.info("CUDA not available. Running on CPU.")
    return device


# Custom CNN Model for MNIST
class SmallCNN(nn.Module):
    def __init__(self):
        super(SmallCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.25)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)
        x = self.dropout(torch.relu(self.fc1(x)))
        x = self.fc2(x)
        return x


# Train & Test Model (MNIST)
@timed_step("Train and Test SmallCNN on MNIST")
def train_test_model(device):
    # -------------------------
    # Check if MNIST data is already downloaded
    # -------------------------
    mnist_raw_dir = os.path.join(DATA_DIR, "MNIST", "raw")
    train_data_file = os.path.join(mnist_raw_dir, "train-images-idx3-ubyte")
    test_data_file = os.path.join(mnist_raw_dir, "t10k-images-idx3-ubyte")
    download = not (os.path.isfile(train_data_file) and os.path.isfile(test_data_file))

    if download:
        logging.info("⬇️ MNIST data not found locally. Will download.")
    else:
        logging.info("✅ Found MNIST dataset locally. Skipping download.")

    # transforms (1-channel, no resize)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # Datasets & Loaders
    trainset = torchvision.datasets.MNIST(root=DATA_DIR, train=True, download=download, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True)

    testset = torchvision.datasets.MNIST(root=DATA_DIR, train=False, download=download, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=False)

    # Model, Loss, Optimizer
    model = SmallCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    # Training Loop
    EPOCHS = 10
    for epoch in range(EPOCHS):
        print(f"\n🔁 Epoch {epoch + 1}/{EPOCHS}")
        epoch_start = time.time()
        running_loss = 0.0

        for i, data in enumerate(trainloader, 0):
            inputs, labels = data[0].to(device), data[1].to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            logging.info(f"[Epoch {epoch + 1}, Batch {i + 1}] Loss: {loss.item():.4f}")

        epoch_time = time.time() - epoch_start
        avg_loss = running_loss / len(trainloader)
        print(f"⏱️  Time: {epoch_time:.2f}s | 📉 Avg Loss: {avg_loss:.4f}")

    # Evaluation
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f"\n🎯 Final Test Accuracy: {accuracy:.2f}%")
    logging.info(f"Test Accuracy: {accuracy:.2f}%")



def main():
    check_versions()
    device = check_cuda()
    train_test_model(device)

if __name__ == "__main__":
    main()
