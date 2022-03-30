import torch
import torch.nn as nn
# from model import resnet18
from torch.utils.data import DataLoader
from dataset import CustomDataset
import config
import torchvision.models as models
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
from torch.cuda.amp import autocast
import warnings

warnings.filterwarnings("ignore")

torch.backends.cudnn.benchmark = True
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)


def my_plot(epochs, loss):
    plt.plot(epochs, loss)


def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False


def save_model(model, filename):
    torch.save(model, filename)


def train_fn(train_loader, model, optimizer, loss_fn):
    loop = tqdm(train_loader, leave=True)
    epoch_loss = []
    for batch_idx, (x, y) in enumerate(loop):
        x, y = x.to(config.DEVICE), y.to(config.DEVICE)
        optimizer.zero_grad()
        with autocast():
            out = model(x)
            loss_value = loss_fn(out, y)
        epoch_loss.append(loss_value.item())
        loss_value.backward()
        optimizer.step()

        # update progress bar
        loop.set_postfix(loss=loss_value.item())
    epoch_loss_value = sum(epoch_loss) / len(epoch_loss)
    print(f"epoch loss was {epoch_loss_value}")
    return epoch_loss_value


def check_accuracy(model, test_loader):
    num_correct = 0
    num_samples = 0
    for (x, y) in test_loader:
        x, y = x.to(config.DEVICE), y.to(config.DEVICE)
        with torch.no_grad():
            out = model(x)
            _, predictions = out.max(1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)

    print(f'Got {num_correct} / {num_samples} with accuracy {float(num_correct) / float(num_samples) * 100:.2f}')
    return float(num_correct) / float(num_samples)


def main():
    model = models.resnet18(pretrained=True)
    set_parameter_requires_grad(model, config.FEATURE_EXTRACT)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, config.NUM_OF_CLASSES)
    model = model.to(config.DEVICE)
    train_data = CustomDataset(config.NUM_OF_CLASSES, "train.csv", config.train_transforms)
    test_data = CustomDataset(config.NUM_OF_CLASSES, "test.csv", config.test_transforms)
    train_loader = DataLoader(dataset=train_data, batch_size=config.BATCH_SIZE, num_workers=config.NUM_WORKERS,
                              pin_memory=config.PIN_MEMORY, shuffle=True, drop_last=True)
    test_loader = DataLoader(dataset=test_data, batch_size=config.BATCH_SIZE, num_workers=config.NUM_WORKERS,
                             pin_memory=config.PIN_MEMORY, shuffle=True, drop_last=True)
    optimizer = optim.Adam(
        model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY
    )
    loss_fn = nn.CrossEntropyLoss()
    loss_vals = []
    best_accuracy = 0
    for epoch in range(config.EPOCHS):
        loss_vals.append(train_fn(train_loader=train_loader, model=model, optimizer=optimizer, loss_fn=loss_fn))
        if not epoch % 5:
            model.eval()
            accuracy = check_accuracy(model=model, test_loader=test_loader)
            if accuracy > best_accuracy:
                save_model(model, config.MODEL_FILENAME)
                best_accuracy = accuracy
            model.train()
    # plotting
    my_plot(np.linspace(1, config.EPOCHS, config.EPOCHS).astype(int), loss_vals)
    plt.show()


if __name__ == "__main__":
    main()
