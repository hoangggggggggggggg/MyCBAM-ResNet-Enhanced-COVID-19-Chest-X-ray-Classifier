import os
import torch
from Covid_dataset_and_calculate_mean_and_std import CovidDataset
from torch.utils.data import  DataLoader
from torchvision.transforms import Compose, ToTensor, Resize, Normalize
import torch.nn as nn
from sklearn.metrics import accuracy_score, confusion_matrix
from tqdm.autonotebook import tqdm
from torch.utils.tensorboard import SummaryWriter
from Covid_model_based_residual_network_and_CBAM import Model
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


# Function to plot and log confusion matrix to TensorBoard
def plot_confusion_matrix(writer, cm, class_names, epoch):
    fig, ax = plt.subplots(figsize=(6, 6), dpi=300)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="coolwarm", xticklabels=class_names, yticklabels=class_names)
    ax.set_xlabel("Predicted Labels")
    ax.set_ylabel("True Labels")
    ax.set_title(f"Confusion Matrix - Epoch {epoch}")
    writer.add_figure(f'confusion_matrix/epoch_{epoch}', fig, epoch)

# Main training function
def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Image transformation pipeline
    transform = Compose([
        ToTensor(),
        Resize((224, 224)),
        Normalize(mean=[0.5143], std=[0.2461])
    ])
    #define number of epochs
    num_epochs = 20
    # Load training dataset
    train_dataset = CovidDataset(root="/content/dataset/latest_updated_covid/Lung Segmentation Data/Lung Segmentation Data", is_train="train", transform=transform)
    train_params = {
        "batch_size": 16,
        "shuffle": True,
        "num_workers": 2,
        "drop_last": False
    }

    train_dataloader = DataLoader(dataset=train_dataset, **train_params)
    num_iters_per_epoch = len(train_dataloader)
    # Load validation dataset
    valid_dataset = CovidDataset(root="/content/dataset/latest_updated_covid/Lung Segmentation Data/Lung Segmentation Data", is_train="validation", transform=transform)
    val_params = {
        "batch_size": 16,
        "shuffle": False,
        "num_workers": 2,
        "drop_last": False
    }
    val_dataloader = DataLoader(dataset=valid_dataset, **val_params)
    # Initialize model
    model = Model()
    # Initialize weights using Kaiming He initialization
    def init_weights(m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    model.apply(init_weights)
    model.to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.999))

    # Load pretrained checkpoint if available
    pretrained_checkpoint_path = None
    if pretrained_checkpoint_path:
        checkpoint = torch.load(pretrained_checkpoint_path)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = checkpoint["epoch"]
        best_acc = checkpoint["best_acc"]
    else:
        start_epoch = 0
        best_acc = -1
    # Logging and checkpoint paths
    log_path = "/content/tensorboard/animal"
    os.makedirs(log_path, exist_ok=True)
    checkpoint_path = "/content/train_models/animal"
    if not os.path.isdir(checkpoint_path):
        os.makedirs(checkpoint_path)

    writer = SummaryWriter(log_dir=log_path)
    class_names = ['COVID-19','Normal','Non-COVID']
    # Training and validation loop
    for epoch in range(start_epoch, num_epochs):
        # Training loop
        model.train()
        train_loss = []
        progress_bar = tqdm(train_dataloader, colour="green")
        for iter, (images, labels) in enumerate(progress_bar):
            images = images.to(device)
            labels = labels.to(device)
            predictions = model(images)
            loss = criterion(predictions, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())
            avg_loss = np.mean(train_loss)
            progress_bar.set_description(f"epoch: {epoch}/{num_epochs-1}. AVGLoss {avg_loss}")
            writer.add_scalar("train/loss", avg_loss, epoch * num_iters_per_epoch + iter)

        # Validation loop
        all_losses = []
        all_predictions = []
        all_labels = []
        model.eval()
        with torch.no_grad():
            progress_bar = tqdm(val_dataloader, colour="blue")
            for images, labels in progress_bar:
                images = images.to(device)
                labels = labels.to(device)
                predictions = model(images)
                loss = criterion(predictions, labels)
                predictions = torch.argmax(predictions, 1)
                all_labels.extend(labels.tolist())
                all_predictions.extend(predictions.tolist())
                all_losses.append(loss.item())
        # Calculate validation metrics
        acc = accuracy_score(all_labels, all_predictions)
        loss = np.mean(all_losses)
        print(f"epoch {epoch}. Validation loss: {loss}. Validation accuracy: {acc}")
        conf_matrix = confusion_matrix(all_labels, all_predictions)
        writer.add_scalar("val/loss", loss, epoch)
        writer.add_scalar("val/accuracy", acc, epoch)
        plot_confusion_matrix(writer, conf_matrix, class_names, epoch)
        # Save model checkpoint
        checkpoint = {
            "epoch": epoch + 1,
            "best_acc": best_acc,
            "model_state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        }
        torch.save(checkpoint, os.path.join(checkpoint_path, "model.pt"))
        # Save best model if accuracy improves
        if acc > best_acc:
            best_acc = acc
            torch.save(checkpoint, os.path.join(checkpoint_path, "best_model.pt"))


if __name__ == '__main__':
    train()