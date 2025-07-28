import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, confusion_matrix
from torch.utils.tensorboard import SummaryWriter

from config import Config
from utils import EarlyStopping


def plot_confusion_matrix(cm, class_names):
    """
    Returns a matplotlib figure showing the confusion matrix.
    """
    fig, ax = plt.subplots(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names,
                yticklabels=class_names)
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    return fig


def train_model(model, train_ds, val_ds, fold=0):
    train_loader = DataLoader(train_ds, batch_size=Config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=Config.BATCH_SIZE)

    model = model.to(Config.DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()
    early_stopper = EarlyStopping(patience=Config.EARLY_STOPPING_PATIENCE)

    # TensorBoard
    log_dir = os.path.join("/content/drive/MyDrive/AccentLogs", f"fold_{fold}")
    writer = SummaryWriter(log_dir)

    for epoch in range(Config.NUM_EPOCHS):
        model.train()
        train_losses = []
        train_preds = []
        train_targets = []

        for X, y in train_loader:
            X, y = X.to(Config.DEVICE), y.to(Config.DEVICE)
            outputs = model(X)
            loss = criterion(outputs, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())
            preds = torch.argmax(outputs, dim=1).detach().cpu().numpy()
            train_preds.extend(preds)
            train_targets.extend(y.cpu().numpy())

        avg_train_loss = np.mean(train_losses)
        train_acc = accuracy_score(train_targets, train_preds)

        # Validation
        model.eval()
        val_losses = []
        val_preds = []
        val_targets = []

        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(Config.DEVICE), y.to(Config.DEVICE)
                outputs = model(X)
                loss = criterion(outputs, y)

                val_losses.append(loss.item())
                preds = torch.argmax(outputs, dim=1).cpu().numpy()
                val_preds.extend(preds)
                val_targets.extend(y.cpu().numpy())

        avg_val_loss = np.mean(val_losses)
        val_acc = accuracy_score(val_targets, val_preds)

        # TensorBoard logging
        writer.add_scalar("Loss/train", avg_train_loss, epoch)
        writer.add_scalar("Loss/val", avg_val_loss, epoch)
        writer.add_scalar("Accuracy/train", train_acc, epoch)
        writer.add_scalar("Accuracy/val", val_acc, epoch)

        # Confusion matrix plot
        cm = confusion_matrix(val_targets, val_preds)
        fig = plot_confusion_matrix(cm, Config.CLASS_NAMES)
        writer.add_figure("ConfusionMatrix", fig, global_step=epoch)

        print(f"Epoch {epoch+1}/{Config.NUM_EPOCHS} "
              f"| Train Loss: {avg_train_loss:.4f} Acc: {train_acc:.4f} "
              f"| Val Loss: {avg_val_loss:.4f} Acc: {val_acc:.4f}")

        if early_stopper.step(val_acc):
            print("Early stopping triggered.")
            break

    writer.close()
    return model
