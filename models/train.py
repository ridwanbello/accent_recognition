import torch
from tqdm.auto import tqdm
from torch import nn
from typing import Dict, List
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report
from tqdm.auto import tqdm
from accent_recognition.models.utils import EarlyStopping
from torch.utils.tensorboard import SummaryWriter
# Setup device 
# Stup device for device-agnostic code
device = "cuda" if torch.cuda.is_available() else "cpu"

# Add tensorboard module



# Create a writer with all default settings
writer = SummaryWriter()

# Define the training step function
def train_step(model: torch.nn.Module,
               dataloader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               optimizer: torch.optim.Optimizer):
    # Put model in train mode
    model.train()

    # Setup train loss and train accuracy values
    train_loss, train_acc = 0, 0

    # Loop through data loader data batches
    for batch, (X, y) in enumerate(dataloader):
        # Send data to target device
        X, y = X.to(device), y.to(device)

        # 1. Forward pass
        y_pred = model(X)

        # 2. Calculate  and accumulate loss
        loss = loss_fn(y_pred, y)
        train_loss += loss.item()

        # 3. Optimizer zero grad
        optimizer.zero_grad()

        # 4. Loss backward
        loss.backward()

        # 5. Optimizer step
        optimizer.step()

        # Calculate and accumulate accuracy metrics across all batches
        y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
        train_acc += (y_pred_class == y).sum().item()/len(y_pred)

    # Adjust metrics to get average loss and accuracy per batch
    train_loss = train_loss / len(dataloader)
    train_acc = train_acc / len(dataloader)
    return train_loss, train_acc

# Define the testing step function
def test_step(model: torch.nn.Module,
              dataloader: torch.utils.data.DataLoader,
              loss_fn: torch.nn.Module):
    # Put model in eval mode
    model.eval()

    # Setup test loss and test accuracy values
    test_loss, test_acc = 0, 0

    # Turn on inference context manager
    with torch.inference_mode():
        # Loop through DataLoader batches
        for batch, (X, y) in enumerate(dataloader):
            # Send data to target device
            X, y = X.to(device), y.to(device)

            # 1. Forward pass
            test_pred_logits = model(X)

            # 2. Calculate and accumulate loss
            loss = loss_fn(test_pred_logits, y)
            test_loss += loss.item()

            # Calculate and accumulate accuracy
            test_pred_labels = test_pred_logits.argmax(dim=1)
            test_acc += ((test_pred_labels == y).sum().item()/len(test_pred_labels))

    # Adjust metrics to get average loss and accuracy per batch
    test_loss = test_loss / len(dataloader)
    test_acc = test_acc / len(dataloader)
    return test_loss, test_acc

# Define the main training function
def train(model: torch.nn.Module,
          train_dataloader: torch.utils.data.DataLoader,
          test_dataloader: torch.utils.data.DataLoader,
          optimizer: torch.optim.Optimizer,
          loss_fn: torch.nn.Module,
          epochs: int,
          writer: SummaryWriter):

    # 2. Create empty results dictionary
    results = {"train_loss": [],
        "train_acc": [],
        "test_loss": [],
        "test_acc": []
    }

    # 3. Loop through training and testing steps for a number of epochs
    for epoch in tqdm(range(epochs)):
        train_loss, train_acc = train_step(model=model,
                                           dataloader=train_dataloader,
                                           loss_fn=loss_fn,
                                           optimizer=optimizer)
        test_loss, test_acc = test_step(model=model,
            dataloader=test_dataloader,
            loss_fn=loss_fn)
        
        # Add early stopping to avoid overfitting
        early_stopper = EarlyStopping(patience=3, min_delta=0.01, mode='min')

        early_stopper(test_loss)
        if early_stopper.early_stop:
            print("Early stopping triggered.")
            break

        # 4. Print out what's happening
        print(
            f"Epoch: {epoch+1} | "
            f"train_loss: {train_loss:.4f} | "
            f"train_acc: {train_acc:.4f} | "
            f"test_loss: {test_loss:.4f} | "
            f"test_acc: {test_acc:.4f}"
        )

    

        # 5. Update results dictionary
        # Ensure all data is moved to CPU and converted to float for storage
        results['model'] = model.__class__.__name__
        results["train_loss"].append(train_loss.item() if isinstance(train_loss, torch.Tensor) else train_loss)
        results["train_acc"].append(train_acc.item() if isinstance(train_acc, torch.Tensor) else train_acc)
        results["test_loss"].append(test_loss.item() if isinstance(test_loss, torch.Tensor) else test_loss)
        results["test_acc"].append(test_acc.item() if isinstance(test_acc, torch.Tensor) else test_acc)

        
        ### New: Experiment tracking ###
        # Add loss results to SummaryWriter
        writer.add_scalars(main_tag="Loss", 
                           tag_scalar_dict={"train_loss": train_loss,
                                            "test_loss": test_loss},
                           global_step=epoch)

        # Add accuracy results to SummaryWriter
        writer.add_scalars(main_tag="Accuracy", 
                           tag_scalar_dict={"train_acc": train_acc,
                                            "test_acc": test_acc}, 
                           global_step=epoch)
        
        # Track the PyTorch model architecture
        writer.add_graph(model=model, input_to_model=torch.randn(32, 1, 13, 298).to(device))
    
        # Close the writer
        writer.close()
    
    ### End new ###

    # 6. Return the filled results at the end of the epochs
    return results

# Model training process with timer
def train_model(seed, num_epochs, model, train_loader, val_loader, loss_fn, optimizer, writer):
  # Set random seeds
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)

  # Start the timer
  from timeit import default_timer as timer
  start_time = timer()

  # Train model_0
  model_results = train(model=model,
                          train_dataloader=train_loader,
                          test_dataloader=val_loader,
                          optimizer=optimizer,
                          loss_fn=loss_fn,
                          epochs=num_epochs,
                          writer=writer)

  # End the timer and print out how long it took
  end_time = timer()
  print(f"Total training time: {end_time-start_time:.3f} seconds")
  return model_results


# Sample Usage
# model_results = train_model(seed=31, num_epochs=3, model=model_0, train_loader=train_loader, val_loader = val_loader, loss_fn = loss_fn, optimizer = optimizer)


# Plot loss curve
def plot_loss_curves(results: Dict[str, List[float]]):
    """Plots training curves of a results dictionary.

    Args:
        results (dict): dictionary containing list of values, e.g.
            {"train_loss": [...],
             "train_acc": [...],
             "test_loss": [...],
             "test_acc": [...]}
    """

    # Get the loss values of the results dictionary (training and test)
    loss = results['train_loss']
    test_loss = results['test_loss']

    # Get the accuracy values of the results dictionary (training and test)
    accuracy = results['train_acc']
    test_accuracy = results['test_acc']

    # Figure out how many epochs there were
    epochs = range(len(results['train_loss']))

    # Setup a plot
    plt.figure(figsize=(15, 7))

    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, loss, label='train_loss')
    plt.plot(epochs, test_loss, label='test_loss')
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.legend()

    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, accuracy, label='train_accuracy')
    plt.plot(epochs, test_accuracy, label='test_accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epochs')
    plt.legend()

# Evaluate model
def evaluate_model(model: torch.nn.Module, dataloader: torch.utils.data.DataLoader, device=device):
  """Return accuracy and classification report on the test dataset"""
  # Put model in eval mode
  model.eval()
  model.to(device)

  all_preds = []
  all_labels = []

  with torch.inference_mode():
    for X, y in dataloader:
      # Send data to target device
      X, y = X.to(device), y.to(device)

       # Forward pass
      test_pred_logits = model(X)

      # Get predicted class
      _, preds = torch.max(test_pred_logits, 1)

      all_preds.extend(preds.cpu().numpy())
      all_labels.extend(y.cpu().numpy())

  # Accuracy and Classification report
  acc = accuracy_score(all_labels, all_preds)
  report = classification_report(all_labels, all_preds)

  print(f"Test Accuracy: {acc:.4f}")
  print("Classification Report:")
  print(report)

  return acc, report


