import torch
from torch import nn
from data import speech_data, train_loader, val_loader, test_loader
from train import train_model, evaluate_model
from architecture import SpeechAccentModelV0

NUM_EPOCHS = 30

# Device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Define training model process
model_0 = SpeechAccentModelV0(input_shape=1, # number of color channels (1 for mono)
                    hidden_units=10,
                    output_shape=len(speech_data.classes)).to(device)

# Setup loss function and optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params=model_0.parameters(), lr=0.001)

# Train the model
model_results = train_model(seed=31, num_epochs=NUM_EPOCHS, model=model_0, loss_fn = loss_fn, optimizer = optimizer)

# Evaluation report
accuracy, report = evaluate_model(model=model_0, dataloader = val_loader, device = device)