import torch
from torch import nn

# Model Architecture with 1D CNN
class SpeechAccentModelV0(nn.Module):
    """
    First baseline model with 1D CNN
    """
    def __init__(self, input_shape: int, hidden_units: int, output_shape: int) -> None:
        super().__init__()
        self.conv_block_1 = nn.Sequential(
            nn.Conv1d(in_channels=input_shape,
                      out_channels=64,
                      kernel_size=3, # how big is the square that's going over the image?
                      stride=1, # default
                      padding=1), # options = "valid" (no padding) or "same" (output has same shape as input) or int for specific number
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.MaxPool1d(kernel_size=2,
                         stride=2) # default stride value is same as kernel_size
        )
        self.conv_block_2 = nn.Sequential(
            nn.Conv1d(in_channels = 64, out_channels = 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            # Where did this in_features shape come from?
            # It's because each layer of our network compresses and changes the shape of our input data.
            nn.LazyLinear(hidden_units),
            nn.Dropout(0.5),
            nn.Linear(hidden_units, hidden_units),
            nn.Dropout(0.5),
            nn.Linear(hidden_units, out_features=output_shape)
        )

    def forward(self, x: torch.Tensor):
        # Compress the shape to (B, C, H*W) for 1D input
        batch_size, channels, height, width = x.shape
        x = x.view(batch_size, channels, height * width)
        x = self.conv_block_1(x)
        x = self.conv_block_2(x)
        x = self.classifier(x)
        # print(x.shape)
        return x

# Model architecture with 2D CNN
class SpeechAccentModelV1(nn.Module):
    """
    First baseline model with 2D CNN
    """
    def __init__(self, input_shape: int, hidden_units: int, output_shape: int) -> None:
        super().__init__()
        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(in_channels=input_shape,
                      out_channels=64,
                      kernel_size=3, # how big is the square that's going over the image?
                      stride=1, # default
                      padding=1), # options = "valid" (no padding) or "same" (output has same shape as input) or int for specific number
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=2,
                         stride=2) # default stride value is same as kernel_size
        )
        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(in_channels = 64, out_channels = 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            # Where did this in_features shape come from?
            # It's because each layer of our network compresses and changes the shape of our input data.
            nn.LazyLinear(hidden_units),
            nn.Dropout(0.5),
            nn.Linear(hidden_units, hidden_units),
            nn.Dropout(0.5),
            nn.Linear(hidden_units, out_features=output_shape)
        )

    def forward(self, x: torch.Tensor):
        # Use shape => B, C, H, W for 2D
        x = self.conv_block_1(x)
        x = self.conv_block_2(x)
        x = self.classifier(x)
        return x
    