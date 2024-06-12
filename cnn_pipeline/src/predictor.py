import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T

class Predictor(nn.Module):
    """
    A Predictor class for making predictions with a pre-trained PyTorch model.

    Parameters
    ----------
    model (torch.nn.Module): The pre-trained model to use for predictions.
    class_names (list): A list of class names for the model's output.
    mean (torch.Tensor): The mean values for normalization.
    std (torch.Tensor): The standard deviation values for normalization.
    """

    def __init__(self, model, class_names, mean, std):
        super().__init__()

        self.model = model.eval()
        self.class_names = class_names

        self.transforms = nn.Sequential(
            T.Resize([256, ]),  # Use single int value inside a list due to torchscript type restrictions
            T.CenterCrop(224),
            T.ConvertImageDtype(torch.float),
            T.Normalize(mean.tolist(), std.tolist())
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for making predictions.

        Parameters
	----------
        x (torch.Tensor): Input tensor representing a batch of images.

        Returns
	-------
        torch.Tensor: Output tensor with prediction probabilities for each class.
        """
        with torch.inference_mode():
            # 1. apply transforms
            x = self.transforms(x)
            # 2. get the logits
            x = self.model(x)
            # 3. apply softmax
            x = F.softmax(x, dim=1)

            return x
