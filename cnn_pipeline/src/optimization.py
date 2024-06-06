import torch
import torch.nn as nn


def get_loss():
    """
    Get an instance of the CrossEntropyLoss (useful for classification),
    optionally moving it to the GPU if use_cuda is set to True
    """
    loss  = nn.CrossEntropyLoss()
    return loss if not torch.cuda.is_available() else loss.cuda()


def get_optimizer(
    model: nn.Module,
    optimizer: str = "SGD",
    learning_rate: float = 0.01,
    momentum: float = 0.5,
    weight_decay: float = 0,
):
    """
    Parameters
    ----------
    model: the model to optimize
    optimizer: one of 'SGD' or 'Adam'
    learning_rate: the learning rate
    momentum: the momentum (if the optimizer uses it)
    weight_decay: regularization coefficient

    Returns
    -------
    an optimizer instance
    """
    if optimizer.lower() == "sgd":
        opt = torch.optim.SGD(
            params= model.parameters(),
            lr= learning_rate,
            weight_decay= weight_decay,
            momentum= momentum
        )

    elif optimizer.lower() == "adam":
        opt = torch.optim.Adam(
            params= model.parameters(),
            lr= learning_rate,
            weight_decay= weight_decay,

        )
    else:
        raise ValueError(f"Optimizer {optimizer} not supported")

    return opt
