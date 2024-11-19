import random
import numpy as np
import torch


def is_confident_state(pred_probs, features):
    """
    判断是否是confident state
    """

    eps = 1e-15
    entropies = -np.sum(pred_probs * np.log(np.clip(pred_probs, eps, 1.0)), axis=1)
    reference_probs = np.array([0.8, 0.05, 0.05, 0.05, 0.05])
    entropy_threshold = -np.sum(reference_probs * np.log(reference_probs))

    is_confident = entropies < entropy_threshold

    return is_confident, pred_probs, entropies


def set_seed(seed):
    """
    Sets the random seed for reproducibility.
    Args:
        seed (int): The random seed to use.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def initialize_weights(model):
    """
    Initializes model weights for reproducibility.
    Args:
        model (torch.nn.Module): The model to initialize.
    """
    for layer in model.modules():
        if isinstance(layer, (torch.nn.Linear, torch.nn.Conv2d)):
            torch.nn.init.xavier_uniform_(layer.weight)
            if layer.bias is not None:
                torch.nn.init.zeros_(layer.bias)
        elif isinstance(layer, torch.nn.LSTM):
            for name, param in layer.named_parameters():
                if "weight" in name:
                    torch.nn.init.xavier_uniform_(param)
                elif "bias" in name:
                    torch.nn.init.zeros_(param)
        elif isinstance(model, torch.nn.LayerNorm):
            model.bias.data.zero_()
            model.weight.data.fill_(1.0)


# Example usage
set_seed(42)  # Set the seed for reproducibility
