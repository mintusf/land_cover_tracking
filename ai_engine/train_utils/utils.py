import torch


def load_checkpoint(checkpoint_path: str, device: str):
    """Load checkpoint from file.
    Args:
        checkpoint_path (str): Path to checkpoint file
        device (str): Device to load checkpoint on
    """
    if "cpu" in device:
        checkpoint = torch.load(checkpoint_path, map_location=torch.device("cpu"))
    else:
        checkpoint = torch.load(checkpoint_path)

    weights = checkpoint["model_state_dict"]

    return weights
