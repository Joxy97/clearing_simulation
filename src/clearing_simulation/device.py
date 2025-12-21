import torch


def get_device(device_preference: str = "auto") -> torch.device:
    """
    Return a torch.device based on availability and preference.

    device_preference:
        "auto" -> CUDA > MPS > CPU
        "cpu", "cuda:0", "mps" -> manual override
    """
    if device_preference is None or device_preference == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda:0")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device_preference)
