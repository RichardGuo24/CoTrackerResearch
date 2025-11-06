import torch

def best_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    # Apple Silicon (MPS)
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"

def autocast_dtype(device: str):
    # Safe default: float16 on cuda, float32 elsewhere.
    if device == "cuda":
        return torch.float16
    return torch.float32
