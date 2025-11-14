"""
mps_utils.py — Utility functions for Apple Silicon (M1–M4) MPS backend.

Use in training scripts:
    from utils.mps_utils import get_best_device, setup_mps

    device = get_best_device()
    setup_mps()
"""


import torch


def get_best_device():
    """
    Returns the best available device:
    - MPS (Apple Silicon)
    - CUDA
    - CPU fallback
    """
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


def setup_mps():
    """
    Enables safe MPS training settings for Apple Silicon.
    Does nothing on CUDA/CPU.
    """
    if torch.backends.mps.is_available():
        torch.set_float32_matmul_precision("medium")
        try:
            torch.backends.cudnn.allow_tf32 = True
        except Exception:
            pass

    return


def info():
    """
    Print a small summary of compute capabilities.
    """
    print("PyTorch:", torch.__version__)
    print("CUDA:", torch.cuda.is_available())
    print("MPS:", torch.backends.mps.is_available())
    print("Using device:", get_best_device())
