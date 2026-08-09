"""Small shared utilities: device selection and seeding."""

import os
import random

import numpy as np
import torch


def get_device(prefer: str = "auto") -> torch.device:
    """
    Pick a compute device.

    prefer:
        "auto"  -> MPS if available (Apple Silicon), else CPU.
        "cpu"   -> force CPU. Used for the theory sanity checks, where we want
                   deterministic float behaviour and exact logit comparisons that
                   MPS does not reliably provide.
        "mps"   -> force MPS.
    """
    if prefer == "cpu":
        return torch.device("cpu")
    if prefer == "mps":
        return torch.device("mps")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def set_seed(seed: int) -> None:
    """Seed every RNG we touch. Note: full determinism on MPS is not guaranteed."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
