"""
Utility functions for Synthia.
"""

import random
import numpy as np
from typing import List, Dict, Any


def set_seed(seed: int = 42):
    """
    Set random seed for reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)


def batch_process(items: List[Any], batch_size: int = 100) -> List[List[Any]]:
    """
    Split items into batches.
    """
    return [items[i:i+batch_size] for i in range(0, len(items), batch_size)]


def format_time(hours: float) -> str:
    """
    Format time in hours to human-readable string.
    """
    if hours < 1:
        return f"{hours * 60:.1f} minutes"
    elif hours < 24:
        return f"{hours:.1f} hours"
    else:
        days = hours / 24
        return f"{days:.1f} days"


def format_concentration(molar: float) -> str:
    """
    Format concentration in molar.
    """
    if molar >= 1:
        return f"{molar:.2f} M"
    elif molar >= 1e-3:
        return f"{molar * 1e3:.2f} mM"
    elif molar >= 1e-6:
        return f"{molar * 1e6:.2f} µM"
    elif molar >= 1e-9:
        return f"{molar * 1e9:.2f} nM"
    else:
        return f"{molar * 1e12:.2f} pM"


def save_json(data: Dict, filename: str):
    """
    Save data to JSON file.
    """
    import json
    with open(filename, 'w') as f:
        json.dump(data, f, indent=2)


def load_json(filename: str) -> Dict:
    """
    Load data from JSON file.
    """
    import json
    with open(filename, 'r') as f:
        return json.load(f)
