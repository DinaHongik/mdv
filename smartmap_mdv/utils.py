
import torch, numpy as np, random
import re
from smartmap_mdv.config import DataConfig

def set_seed(seed=13):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def substitute_placeholders(text: str, config: DataConfig = DataConfig()) -> str:
    """
    Substitutes placeholders for IPs, ports, and timestamps in a string.

    Args:
        text: The input string.
        config: DataConfig object with regex patterns.

    Returns:
        The string with placeholders.
    """
    if not text or config.no_placeholder:
        return text
    text = re.sub(config.ip_pattern, '[IP]', text)
    text = re.sub(config.port_pattern, ':[PORT]', text)
    text = re.sub(config.time_pattern, '[TIME]', text)
    return text
