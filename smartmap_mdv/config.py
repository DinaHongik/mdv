from dataclasses import dataclass, asdict
from typing import Literal


@dataclass
class TrainConfig:
    # Backbone / input
    encoder_model: str = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
    mlm_model: str = "bert-base-multilingual-cased"
    input_mode: Literal["raw_msg", "flat_field", "nmo"] = "nmo"
    max_len: int = 512

    # Optimization
    batch_size: int = 8
    lr: float = 2e-5
    epochs: int = 10
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    grad_clip: float = 1.0

    # Loss
    temperature: float = 0.05
    rtd_weight: float = 0.5
    varclr_weight: float = 0.05
    varclr_gamma: float = 1.0
    varclr_cov_weight: float = 1.0

    # System
    seed: int = 42
    device: str = "cuda"
    deterministic: bool = True
    use_augment: bool = True
    data_multiplier: int = 1
    save_every: int = 0

    # Ablation
    ablation: Literal["M", "MD", "MDV"] = "MDV"


@dataclass
class EvalConfig:
    batch_size: int = 32
    max_len: int = 512
    device: str = "cuda"

    # Calibration
    compute_ece: bool = True
    ece_bins: int = 15
    calibration_method: Literal["none", "isotonic", "temperature"] = "isotonic"


@dataclass
class ScoreWeights:
    alpha_cos: float = 1.0
    beta_type: float = 0.3
    gamma_path: float = 0.2
    delta_lex: float = 0.1


@dataclass
class DataConfig:
    mask_name: bool = False
    drop_type: bool = False
    drop_path: bool = False
    drop_desc: bool = False
    drop_example: bool = False
    no_placeholder: bool = False
    encoding: str = "utf-8"

    ip_pattern: str = r"\b(?:\d{1,3}\.){3}\d{1,3}\b"
    port_pattern: str = r":\d{1,5}\b"
    time_pattern: str = r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}"
