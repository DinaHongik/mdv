
from dataclasses import dataclass
from typing import Optional

@dataclass
class TrainConfig:
    # Model configurations
    encoder_model: str = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
    mlm_model: str = "bert-base-multilingual-cased"
    max_len: int = 512
    
    # Training hyperparameters
    batch_size: int = 8
    lr: float = 2e-5
    epochs: int = 10
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    
    # Loss weights
    varclr_weight: float = 0.05
    rtd_weight: float = 0.5
    temperature: float = 0.05
    
    # Training settings
    seed: int = 42
    device: str = "cuda"
    use_augment: bool = True
    save_every: int = 100
    data_multiplier: int = 1
    
    # Ablation settings
    ablation: str = "MDV"  # M, MD, MDV
    input_mode: str = "nmo"  # nmo, msg

@dataclass
class EvalConfig:
    # Evaluation settings
    batch_size: int = 32
    max_len: int = 512
    device: str = "cuda"
    
    # Metrics
    compute_ece: bool = True
    ece_bins: int = 15
    ece_tau: float = 1.0
    bootstrap_samples: int = 1000
    
    # Calibration
    temperature_scaling: bool = False

@dataclass
class ScoreWeights:
    # Scoring weights for field matching
    alpha_cos: float = 1.0      # Cosine similarity weight
    beta_type: float = 0.3      # Type matching weight  
    gamma_path: float = 0.2    # Path similarity weight
    delta_lex: float = 0.1     # Lexical similarity weight
    
@dataclass
class DataConfig:
    # Data processing settings
    mask_name: bool = False
    drop_type: bool = False
    drop_path: bool = False
    no_placeholder: bool = False
    encoding: str = "utf-8"
    
    # Placeholder patterns
    ip_pattern: str = r'\b(?:\d{1,3}\.){3}\d{1,3}\b'
    port_pattern: str = r':\d{1,5}\b'
    time_pattern: str = r'\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}'
