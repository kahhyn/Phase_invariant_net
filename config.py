from dataclasses import dataclass


@dataclass
class DataConfig:
    num_ofdm_symbols: int = 14
    num_subcarriers: int = 72
    bits_per_symbol: int = 2

    snr_db_min: float = 0.0
    snr_db_max: float = 20.0

    channel_error_std: float = 0.05
    channel_kind: str = "smooth"  # "smooth" or "iid"

    phase_mode: str = "fixed"     # "fixed", "narrow", "uniform"
    narrow_phase_range: float = 0.3926990817  # pi/8

    num_samples: int = 10000
    seed: int = 0


@dataclass
class TrainConfig:
    batch_size: int = 64
    epochs: int = 10
    lr: float = 1e-3
    weight_decay: float = 0.0
    num_workers: int = 0
    device: str = "cuda"
    log_interval: int = 50
