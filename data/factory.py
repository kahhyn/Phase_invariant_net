from torch.utils.data import DataLoader

from .ofdm_dataset import OFDMDataset


DATASET_CHOICES = ["ofdm"]


def build_dataset(dataset_type="ofdm", **kwargs):
    if dataset_type == "ofdm":
        return OFDMDataset(**kwargs)
    raise ValueError(f"Unknown dataset_type: {dataset_type}")


def build_loader(dataset, batch_size, shuffle, num_workers, device):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )


def _get(args, name, default=None, prefix=""):
    prefixed = f"{prefix}{name}" if prefix else name
    return getattr(args, prefixed, getattr(args, name, default))


def ofdm_kwargs_from_args(
    args,
    num_samples,
    phase_mode,
    seed,
    snr_db_min=None,
    snr_db_max=None,
    prefix="",
):
    return {
        "num_samples": num_samples,
        "snr_db_min": _get(args, "snr_db_min", snr_db_min, prefix) if snr_db_min is None else snr_db_min,
        "snr_db_max": _get(args, "snr_db_max", snr_db_max, prefix) if snr_db_max is None else snr_db_max,
        "channel_error_std": _get(args, "channel_error_std", 0.05, prefix),
        "h_hat_mode": _get(args, "h_hat_mode", "dmrs_ls_interp", prefix),
        "phase_mode": phase_mode,
        "dmrs_freq_spacing": _get(args, "dmrs_freq_spacing", 1, prefix),
        "dmrs_freq_offset": _get(args, "dmrs_freq_offset", 0, prefix),
        "channel_kind": _get(args, "channel_kind", "tdl", prefix),
        "subcarrier_spacing_hz": _get(args, "subcarrier_spacing_hz", 30e3, prefix),
        "num_paths": _get(args, "num_paths", 12, prefix),
        "rms_delay_spread_s": _get(args, "rms_delay_spread_s", 10e-9, prefix),
        "max_doppler_hz": _get(args, "max_doppler_hz", 200.0, prefix),
        "rician_k_db": _get(args, "rician_k_db", None, prefix),
        "seed": seed,
    }
