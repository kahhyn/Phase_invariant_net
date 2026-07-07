import math
from typing import Dict

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


class OFDMDataset(Dataset):
    """
    Synthetic SISO-OFDM/QPSK dataset.

    Model:
        Y[k,l] = H[k,l] X[k,l] + N[k,l]

    H_hat modes:
        oracle_noisy:
            H_hat = H + E

        dmrs_ls_interp:
            H_hat is estimated from DMRS REs:
                H_LS[pilot] = Y[pilot] / X_pilot[pilot]
            Then non-DMRS symbols are filled by linear interpolation along
            the time dimension.

    Current DMRS pattern:
        Two full-DMRS OFDM symbols across all subcarriers.
        Therefore, this first version only needs time interpolation.
    """
    def __init__(
        self,
        num_samples=10000,
        num_ofdm_symbols=14,
        num_subcarriers=72,
        bits_per_symbol=2,
        snr_db_min=0.0,
        snr_db_max=20.0,
        channel_error_std=0.05,
        channel_kind="smooth",
        h_hat_mode="oracle_noisy",
        phase_mode="fixed",
        narrow_phase_range=math.pi / 8,
        seed=0,
    ):
        if bits_per_symbol != 2:
            raise ValueError("This first version supports QPSK only, bits_per_symbol=2.")

        if h_hat_mode not in ["oracle_noisy", "dmrs_ls_interp"]:
            raise ValueError("h_hat_mode must be 'oracle_noisy' or 'dmrs_ls_interp'.")

        self.num_samples = num_samples
        self.T = num_ofdm_symbols
        self.F = num_subcarriers
        self.bits_per_symbol = bits_per_symbol
        self.snr_db_min = float(snr_db_min)
        self.snr_db_max = float(snr_db_max)
        self.channel_error_std = float(channel_error_std)
        self.channel_kind = channel_kind
        self.h_hat_mode = h_hat_mode
        self.phase_mode = phase_mode
        self.narrow_phase_range = float(narrow_phase_range)
        self.seed = int(seed)

        dmrs_symbols = [2, 11] if self.T >= 12 else [self.T // 2]
        P = torch.zeros(1, self.T, self.F, dtype=torch.float32)
        for s in dmrs_symbols:
            P[:, s, :] = 1.0

        self.dmrs_symbols = dmrs_symbols
        self.P = P
        self.loss_mask = 1.0 - P

    def __len__(self):
        return self.num_samples

    def _generator(self, idx):
        g = torch.Generator()
        g.manual_seed(self.seed + int(idx))
        return g

    def _make_bits_and_symbols(self, g):
        bits = torch.randint(
            low=0,
            high=2,
            size=(self.bits_per_symbol, self.T, self.F),
            generator=g,
            dtype=torch.float32,
        )

        b0 = bits[0]
        b1 = bits[1]
        x = ((2.0 * b0 - 1.0) + 1j * (2.0 * b1 - 1.0)) / math.sqrt(2.0)
        x = x.to(torch.complex64)

        # Known DMRS pilot symbol. The corresponding bits are ignored by loss_mask.
        pilot = torch.ones_like(x)
        x = torch.where(self.P[0].bool(), pilot, x)

        bits = bits * self.loss_mask
        return bits, x

    def _smooth_complex_grid(self, z):
        zr = z.real.unsqueeze(0).unsqueeze(0)
        zi = z.imag.unsqueeze(0).unsqueeze(0)

        for kernel in [(3, 9), (3, 5)]:
            pad_t = kernel[0] // 2
            pad_f = kernel[1] // 2
            zr = F.avg_pool2d(zr, kernel_size=kernel, stride=1, padding=(pad_t, pad_f))
            zi = F.avg_pool2d(zi, kernel_size=kernel, stride=1, padding=(pad_t, pad_f))

        out = torch.complex(zr[0, 0], zi[0, 0])
        power = torch.mean(torch.abs(out) ** 2)
        return (out / torch.sqrt(power + 1e-12)).to(torch.complex64)

    def _make_channel(self, g):
        h = torch.randn(self.T, self.F, generator=g) + 1j * torch.randn(self.T, self.F, generator=g)
        h = h.to(torch.complex64) / math.sqrt(2.0)

        if self.channel_kind == "smooth":
            h = self._smooth_complex_grid(h)
        elif self.channel_kind == "iid":
            power = torch.mean(torch.abs(h) ** 2)
            h = h / torch.sqrt(power + 1e-12)
        else:
            raise ValueError(f"Unknown channel_kind: {self.channel_kind}")

        return h

    def _sample_snr_and_noise(self, h, x, g):
        snr_db = self.snr_db_min + (self.snr_db_max - self.snr_db_min) * torch.rand((), generator=g)
        snr_linear = 10.0 ** (snr_db / 10.0)

        signal_power = torch.mean(torch.abs(h * x) ** 2)
        n0 = signal_power / snr_linear

        noise = torch.sqrt(n0 / 2.0) * (
            torch.randn(self.T, self.F, generator=g) + 1j * torch.randn(self.T, self.F, generator=g)
        )
        return n0.to(torch.float32), noise.to(torch.complex64)

    def _make_h_hat_oracle_noisy(self, h, g):
        e = self.channel_error_std / math.sqrt(2.0) * (
            torch.randn(self.T, self.F, generator=g) + 1j * torch.randn(self.T, self.F, generator=g)
        )
        return h + e.to(torch.complex64)

    def _make_h_hat_dmrs_ls_interp(self, y, x):
        """
        LS on DMRS symbols and linear interpolation along time.

        Since the DMRS symbols occupy all subcarriers:
            H_LS[t_p, f] = Y[t_p, f] / X[t_p, f]
        """
        p_mask = self.P[0].bool()
        h_ls = torch.zeros_like(y)
        h_ls[p_mask] = y[p_mask] / (x[p_mask] + 1e-12)

        pilot_ts = self.dmrs_symbols
        h_hat = torch.empty_like(y)

        if len(pilot_ts) == 1:
            h_hat[:, :] = h_ls[pilot_ts[0], :].unsqueeze(0)
            return h_hat.to(torch.complex64)

        for t in range(self.T):
            if t <= pilot_ts[0]:
                h_hat[t, :] = h_ls[pilot_ts[0], :]
            elif t >= pilot_ts[-1]:
                h_hat[t, :] = h_ls[pilot_ts[-1], :]
            else:
                for i in range(len(pilot_ts) - 1):
                    t0 = pilot_ts[i]
                    t1 = pilot_ts[i + 1]
                    if t0 <= t <= t1:
                        alpha = float(t - t0) / float(t1 - t0)
                        h_hat[t, :] = (1.0 - alpha) * h_ls[t0, :] + alpha * h_ls[t1, :]
                        break

        return h_hat.to(torch.complex64)

    def _make_h_hat(self, h, y, x, g):
        if self.h_hat_mode == "oracle_noisy":
            return self._make_h_hat_oracle_noisy(h, g)
        if self.h_hat_mode == "dmrs_ls_interp":
            return self._make_h_hat_dmrs_ls_interp(y, x)
        raise ValueError(f"Unknown h_hat_mode: {self.h_hat_mode}")

    def _sample_phase(self, g):
        if self.phase_mode == "fixed":
            phi = torch.tensor(0.0)
        elif self.phase_mode == "narrow":
            r = torch.rand((), generator=g)
            phi = (2.0 * r - 1.0) * self.narrow_phase_range
        elif self.phase_mode == "uniform":
            phi = 2.0 * math.pi * torch.rand((), generator=g)
        else:
            raise ValueError(f"Unknown phase_mode: {self.phase_mode}")

        rot = torch.cos(phi) + 1j * torch.sin(phi)
        return rot.to(torch.complex64), phi.to(torch.float32)

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        g = self._generator(idx)

        bits, x = self._make_bits_and_symbols(g)
        h = self._make_channel(g)
        n0, noise = self._sample_snr_and_noise(h, x, g)

        y = h * x + noise
        h_hat = self._make_h_hat(h, y, x, g)

        # Apply common phase rotation. If h_hat is obtained by DMRS-LS,
        # rotating h_hat after LS is equivalent to rotating Y before LS.
        rot, phi = self._sample_phase(g)
        y = rot * y
        h = rot * h
        h_hat = rot * h_hat

        return {
            "Y": y,
            "H_hat": h_hat,
            "P": self.P.clone(),
            "N0": n0.view(1),
            "bits": bits,
            "X": x,
            "H": h,
            "loss_mask": self.loss_mask.clone(),
            "phi": phi.view(1),
        }
