import torch
import torch.nn as nn

from .complex_layers import ChargeBranch, EquivariantInteraction


def _prepare_complex_input(Y, H_hat):
    """
    Accepts Y and H_hat with shape:
        (B,T,F) or (B,C,T,F)
    Returns charge +1 input with shape:
        (B, C_total, T, F)
    """
    if Y.dim() == 3:
        Y = Y.unsqueeze(1)
    if H_hat.dim() == 3:
        H_hat = H_hat.unsqueeze(1)

    if not (torch.is_complex(Y) and torch.is_complex(H_hat)):
        raise TypeError("Y and H_hat must be complex tensors.")

    return torch.cat([Y, H_hat], dim=1)


def _prepare_zero_features(P, N0, batch_size, t, f, device):
    if P.dim() == 3:
        P = P.unsqueeze(1)
    P = P.to(device=device, dtype=torch.float32)

    if N0.dim() == 1:
        N0 = N0.view(batch_size, 1, 1, 1)
    elif N0.dim() == 2:
        N0 = N0.view(batch_size, 1, 1, 1)
    elif N0.dim() == 4:
        pass
    else:
        raise ValueError(f"Unsupported N0 shape: {N0.shape}")

    N0_grid = torch.log(N0.to(device=device, dtype=torch.float32) + 1e-12)
    N0_grid = N0_grid.expand(batch_size, 1, t, f)
    return P, N0_grid


class PhaseInvariantReceiver(nn.Module):
    """
    U(1)-invariant receiver prototype.

    Charge convention:
        Y, H_hat are charge +1 features:
            F -> exp(j phi) F

        conj(Y), conj(H_hat) are charge -1 features:
            F -> exp(-j phi) F

        charge +1 * charge -1 -> charge 0.

    The final LLR head reads only charge-0 features, P and log(N0).
    Therefore, under a common phase rotation:
        Y -> exp(j phi)Y
        H_hat -> exp(j phi)H_hat
    the output LLR is invariant up to numerical precision.
    """
    def __init__(
        self,
        hidden_complex=16,
        zero_complex=16,
        hidden_real=32,
        bits_per_symbol=2,
        branch_layers=2,
        kernel_size=3,
    ):
        super().__init__()

        in_channels = 2  # Y, H_hat

        self.pos_branch = ChargeBranch(
            in_channels=in_channels,
            hidden_channels=hidden_complex,
            num_layers=branch_layers,
            kernel_size=kernel_size,
        )

        self.neg_branch = ChargeBranch(
            in_channels=in_channels,
            hidden_channels=hidden_complex,
            num_layers=branch_layers,
            kernel_size=kernel_size,
        )

        self.interaction = EquivariantInteraction(
            c_pos=hidden_complex,
            c_neg=hidden_complex,
            c_zero=zero_complex,
        )

        # Re(F0), Im(F0), P, log(N0)
        llr_in_channels = 2 * zero_complex + 2

        self.llr_head = nn.Sequential(
            nn.Conv2d(llr_in_channels, hidden_real, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_real, hidden_real, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_real, bits_per_symbol, kernel_size=1),
        )

    def forward(self, Y, H_hat, P, N0):
        feat_p1_in = _prepare_complex_input(Y, H_hat)
        feat_n1_in = torch.conj(feat_p1_in)

        b, _, t, f = feat_p1_in.shape
        P, N0_grid = _prepare_zero_features(P, N0, b, t, f, feat_p1_in.device)

        feat_p1 = self.pos_branch(feat_p1_in)
        feat_n1 = self.neg_branch(feat_n1_in)

        zero_feat = self.interaction(feat_p1, feat_n1)

        zero_real = torch.cat([zero_feat.real, zero_feat.imag], dim=1)
        zero_cond = torch.cat([zero_real, P, N0_grid], dim=1)

        return self.llr_head(zero_cond)
