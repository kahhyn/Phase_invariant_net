import torch
import torch.nn as nn


def _ensure_complex_grid(x):
    """
    Accepts (B,T,F) or (B,1,T,F) complex tensor.
    Returns (B,T,F) complex tensor for the baseline models.
    """
    if x.dim() == 4:
        if x.shape[1] != 1:
            raise ValueError("Baseline models currently expect single complex channel.")
        x = x[:, 0]
    return x


def _prepare_zero_features(P, N0, batch_size, t, f, device):
    """
    P:  (B,1,T,F) or (B,T,F), real
    N0: (B,), (B,1), or (B,1,1,1), real
    Returns P and log(N0) grid, each shaped (B,1,T,F).
    """
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


class RealImagCNN(nn.Module):
    """
    Ordinary real-valued CNN baseline.

    Input channels:
        Re(Y), Im(Y), Re(H_hat), Im(H_hat), P, log(N0)
    This model does NOT structurally guarantee common phase invariance.
    """
    def __init__(self, hidden=32, bits_per_symbol=2):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(6, hidden, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden, hidden, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden, bits_per_symbol, kernel_size=1),
        )

    def forward(self, Y, H_hat, P, N0):
        Y = _ensure_complex_grid(Y)
        H_hat = _ensure_complex_grid(H_hat)

        b, t, f = Y.shape
        P, N0_grid = _prepare_zero_features(P, N0, b, t, f, Y.device)

        x = torch.cat(
            [
                Y.real.unsqueeze(1),
                Y.imag.unsqueeze(1),
                H_hat.real.unsqueeze(1),
                H_hat.imag.unsqueeze(1),
                P,
                N0_grid,
            ],
            dim=1,
        )

        return self.net(x)


class PhysicalFeatureCNN_OLD(nn.Module):
    """
    Baseline using hand-crafted phase-invariant physical features.

    Input channels:
        Re(conj(H_hat)*Y), Im(conj(H_hat)*Y), |H_hat|^2, |Y|^2, P, log(N0)

    This is an important baseline because it obtains phase invariance through
    feature construction rather than through network equivariance.
    """
    def __init__(self, hidden=32, bits_per_symbol=2):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(6, hidden, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden, hidden, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden, bits_per_symbol, kernel_size=1),
        )

    def forward(self, Y, H_hat, P, N0):
        Y = _ensure_complex_grid(Y)
        H_hat = _ensure_complex_grid(H_hat)

        b, t, f = Y.shape
        P, N0_grid = _prepare_zero_features(P, N0, b, t, f, Y.device)

        matched = torch.conj(H_hat) * Y
        h_power = torch.abs(H_hat) ** 2
        y_power = torch.abs(Y) ** 2

        x = torch.cat(
            [
                matched.real.unsqueeze(1),
                matched.imag.unsqueeze(1),
                h_power.unsqueeze(1),
                y_power.unsqueeze(1),
                P,
                N0_grid,
            ],
            dim=1,
        )

        return self.net(x)




class PhysicalFeatureCNN(nn.Module):
    """
    Parameter/depth-matched physical feature CNN.

    It uses hand-crafted phase-invariant physical features:
        Re(conj(H_hat)*Y), Im(conj(H_hat)*Y), |H_hat|^2, |Y|^2

    Then it mirrors the macro-depth of PhaseInvariantReceiver:

        physical zero-order features
            ↓
        branch_layers real Conv blocks
            ↓
        1x1 compression to 2 * zero_complex channels
            ↓
        concat P and log(N0)
            ↓
        real-valued LLR head

    This is a stronger and fairer physical baseline than a shallow CNN.
    """

    def __init__(
        self,
        hidden=42,
        zero_complex=16,
        hidden_real=32,
        bits_per_symbol=2,
        branch_layers=2,
        kernel_size=3,
    ):
        super().__init__()

        padding = kernel_size // 2

        layers = []

        # Input physical invariant channels:
        # Re(H*Y), Im(H*Y), |H|^2, |Y|^2
        in_channels = 4

        layers.append(
            nn.Conv2d(
                in_channels,
                hidden,
                kernel_size=kernel_size,
                padding=padding,
            )
        )
        layers.append(nn.ReLU())

        for _ in range(branch_layers - 1):
            layers.append(
                nn.Conv2d(
                    hidden,
                    hidden,
                    kernel_size=kernel_size,
                    padding=padding,
                )
            )
            layers.append(nn.ReLU())

        # Match the real dimension of zero_complex complex zero-order features.
        # PhaseInvariantReceiver has zero_complex complex channels,
        # then concatenates real and imag -> 2 * zero_complex real channels.
        layers.append(
            nn.Conv2d(
                hidden,
                2 * zero_complex,
                kernel_size=1,
            )
        )

        self.feature_net = nn.Sequential(*layers)

        # Then concat P and log(N0), so input is 2 * zero_complex + 2.
        llr_in_channels = 2 * zero_complex + 2

        self.llr_head = nn.Sequential(
            nn.Conv2d(llr_in_channels, hidden_real, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_real, hidden_real, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_real, bits_per_symbol, kernel_size=1),
        )

    def forward(self, Y, H_hat, P, N0):
        Y = _ensure_complex_grid(Y)
        H_hat = _ensure_complex_grid(H_hat)

        b, t, f = Y.shape
        P, N0_grid = _prepare_zero_features(P, N0, b, t, f, Y.device)

        matched = torch.conj(H_hat) * Y
        h_power = torch.abs(H_hat) ** 2
        y_power = torch.abs(Y) ** 2

        physical_features = torch.cat(
            [
                matched.real.unsqueeze(1),
                matched.imag.unsqueeze(1),
                h_power.unsqueeze(1),
                y_power.unsqueeze(1),
            ],
            dim=1,
        )

        zero_feat = self.feature_net(physical_features)

        zero_cond = torch.cat(
            [
                zero_feat,
                P,
                N0_grid,
            ],
            dim=1,
        )

        return self.llr_head(zero_cond)
