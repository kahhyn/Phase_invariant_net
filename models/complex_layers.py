import torch
import torch.nn as nn


class ComplexConv2d(nn.Module):
    """
    Complex-valued 2D convolution implemented by two real convolutions.

    Let z = zr + j zi, W = A + j B.
    Then Wz = (A*zr - B*zi) + j(A*zi + B*zr).

    For non-zero charge features, set bias=False. A complex bias breaks
    phase equivariance for charge +1 / -1 / etc.
    For zero-charge features, bias=True is allowed.
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, bias=False):
        super().__init__()
        self.real_conv = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=kernel_size, padding=padding, bias=False
        )
        self.imag_conv = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=kernel_size, padding=padding, bias=False
        )

        if bias:
            self.bias_real = nn.Parameter(torch.zeros(out_channels))
            self.bias_imag = nn.Parameter(torch.zeros(out_channels))
        else:
            self.register_parameter("bias_real", None)
            self.register_parameter("bias_imag", None)

    def forward(self, z):
        if not torch.is_complex(z):
            raise TypeError("ComplexConv2d expects a complex tensor.")

        zr = z.real
        zi = z.imag

        real = self.real_conv(zr) - self.imag_conv(zi)
        imag = self.real_conv(zi) + self.imag_conv(zr)

        if self.bias_real is not None:
            real = real + self.bias_real.view(1, -1, 1, 1)
            imag = imag + self.bias_imag.view(1, -1, 1, 1)

        return torch.complex(real, imag)


class AmplitudeGate(nn.Module):
    """
    Phase-equivariant nonlinearity:
        z -> gate(|z|) * z

    Since gate is real-valued and depends only on |z|, this preserves charge.
    """
    def __init__(self, channels):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(1, channels, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, z):
        gate = torch.sigmoid(self.scale * torch.abs(z) + self.bias)
        return gate * z


class ChargeBranch(nn.Module):
    """
    Branch for a fixed non-zero charge, e.g. +1 or -1.
    All operations preserve the charge.
    """
    def __init__(self, in_channels, hidden_channels, num_layers=2, kernel_size=3):
        super().__init__()
        padding = kernel_size // 2

        layers = []
        c_in = in_channels
        for _ in range(num_layers):
            layers.append(
                ComplexConv2d(
                    c_in, hidden_channels,
                    kernel_size=kernel_size,
                    padding=padding,
                    bias=False,  # critical for non-zero charge
                )
            )
            layers.append(AmplitudeGate(hidden_channels))
            c_in = hidden_channels

        self.layers = nn.ModuleList(layers)

    def forward(self, z):
        for layer in self.layers:
            z = layer(z)
        return z


class EquivariantInteraction(nn.Module):
    """
    Equivariant interaction:
        charge +1 feature * charge -1 feature -> charge 0 feature.

    The full outer-product interaction produces C_pos * C_neg channels.
    For large C this can be heavy, so keep hidden_complex modest in experiments.
    """
    def __init__(self, c_pos, c_neg, c_zero):
        super().__init__()
        self.c_pos = c_pos
        self.c_neg = c_neg

        self.compress = ComplexConv2d(
            c_pos * c_neg,
            c_zero,
            kernel_size=1,
            padding=0,
            bias=True,  # zero-charge features may use bias
        )

    def forward(self, feat_p1, feat_n1):
        if not (torch.is_complex(feat_p1) and torch.is_complex(feat_n1)):
            raise TypeError("EquivariantInteraction expects complex tensors.")

        b, c_pos, h, w = feat_p1.shape
        _, c_neg, _, _ = feat_n1.shape

        if c_pos != self.c_pos or c_neg != self.c_neg:
            raise ValueError(
                f"Expected channels ({self.c_pos}, {self.c_neg}), "
                f"got ({c_pos}, {c_neg})."
            )

        p = feat_p1.unsqueeze(2)   # (B, C_pos, 1, H, W)
        n = feat_n1.unsqueeze(1)   # (B, 1, C_neg, H, W)

        zero_raw = p * n           # (B, C_pos, C_neg, H, W)
        zero_raw = zero_raw.reshape(b, c_pos * c_neg, h, w)

        return self.compress(zero_raw)
