import torch
import torch.nn.functional as F


def masked_bce_with_logits(logits, bits, loss_mask):
    """
    logits:    (B, bits_per_symbol, T, F)
    bits:      (B, bits_per_symbol, T, F), float in {0, 1}
    loss_mask: (B, 1, T, F), 1 for data RE, 0 for ignored RE
    """
    bce = F.binary_cross_entropy_with_logits(logits, bits, reduction="none")
    denom = loss_mask.sum() * logits.shape[1] + 1e-12
    return (bce * loss_mask).sum() / denom


@torch.no_grad()
def masked_ber(logits, bits, loss_mask):
    """
    Decision rule:
    logits > 0 -> bit 1
    logits < 0 -> bit 0
    """
    pred = (logits > 0).to(bits.dtype)
    err = (pred != bits).to(bits.dtype)
    denom = loss_mask.sum() * logits.shape[1] + 1e-12
    return (err * loss_mask).sum() / denom


@torch.no_grad()
def max_mean_abs_diff(a, b):
    diff = torch.abs(a - b)
    return diff.max().item(), diff.mean().item()
