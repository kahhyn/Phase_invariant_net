import torch

from utils.metrics import (
    masked_bce_sum_and_count,
    masked_bce_with_logits,
    masked_bit_errors_and_count,
)


def move_batch(batch, device):
    out = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            out[k] = v.to(device)
        else:
            out[k] = v
    return out


def forward_receiver(model, batch):
    return model(batch["Y"], batch["H_hat"], batch["P"], batch["N0"])


def _update_metric_totals(logits, bits, loss_mask):
    loss_sum, bit_count = masked_bce_sum_and_count(logits, bits, loss_mask)
    err_sum, err_count = masked_bit_errors_and_count(logits, bits, loss_mask)
    return (
        loss_sum.detach().item(),
        bit_count.detach().item(),
        err_sum.detach().item(),
        err_count.detach().item(),
    )


def _finalize_metrics(loss_sum, bit_count, err_sum, err_count):
    loss = loss_sum / max(bit_count, 1.0)
    ber = err_sum / max(err_count, 1.0)
    return loss, ber


def train_one_epoch(model, loader, optimizer, device, log_interval):
    model.train()

    total_loss_sum = 0.0
    total_bit_count = 0.0
    total_err_sum = 0.0
    total_err_count = 0.0

    for step, batch in enumerate(loader, start=1):
        batch = move_batch(batch, device)

        logits = forward_receiver(model, batch)
        loss = masked_bce_with_logits(logits, batch["bits"], batch["loss_mask"])

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        loss_sum, bit_count, err_sum, err_count = _update_metric_totals(
            logits.detach(), batch["bits"], batch["loss_mask"]
        )
        total_loss_sum += loss_sum
        total_bit_count += bit_count
        total_err_sum += err_sum
        total_err_count += err_count

        if log_interval > 0 and step % log_interval == 0:
            avg_loss, avg_ber = _finalize_metrics(
                total_loss_sum, total_bit_count, total_err_sum, total_err_count
            )
            print(f"  step {step:05d} | loss {avg_loss:.5f} | BER {avg_ber:.5f}")

    return _finalize_metrics(total_loss_sum, total_bit_count, total_err_sum, total_err_count)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()

    total_loss_sum = 0.0
    total_bit_count = 0.0
    total_err_sum = 0.0
    total_err_count = 0.0

    for batch in loader:
        batch = move_batch(batch, device)
        logits = forward_receiver(model, batch)

        loss_sum, bit_count, err_sum, err_count = _update_metric_totals(
            logits, batch["bits"], batch["loss_mask"]
        )
        total_loss_sum += loss_sum
        total_bit_count += bit_count
        total_err_sum += err_sum
        total_err_count += err_count

    return _finalize_metrics(total_loss_sum, total_bit_count, total_err_sum, total_err_count)


def save_checkpoint(path, model, optimizer=None, args=None, **metadata):
    ckpt = {
        "model_state": model.state_dict(),
        **metadata,
    }
    if args is not None:
        ckpt["args"] = vars(args)
        ckpt.setdefault("model_name", getattr(args, "model", None))
    if optimizer is not None:
        ckpt["optimizer_state"] = optimizer.state_dict()
    torch.save(ckpt, path)
