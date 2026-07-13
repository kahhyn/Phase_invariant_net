import argparse
import csv
from pathlib import Path

import torch
from torch.utils.data import ConcatDataset

from data import build_dataset, build_loader
from engine import evaluate, save_checkpoint, train_one_epoch
from models import MODEL_CHOICES, build_model_from_args


def optional_float(text):
    if text is None or str(text).lower() in ["none", "null"]:
        return None
    return float(text)


def parse_float_list(text):
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def default_domain_values(drift_type):
    if drift_type == "delay_spread":
        return [10e-9, 30e-9, 60e-9, 100e-9, 150e-9]
    if drift_type == "doppler":
        return [200.0, 400.0, 600.0, 800.0, 1000.0]
    raise ValueError(f"Unknown drift_type: {drift_type}")


def domain_name(drift_type, value):
    if drift_type == "delay_spread":
        return f"delay_{value * 1e9:.0f}ns"
    if drift_type == "doppler":
        return f"doppler_{value:.0f}hz"
    return f"domain_{value:g}"


def domain_kwargs(args, domain_value):
    kwargs = {
        "snr_db_min": args.snr_db_min,
        "snr_db_max": args.snr_db_max,
        "channel_error_std": args.channel_error_std,
        "h_hat_mode": args.h_hat_mode,
        "phase_mode": args.phase_mode,
        "dmrs_freq_spacing": args.dmrs_freq_spacing,
        "dmrs_freq_offset": args.dmrs_freq_offset,
        "channel_kind": args.channel_kind,
        "subcarrier_spacing_hz": args.subcarrier_spacing_hz,
        "num_paths": args.num_paths,
        "rms_delay_spread_s": args.rms_delay_spread_s,
        "max_doppler_hz": args.max_doppler_hz,
        "rician_k_db": args.rician_k_db,
    }

    if args.drift_type == "delay_spread":
        kwargs["rms_delay_spread_s"] = domain_value
    elif args.drift_type == "doppler":
        kwargs["max_doppler_hz"] = domain_value
    else:
        raise ValueError(f"Unknown drift_type: {args.drift_type}")

    return kwargs


def make_dataset(args, domain_idx, domain_value, num_samples, seed):
    return build_dataset(
        "ofdm",
        num_samples=num_samples,
        seed=seed,
        **domain_kwargs(args, domain_value),
    )


def make_loader(args, dataset, device, shuffle):
    return build_loader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.num_workers,
        device=device,
    )


def load_model(args, device):
    model = build_model_from_args(args, bits_per_symbol=2).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=True)
    state = ckpt["model_state"] if isinstance(ckpt, dict) and "model_state" in ckpt else ckpt
    model.load_state_dict(state)
    return model, ckpt if isinstance(ckpt, dict) else {}


def apply_freeze_mode(model, freeze_mode):
    for param in model.parameters():
        param.requires_grad = True

    if freeze_mode == "none":
        return

    for param in model.parameters():
        param.requires_grad = False

    if freeze_mode == "llr_head":
        for name, param in model.named_parameters():
            if name.startswith("llr_head."):
                param.requires_grad = True
        return

    raise ValueError(f"Unknown freeze_mode: {freeze_mode}")


def count_trainable_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def build_optimizer(args, model):
    params = [p for p in model.parameters() if p.requires_grad]
    if not params:
        raise ValueError("No trainable parameters. Check freeze_mode.")
    return torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def build_adapt_dataset(args, domain_values, domain_idx, epoch):
    current_seed = args.adapt_seed + domain_idx * args.seed_stride
    if args.resample_each_epoch:
        current_seed += epoch * args.num_adapt

    datasets = [
        make_dataset(
            args,
            domain_idx=domain_idx,
            domain_value=domain_values[domain_idx],
            num_samples=args.num_adapt,
            seed=current_seed,
        )
    ]

    if args.replay_per_old_domain > 0:
        for old_idx in range(0, domain_idx):
            replay_seed = (
                args.replay_seed
                + domain_idx * args.seed_stride
                + old_idx * args.replay_per_old_domain
            )
            if args.resample_each_epoch:
                replay_seed += epoch * args.replay_per_old_domain
            datasets.append(
                make_dataset(
                    args,
                    domain_idx=old_idx,
                    domain_value=domain_values[old_idx],
                    num_samples=args.replay_per_old_domain,
                    seed=replay_seed,
                )
            )

    if len(datasets) == 1:
        return datasets[0]
    return ConcatDataset(datasets)


@torch.no_grad()
def evaluate_seen_domains(args, model, device, domain_values, seen_until, stage, after_domain_idx):
    rows = []
    eval_until = len(domain_values) - 1 if args.eval_all_domains else seen_until

    for eval_idx in range(0, eval_until + 1):
        seed = args.val_seed + eval_idx * args.seed_stride
        dataset = make_dataset(
            args,
            domain_idx=eval_idx,
            domain_value=domain_values[eval_idx],
            num_samples=args.num_val,
            seed=seed,
        )
        loader = make_loader(args, dataset, device, shuffle=False)
        val_loss, val_ber = evaluate(model, loader, device)

        rows.append(
            {
                "stage": stage,
                "after_domain_idx": after_domain_idx,
                "after_domain": "source" if after_domain_idx < 0 else domain_name(args.drift_type, domain_values[after_domain_idx]),
                "eval_domain_idx": eval_idx,
                "eval_domain": domain_name(args.drift_type, domain_values[eval_idx]),
                "drift_type": args.drift_type,
                "eval_domain_value": domain_values[eval_idx],
                "freeze_mode": args.freeze_mode,
                "replay_per_old_domain": args.replay_per_old_domain,
                "val_loss": val_loss,
                "val_ber": val_ber,
            }
        )

    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, choices=MODEL_CHOICES)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")

    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--hidden_complex", type=int, default=32)
    parser.add_argument("--zero_complex", type=int, default=32)
    parser.add_argument("--branch_layers", type=int, default=3)
    parser.add_argument("--kernel_size", type=int, default=3)
    parser.add_argument("--no_norm", action="store_true")
    parser.add_argument("--gate_type", type=str, default="swiglu",
                        choices=["sigmoid", "swiglu"])
    parser.add_argument("--single_readout_mode", type=str, default="low_rank",
                        choices=["low_rank", "full"])

    parser.add_argument("--drift_type", type=str, default="delay_spread",
                        choices=["delay_spread", "doppler"])
    parser.add_argument("--domain_values", type=str, default="")
    parser.add_argument("--start_domain", type=int, default=1)
    parser.add_argument("--eval_all_domains", action="store_true")

    parser.add_argument("--num_adapt", type=int, default=256)
    parser.add_argument("--num_val", type=int, default=4000)
    parser.add_argument("--epochs_per_domain", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--log_interval", type=int, default=20)
    parser.add_argument("--freeze_mode", type=str, default="none",
                        choices=["none", "llr_head"])
    parser.add_argument("--reset_optimizer_each_domain", action="store_true")
    parser.add_argument("--resample_each_epoch", action="store_true")
    parser.add_argument("--replay_per_old_domain", type=int, default=0)

    parser.add_argument("--phase_mode", type=str, default="uniform",
                        choices=["fixed", "narrow", "uniform"])
    parser.add_argument("--snr_db_min", type=float, default=-5.0)
    parser.add_argument("--snr_db_max", type=float, default=20.0)
    parser.add_argument("--channel_error_std", type=float, default=0.05)
    parser.add_argument("--h_hat_mode", type=str, default="dmrs_ls_interp",
                        choices=["oracle_noisy", "dmrs_ls_interp"])
    parser.add_argument("--dmrs_freq_spacing", type=int, default=1)
    parser.add_argument("--dmrs_freq_offset", type=int, default=0)
    parser.add_argument("--channel_kind", type=str, default="tdl",
                        choices=["smooth", "iid", "tdl"])
    parser.add_argument("--subcarrier_spacing_hz", type=float, default=30e3)
    parser.add_argument("--num_paths", type=int, default=12)
    parser.add_argument("--rms_delay_spread_s", type=float, default=10e-9)
    parser.add_argument("--max_doppler_hz", type=float, default=200.0)
    parser.add_argument("--rician_k_db", type=optional_float, default=None)

    parser.add_argument("--adapt_seed", type=int, default=200000)
    parser.add_argument("--replay_seed", type=int, default=250000)
    parser.add_argument("--val_seed", type=int, default=300000)
    parser.add_argument("--seed_stride", type=int, default=100000)

    args = parser.parse_args()

    domain_values = parse_float_list(args.domain_values) if args.domain_values else default_domain_values(args.drift_type)
    if not (0 <= args.start_domain < len(domain_values)):
        raise ValueError("--start_domain must select one of the configured domains.")

    device = torch.device(args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu")
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    model, source_ckpt = load_model(args, device)
    apply_freeze_mode(model, args.freeze_mode)
    optimizer = build_optimizer(args, model)

    trainable_params = count_trainable_params(model)
    eval_rows = []
    train_rows = []

    print(f"Device: {device}")
    print(f"Model: {args.model}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Drift type: {args.drift_type}")
    print("Domains: " + ", ".join(domain_name(args.drift_type, v) for v in domain_values))
    print(f"Start adaptation at domain index: {args.start_domain}")
    print(f"Freeze mode: {args.freeze_mode} | trainable params: {trainable_params}")
    print(f"Replay per old domain: {args.replay_per_old_domain}")

    initial_seen = max(0, args.start_domain - 1)
    print("\nInitial evaluation")
    rows = evaluate_seen_domains(
        args,
        model,
        device,
        domain_values,
        seen_until=initial_seen,
        stage=0,
        after_domain_idx=-1,
    )
    eval_rows.extend(rows)
    for row in rows:
        print(f"  eval {row['eval_domain']:>14s} | BCE {row['val_loss']:.6f} | BER {row['val_ber']:.6e}")

    for domain_idx in range(args.start_domain, len(domain_values)):
        if args.reset_optimizer_each_domain:
            optimizer = build_optimizer(args, model)

        print(f"\nAdapting to domain {domain_idx}: {domain_name(args.drift_type, domain_values[domain_idx])}")
        for epoch in range(1, args.epochs_per_domain + 1):
            dataset = build_adapt_dataset(args, domain_values, domain_idx, epoch)
            loader = make_loader(args, dataset, device, shuffle=True)
            train_loss, train_ber = train_one_epoch(
                model,
                loader,
                optimizer,
                device,
                args.log_interval,
            )

            print(
                f"  epoch {epoch:03d}/{args.epochs_per_domain} | "
                f"adapt loss {train_loss:.5f} | adapt BER {train_ber:.5f}"
            )
            train_rows.append(
                {
                    "adapt_domain_idx": domain_idx,
                    "adapt_domain": domain_name(args.drift_type, domain_values[domain_idx]),
                    "drift_type": args.drift_type,
                    "adapt_domain_value": domain_values[domain_idx],
                    "epoch": epoch,
                    "freeze_mode": args.freeze_mode,
                    "replay_per_old_domain": args.replay_per_old_domain,
                    "train_loss": train_loss,
                    "train_ber": train_ber,
                }
            )

        stage = domain_idx - args.start_domain + 1
        rows = evaluate_seen_domains(
            args,
            model,
            device,
            domain_values,
            seen_until=domain_idx,
            stage=stage,
            after_domain_idx=domain_idx,
        )
        eval_rows.extend(rows)
        for row in rows:
            print(f"  eval {row['eval_domain']:>14s} | BCE {row['val_loss']:.6f} | BER {row['val_ber']:.6e}")

        save_checkpoint(
            save_dir / f"after_domain_{domain_idx}.pt",
            model,
            optimizer=optimizer,
            args=args,
            source_args=source_ckpt.get("args", {}),
            drift_type=args.drift_type,
            domain_values=domain_values,
            after_domain_idx=domain_idx,
        )
        save_checkpoint(
            save_dir / "last.pt",
            model,
            optimizer=optimizer,
            args=args,
            source_args=source_ckpt.get("args", {}),
            drift_type=args.drift_type,
            domain_values=domain_values,
            after_domain_idx=domain_idx,
        )

        write_csv(save_dir / "continual_eval.csv", eval_rows)
        write_csv(save_dir / "continual_train.csv", train_rows)

    write_csv(save_dir / "continual_eval.csv", eval_rows)
    write_csv(save_dir / "continual_train.csv", train_rows)
    print(f"\nSaved continual learning results to {save_dir}")


if __name__ == "__main__":
    main()
