import argparse
from pathlib import Path

import torch

from data import DATASET_CHOICES, build_dataset, build_loader, ofdm_kwargs_from_args
from engine import evaluate, move_batch, save_checkpoint, train_one_epoch
from models import MODEL_CHOICES, build_model, build_model_from_args
from utils.seed import set_seed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="phase_invariant",
                        choices=MODEL_CHOICES)
    parser.add_argument("--dataset_type", type=str, default="ofdm",
                        choices=DATASET_CHOICES)

    parser.add_argument("--train_phase_mode", type=str, default="fixed",
                        choices=["fixed", "narrow", "uniform"])
    parser.add_argument("--val_phase_mode", type=str, default="uniform",
                        choices=["fixed", "narrow", "uniform"])

    parser.add_argument("--num_train", type=int, default=10000)
    parser.add_argument("--num_val", type=int, default=2000)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=64)

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
    parser.add_argument("--rician_k_db", type=float, default=None)

    parser.add_argument("--hidden", type=int, default=32)
    parser.add_argument("--hidden_complex", type=int, default=16)
    parser.add_argument("--zero_complex", type=int, default=16)
    parser.add_argument("--branch_layers", type=int, default=2)
    parser.add_argument("--kernel_size", type=int, default=3)
    parser.add_argument("--no_norm", action="store_true")
    parser.add_argument("--gate_type", type=str, default="swiglu",
                        choices=["sigmoid", "swiglu"])
    parser.add_argument("--single_readout_mode", type=str, default="low_rank",
                        choices=["low_rank", "full"])

    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--log_interval", type=int, default=50)

    parser.add_argument("--save_dir", type=str, default="runs/debug/")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--train_seed", type=int, default=None)
    parser.add_argument("--val_seed", type=int, default=None)
    parser.add_argument("--deterministic", action="store_true")

    args = parser.parse_args()
    set_seed(args.seed, deterministic=args.deterministic)

    device = torch.device(args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu")
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    train_set = build_dataset(
        args.dataset_type,
        **ofdm_kwargs_from_args(
            args,
            num_samples=args.num_train,
            phase_mode=args.train_phase_mode,
            seed=args.train_seed if args.train_seed is not None else args.seed,
        ),
    )
    val_set = build_dataset(
        args.dataset_type,
        **ofdm_kwargs_from_args(
            args,
            num_samples=args.num_val,
            phase_mode=args.val_phase_mode,
            seed=args.val_seed if args.val_seed is not None else args.seed + 100000,
        ),
    )

    train_loader = build_loader(
        train_set, args.batch_size, shuffle=True, num_workers=args.num_workers, device=device
    )
    val_loader = build_loader(
        val_set, args.batch_size, shuffle=False, num_workers=args.num_workers, device=device
    )

    model = build_model_from_args(args, bits_per_symbol=2).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    best_val_loss = float("inf")

    print(f"Device: {device}")
    print(f"Model: {args.model}")
    print(f"Dataset: {args.dataset_type} | Seed: {args.seed}")
    print(f"Train phase mode: {args.train_phase_mode} | Val phase mode: {args.val_phase_mode}")

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")

        train_loss, train_ber = train_one_epoch(
            model, train_loader, optimizer, device, args.log_interval
        )

        val_loss, val_ber = evaluate(model, val_loader, device)

        print(
            f"Epoch {epoch:03d} | "
            f"train loss {train_loss:.5f} | train BER {train_ber:.5f} | "
            f"val loss {val_loss:.5f} | val BER {val_ber:.5f}"
        )

        save_checkpoint(save_dir / "last.pt", model, args=args)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(save_dir / "best.pt", model, args=args, best_val_loss=best_val_loss)
            print(f"  saved best checkpoint to {save_dir / 'best.pt'}")


if __name__ == "__main__":
    main()
