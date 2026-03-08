from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from data_utils import (
    encode_sequence,
    generate_windows_and_labels,
    parse_fasta,
    parse_gff3,
    set_seed,
)
from model import GenePredictionTransformer


class DNASequenceDataset(Dataset):
    def __init__(self, sequences: list[list[int]], labels: list[int]) -> None:
        self.x = [torch.tensor(seq, dtype=torch.long) for seq in sequences]
        self.y = torch.tensor(labels, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int):
        return self.x[idx], self.y[idx]


def run_epoch(model, loader, criterion, device, optimizer=None):
    is_train = optimizer is not None
    model.train(is_train)

    total_loss = 0.0
    preds = []
    labels = []

    context = torch.enable_grad() if is_train else torch.no_grad()
    with context:
        for inputs, y in tqdm(loader, disable=False):
            inputs = inputs.to(device)
            y = y.to(device)

            if is_train:
                optimizer.zero_grad()

            logits = model(inputs)
            loss = criterion(logits, y)

            if is_train:
                loss.backward()
                optimizer.step()

            total_loss += loss.item()
            preds.extend(torch.argmax(logits, dim=1).detach().cpu().tolist())
            labels.extend(y.detach().cpu().tolist())

    avg_loss = total_loss / max(1, len(loader))
    acc = accuracy_score(labels, preds)
    return avg_loss, acc, labels, preds


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train transformer for coding/non-coding prediction.")
    parser.add_argument(
        "--fasta",
        type=str,
        default="data/ergobibamus_cyprinoides_genome.fasta",
        help="Path to FASTA file (default: data/ergobibamus_cyprinoides_genome.fasta)",
    )
    parser.add_argument(
        "--gff3",
        type=str,
        default="data/ergobibamus_cyprinoides_labels.gff3",
        help="Path to GFF3 file (default: data/ergobibamus_cyprinoides_labels.gff3)",
    )
    parser.add_argument("--output-dir", type=str, default="outputs")

    parser.add_argument("--window-size", type=int, default=1000)
    parser.add_argument("--stride", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)

    parser.add_argument("--embed-dim", type=int, default=256)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)

    parser.add_argument("--val-size", type=float, default=0.1)
    parser.add_argument("--test-size", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-samples", type=int, default=0, help="0 means use all windows.")
    parser.add_argument(
        "--early-stop-patience",
        type=int,
        default=3,
        help="Stop if validation accuracy does not improve for this many epochs.",
    )
    parser.add_argument(
        "--early-stop-min-delta",
        type=float,
        default=1e-4,
        help="Minimum validation accuracy improvement to reset early stopping counter.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fasta_path = Path(args.fasta)
    gff3_path = Path(args.gff3)
    if not fasta_path.exists():
        raise FileNotFoundError(
            f"FASTA file not found: {fasta_path}. "
            "Place your file in data/ or pass --fasta <path>."
        )
    if not gff3_path.exists():
        raise FileNotFoundError(
            f"GFF3 file not found: {gff3_path}. "
            "Place your file in data/ or pass --gff3 <path>."
        )

    print("Loading FASTA/GFF3...")
    sequences = parse_fasta(fasta_path)
    gff3_df = parse_gff3(gff3_path)

    print("Generating windows and labels...")
    features, labels = generate_windows_and_labels(
        sequences=sequences,
        gff3_df=gff3_df,
        window_size=args.window_size,
        stride=args.stride,
    )

    if not features:
        raise RuntimeError("No windows were generated. Check input files and window/stride values.")

    if args.max_samples > 0 and args.max_samples < len(features):
        idx = np.random.permutation(len(features))[: args.max_samples]
        features = [features[i] for i in idx]
        labels = [labels[i] for i in idx]

    encoded = [encode_sequence(seq) for seq in features]

    y = np.array(labels)
    stratify = y if len(np.unique(y)) > 1 else None

    holdout = args.val_size + args.test_size
    if not 0.0 < holdout < 1.0:
        raise ValueError("val_size + test_size must be between 0 and 1.")

    x_train, x_holdout, y_train, y_holdout = train_test_split(
        encoded,
        labels,
        test_size=holdout,
        random_state=args.seed,
        stratify=stratify,
    )

    test_fraction_of_holdout = args.test_size / holdout
    holdout_stratify = np.array(y_holdout) if len(set(y_holdout)) > 1 else None
    x_val, x_test, y_val, y_test = train_test_split(
        x_holdout,
        y_holdout,
        test_size=test_fraction_of_holdout,
        random_state=args.seed,
        stratify=holdout_stratify,
    )

    train_ds = DNASequenceDataset(x_train, y_train)
    val_ds = DNASequenceDataset(x_val, y_val)
    test_ds = DNASequenceDataset(x_test, y_test)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GenePredictionTransformer(
        embed_dim=args.embed_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        dropout=args.dropout,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = -1.0
    best_path = output_dir / "best_model.pth"
    epochs_without_improve = 0

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc, _, _ = run_epoch(model, train_loader, criterion, device, optimizer)
        val_loss, val_acc, _, _ = run_epoch(model, val_loader, criterion, device, optimizer=None)

        print(
            f"Epoch {epoch}/{args.epochs} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )

        if val_acc > (best_val_acc + args.early_stop_min_delta):
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_path)
            epochs_without_improve = 0
        else:
            epochs_without_improve += 1
            if epochs_without_improve >= args.early_stop_patience:
                print(
                    f"Early stopping at epoch {epoch}: "
                    f"no val_acc improvement > {args.early_stop_min_delta} "
                    f"for {args.early_stop_patience} consecutive epochs."
                )
                break

    print(f"Loading best checkpoint from: {best_path}")
    model.load_state_dict(torch.load(best_path, map_location=device))

    test_loss, test_acc, y_true, y_pred = run_epoch(model, test_loader, criterion, device, optimizer=None)

    report = classification_report(y_true, y_pred, target_names=["Non-coding", "Coding"], output_dict=True)
    cm = confusion_matrix(y_true, y_pred).tolist()

    metrics = {
        "test_loss": test_loss,
        "test_accuracy": test_acc,
        "classification_report": report,
        "confusion_matrix": cm,
        "num_samples": {
            "train": len(train_ds),
            "val": len(val_ds),
            "test": len(test_ds),
        },
        "config": vars(args),
    }

    with open(output_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    np.savez_compressed(output_dir / "test_split.npz", X=np.array(x_test), y=np.array(y_test))

    with open(output_dir / "label_mapping.json", "w", encoding="utf-8") as f:
        json.dump({"0": "Non-coding", "1": "Coding"}, f, indent=2)

    print("Training complete.")
    print(f"Test accuracy: {test_acc:.4f}")
    print(f"Artifacts saved in: {output_dir}")


if __name__ == "__main__":
    main()
