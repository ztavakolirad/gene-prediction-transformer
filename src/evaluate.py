from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
)
from torch.utils.data import DataLoader, Dataset

from model import GenePredictionTransformer


class DNASequenceDataset(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray) -> None:
        self.x = torch.tensor(x, dtype=torch.long)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int):
        return self.x[idx], self.y[idx]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained model on saved test split.")
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--test-split", type=str, required=True, help="Path to test_split.npz")
    parser.add_argument("--batch-size", type=int, default=64)

    parser.add_argument("--embed-dim", type=int, default=256)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)

    parser.add_argument("--output-json", type=str, default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data = np.load(args.test_split)
    x = data["X"]
    y = data["y"]

    ds = DNASequenceDataset(x, y)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GenePredictionTransformer(
        embed_dim=args.embed_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        dropout=args.dropout,
    ).to(device)

    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()

    y_true = []
    y_pred = []
    y_scores = []

    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            logits = model(inputs)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            preds = np.argmax(probs, axis=1)
            y_pred.extend(preds.tolist())
            y_true.extend(labels.numpy().tolist())
            y_scores.extend(probs[:, 1].tolist())

    accuracy = accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred, target_names=["Non-coding", "Coding"], output_dict=True)
    cm = confusion_matrix(y_true, y_pred).tolist()

    result = {
        "accuracy": accuracy,
        "classification_report": report,
        "confusion_matrix": cm,
    }

    print(f"Accuracy: {accuracy:.4f}")
    print(json.dumps(result, indent=2))

    # Save plots next to output-json (or near model path if output-json is empty)
    if args.output_json:
        base_dir = Path(args.output_json).parent
    else:
        base_dir = Path(args.model_path).parent
    base_dir.mkdir(parents=True, exist_ok=True)

    cm_arr = np.array(cm)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm_arr, cmap="Blues")
    for (r, c), v in np.ndenumerate(cm_arr):
        ax.text(c, r, str(v), ha="center", va="center")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Non-coding", "Coding"])
    ax.set_yticklabels(["Non-coding", "Coding"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(base_dir / "confusion_matrix.png", dpi=180)
    plt.close(fig)

    precision, recall, _ = precision_recall_curve(np.array(y_true), np.array(y_scores))
    ap = average_precision_score(np.array(y_true), np.array(y_scores))
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(recall, precision, label=f"AP={ap:.3f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(base_dir / "precision_recall_curve.png", dpi=180)
    plt.close(fig)

    if args.output_json:
        output = Path(args.output_json)
        output.parent.mkdir(parents=True, exist_ok=True)
        with open(output, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()
