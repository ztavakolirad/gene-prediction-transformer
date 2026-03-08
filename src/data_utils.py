from __future__ import annotations

import os
import random
import zipfile
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from Bio import SeqIO


GFF3_COLUMNS = [
    "seq_id",
    "source",
    "type",
    "start",
    "end",
    "score",
    "strand",
    "phase",
    "attributes",
]

CODING_FEATURES = {"CDS", "exon", "start_codon", "stop_codon"}
DNA_ENCODING = {"A": 0, "C": 1, "G": 2, "T": 3, "N": 4}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def parse_gff3(gff3_file: str | os.PathLike[str]) -> pd.DataFrame:
    records: list[dict] = []
    with open(gff3_file, "r", encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("#") or line.startswith("seq_id"):
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) != 9:
                continue
            try:
                start = int(parts[3])
                end = int(parts[4])
            except ValueError:
                continue
            records.append(
                {
                    "seq_id": parts[0],
                    "source": parts[1],
                    "type": parts[2],
                    "start": start,
                    "end": end,
                    "score": parts[5],
                    "strand": parts[6],
                    "phase": parts[7],
                    "attributes": parts[8],
                }
            )
    return pd.DataFrame(records, columns=GFF3_COLUMNS)


def parse_fasta(fasta_file: str | os.PathLike[str]) -> dict[str, str]:
    sequences: dict[str, str] = {}
    for record in SeqIO.parse(str(fasta_file), "fasta"):
        sequences[record.id] = str(record.seq).upper()
    return sequences


def encode_sequence(sequence: str) -> list[int]:
    return [DNA_ENCODING.get(base, 4) for base in sequence]


def extract_zip(zip_path: str | os.PathLike[str], output_dir: str | os.PathLike[str]) -> Path:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(output)
    return output


def merge_text_files(file_list: Iterable[str | os.PathLike[str]], output_file: str | os.PathLike[str]) -> None:
    file_list = list(file_list)
    with open(output_file, "w", encoding="utf-8") as out:
        for idx, fname in enumerate(file_list):
            with open(fname, "r", encoding="utf-8") as src:
                out.write(src.read())
            if idx != len(file_list) - 1:
                out.write("\n")


def merge_genome_files(input_dir: str | os.PathLike[str], output_dir: str | os.PathLike[str]) -> tuple[Path, Path]:
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    gff3_files = sorted(input_path.glob("*.gff3"))
    fasta_files = sorted(input_path.glob("*.fasta"))

    if not gff3_files:
        raise FileNotFoundError(f"No .gff3 files found in {input_path}")
    if not fasta_files:
        raise FileNotFoundError(f"No .fasta files found in {input_path}")

    merged_gff3 = output_path / "merged.gff3"
    merged_fasta = output_path / "merged.fasta"

    merge_text_files(gff3_files, merged_gff3)
    merge_text_files(fasta_files, merged_fasta)

    return merged_gff3, merged_fasta


def generate_windows_and_labels(
    sequences: dict[str, str],
    gff3_df: pd.DataFrame,
    window_size: int = 1000,
    stride: int | None = None,
    coding_features: set[str] | None = None,
) -> tuple[list[str], list[int]]:
    if stride is None:
        stride = window_size
    coding_features = coding_features or CODING_FEATURES

    features: list[str] = []
    labels: list[int] = []

    coding_df = gff3_df[gff3_df["type"].isin(coding_features)].copy()

    for seq_id, sequence in sequences.items():
        seq_len = len(sequence)
        if seq_len < window_size:
            continue

        ann = coding_df[coding_df["seq_id"] == seq_id][["start", "end"]]
        ann_values = ann.to_numpy(dtype=np.int64)

        # start0/end0 are 0-based [start0, end0) for Python slicing.
        # GFF3 is 1-based inclusive, so we compare against [start0+1, end0].
        for start0 in range(0, seq_len - window_size + 1, stride):
            end0 = start0 + window_size
            window_seq = sequence[start0:end0]

            w_start_1b = start0 + 1
            w_end_1b = end0

            is_coding = 0
            if ann_values.size > 0:
                overlaps = (ann_values[:, 0] <= w_end_1b) & (ann_values[:, 1] >= w_start_1b)
                is_coding = int(np.any(overlaps))

            features.append(window_seq)
            labels.append(is_coding)

    return features, labels
