from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score


# ====== Dataset config ======
CSV_FILENAME = "114_commands_double_labeled_37_classes_counted.csv"

# ====== Your highlight: custom token pattern ======
# Designed for command-lines: URLs, IPs, paths, flags, env vars, etc.
CUSTOM_TOKEN_PATTERN = (
    r"(?u)"
    r"(?:https?://[^\s]+)"                                  # URLs
    r"|(?:\b(?:\d{1,3}\.){3}\d{1,3}\b)"                     # IPv4
    r"|(?:[A-Za-z]:\\[^\s]+)"                               # Windows paths
    r"|(?:/(?:[^ \t\n\r\f\v/]+/)*[^ \t\n\r\f\v]+)"          # Unix paths
    r"|(?:--?[\w-]+(?:=[^\s]+)?)"                           # flags: -a, --color=auto
    r"|(?:\$\{?\w+\}?)"                                     # env vars: $HOME, ${PATH}
    r"|(?:0x[0-9a-fA-F]+)"                                  # hex
    r"|(?:[A-Za-z_]\w+)"                                    # words
    r"|(?:\d+)"                                             # numbers
)


@dataclass(frozen=True)
class Paths:
    root: Path
    raw_csv: Path
    processed_dir: Path
    df_final_csv: Path
    models_dir: Path

def get_paths() -> Paths:
    # command_classifier/common.py -> repo root is two levels up
    root = Path(__file__).resolve().parents[1]
    return Paths(
        root=root,
        raw_csv=root / "data" / "raw" / CSV_FILENAME,
        processed_dir=root / "data" / "processed",
        df_final_csv=root / "data" / "processed" / "df_final.csv",
        models_dir=root / "models",
    )

def ensure_columns(df: pd.DataFrame) -> None:
    needed = {"proctitle", "technique_grouped", "count"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"Dataset missing columns: {sorted(missing)}. Found: {list(df.columns)}")

def build_df_final(df_raw: pd.DataFrame) -> pd.DataFrame:
    # keep only what we need (drop any extra)
    df = df_raw.copy()

    # Some versions include "technique"; we don't need it.
    if "technique" in df.columns:
        df = df.drop(columns=["technique"])

    ensure_columns(df)

    df["proctitle"] = df["proctitle"].fillna("").astype(str)
    df["technique_grouped"] = df["technique_grouped"].fillna("").astype(str)
    df["count"] = pd.to_numeric(df["count"], errors="coerce").fillna(0).astype(int)

    # Merge duplicates by summing count
    df_final = (
        df.groupby(["proctitle", "technique_grouped"], as_index=False)["count"]
          .sum()
    )
    return df_final

def load_df_final(prefer_processed: bool = True, save_processed: bool = True) -> pd.DataFrame:
    p = get_paths()

    if prefer_processed and p.df_final_csv.exists():
        df_final = pd.read_csv(p.df_final_csv)
        ensure_columns(df_final)
        return df_final

    if not p.raw_csv.exists():
        raise FileNotFoundError(
            f"Raw dataset not found: {p.raw_csv}\n"
            f"Expected file name: {CSV_FILENAME}\n"
            f"Put it in: data/raw/"
        )

    df_raw = pd.read_csv(p.raw_csv)
    df_final = build_df_final(df_raw)

    if save_processed:
        p.processed_dir.mkdir(parents=True, exist_ok=True)
        df_final.to_csv(p.df_final_csv, index=False)

    return df_final

def split_60_20_20(df_final: pd.DataFrame, random_state: int = 42):
    ensure_columns(df_final)

    X = df_final["proctitle"]
    y = df_final["technique_grouped"]
    w = df_final["count"]

    # 60/40 then 20/20
    X_train, X_tmp, y_train, y_tmp, w_train, w_tmp = train_test_split(
        X, y, w, test_size=0.4, random_state=random_state, stratify=y
    )
    X_val, X_test, y_val, y_test, w_val, w_test = train_test_split(
        X_tmp, y_tmp, w_tmp, test_size=0.5, random_state=random_state, stratify=y_tmp
    )
    return X_train, X_val, X_test, y_train, y_val, y_test, w_train, w_val, w_test

def weighted_scores(y_true, y_pred, sample_weight):
    acc = accuracy_score(y_true, y_pred, sample_weight=sample_weight)
    f1w = f1_score(y_true, y_pred, average="weighted", sample_weight=sample_weight)
    return acc, f1w
