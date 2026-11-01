"""Export each tour dataset's raw CSV as web/public/data/<slug>.dataset.json (CSV only, no training)."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "web" / "public" / "data"


def build_payload(slug: str, label: str, csv_path: Path) -> dict:
    df = pd.read_csv(csv_path)
    n_rows, n_cols = df.shape
    has_target = "target" in df.columns
    n_features = n_cols - 1 if has_target else n_cols
    positives = int(df["target"].sum()) if has_target else 0
    desc = df.describe().T.round(4)
    stats = [
        {
            "column": str(col),
            "count": float(r["count"]), "mean": float(r["mean"]), "std": float(r["std"]),
            "min": float(r["min"]), "p25": float(r["25%"]), "p50": float(r["50%"]),
            "p75": float(r["75%"]), "max": float(r["max"]),
        }
        for col, r in desc.iterrows()
    ]
    rows = []
    for rec in df.itertuples(index=False):
        row = []
        for col, v in zip(df.columns, rec):
            row.append(int(v) if col == "target" else round(float(v), 4))
        rows.append(row)
    return {
        "slug": slug, "label": label, "columns": [str(c) for c in df.columns], "rows": rows,
        "stats": stats, "n_rows": int(n_rows), "n_features": int(n_features),
        "positives": positives, "negatives": int(n_rows - positives),
    }


def main() -> None:
    index = json.loads((OUT_DIR / "index.json").read_text())
    for item in index:
        slug = item["slug"]
        payload = build_payload(slug, item["label"], DATA_DIR / f"{slug}.csv")
        out = OUT_DIR / f"{slug}.dataset.json"
        out.write_text(json.dumps(payload, separators=(",", ":"), allow_nan=False))
        print(f"{out.name}: {out.stat().st_size} bytes")


if __name__ == "__main__":
    main()
