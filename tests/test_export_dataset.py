import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from export_dataset_tables import build_payload  # noqa: E402


def test_vertebral_column_payload():
    csv = ROOT / "data" / "vertebral_column.csv"
    p = build_payload("vertebral_column", "Vertebral Column (Orthopedics)", csv)
    assert p["n_rows"] == 310 and len(p["rows"]) == 310
    assert p["n_features"] == 6
    assert p["columns"][-1] == "target" and len(p["columns"]) == 7
    assert p["positives"] + p["negatives"] == p["n_rows"]
    assert all(isinstance(r[-1], int) for r in p["rows"])
    d = pd.read_csv(csv).describe().T.round(4)
    s = next(x for x in p["stats"] if x["column"] == "pelvic_incidence")
    r = d.loc["pelvic_incidence"]
    assert (s["count"], s["mean"], s["std"], s["min"], s["max"]) == (r["count"], r["mean"], r["std"], r["min"], r["max"])
    assert (s["p25"], s["p50"], s["p75"]) == (r["25%"], r["50%"], r["75%"])
