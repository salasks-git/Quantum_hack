"""Elite export (Streamlit Tab 8 parity) on vertebral_column. Run with: pytest tests/test_export_elite.py"""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from export_tour_elite import export_dataset, metrics_of  # noqa: E402
from qmed.config import DATASETS  # noqa: E402

SLUG = "vertebral_column"
DATA = ROOT / "web/public/data"


@pytest.fixture(scope="module")
def elite():
    path = DATA / f"{SLUG}.elite.json"
    if path.exists():
        return json.loads(path.read_text())
    label, csv = next((l, c) for l, c in DATASETS.items() if Path(c).stem == SLUG)
    return json.loads(json.dumps(export_dataset(label, csv)))


def test_both_approaches(elite):
    assert {"sandwich", "ckta"} <= set(elite["approaches"])
    for a in ("sandwich", "ckta"):
        assert elite["approaches"][a]["description"]


def test_test_length_matches_main_json(elite):
    main = json.loads((DATA / f"{SLUG}.json").read_text())
    for a in elite["approaches"].values():
        assert len(a["test"]) == len(main["test"])
        assert [t["label"] for t in a["test"]] == [t["label"] for t in main["test"]]


def test_probs_in_unit_interval(elite):
    for a in elite["approaches"].values():
        assert all(0.0 <= t["prob"] <= 1.0 for t in a["test"])


def test_leaderboard_rows(elite):
    names = {r["model"] for r in elite["leaderboard"]}
    assert {"QuHaMA", "Sandwich", "CKTA"} <= names


def test_metrics_match_test_preds(elite):
    key = {"QuHaMA": "quhama", "Sandwich": "sandwich", "CKTA": "ckta"}
    for r in elite["leaderboard"]:
        if r["model"] not in key:
            continue
        t = elite["approaches"][key[r["model"]]]["test"]
        m = metrics_of([x["label"] for x in t], np.array([x["pred"] for x in t]),
                       np.array([x["prob"] for x in t]))
        for k, v in m.items():
            assert abs(v - r[k]) < 1e-6, (r["model"], k)
