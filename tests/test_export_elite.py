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


@pytest.mark.parametrize("slug", [Path(c).stem for c in DATASETS.values()])
def test_elite_quhama_equals_quhama_json(slug):
    """Scene 8 shows ONE set of QuHaMA numbers: elite.json reuses the <slug>.quhama.json run."""
    e = json.loads((DATA / f"{slug}.elite.json").read_text())
    q = json.loads((DATA / f"{slug}.quhama.json").read_text())
    names = {"accuracy": "Accuracy", "f2": "F2-Score", "precision": "Precision", "recall": "Recall", "roc_auc": "ROC-AUC"}
    m = e["approaches"]["quhama"]["metrics"]
    row = next(r for r in e["leaderboard"] if r["key"] == "quhama")
    for k, qk in names.items():
        assert abs(m[k] - q["metrics"][qk]) < 1e-5, (slug, k)
        assert row[k] == m[k]
    t_e, t_q = e["approaches"]["quhama"]["test"], q["test"]
    assert [r["pred"] for r in t_e] == [r["pred"] for r in t_q]
    assert all(abs(a["prob"] - b["p_final"]) < 1e-9 for a, b in zip(t_e, t_q))
    assert e["approaches"]["quhama"]["architecture"]["epochs_encoder"] == q["settings"]["epochs_encoder"]


def test_quhama_section_reuses_export_without_training(monkeypatch):
    import export_tour_elite as ete
    monkeypatch.setattr(ete, "export_quhama", lambda *a, **k: (_ for _ in ()).throw(AssertionError("retrained")))
    q = json.loads((DATA / f"{SLUG}.quhama.json").read_text())
    sec = ete.quhama_section(q, [r["label"] for r in q["test"]])
    assert sec["metrics"]["f2"] == pytest.approx(q["metrics"]["F2-Score"], abs=1e-5)
    assert ete.load_quhama(DATA / f"{SLUG}.csv", None, DATA)["slug"] == SLUG
