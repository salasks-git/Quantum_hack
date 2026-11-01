# AGENTS.md — Autonomous Agent Setup & Architecture Instructions

## 1. Project Context
- **Event:** Qiskit Fall Fest 2026 · Quantum Hack Day Kochi (Sunday, Nov 1, 2026)
- **Track:** Challenge 02 — Quantum Machine Learning (QML) for Healthcare Disease Classification
- **Goal:** Build a reproducible hybrid quantum-classical diagnostic pipeline comparing Quantum Kernel SVMs (`QSVC`), Variational Quantum Classifiers (`VQC`), and Classical Baselines (`SVC`) across clinical datasets.

---

## 2. Environment Setup Commands (Execute First)
When initializing or setting up this repository on a teammate's machine, execute the following commands sequentially:

```bash
# 1. Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# 2. Install pinned Qiskit 1.x and ML dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 3. Verify or populate local offline clinical datasets in ./data/
python3 scripts/download_datasets.py

# 4. Launch the Streamlit application
streamlit run app.py
```

---

## 3. Commit & Push So Your Teammates Can Pull Right Now

Once you've run the two blocks above (and saved `app.py`), push the setup to GitHub so your teammates and their agents can pull it immediately:

```bash
git add .
git commit -m "chore: add Qiskit 1.x setup, offline clinical datasets, and AGENTS.md"
git push origin main
```
