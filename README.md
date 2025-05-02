# OFI Feature Construction—Blockhouse Capital Work‑Trial  
Babak Badnava · May 2025

This repository contains **clean, modular Python code** for constructing  
Order‑Flow Imbalance (OFI) features exactly as defined in **Cont, Cucuringu & Zhang (2023)**:

* **Per‑message OF / OFI** for the top 10 limit‑order‑book levels  
* **Bar‑level aggregation & depth normalisation** (Eq. 3)  
* **Integrated OFI** via PCA (Eq. 4)  
* **Multi‑asset alignment & sparsity‑ready design matrix**

A short LaTeX PDF in `/doc/` answers the three conceptual questions requested in the task.

---

## Quick start

```bash
# clone and create a fresh env
git clone https://github.com/badnava-babak/Order-Flow-Imbalance.git
cd Order-Flow-Imbalance
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# build single‑asset features and plot the correlation heat‑map
python main.py --csv data/first_25000_rows.csv --symbol AAPL --freq 1min
# outputs:
#   AAPL_ofi_1min.parquet
#   AAPL_corr.png

# build a toy multi‑asset matrix (see §Dataset note below)
python main.py --csv data/multi-asset-dataset.csv --symbols AAPL,JPM
```
A fully‑worked Jupyter notebook lives in /notebooks/reproduce.ipynb.

## Project layout
```
ofi-features/
├─ src/                  # pure‑python modules
│  ├─ data_loader.py     # CSV ↔ DataFrame helpers
│  ├─ ofi.py             # OF / OFI construction + bar aggregation
│  └─ plots.py           # Matplotlib visuals
├─ main.py               # command‑line entry‑point (runs end‑to‑end)
├─ notebooks/            # interactive demo
├─ data/                 # sample raw CSV(s)
├─ ofi-data/             # saved OFI data
├─ doc/                  # A short LaTeX PDF
├─ requirements.txt
└─ README.md
```
Each ```src/``` function is fully typed and documented; there is no global state, so the code can be unit‑tested in isolation.

## Dataset note

The original CSV supplied with the task contains trade & quote messages only for ```AAPL```.
To demonstrate multi‑asset functionality the notebook:

* Duplicates AAPL’s raw rows and
* Re‑labels the symbol column to JPM.

This synthetic step is clearly flagged in the notebook and in comments; no statistical conclusions are drawn from the duplicated data.
When you supply genuine multi‑asset data, the same code paths will work unmodified.


## Reproducibility & licence

* All dependencies are pinned in ```requirements.txt``` (```pandas```, ```numpy```, ```scikit‑learn```, ```matplotlib```).
* Running ```python main.py``` or executing the notebook end‑to‑end creates every intermediate artefact in a deterministic manner.
* Code is MIT‑licensed; feel free to reuse or extend.

Questions? Open an issue or reach me at ```babak.badnava@gmail.com```.