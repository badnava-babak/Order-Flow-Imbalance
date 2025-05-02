"""
I/O utilities for the OFI feature project.
------------------------------------------

Only minimal cleaning is performed here: time‑sorting and timestamp
conversion.  Heavier processing lives in ``ofi.py`` and friends so that
each layer is testable in isolation.
"""
from __future__ import annotations

from pathlib import Path
import pandas as pd


def load_raw(path: str | Path) -> pd.DataFrame:
    """
    Load an ITCH‑style CSV and perform basic cleaning.

    Parameters
    ----------
    path :
        Path to the raw CSV file.

    Returns
    -------
    pandas.DataFrame
        Cleaned frame sorted by ``ts_event`` and with that column converted
        to ``datetime64[ns]``.  *No* index is set here—down‑stream code
        decides how to index.
    """
    df = (pd.read_csv(path)
          .sort_values("ts_event")
          .reset_index(drop=True))
    df["ts_event"] = pd.to_datetime(df["ts_event"], unit="ns")
    return df
