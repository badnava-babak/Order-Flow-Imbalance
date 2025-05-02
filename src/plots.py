"""
Plotting helpers (Matplotlib only).
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from .ofi import LEVELS


def corr_heatmap(ofi_bar: pd.DataFrame, path: str | Path | None = None) -> None:
    """
    Plot and optionally save the correlation matrix of multi‑level OFI.

    Parameters
    ----------
    ofi_bar :
        Bar‑level DataFrame that includes ``ofi_<level>`` columns.
    path :
        If supplied, PNG is written there (300dpi); otherwise only shown.
    """
    corr = ofi_bar[[f"ofi_{m:02d}" for m in LEVELS]].corr()
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(corr, cmap="coolwarm")
    ax.set_xticks(LEVELS)
    ax.set_yticks(LEVELS)
    ax.set_xticklabels(range(1, 11))
    ax.set_yticklabels(range(1, 11))
    ax.set_title("Correlation Matrix of Multi‑Level OFI")
    fig.colorbar(im, ax=ax, shrink=0.8)
    plt.tight_layout()
    if path is not None:
        fig.savefig(path, dpi=300)
    plt.show()
