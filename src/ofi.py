"""
Core functions for Order‑Flow & Order‑Flow Imbalance (OFI) construction.
"""
from __future__ import annotations
from typing import Final, Literal, Sequence
from collections.abc import Iterable

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

# ----------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------
#: Depth levels supported by the paper.  Adjust if you need fewer/more.
LEVELS: Final[Sequence[int]] = tuple(range(10))


# ----------------------------------------------------------------------
# Private helpers
# ----------------------------------------------------------------------
def _side_flow(
        df: pd.DataFrame, side: Literal["bid", "ask"], level: int
) -> pd.Series:
    """
    Compute per‑message signed order‑flow for one side/level.

    Implements the three‑branch rule in Eq.(1) of the paper.

    Parameters
    ----------
    df :
        Frame containing *at least* ``{side}_px_<level>`` and
        ``{side}_sz_<level>``.
    side :
        ``"bid"`` or ``"ask"``.
    level :
        Book depth(0= inside quote).

    Returns
    -------
    pandas.Series
        Signed size change at each message; ``index`` matches ``df``.
    """

    p = df[f"{side}_px_{level:02d}"]
    q = df[f"{side}_sz_{level:02d}"]
    dp = p.diff()
    dq = q.diff()

    if side == "bid":
        of = np.where(dp > 0, q,  # price improved
                      np.where(dp == 0, dq,  # same price, size change
                               -q))  # price deteriorated
    else:  # 'ask'
        of = np.where(dp < 0, -q,  # price improved (ask ↓)
                      np.where(dp == 0, dq,
                               q))  # price deteriorated (ask ↑)
    return pd.Series(of, index=df.index)


# ----------------------------------------------------------------------
# Public API
# ----------------------------------------------------------------------
def add_level_ofi(df: pd.DataFrame) -> pd.DataFrame:
    """
    Append per‑message OF / OFI columns for all levels.

    The original frame is **not** modified; a copy is returned.

    Returns
    -------
    pandas.DataFrame
        Input frame plus:

        * ``of_bid_<level>``, ``of_ask_<level>``
        * ``ofi_<level>``
    """
    out = df.copy()
    for m in LEVELS:
        out[f"of_bid_{m:02d}"] = _side_flow(out, "bid", m)
        out[f"of_ask_{m:02d}"] = _side_flow(out, "ask", m)
        out[f"ofi_{m:02d}"] = out[f"of_bid_{m:02d}"] - out[f"of_ask_{m:02d}"]
    return out


def aggregate_ofi(
        df: pd.DataFrame,
        freq: str = "1min",
        *,
        normalise: bool = True,
) -> pd.DataFrame:
    """
    Aggregate per‑message OFI into physical‑time bars.

    Parameters
    ----------
    df :
        Frame *already* containing ``ofi_<level>`` columns and a
        ``ts_event`` datetime column.
    freq :
        Resampling frequency (“1min”, “5s”, …).
    normalise :
        If ``True`` divide by average top‑10 depth (paper Eq.3).

    Returns
    -------
    pandas.DataFrame
        Bar‑level multi‑column DataFrame indexed by bar end‑time.
    """
    ofi_cols = [f"ofi_{m:02d}" for m in LEVELS]
    bar = (
        df.set_index("ts_event")[ofi_cols]
        .resample(freq)
        .sum(min_count=1)
        .dropna()
    )

    if not normalise:
        return bar

    depth_cols = [f"depth_{m:02d}" for m in LEVELS]
    for m in LEVELS:
        # Calculate sum of order quantities for all rows
        df[f'depth_{m:02d}'] = df[f"bid_sz_{m:02d}"] + df[f"ask_sz_{m:02d}"]

    # Aggregate over 1min time intervals as Q in Eq. 3 in the paper (This only calculates the inner summation)
    depth = df.set_index("ts_event")[depth_cols].resample(freq).mean().dropna()
    # Take average over all levels
    depth_bar = depth.mean(axis=1) / 2

    return bar.div(depth_bar, axis=0)


def integrated_ofi(ofi_bar: pd.DataFrame, n_components: int) -> pd.Series:
    """
    Compute Integrated OFI (first PC, L¹‑normalised) as in Eq.(4).

    Parameters
    ----------
    ofi_bar :
        Bar‑level DataFrame with columns ``ofi_<level>``.

    Returns
    -------
    pandas.Series
        One value per bar.
    """
    pca = PCA(n_components=n_components, svd_solver="full").fit(ofi_bar)
    w = pca.components_[0]
    w /= np.sum(np.abs(w))  # L-1 normalisation
    return ofi_bar @ w


# ----------------------------------------------------------------------
# Multi‑asset convenience
# ----------------------------------------------------------------------


def multi_asset_ofi(
        raw_df: pd.DataFrame,
        symbols: Iterable[str],
        *,
        freq: str = "1min",
        normalise: bool = True,
) -> pd.DataFrame:
    """
    Build bar‑level OFI for several symbols and align them on time.

    Parameters
    ----------
    raw_df :
        Raw ITCH‑style DataFrame containing all symbols.
    symbols :
        Iterable of ticker symbols to include (e.g. ``["AAPL", "MSFT"]``).
    freq :
        Resampling frequency for bar aggregation.
    normalise :
        Whether to apply depth normalisation (paper Eq.3).

    Returns
    -------
    pandas.DataFrame
        Columns have a two‑level Index (symbol, ``ofi_XX``); the index is
        the bar end‑time.  Only timestamps present for *all* symbols are
        retained (inner join), so downstream models have no NaNs.

    Notes
    -----
    *If* one of the symbols is missing from ``raw_df`` a ``KeyError`` is
    raised—the caller can catch this and drop or warn as needed.
    """

    frames: dict[str, pd.DataFrame] = {}
    for sym in symbols:
        sym_df = raw_df[raw_df["symbol"] == sym]
        if sym_df.empty:
            raise KeyError(f"symbol {sym!r} not found in raw_df")
        msg_df = add_level_ofi(sym_df)
        bar_df = aggregate_ofi(msg_df, freq=freq, normalise=normalise)
        frames[sym] = bar_df

    # Align on the bar index; inner join removes missing timestamps
    aligned = pd.concat(frames, axis=1, join="inner")
    return aligned
