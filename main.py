#!/usr/bin/env python
"""
Command‑line entry for OFI feature construction.

Example
-------
>>> python main.py --csv data/multi-asset-dataset.csv --symbol AAPL --freq "1min"
"""
from __future__ import annotations

import argparse
from pathlib import Path

from src.data_loader import load_raw
from src.ofi import add_level_ofi, aggregate_ofi, integrated_ofi, multi_asset_ofi
from src.plots import corr_heatmap


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build OFI features.")
    p.add_argument("--csv", required=True, help="Path to raw ITCH‑like CSV")
    p.add_argument("--symbol", required=True, help="Ticker symbol (e.g. AAPL)")
    p.add_argument("--freq", default="1min", help="Resample frequency")
    p.add_argument("--n_components", default=10, help="Number of components for PCA")
    p.add_argument("--symbols", help="Comma‑separated tickers for multi-asset OFI calculation (e.g. AAPL,JPM)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    raw = load_raw(args.csv)
    sym_df = raw[raw["symbol"] == args.symbol]

    msg_df = add_level_ofi(sym_df)
    bar_df = aggregate_ofi(msg_df, freq=args.freq, normalise=True)
    bar_df["ofi_integrated"] = integrated_ofi(bar_df, args.n_components)

    out_csv = Path(f"ofi-data/{args.symbol}_ofi_{args.freq}.parquet")
    bar_df.to_parquet(out_csv)
    print(f"Bar‑level OFI written to {out_csv}")

    corr_heatmap(bar_df, path=f"ofi-data/{args.symbol}_corr.png")

    if args.symbols:
        syms = [s.strip().upper() for s in args.symbols.split(",")]
        multi_bar = multi_asset_ofi(raw, syms, freq=args.freq, normalise=True)
        multi_bar.to_parquet("ofi-data/multi_asset_ofi.parquet")
        print(f"Multi-asset bar‑level OFI written to ofi-data/multi_asset_ofi.parquet")


if __name__ == "__main__":
    main()
