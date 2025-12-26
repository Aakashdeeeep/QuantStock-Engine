"""smart_investing.py

Standalone script that implements the value screen and portfolio allocation
originally in `smart investing.ipynb`.

Usage (examples):
    python3 smart_investing.py --tickers-file top_50_indian_stocks.csv --top-n 10 --portfolio-size 100000 --weighting equal
    python3 smart_investing.py --tickers TCS.NS,INFY.NS --top-n 2 --portfolio-size 100000 --save allocation.csv
"""

from __future__ import annotations
import argparse
import math
import time
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Optional

import pandas as pd
import numpy as np
import yfinance as yf

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def safe_get(info: dict, key: str, default=np.nan):
    try:
        return info.get(key, default)
    except Exception:
        return default


def fetch_single_ticker(ticker: str, retries: int = 2, delay: float = 1.0) -> dict:
    for attempt in range(retries + 1):
        try:
            stock = yf.Ticker(ticker)
            hist = stock.history(period='1d')
            price = float(hist['Close'].iloc[-1]) if not hist.empty else np.nan
            info = stock.info or {}

            pe_ratio = safe_get(info, 'forwardPE')
            pb_ratio = safe_get(info, 'priceToBook')
            ps_ratio = safe_get(info, 'priceToSalesTrailing12Months')
            ev = safe_get(info, 'enterpriseValue')
            ebitda = safe_get(info, 'ebitda')
            ev_ebitda = (ev / ebitda) if (ev and ebitda) else np.nan

            gross_margin = safe_get(info, 'grossMargins', np.nan)
            total_revenue = safe_get(info, 'totalRevenue', np.nan)
            grossprofit = (gross_margin * total_revenue) if (not np.isnan(gross_margin) and not np.isnan(total_revenue)) else np.nan
            ev_gp = (ev / grossprofit) if (ev and grossprofit) else np.nan

            return {
                'Ticker': ticker,
                'Price': price,
                'PE-ratio': pe_ratio if pe_ratio is not None else np.nan,
                'PB-ratio': pb_ratio if pb_ratio is not None else np.nan,
                'PS-ratio': ps_ratio if ps_ratio is not None else np.nan,
                'EV/EBITDA': ev_ebitda,
                'EV/GP': ev_gp
            }
        except Exception as exc:
            logger.debug(f"Error fetching {ticker} (attempt {attempt+1}): {exc}")
            if attempt < retries:
                time.sleep(delay)
            else:
                return {
                    'Ticker': ticker,
                    'Price': np.nan,
                    'PE-ratio': np.nan,
                    'PB-ratio': np.nan,
                    'PS-ratio': np.nan,
                    'EV/EBITDA': np.nan,
                    'EV/GP': np.nan
                }


def fetch_value_of_stocks(tickers: List[str], max_workers: int = 8) -> pd.DataFrame:
    """Fetch metrics for tickers concurrently."""
    results = []
    logger.info(f"Fetching {len(tickers)} tickers using {max_workers} workers...")
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(fetch_single_ticker, t): t for t in tickers}
        for f in as_completed(futures):
            results.append(f.result())
    df = pd.DataFrame(results)
    return df


def process_value_df(df: pd.DataFrame, value_cols: Optional[List[str]] = None) -> pd.DataFrame:
    if value_cols is None:
        value_cols = ["PE-ratio", "PB-ratio", "PS-ratio", "EV/EBITDA", "EV/GP"]

    for col in value_cols:
        if col in df.columns:
            median = df[col].median(skipna=True)
            df[col] = df[col].fillna(median)

    for col in value_cols:
        perc_col = f"{col}_Percentile"
        df[perc_col] = df[col].rank(pct=True, method='average')

    percentile_cols = [f"{c}_Percentile" for c in value_cols]
    df['Value Score'] = df[percentile_cols].mean(axis=1)

    df = df.sort_values(by='Value Score', ascending=False).reset_index(drop=True)
    return df


def allocate_positions(df: pd.DataFrame, portfolio_size: float, top_n: int = 10, weighting: str = 'equal') -> pd.DataFrame:
    df_top = df.head(top_n).copy().reset_index(drop=True)
    n = len(df_top)

    if weighting.lower() in ('equal', 'equal_weight', 'equal-weight'):
        df_top['Weight'] = 1 / n if n else 0
    else:
        ws = df_top['Value Score']
        df_top['Weight'] = ws / ws.sum()

    df_top['Position Size (₹)'] = df_top['Weight'] * portfolio_size
    df_top['Number of shares can buy'] = df_top.apply(lambda r: math.floor(r['Position Size (₹)'] / r['Price']) if r['Price'] and not np.isnan(r['Price']) and r['Price']>0 else 0, axis=1)
    df_top['Traded Value (₹)'] = df_top['Number of shares can buy'] * df_top['Price']
    traded_total = df_top['Traded Value (₹)'].sum()
    df_top.attrs['cash_remaining'] = portfolio_size - traded_total

    return df_top


def load_ticker_list_from_file(path: str) -> List[str]:
    df = pd.read_csv(path)
    if 'Ticker' not in df.columns:
        raise ValueError("CSV must contain a 'Ticker' column")
    return df['Ticker'].dropna().astype(str).tolist()


def main(argv: Optional[List[str]] = None):
    parser = argparse.ArgumentParser(description='Smart investing value screener')
    parser.add_argument('--tickers-file', type=str, default='top_50_indian_stocks.csv', help='CSV file with a Ticker column')
    parser.add_argument('--tickers', type=str, default=None, help='Comma separated list of tickers (overrides tickers-file)')
    parser.add_argument('--top-n', type=int, default=10, help='Number of top tickers to keep')
    parser.add_argument('--portfolio-size', type=float, default=100000, help='Total portfolio capital (₹)')
    parser.add_argument('--weighting', choices=['equal', 'value'], default='equal', help='Weighting scheme')
    parser.add_argument('--max-workers', type=int, default=8, help='Max concurrent workers for fetching')
    parser.add_argument('--save', type=str, default=None, help='Path to save allocation CSV')

    args = parser.parse_args(argv)

    if args.tickers:
        tickers = [t.strip() for t in args.tickers.split(',') if t.strip()]
    else:
        tickers = load_ticker_list_from_file(args.tickers_file)

    df = fetch_value_of_stocks(tickers, max_workers=args.max_workers)
    df = process_value_df(df)
    df_top = allocate_positions(df, portfolio_size=args.portfolio_size, top_n=args.top_n, weighting=args.weighting)

    # Print concise summary
    print('\nTop allocations:')
    print(df_top[['Ticker','Price','Number of shares can buy','Traded Value (₹)','Weight','Value Score']].to_string(index=False, justify='left', formatters={
        'Price': '{:.2f}'.format, 'Traded Value (₹)': '{:.2f}'.format, 'Value Score': '{:.4f}'.format, 'Weight': lambda x: f"{x:.2%}"
    }))

    cash_remaining = df_top.attrs.get('cash_remaining', 0.0)
    print(f"\nEstimated cash remaining: ₹{cash_remaining:,.2f}")

    if args.save:
        df_top.to_csv(args.save, index=False)
        print(f"Saved allocation to {args.save}")

    return df_top


if __name__ == '__main__':
    main()
