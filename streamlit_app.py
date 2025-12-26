import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import math
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
from io import StringIO


def safe_get(info, key, default=np.nan):
    try:
        return info.get(key, default)
    except Exception:
        return default


def fetch_single_ticker(ticker: str, retries: int = 2, delay: float = 1.0):
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
        except Exception:
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


def fetch_value_of_stocks_concurrent(tickers: list, max_workers: int = 8):
    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(fetch_single_ticker, t): t for t in tickers}
        for future in as_completed(futures):
            results.append(future.result())
    df = pd.DataFrame(results)
    return df

def main():
    st.set_page_config(page_title="Smart Investing — Value Screener", layout="wide")

    # Simple CSS for Streamlit
    APP_CSS = """
    <style>
    body {background-color: #f6f8fa}
    .stButton>button {background-color:#0b6cff;color:white}
    h1 {color: #0b6cff}
    .dataframe th {background-color: #0b6cff; color: white}
    </style>
    """

    st.markdown(APP_CSS, unsafe_allow_html=True)

    st.title("Smart Investing — Value Screener 💡")
    st.write("A compact Streamlit UI to run the value screen and build a simple position sizing plan using the same logic as your notebook.")

    # Helpers
    @st.cache_data(show_spinner=False)
    def load_ticker_list(uploaded_file):
        if uploaded_file is None:
            return pd.read_csv('top_50_indian_stocks.csv')['Ticker'].dropna().astype(str).tolist()
        else:
            try:
                df = pd.read_csv(uploaded_file)
                if 'Ticker' in df.columns:
                    return df['Ticker'].dropna().astype(str).tolist()
                else:
                    st.error("Uploaded CSV must contain a 'Ticker' column")
                    return []
            except Exception as e:
                st.error(f"Could not read uploaded file: {e}")
                return []

    # UI: Sidebar inputs
    st.sidebar.header("Inputs")
    uploaded_file = st.sidebar.file_uploader("Upload CSV with 'Ticker' column", type=['csv'])

    ticker_list = load_ticker_list(uploaded_file)

    n_top = st.sidebar.slider("Top N stocks to keep", min_value=1, max_value=min(50, len(ticker_list) or 50), value=10)
    portfolio_size = st.sidebar.number_input("Portfolio size (₹)", min_value=1, value=100000, step=1000)
    weighting = st.sidebar.selectbox("Weighting scheme", options=["Equal Weight", "Value Score Weight"], index=0)

    run_button = st.sidebar.button("Run screen")

    if run_button:
        if len(ticker_list) == 0:
            st.error("No tickers available. Upload a CSV or ensure the default CSV 'top_50_indian_stocks.csv' exists.")
        else:
            with st.spinner('Fetching stock data (this may take a moment)...'):
                df = fetch_value_of_stocks_concurrent(ticker_list)

            # Clean & fill
            value_cols = ["PE-ratio", "PB-ratio", "PS-ratio", "EV/EBITDA", "EV/GP"]
            for col in value_cols:
                if col in df.columns:
                    median = df[col].median(skipna=True)
                    df[col] = df[col].fillna(median)

            # Percentile (rank pct is faster and stable)
            for col in value_cols:
                perc_col = f"{col}_Percentile"
                df[perc_col] = df[col].rank(pct=True, method='average')

            percentile_cols = [f"{c}_Percentile" for c in value_cols]
            df['Value Score'] = df[percentile_cols].mean(axis=1)

            # Sort & filter top N
            df = df.sort_values(by='Value Score', ascending=False)
            df_top = df.head(n_top).reset_index(drop=True)

            st.subheader("Top candidates")
            st.dataframe(df_top.style.format({"Price": "{:.2f}", "Value Score": "{:.3f}"}))

            # Position sizing
            n = len(df_top)
            position_size = portfolio_size / n if n else 0
            df_top['Position Size (₹)'] = position_size

            if weighting == "Equal Weight":
                df_top['Weight'] = 1/n if n else 0
            else:
                ws = df_top['Value Score']
                df_top['Weight'] = ws / ws.sum()
                df_top['Position Size (₹)'] = df_top['Weight'] * portfolio_size

            # Shares calculation and cash remainder
            df_top['Number of shares can buy'] = df_top.apply(lambda r: math.floor(r['Position Size (₹)'] / r['Price']) if r['Price'] and not np.isnan(r['Price']) else 0, axis=1)
            df_top['Traded Value (₹)'] = df_top['Number of shares can buy'] * df_top['Price']
            traded_total = df_top['Traded Value (₹)'].sum()
            cash_remaining = portfolio_size - traded_total

            st.markdown(f"**Estimated cash remaining:** ₹{cash_remaining:,.2f}")

            st.subheader("Final allocation")
            st.dataframe(df_top[['Ticker','Price','Number of shares can buy','Traded Value (₹)','Weight','Value Score']].reset_index(drop=True).style.format({
                'Price':'{:.2f}','Traded Value (₹)':'{:.2f}','Value Score':'{:.3f}','Weight':'{:.3%}'
            }))

            # Export
            csv = df_top.to_csv(index=False)
            st.download_button(label='Download allocation (CSV)', data=csv, file_name='allocation.csv', mime='text/csv')

            # Save cache optionally
            if st.button('Save results to cache file'):
                df_top.to_csv('value_screen_results.csv', index=False)
                st.success('Saved to value_screen_results.csv')

            st.success('Done')


if __name__ == "__main__":
    main()
