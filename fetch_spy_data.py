"""
Fetch SPY 1-minute bar data from Alpaca for backtesting.

Usage:
    1. Make sure you have a .env file in your trading-system root with:
           ALPACA_API_KEY=your_key_here
           ALPACA_API_SECRET=your_secret_here

    2. Install dependencies (if not already):
           pip install alpaca-trade-api pandas python-dotenv

    3. Run this script from your trading-system directory:
           python fetch_spy_data.py

    This will create two CSV files in the data/ folder:
        - SPY_1Min_train.csv  (Jan 2024 - Jun 2025)
        - SPY_1Min_test.csv   (Jul 2025 - Jan 2026)
"""

import os
from pathlib import Path
from dotenv import load_dotenv
import alpaca_trade_api as tradeapi
import pandas as pd

# ── CONFIG ──────────────────────────────────────────────────────────
SYMBOL = "SPY"

# Train/test split periods
TRAIN_START = "2024-01-01"
TRAIN_END = "2025-06-30"
TEST_START = "2025-07-01"
TEST_END = "2026-01-31"

OUTPUT_DIR = Path("data")
# ────────────────────────────────────────────────────────────────────


def load_api():
    """Load Alpaca API credentials from .env file."""
    for env_path in [Path.cwd() / ".env", Path(__file__).resolve().parent / ".env"]:
        if env_path.exists():
            load_dotenv(env_path, override=False)
            break
    else:
        load_dotenv(override=False)

    api_key = os.environ.get("ALPACA_API_KEY")
    api_secret = os.environ.get("ALPACA_API_SECRET")

    if not api_key or not api_secret:
        raise RuntimeError(
            "Missing ALPACA_API_KEY or ALPACA_API_SECRET.\n"
            "Create a .env file with:\n"
            "  ALPACA_API_KEY=your_key\n"
            "  ALPACA_API_SECRET=your_secret"
        )

    base_url = os.environ.get("ALPACA_API_URL", "https://paper-api.alpaca.markets")
    return tradeapi.REST(api_key, api_secret, base_url, api_version="v2")


def fetch_bars_chunked(api, symbol, start, end):
    """
    Fetch all 1-minute bars by iterating month by month.
    This avoids hitting the 10,000 bar per-request limit.
    """
    feed = os.environ.get("ALPACA_DATA_FEED", "iex")
    tf = tradeapi.TimeFrame.Minute

    all_dfs = []
    current_start = pd.Timestamp(start, tz="UTC")
    final_end = pd.Timestamp(end, tz="UTC")

    while current_start < final_end:
        # Fetch one month at a time
        chunk_end = current_start + pd.DateOffset(months=1)
        if chunk_end > final_end:
            chunk_end = final_end

        start_str = current_start.strftime("%Y-%m-%d")
        end_str = chunk_end.strftime("%Y-%m-%d")

        print(f"  Fetching {start_str} to {end_str} ...", end=" ")

        try:
            bars = api.get_bars(
                symbol,
                tf,
                start=start_str,
                end=end_str,
                feed=feed,
                limit=10000,
            ).df

            if not bars.empty:
                all_dfs.append(bars)
                print(f"got {len(bars):,} bars")
            else:
                print("no data")
        except Exception as e:
            print(f"error: {e}")

        current_start = chunk_end

    if not all_dfs:
        print("  WARNING: No bars returned at all!")
        return pd.DataFrame()

    # Combine all chunks
    combined = pd.concat(all_dfs)
    combined = combined.reset_index()

    # Rename columns
    rename_map = {}
    for col in combined.columns:
        cl = str(col).lower()
        if cl in {"timestamp", "time", "t", "index"}:
            rename_map[col] = "Datetime"
        elif cl == "open":
            rename_map[col] = "Open"
        elif cl == "high":
            rename_map[col] = "High"
        elif cl == "low":
            rename_map[col] = "Low"
        elif cl == "close":
            rename_map[col] = "Close"
        elif cl == "volume":
            rename_map[col] = "Volume"
    combined.rename(columns=rename_map, inplace=True)

    keep = [c for c in ["Datetime", "Open", "High", "Low", "Close", "Volume"] if c in combined.columns]
    combined = combined[keep]

    combined["Datetime"] = pd.to_datetime(combined["Datetime"], utc=True)
    combined.drop_duplicates(subset=["Datetime"], inplace=True)
    combined.sort_values("Datetime", inplace=True)
    combined.reset_index(drop=True, inplace=True)

    print(f"  Total: {len(combined):,} bars ({combined['Datetime'].iloc[0]} to {combined['Datetime'].iloc[-1]})")
    return combined


def main():
    print("=" * 60)
    print("SPY Data Fetcher for Backtesting")
    print("=" * 60)

    api = load_api()
    print("API connection OK\n")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Fetch training data ──
    print("[1/2] TRAINING DATA")
    df_train = fetch_bars_chunked(api, SYMBOL, TRAIN_START, TRAIN_END)
    if not df_train.empty:
        train_path = OUTPUT_DIR / "SPY_1Min_train.csv"
        df_train.to_csv(train_path, index=False)
        print(f"  Saved to {train_path}\n")

    # ── Fetch testing data ──
    print("[2/2] TESTING DATA")
    df_test = fetch_bars_chunked(api, SYMBOL, TEST_START, TEST_END)
    if not df_test.empty:
        test_path = OUTPUT_DIR / "SPY_1Min_test.csv"
        df_test.to_csv(test_path, index=False)
        print(f"  Saved to {test_path}\n")

    # ── Summary ──
    print("=" * 60)
    print("SUMMARY")
    print(f"  Training: {len(df_train):,} bars  ({TRAIN_START} to {TRAIN_END})")
    print(f"  Testing:  {len(df_test):,} bars   ({TEST_START} to {TEST_END})")
    print(f"  Files saved in: {OUTPUT_DIR.resolve()}")
    print("=" * 60)


if __name__ == "__main__":
    main()