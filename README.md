# Trading System
## What this includes
- Alpaca paper-data download + cleaning helpers.
- A paper-trading runner that submits orders through Alpaca.
- A lightweight backtester with a market data gateway, order book, order manager,
  and matching engine (offline only).
- Two strategies only:
  - TemplateStrategy: a starter template to customize.
  - MovingAverageStrategy: a simple moving average crossover baseline.

## Accessing Alpaca
Reminders:
   Do not add real money to your Alpaca account.
   Do not share your API keys.
   Keep it simple if you're short on time.

1. Sign up at alpaca.markets.
   Complete identity verification and confirm your email. 
2. Configure the paper trading option. Your starting equity should be $1,000,000.
3. Obtain your API Key ID and Secret Key at https://app.alpaca.markets/dashboard/overview 
   by scrolling down and looking at the right side of the screen. Generate the endpoint, key, and secret.
4. Retrieve Market Data

In your terminal, install Alpaca SDK:

```bash
pip install alpaca-trade-api
```
Sample usage:
```python
import alpaca_trade_api as tradeapi

api = tradeapi.REST('your_api_key', 'your_api_secret', 'https://paper-api.alpaca.markets')

bars = api.get_bars('AAPL', '1Min', limit=1).df
```
   Review Alpaca's API docs and GitHub for more endpoints.

5. Save Market Data via flat files CSV, Pickle, OR parquet.

## Run modes

Backtest (offline CSV):
```
python run_backtest.py --csv data/AAPL_1Min_stock_alpaca_clean.csv --strategy ma
```
To run your strategy, replace `ma` with `template` or your class name.

Live paper trading (Alpaca):
```
python run_live.py --symbol AAPL --asset-class stock --strategy ma --timeframe 1Min --live
```

## Quick start (Alpaca paper trading)
0) Clone this repository by using an IDE (VS Code, Cursor) and running in powershell / terminal:
```bash
git clone https://github.com/Maroon-Capital-Trading-Project/Trading-System.git
git pull
```
Alternatively, you can use Github Desktop for a user-friendly interface.

1) Install dependencies:
```
pip install -r requirements.txt
```
2) Create a `.env` file in the project root:
```
ALPACA_API_KEY=your_key_here
ALPACA_API_SECRET=your_secret_here
ALPACA_API_URL=https://paper-api.alpaca.markets
ALPACA_DATA_FEED=iex
```
The scripts load this file automatically.
ALPACA_DATA_FEED is optional and applies to stock data.
3) Run the paper-trading loop:
```
python run_live.py --symbol AAPL --asset-class stock --strategy ma --timeframe 1Min
```

You will see trade-by-trade output in the terminal, followed by a summary.
Use `--save-data` to write raw and cleaned CSVs to `data/`.
Use `--dry-run` to preview decisions without placing orders.
For crypto, use symbols like `BTCUSD` with `--asset-class crypto`.
This script submits paper orders to Alpaca.
To run continuously, add `--live` and stop with Ctrl+C.

Optional local smoke test (synthetic data):
```
python test_system.py
```

## Fetch data with Alpaca
Use the notebooks in `notebooks/`:
- `notebooks/fetch_data_stock.ipynb`
- `notebooks/fetch_data_crypto.ipynb`

They download bars from Alpaca and save raw/clean CSVs to `data/`.

## Build your own strategy
Open `strategies/strategy_base.py` and edit `TemplateStrategy` (recommended), or add your own class.
The backtester expects:
- `signal`: 1 for buy, -1 for sell, 0 for no action.
- `target_qty`: the quantity to trade when a signal triggers.
- Optionally, `limit_price` if you want a limit price different from `Close`.

The Alpaca runner uses these same fields to submit paper orders.
To run your custom class from the CLI, give it a no-arg constructor and call (case-insensitive):
```
python run_backtest.py --csv data/AAPL_1Min_stock_alpaca_clean.csv --strategy MyStrategy
```
or
```
python run_live.py --symbol AAPL --asset-class stock --strategy MyStrategy --timeframe 1Min --live
```

Example signal logic inside `generate_signals`:
```python
df["signal"] = 0
buy = df["momentum"] > 0.01
sell = df["momentum"] < -0.01
df.loc[buy, "signal"] = 1
df.loc[sell, "signal"] = -1
df["position"] = df["signal"].replace(0, np.nan).ffill().fillna(0)
df["target_qty"] = df["position"].abs() * self.position_size
```

## Trade output format (live)
Each fill prints a line like:
```
2024-01-01 09:31:00 | BUY 10 AAPL @ 101.23 | order_id=1234 | net_pnl=+12.50
```

## Project structure
```
core/
pipeline/
strategies/
data/
notebooks/
.env
run_backtest.py
run_live.py
test_system.py
```
