#this is not an official file it was added by lucas li to try and run the test with the raw data

"""
Used for running with raw data (in an attempt to get the time and date correct)

Usage:
    python run_backtest_raw_data.py --csv data\\AAPL_1Min_stock_alpaca_clean.csv --strategy ma --plot

"""

"""
Run backtest on raw data files (auto-fixes them first)
"""

import argparse
from pathlib import Path
import pandas as pd

from core.backtester_with_plots import Backtester, PerformanceAnalyzer, plot_equity
from core.gateway import MarketDataGateway
from core.matching_engine import MatchingEngine
from core.order_book import OrderBook
from core.order_manager import OrderLoggingGateway, OrderManager
from strategies import get_strategy_class


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, required=True)
    parser.add_argument("--strategy", default="momentumstrategy")
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--capital", type=float, default=50000)
    parser.add_argument("--position-size", type=float, default=10.0)
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Fix raw data
    print(f"Reading and fixing raw data from {args.csv}...")
    df = pd.read_csv(args.csv, skiprows=[1, 2])
    df.rename(columns={'Price': 'Datetime'}, inplace=True)
    df = df[['Datetime', 'Open', 'High', 'Low', 'Close', 'Volume']]
    df.dropna(inplace=True)
    
    # Save to temp
    temp_path = Path('data/temp_fixed.csv')
    df.to_csv(temp_path, index=False)
    print(f"Fixed {len(df)} rows, saved to {temp_path}")
    
    # Run backtest
    strategy_cls = get_strategy_class(args.strategy)
    strategy = strategy_cls()
    
    gateway = MarketDataGateway(temp_path)
    order_book = OrderBook()
    order_manager = OrderManager(capital=args.capital, max_long_position=1000, max_short_position=1000)
    matching_engine = MatchingEngine()
    logger = OrderLoggingGateway()
    
    backtester = Backtester(
        data_gateway=gateway,
        strategy=strategy,
        order_manager=order_manager,
        order_book=order_book,
        matching_engine=matching_engine,
        logger=logger,
        default_position_size=int(args.position_size),
    )
    
    equity_df = backtester.run()
    analyzer = PerformanceAnalyzer(equity_df["equity"].tolist(), backtester.trades)
    
    print("\n=== Backtest Summary ===")
    print(f"Equity data points: {len(equity_df)}")
    print(f"Trades executed: {sum(1 for t in backtester.trades if t.qty > 0)}")
    print(f"Final portfolio value: {equity_df.iloc[-1]['equity']:.2f}")
    print(f"PnL: {analyzer.pnl():.2f}")
    print(f"Sharpe: {analyzer.sharpe():.2f}")
    print(f"Max Drawdown: {analyzer.max_drawdown():.4f}")
    print(f"Win Rate: {analyzer.win_rate():.2%}")
    
    if args.plot:
        plot_equity(equity_df)


if __name__ == "__main__":
    main()