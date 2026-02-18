"""
Alpaca paper-trading runner (updated for new MovingAverageStrategy params).

Requires .env file with:
    ALPACA_API_KEY      (required)
    ALPACA_API_SECRET   (required)

Usage:
    python run_live_live.py --symbol AAPL --strategy ma
    python run_live_live.py --symbol AAPL --strategy ma --live
    python run_live_live.py --symbol AAPL --strategy ma --dry-run
    python run_live_live.py --symbol BTCUSD --asset-class crypto --strategy crypto_trend --live

Logs are saved to: logs/trades.csv, logs/signals.csv, logs/system.log
"""

from __future__ import annotations

import argparse
import sys
import time

from core.alpaca_trader import AlpacaTrader
from core.logger import get_logger, get_trade_logger
from pipeline.alpaca import clean_market_data, save_bars
from strategies.strategy_base import MovingAverageStrategy
from strategies import TemplateStrategy, CryptoTrendStrategy, DemoStrategy, get_strategy_class, list_strategies
import importlib.util

logger = get_logger("run_live")

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a paper-trading loop with Alpaca.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Available strategies: {', '.join(list_strategies())}

Examples:
  python run_live_live.py --symbol AAPL --strategy ma --live
  python run_live_live.py --symbol BTCUSD --asset-class crypto --strategy crypto_trend --live
  python run_live_live.py --symbol AAPL --strategy ma --dry-run --iterations 5
        """,
    )
    parser.add_argument("--symbol", default="AAPL", help="Ticker or crypto symbol (default: AAPL)")
    parser.add_argument("--asset-class", choices=["stock", "crypto"], default="stock", help="Asset class (default: stock)")
    parser.add_argument("--timeframe", default="1Min", help="Alpaca timeframe: 1Min, 5Min, 15Min, 1H, 1D (default: 1Min)")
    parser.add_argument("--lookback", type=int, default=200, help="Bars to fetch each iteration (default: 200)")
    parser.add_argument("--strategy", default="ma", help="Strategy name (default: ma)")
    parser.add_argument("--short-window", type=int, default=20, help="Short MA window (default: 20)")
    parser.add_argument("--long-window", type=int, default=60, help="Long MA window (default: 60)")
    parser.add_argument("--max-capital", type=float, default=10000.0, help="Max capital per trade (default: 10000.0)")
    parser.add_argument("--risk-pct", type=float, default=0.01, help="Risk percent per trade (default: 0.01)")
    parser.add_argument("--max-order-notional", type=float, default=None, help="Max notional per order (crypto only)")
    parser.add_argument("--momentum-lookback", type=int, default=14, help="Momentum lookback for template strategy (default: 14)")
    parser.add_argument("--buy-threshold", type=float, default=0.01, help="Buy threshold for template strategy (default: 0.01)")
    parser.add_argument("--sell-threshold", type=float, default=-0.01, help="Sell threshold for template strategy (default: -0.01)")
    parser.add_argument("--iterations", type=int, default=1, help="Number of loops to run (default: 1)")
    parser.add_argument("--sleep", type=int, default=60, help="Seconds between loops (default: 60)")
    parser.add_argument("--live", action="store_true", help="Run continuously until Ctrl+C")
    parser.add_argument("--save-data", action="store_true", help="Save raw+clean CSVs to data/")
    parser.add_argument("--dry-run", action="store_true", help="Print decisions without placing orders")
    parser.add_argument("--feed", default=None, help="Data feed (iex or sip for stocks)")
    parser.add_argument("--list-strategies", action="store_true", help="List available strategies and exit")
    return parser.parse_args()

def main() -> None:
    args = parse_args()

    # Handle --list-strategies
    if args.list_strategies:
        print("Available strategies:")
        for name in list_strategies():
            print(f"  - {name}")
        sys.exit(0)

    # Build strategy
    strategy_cls = get_strategy_class(args.strategy)
    # Portfolio mode if symbol is 'PORTFOLIO' or comma-separated list
    if strategy_cls is MovingAverageStrategy and (args.symbol.upper() == "PORTFOLIO" or args.symbol.lower() == "portfolio_tickers" or "," in args.symbol):
        # Multi-stock portfolio mode
        if args.symbol.lower() == "portfolio_tickers":
            # Dynamically import the ticker list
            import importlib.util
            import os
            ticker_path = os.path.join(os.path.dirname(__file__), "data", "portfolio_tickers.py")
            spec = importlib.util.spec_from_file_location("portfolio_tickers", ticker_path)
            pt = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(pt)
            symbols = pt.PORTFOLIO_TICKERS
        else:
            symbols = [s.strip().upper() for s in args.symbol.split(",") if s.strip()]
            if args.symbol.upper() == "PORTFOLIO":
                symbols = ["AAPL", "MSFT", "NVDA", "TSLA", "GOOGL"]
        strategy = MovingAverageStrategy(
            short_window=args.short_window,
            long_window=args.long_window,
            max_capital=args.max_capital,
            risk_pct=args.risk_pct,
            symbols=symbols,
                # max_positions=5,  # Removed as the class no longer accepts this argument
        )
        import pandas as pd
        from pipeline.alpaca import fetch_stock_bars, get_rest
        api = get_rest()
        trade_logger = get_trade_logger()
        start_equity = None
        iteration_count = 0
        # Persistent state for portfolio trailing stops/positions
        portfolio_state = {}
        def handle_iteration_portfolio():
            nonlocal iteration_count, portfolio_state
            iteration_count += 1
            logger.info(f"[Loop] Iteration {iteration_count}: fetching data for portfolio: {symbols}")
            stock_dfs = {}
            for symbol in symbols:
                try:
                    df = fetch_stock_bars(symbol, timeframe=args.timeframe, limit=args.lookback, api=api)
                    stock_dfs[symbol] = df
                except Exception as e:
                    logger.debug(f"Skipping {symbol}: {e}")
            signal_dfs, portfolio_state = strategy.generate_signals(stock_dfs, state=portfolio_state)
            trades_this_iter = 0
            for symbol, df in signal_dfs.items():
                if df is not None and not df.empty:
                    latest = df.iloc[-1]
                    signal = int(latest.get("signal", 0))
                    qty = float(latest.get("target_qty", 0))
                    price = float(latest.get("Close", 0))
                    if signal != 0 and qty > 0:
                        side = "buy" if signal > 0 else "sell"
                        logger.info(f"{side.upper()} {qty:.2f} {symbol} @ {price:.2f}")
                        trades_this_iter += 1
                        if not args.dry_run:
                            try:
                                api.submit_order(symbol=symbol, qty=int(qty), side=side, type="market", time_in_force="day")
                            except Exception as e:
                                logger.warning(f"Order failed for {symbol}: {e}")
                        trade_logger.log_trade(symbol=symbol, side=side, qty=qty, price=price, order_type="market", status="dry_run" if args.dry_run else "submitted", strategy=args.strategy)
                    else:
                        # No trade for this symbol this iteration
                        trade_logger.log_trade(symbol=symbol, side="hold", qty=qty, price=price, order_type="market", status="dry_run" if args.dry_run else "submitted", strategy=args.strategy)
            if trades_this_iter == 0:
                logger.info("No trades executed this iteration.")
        def print_summary_portfolio():
            logger.info("")
            logger.info("=" * 60)
            logger.info("                    PORTFOLIO SESSION SUMMARY")
            logger.info("=" * 60)
            logger.info(f"  Iterations:      {iteration_count}")
            logger.info("=" * 60)
            logger.info("Logs: logs/trades.csv, logs/system.log")
        import datetime
        import pytz
        eastern = pytz.timezone('US/Eastern')
        market_open = datetime.time(9, 30)
        market_close = datetime.time(16, 0)
        def is_market_open():
            now = datetime.datetime.now(eastern)
            return market_open <= now.time() <= market_close and now.weekday() < 5
        def seconds_until_open():
            now = datetime.datetime.now(eastern)
            today_open = now.replace(hour=9, minute=30, second=0, microsecond=0)
            if now.time() < market_open:
                return (today_open - now).total_seconds()
            # If after close, sleep until next weekday open
            next_open = today_open + datetime.timedelta(days=1)
            while next_open.weekday() >= 5:
                next_open += datetime.timedelta(days=1)
            return (next_open - now).total_seconds()
        if args.live:
            logger.info(f"Running continuously (Ctrl+C to stop). Will only trade during US market hours. Sleep: {args.sleep}s between iterations.")
            try:
                while True:
                    if is_market_open():
                        handle_iteration_portfolio()
                        time.sleep(args.sleep)
                    else:
                        wait = max(1, int(seconds_until_open()))
                        logger.info(f"Market closed. Sleeping {wait//60} min {wait%60} sec until next open.")
                        time.sleep(wait)
            except KeyboardInterrupt:
                logger.info("Received stop signal.")
                print_summary_portfolio()
        else:
            logger.info(f"Running {args.iterations} iteration(s)...")
            for i in range(args.iterations):
                handle_iteration_portfolio()
                if i < args.iterations - 1:
                    time.sleep(args.sleep)
            print_summary_portfolio()
        return
    # ...existing code...

    trade_logger = get_trade_logger()
    start_equity = trader.starting_equity
    iteration_count = 0

    def handle_iteration() -> None:
        nonlocal iteration_count
        iteration_count += 1
        logger.info(f"[Loop] Iteration {iteration_count}: fetching data for {args.symbol}")
        df = trader.run_once()
        if args.save_data and df is not None:
            raw_path = save_bars(df, args.symbol, args.timeframe, args.asset_class)
            clean_market_data(raw_path)

    def print_summary() -> None:
        summary = trade_logger.get_session_summary(start_equity)
        logger.info("")
        logger.info("=" * 60)
        logger.info("                    SESSION SUMMARY")
        logger.info("=" * 60)
        logger.info(f"  Iterations:      {iteration_count}")
        logger.info(f"  Total Trades:    {summary['total_trades']}")
        logger.info(f"  Buys / Sells:    {summary['buys']} / {summary['sells']}")
        logger.info("-" * 60)
        logger.info(f"  Wins / Losses:   {summary['wins']} / {summary['losses']}")
        logger.info(f"  Win Rate:        {summary['win_rate']:.1f}%")
        logger.info(f"  Avg Trade P&L:   ${summary['avg_trade_pnl']:+,.2f}")
        logger.info("-" * 60)
        logger.info(f"  Start Equity:    ${summary['start_equity']:,.2f}")
        logger.info(f"  End Equity:      ${summary['end_equity']:,.2f}")
        logger.info(f"  Net P&L:         ${summary['net_pnl']:+,.2f}")
        logger.info("-" * 60)
        logger.info(f"  Sharpe Ratio:    {summary['sharpe_ratio']:.2f}")
        logger.info(f"  Volatility:      {summary['volatility']:.2f}%")
        logger.info(f"  Max Drawdown:    {summary['max_drawdown']:.2f}%")
        logger.info("=" * 60)
        logger.info("Logs: logs/trades.csv, logs/system.log")

        import datetime
        import pytz
        eastern = pytz.timezone('US/Eastern')
        market_open = datetime.time(9, 30)
        market_close = datetime.time(16, 0)
        def is_market_open():
            now = datetime.datetime.now(eastern)
            return market_open <= now.time() <= market_close and now.weekday() < 5
        def seconds_until_open():
            now = datetime.datetime.now(eastern)
            today_open = now.replace(hour=9, minute=30, second=0, microsecond=0)
            if now.time() < market_open:
                return (today_open - now).total_seconds()
            # If after close, sleep until next weekday open
            next_open = today_open + datetime.timedelta(days=1)
            while next_open.weekday() >= 5:
                next_open += datetime.timedelta(days=1)
            return (next_open - now).total_seconds()
        if args.live:
            logger.info(f"Running continuously (Ctrl+C to stop). Will only trade during US market hours. Sleep: {args.sleep}s between iterations.")
            try:
                while True:
                    if is_market_open():
                        handle_iteration()
                        time.sleep(args.sleep)
                    else:
                        wait = max(1, int(seconds_until_open()))
                        logger.info(f"Market closed. Sleeping {wait//60} min {wait%60} sec until next open.")
                        time.sleep(wait)
            except KeyboardInterrupt:
                logger.info("Received stop signal.")
                print_summary()
        else:
            logger.info(f"Running {args.iterations} iteration(s)...")
            for i in range(args.iterations):
                handle_iteration()
                if i < args.iterations - 1:
                    time.sleep(args.sleep)
            print_summary()


if __name__ == "__main__":
    main()
