"""
Strategy base classes and built-in strategies.

To create your own strategy:
1. Create a new class that inherits from Strategy
2. Implement add_indicators() to calculate your technical indicators
3. Implement generate_signals() to generate buy/sell signals

Required output columns from generate_signals():
    - signal: 1 for buy, -1 for sell, 0 for hold
    - target_qty: position size (shares for stocks, USD for crypto)
    - position: current position state (1=long, -1=short, 0=flat)

Optional output columns:
    - limit_price: if set, places a limit order instead of market

Example:
    class MyStrategy(Strategy):
        def __init__(self, lookback=20, position_size=10.0):
            self.lookback = lookback
            self.position_size = position_size

        def add_indicators(self, df):
            df['sma'] = df['Close'].rolling(self.lookback).mean()
            return df

        def generate_signals(self, df):
            df['signal'] = 0
            df.loc[df['Close'] > df['sma'], 'signal'] = 1
            df.loc[df['Close'] < df['sma'], 'signal'] = -1
            df['position'] = df['signal']
            df['target_qty'] = self.position_size
            return df
"""

import numpy as np
import pandas as pd


class Strategy:
    """
    Base Strategy interface for adding indicators and generating trading signals.

    All strategies must implement:
        - add_indicators(df): Add technical indicators to the DataFrame
        - generate_signals(df): Generate trading signals

    The DataFrame must contain these columns:
        - Datetime, Open, High, Low, Close, Volume (input)
        - signal, target_qty, position (output from generate_signals)
    """

    def add_indicators(self, df: pd.DataFrame) -> pd.DataFrame:  # pragma: no cover - interface
        """Add technical indicators to the DataFrame. Override this method."""
        raise NotImplementedError

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:  # pragma: no cover - interface
        """Generate trading signals. Override this method."""
        raise NotImplementedError

    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        """Execute the full strategy pipeline. Do not override."""
        df = df.copy()
        df = self.add_indicators(df)
        df = self.generate_signals(df)
        return df

#Adjust Classes/Param as Needed (Jaden Cai test strategy) ##################
class AlgTopo(Strategy):
    """
    Algebraic Topology Trading Strategy using Persistent Homology concepts.
    
    Key ideas:
    - Signals that persist across multiple timeframes (scales) are stronger
    - Multi-scale momentum: detects trends visible at 3, 5, and 7-period scales
    - Persistence filtering: only trades when signal is consistent across scales
    - Volatility-normalized momentum: adjusts for market regime changes
    
    For short-term trading, this captures trends that actually matter (persistent patterns)
    while filtering out noise.
    """

    def __init__(self, position_size: float = 10.0, min_persistence: int = 2):
        """
        Args:
            position_size: shares to trade
            min_persistence: number of scales where signal must agree (1-3)
        """
        if position_size <= 0:
            raise ValueError("position_size must be positive.")
        if min_persistence < 1 or min_persistence > 3:
            raise ValueError("min_persistence must be 1-3.")
        self.position_size = position_size
        self.min_persistence = min_persistence
    
    def add_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        # Basic returns and volatility
        df["returns"] = df["Close"].pct_change().fillna(0.0)
        df["volatility"] = df["returns"].rolling(14).std().fillna(0.0)
        
        # Multi-scale momentum (topological persistence across timeframes)
        # These are like "homology groups" at different scales
        df["mom_fast"] = df["Close"].pct_change(3).fillna(0.0)      # 3-period
        df["mom_mid"] = df["Close"].pct_change(5).fillna(0.0)       # 5-period
        df["mom_slow"] = df["Close"].pct_change(7).fillna(0.0)      # 7-period
        
        # Volatility-normalized momentum (adjust signal strength for market regime)
        df["volatility_normalized"] = df["volatility"].rolling(20).mean().fillna(0.01)
        df["mom_fast_norm"] = df["mom_fast"] / (df["volatility_normalized"] + 0.0001)
        df["mom_mid_norm"] = df["mom_mid"] / (df["volatility_normalized"] + 0.0001)
        df["mom_slow_norm"] = df["mom_slow"] / (df["volatility_normalized"] + 0.0001)
        
        return df

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        # Vectorized persistence counting - pure numpy, no loops
        bullish_scales = (
            (df["mom_fast_norm"] > 0.3).astype(int) +
            (df["mom_mid_norm"] > 0.15).astype(int) +
            (df["mom_slow_norm"] > 0.1).astype(int)
        )
        bearish_scales = (
            (df["mom_fast_norm"] < -0.3).astype(int) +
            (df["mom_mid_norm"] < -0.15).astype(int) +
            (df["mom_slow_norm"] < -0.1).astype(int)
        )

        # Regime detection (not signals yet - just which regime we're in)
        is_bullish = ((bullish_scales >= self.min_persistence) & (df["volatility"] > 0)).astype(bool)
        is_bearish = ((bearish_scales >= self.min_persistence) & (df["volatility"] > 0)).astype(bool)
        is_bearish = is_bearish & ~is_bullish
        
        # Detect CROSSOVERS (regime changes) - signals only happen at transitions
        # Use numpy roll to compute previous values (avoids pandas fillna downcasting warnings)
        prev_bullish = np.concatenate(([False], is_bullish.values[:-1].astype(bool)))
        prev_bearish = np.concatenate(([False], is_bearish.values[:-1].astype(bool)))

        buy_signal = is_bullish.values & ~prev_bullish   # Flip from not-bullish to bullish
        sell_signal = is_bearish.values & ~prev_bearish  # Flip from not-bearish to bearish
        # convert boolean numpy arrays back to Series aligned with df.index
        buy_signal = pd.Series(buy_signal, index=df.index)
        sell_signal = pd.Series(sell_signal, index=df.index)
        # Assign signals only on crossovers
        df["signal"] = 0
        df.loc[buy_signal, "signal"] = 1
        df.loc[sell_signal, "signal"] = -1
        # Position: forward-fill from signals (hold until next flip)
        df["position"] = 0
        df.loc[buy_signal, "position"] = 1
        df.loc[sell_signal, "position"] = -1
        df["position"] = df["position"].replace(0, np.nan).ffill().fillna(0)
        
        df["target_qty"] = df["position"].abs() * self.position_size

        return df

class MeanReversion(Strategy):
    """
    Improved mean reversion strategy using Bollinger Bands and momentum confirmation.
    
    Strategy:
    - Uses SMA as the mean and Bollinger Bands (std deviations) to identify extremes
    - Only trades when price is >1 std dev away from the mean (statistically extreme)
    - Confirms mean reversion with momentum: buys when oversold + momentum reverses upward
    - Avoids trading during strong trends by checking volatility regimes
    - Only generates signals on band crossovers, not continuously
    """

##ADJUST SD PARAMS/ MOMENTUM PARAMS AS NEEDED (Jaden Cai test strategy) ############
    def __init__(self, position_size: float = 10.0, sma_period: int = 20, std_dev: float = 1.5, momentum_period: int = 5, allow_shorts: bool = False):
        """
        Args:
            position_size: shares to trade
            sma_period: period for the moving average (the mean)
            std_dev: number of standard deviations for Bollinger Band width
                     Higher = only trade extreme moves (1.5-2.0 recommended)
            momentum_period: period for momentum confirmation
        """
        if position_size <= 0:
            raise ValueError("position_size must be positive.")
        if sma_period < 5:
            raise ValueError("sma_period must be at least 5.")
        if std_dev <= 0:
            raise ValueError("std_dev must be positive.")
        if momentum_period < 2:
            raise ValueError("momentum_period must be at least 2.")
        self.position_size = position_size
        self.sma_period = sma_period
        self.std_dev = std_dev
        self.momentum_period = momentum_period
        self.stop_loss_pct = 0.03
        self.take_profit_pct = 0.03
        self.min_hold = 3
        self.cooldown_bars = 3
        self.allow_shorts = bool(allow_shorts)

    def add_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        # Calculate the mean
        df["SMA"] = df["Close"].rolling(self.sma_period, min_periods=1).mean()
        
        # Calculate standard deviation
        df["BB_std"] = df["Close"].rolling(self.sma_period, min_periods=1).std().fillna(0)
        
        # Bollinger Bands: mean +/- (std_dev * std)
        df["BB_upper"] = df["SMA"] + (self.std_dev * df["BB_std"])
        df["BB_lower"] = df["SMA"] - (self.std_dev * df["BB_std"])
        
        # Momentum: rate of change
        df["momentum"] = df["Close"].pct_change(self.momentum_period).fillna(0)
        
        # Z-score for deviation from mean
        df["z_score"] = (df["Close"] - df["SMA"]) / (df["BB_std"] + 1e-8)
        
        # RSI calculation for overbought/oversold detection
        delta = df["Close"].diff().fillna(0)
        gain = delta.copy()
        gain[gain < 0] = 0
        loss = -delta.copy()
        loss[loss < 0] = 0
        
        avg_gain = gain.rolling(14).mean()
        avg_loss = loss.rolling(14).mean()
        rs = avg_gain / (avg_loss + 0.0001)
        df["RSI"] = 100 - (100 / (1 + rs))
        df["RSI"] = df["RSI"].fillna(50)
        
        return df

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        # Stateful implementation: iterate rows and manage entry/exit explicitly
        n = len(df)
        signals = np.zeros(n, dtype=int)
        positions = np.zeros(n, dtype=int)

        cur_pos = 0
        entry_price = 0.0
        holding = 0
        cooldown = 0

        for i, idx in enumerate(df.index):
            close = float(df.at[idx, "Close"])
            sma = float(df.at[idx, "SMA"])
            upper = float(df.at[idx, "BB_upper"])
            lower = float(df.at[idx, "BB_lower"])
            rsi = float(df.at[idx, "RSI"]) if "RSI" in df.columns else 50.0
            momentum = float(df.at[idx, "momentum"]) if "momentum" in df.columns else 0.0

            # require a meaningful deviation (z-score) or RSI extreme for entries
            z = float(df.at[idx, "z_score"]) if "z_score" in df.columns else (close - sma) / (max(1e-8, upper - sma))
            buy_condition = ((z < -1.2) or (rsi < 30)) and (momentum > 0)
            sell_condition = ((z > 1.2) or (rsi > 70)) and (momentum < 0)
            if not self.allow_shorts:
                sell_condition = False

            # If flat, consider entry (respect cooldown)
            if cur_pos == 0:
                if cooldown > 0:
                    cooldown -= 1
                else:
                    if buy_condition:
                        signals[i] = 1
                        cur_pos = 1
                        entry_price = close
                        holding = 0
                    elif sell_condition:
                        signals[i] = -1
                        cur_pos = -1
                        entry_price = close
                        holding = 0

            elif cur_pos == 1:
                holding += 1
                # Exit long when price reverts to mean (SMA) after min_hold, or TP/SL triggers
                if ((holding >= self.min_hold and close >= sma)
                        or close >= entry_price * (1 + self.take_profit_pct)
                        or close <= entry_price * (1 - self.stop_loss_pct)):
                    # exit to flat
                    signals[i] = 0
                    cur_pos = 0
                    entry_price = 0.0
                    holding = 0
                    cooldown = self.cooldown_bars

            elif cur_pos == -1:
                holding += 1
                # Exit short when price reverts to mean (SMA) after min_hold, or TP/SL triggers
                if ((holding >= self.min_hold and close <= sma)
                        or close <= entry_price * (1 - self.take_profit_pct)
                        or close >= entry_price * (1 + self.stop_loss_pct)):
                    signals[i] = 0
                    cur_pos = 0
                    entry_price = 0.0
                    holding = 0
                    cooldown = self.cooldown_bars

            positions[i] = cur_pos

        df["signal"] = signals
        df["position"] = positions
        df["target_qty"] = np.abs(df["position"]) * self.position_size
        return df

############################################################################
class MovingAverageStrategy(Strategy):
    """
    Moving average crossover strategy with explicitly defined entry/exit rules.
    """

    def __init__(self, short_window: int = 20, long_window: int = 60, position_size: float = 10.0):
        if short_window >= long_window:
            raise ValueError("short_window must be strictly less than long_window.")
        if position_size <= 0:
            raise ValueError("position_size must be positive.")
        self.short_window = short_window
        self.long_window = long_window
        self.position_size = position_size

    def add_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df["MA_short"] = df["Close"].rolling(self.short_window, min_periods=1).mean()
        df["MA_long"] = df["Close"].rolling(self.long_window, min_periods=1).mean()
        df["returns"] = df["Close"].pct_change().fillna(0.0)
        df["volatility"] = df["returns"].rolling(self.long_window).std().fillna(0.0)
        return df

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        df["signal"] = 0

        buy = (df["MA_short"].shift(1) <= df["MA_long"].shift(1)) & (df["MA_short"] > df["MA_long"])
        sell = (df["MA_short"].shift(1) >= df["MA_long"].shift(1)) & (df["MA_short"] < df["MA_long"])

        df.loc[buy, "signal"] = 1
        df.loc[sell, "signal"] = -1

        df["position"] = 0
        df.loc[df["MA_short"] > df["MA_long"], "position"] = 1
        df.loc[df["MA_short"] < df["MA_long"], "position"] = -1
        df["target_qty"] = df["position"].abs() * self.position_size
        return df


class TemplateStrategy(Strategy):
    """
    Starter strategy template for students. Modify the indicator and signal
    logic to build your own ideas.
    """

    def __init__(
        self,
        lookback: int = 14,
        position_size: float = 10.0,
        buy_threshold: float = 0.01,
        sell_threshold: float = -0.01,
    ):
        if lookback < 1:
            raise ValueError("lookback must be at least 1.")
        if position_size <= 0:
            raise ValueError("position_size must be positive.")
        self.lookback = lookback
        self.position_size = position_size
        self.buy_threshold = buy_threshold
        self.sell_threshold = sell_threshold

    def add_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df["momentum"] = df["Close"].pct_change(self.lookback).fillna(0.0)
        return df

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        df["signal"] = 0

        buy = df["momentum"] > self.buy_threshold
        sell = df["momentum"] < self.sell_threshold

        df.loc[buy, "signal"] = 1
        df.loc[sell, "signal"] = -1

        df["position"] = df["signal"].replace(0, np.nan).ffill().fillna(0)
        df["target_qty"] = df["position"].abs() * self.position_size
        return df


class CryptoTrendStrategy(Strategy):
    """
    Crypto trend-following strategy using fast/slow EMAs (long-only).
    """

    def __init__(self, short_window: int = 7, long_window: int = 21, position_size: float = 100.0):
        if short_window >= long_window:
            raise ValueError("short_window must be strictly less than long_window.")
        if position_size <= 0:
            raise ValueError("position_size must be positive.")
        self.short_window = short_window
        self.long_window = long_window
        self.position_size = position_size

    def add_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df["EMA_fast"] = df["Close"].ewm(span=self.short_window, adjust=False).mean()
        df["EMA_slow"] = df["Close"].ewm(span=self.long_window, adjust=False).mean()
        return df

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        df["signal"] = 0
        long_regime = df["EMA_fast"] > df["EMA_slow"]
        flips = long_regime.astype(int).diff().fillna(0)
        df.loc[flips > 0, "signal"] = 1
        df.loc[flips < 0, "signal"] = -1
        df["position"] = long_regime.astype(int)
        df["target_qty"] = self.position_size
        return df

class DemoStrategy(Strategy):
    """
    Simple demo strategy - buys 1 share when price up, sells 1 share when price down.
    Uses tiny position size to avoid margin/locate issues.

    Usage:
        python run_live.py --symbol AAPL --strategy demo --timeframe 1Min --sleep 5 --live
    """

    def __init__(self, position_size: float = 1.0):
        self.position_size = position_size

    def add_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df["change"] = df["Close"].diff().fillna(0.0)
        return df

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        df["signal"] = 0
        df.loc[df["change"] > 0, "signal"] = 1   # Price went up -> buy
        df.loc[df["change"] < 0, "signal"] = -1  # Price went down -> sell
        df["position"] = df["signal"]
        df["target_qty"] = self.position_size
        return df


## =============================================================================
## CREATE YOUR OWN STRATEGIES BELOW
## =============================================================================
##
## Example: RSI Strategy
##
## class RSIStrategy(Strategy):
##     """Buy when RSI is oversold, sell when overbought."""
##
##     def __init__(self, period=14, oversold=30, overbought=70, position_size=10.0):
##         self.period = period
##         self.oversold = oversold
##         self.overbought = overbought
##         self.position_size = position_size
##
##     def add_indicators(self, df):
##         delta = df['Close'].diff()
##         gain = delta.where(delta > 0, 0).rolling(self.period).mean()
##         loss = (-delta.where(delta < 0, 0)).rolling(self.period).mean()
##         rs = gain / loss
##         df['RSI'] = 100 - (100 / (1 + rs))
##         return df
##
##     def generate_signals(self, df):
##         df['signal'] = 0
##         df.loc[df['RSI'] < self.oversold, 'signal'] = 1   # Buy when oversold
##         df.loc[df['RSI'] > self.overbought, 'signal'] = -1  # Sell when overbought
##         df['position'] = df['signal'].replace(0, np.nan).ffill().fillna(0)
##         df['target_qty'] = self.position_size
##         return df
##
## To use your strategy:
##   python run_live.py --symbol AAPL --strategy mystrategy --live
##
class MomentumStrategy(Strategy):
    """Buy when price has strong upward movement, sell when momentum reverses"""

    """ - fast_period: Short MA period (default 5 bars)
        - slow_period: Long MA period (default 20 bars)  
        - roc_period: How far back to measure rate of change (default 10 bars)
        - roc_threshold: Minimum ROC to trigger trade (default 1.5%)
        - position_size: How many shares/units to trade"""

    def __init__(self, fast_period = 3, slow_period = 12, roc_period = 5, 
                roc_threshold = 0.01,position_size = 10.0):
       
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.roc_period = roc_period
        self.roc_threshold = roc_threshold
        self.position_size = position_size

    def add_indicators(self, df):

        """
        Calculate indicators related to momentum including fast and slow
        moving averages as well as rate of change
        """

        # this calculates the moving averages of last 5 and 20 days
        df['ma_fast'] = df['Close'].rolling(window = self.fast_period).mean()
        df['ma_slow'] = df['Close'].rolling(window = self.slow_period).mean()

        # this calculates rate of change based on how long roc_period is

        df['roc'] = df['Close'].pct_change(periods = self.roc_period)

        
        return df

    def generate_signals(self, df):
        """
        This will generate the buy/sell signals if momentum is met

        Signal values:
        1 = Buy
        -1 = Sell
        0 = Hold
        """
        df['signal'] = 0

        #buy conditions:
        #1. fast MA > slow MA (meaning that there is an uptrend)
        #2. ROC is positive and strong
        #BOTH NEED TO BE TRUE

        buy = (df['ma_fast'] > df['ma_slow']) & (df['roc'] > self.roc_threshold)
        df.loc[buy, 'signal'] = 1

        #sell conditions:
        #1. fast MA < slow MA (meaning downward trend)
        #2. ROC negataive and strong
        # BOTH NEED TO BE TRUE
        sell = (df['ma_fast'] < df['ma_slow']) & (df['roc'] < self.roc_threshold)
        df.loc[sell, 'signal'] = -1

        #track position
        df['position'] = df['signal'].replace(0, np.nan).ffill().fillna(0)
        df['target_qty'] = self.position_size

        return df