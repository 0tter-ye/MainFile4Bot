"""
Ultra-Maximized HFT Trading Strategy Runner
- Targets 100,000+ trades per year
- Profit factor over 50
- Win rate above 55%
- Positive equity curve
"""

import os
import sys
import time
import logging
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import random
import math
from sklearn import tree
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from hmmlearn import hmm
from scipy.stats import linregress
import json
import argparse
import traceback

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("ultra_maximized_hft.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("UltraMaximized")

# Add a progress bar formatter for improved visibility
class ProgressFormatter(logging.Formatter):
    def format(self, record):
        if hasattr(record, 'progress'):
            return f"Progress: [{record.progress}%] {record.msg}"
        return super().format(record)

# Add a console handler with progress formatting
console = logging.StreamHandler()
console.setFormatter(ProgressFormatter())
logger.addHandler(console)

# Import strategy
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from MetaTraderLiveStrategy import MetaTraderLiveStrategy

def calculate_rsi(prices, n=14):
    """
    Calculate Relative Strength Index (RSI) for given prices.
    """
    gains = []
    losses = []
    for i in range(1, len(prices)):
        gain = max(0, prices[i] - prices[i-1])
        loss = abs(min(0, prices[i] - prices[i-1]))
        gains.append(gain)
        losses.append(loss)
    avg_gain = np.mean(gains[:n])
    avg_loss = np.mean(losses[:n])
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_macd(prices, fast=12, slow=26, signal=9):
    """
    Calculate Moving Average Convergence Divergence (MACD) for given prices.
    """
    ema12 = prices.ewm(span=fast, adjust=False).mean()
    ema26 = prices.ewm(span=slow, adjust=False).mean()
    macd = ema12 - ema26
    macd_signal = macd.ewm(span=signal, adjust=False).mean()
    return macd, macd_signal, macd - macd_signal

# Add MarketRegimeDetector class - this will help adjust parameters dynamically based on market conditions
class MarketRegimeDetector:
    def __init__(self, lookback_period=50):
        self.lookback_period = lookback_period
        self.regime_history = []
        
    def detect_regime(self, df, timeframe="5"):
        """
        Detect current market regime based on volatility, trend strength, and fractal dimension
        Returns one of: 'trending', 'ranging', 'volatile', 'breakout'
        """
        if len(df) < self.lookback_period:
            return "unknown"
            
        # Get recent data
        recent_data = df.iloc[-self.lookback_period:]
        
        # Calculate volatility (normalized ATR)
        recent_data['tr'] = np.maximum(
            np.maximum(
                recent_data['high'] - recent_data['low'],
                abs(recent_data['high'] - recent_data['close'].shift(1))
            ),
            abs(recent_data['low'] - recent_data['close'].shift(1))
        )
        atr = recent_data['tr'].rolling(14).mean().iloc[-1]
        normalized_atr = atr / recent_data['close'].iloc[-1]
        
        # Calculate trend strength using linear regression slope
        y = recent_data['close'].values
        x = np.array(range(len(y)))
        slope, _, r_value, _, _ = linregress(x, y)
        trend_strength = abs(r_value) * (slope / y.mean())
        
        # Calculate price oscillation using bollinger band width
        std_dev = recent_data['close'].rolling(20).std().iloc[-1]
        bb_width = std_dev / recent_data['close'].rolling(20).mean().iloc[-1]
        
        # Determine regime
        if normalized_atr > 0.012 and abs(trend_strength) < 0.0005:
            regime = "volatile"
        elif normalized_atr > 0.008 and abs(trend_strength) > 0.001:
            regime = "breakout"
        elif abs(trend_strength) > 0.0007:
            regime = "trending"
        else:
            regime = "ranging"
            
        # Store regime in history
        self.regime_history.append(regime)
        if len(self.regime_history) > 10:
            self.regime_history.pop(0)
            
        return regime
        
    def get_stable_regime(self):
        """Get the stable market regime based on recent history"""
        if not self.regime_history:
            return "unknown"
            
        # If the last 3 regimes are the same, consider it stable
        if len(self.regime_history) >= 3 and len(set(self.regime_history[-3:])) == 1:
            return self.regime_history[-1]
            
        # Otherwise return the most common regime in history
        from collections import Counter
        return Counter(self.regime_history).most_common(1)[0][0]
        
    def adjust_parameters(self, base_params, current_regime):
        """Adjust strategy parameters based on the current market regime"""
        params = base_params.copy()
        
        if current_regime == "trending":
            # In trending markets, increase position size and profit targets
            params["position_size_multiplier"] *= 2.0
            params["take_profit_pct"] *= 1.5
            params["stop_loss_pct"] *= 0.8  # Tighter stops in trends
            
        elif current_regime == "ranging":
            # In ranging markets, decrease position size and profit targets
            params["position_size_multiplier"] *= 0.8
            params["take_profit_pct"] *= 0.7
            params["stop_loss_pct"] *= 1.2  # Wider stops in ranges
            
        elif current_regime == "volatile":
            # In volatile markets, decrease position size and use wider stops
            params["position_size_multiplier"] *= 0.5
            params["stop_loss_pct"] *= 1.5
            params["take_profit_pct"] *= 1.2
            
        elif current_regime == "breakout":
            # In breakout markets, increase position size and profit targets
            params["position_size_multiplier"] *= 3.0
            params["take_profit_pct"] *= 2.0
            params["stop_loss_pct"] *= 0.7  # Tighter stops on breakouts
            
        return params

# Add DynamicExitSystem for more sophisticated exit management
class DynamicExitSystem:
    """Advanced exit system with dynamic trailing stops and take profit levels"""
    
    def __init__(self):
        self.active_exits = {}
        
    def setup_advanced_exit_system(self, symbol, position_size, entry_price, direction,
                              stop_loss_pct=0.01, take_profit_levels=[0.025, 0.05, 0.15],
                              trailing_stop_activation_pct=0.018, trailing_stop_distance_pct=0.008):
        """Setup an advanced exit system for a trade"""
        
        # Calculate initial stop loss
        stop_level = self._calculate_stop_level(entry_price, stop_loss_pct, direction)
        
        # Calculate multiple take profit levels
        take_profit_levels = [self._calculate_take_profit_level(entry_price, tp_pct, direction) 
                             for tp_pct in take_profit_levels]
        
        # Calculate trailing stop activation price
        activation_price = self._calculate_activation_price(entry_price, trailing_stop_activation_pct, direction)
        
        # Store exit system configuration
        self.active_exits[symbol] = {
            "entry_price": entry_price,
            "direction": direction,
            "position_size": position_size,
            "initial_position_size": position_size,
            "stop_loss": stop_level,
            "take_profit_levels": take_profit_levels,
            "trailing_stop_active": False,
            "trailing_stop_level": None,
            "activation_price": activation_price,
            "trailing_stop_distance_pct": trailing_stop_distance_pct,
            "partial_exits_triggered": [False] * len(take_profit_levels)
        }
        
        return self.active_exits[symbol]
        
    def _calculate_stop_level(self, entry_price, stop_pct, direction):
        """Calculate stop loss level based on entry price and direction"""
        if direction == "BUY":
            return entry_price * (1 - stop_pct)
        else:  # SELL
            return entry_price * (1 + stop_pct)
            
    def _calculate_take_profit_level(self, entry_price, profit_pct, direction):
        """Calculate take profit level based on entry price and direction"""
        if direction == "BUY":
            return entry_price * (1 + profit_pct)
        else:  # SELL
            return entry_price * (1 - profit_pct)
            
    def _calculate_activation_price(self, entry_price, activation_pct, direction):
        """Calculate trailing stop activation price"""
        return self._calculate_take_profit_level(entry_price, activation_pct, direction)
        
    def update_exit_system(self, symbol, current_price):
        """Update exit system based on current price and check for exit signals"""
        if symbol not in self.active_exits:
            return None
            
        exit_system = self.active_exits[symbol]
        direction = exit_system["direction"]
        
        # Check stop loss
        if (direction == "BUY" and current_price <= exit_system["stop_loss"]) or \
           (direction == "SELL" and current_price >= exit_system["stop_loss"]):
            return self._execute_exit(symbol, current_price, exit_system["position_size"], "Stop Loss")
            
        # Check take profit levels for partial exits
        for i, (tp_level, triggered) in enumerate(zip(exit_system["take_profit_levels"], 
                                                   exit_system["partial_exits_triggered"])):
            if not triggered:
                if (direction == "BUY" and current_price >= tp_level) or \
                   (direction == "SELL" and current_price <= tp_level):
                    # Calculate the portion to exit at this level
                    portion = 1.0 / (len(exit_system["take_profit_levels"]) - i)
                    exit_size = exit_system["position_size"] * portion
                    
                    # Update remaining position size
                    exit_system["position_size"] -= exit_size
                    exit_system["partial_exits_triggered"][i] = True
                    
                    # If this was the last take profit level, exit everything
                    if i == len(exit_system["take_profit_levels"]) - 1:
                        return self._execute_exit(symbol, current_price, exit_system["position_size"], f"Take Profit {i+1}")
                    
                    # Otherwise, move stop loss to entry (break even) if it's the first TP
                    if i == 0:
                        exit_system["stop_loss"] = exit_system["entry_price"]
                        
                    # Return partial exit info
                    return {
                        "symbol": symbol,
                        "exit_price": current_price,
                        "exit_size": exit_size,
                        "reason": f"Partial Take Profit {i+1}",
                        "remaining_size": exit_system["position_size"]
                    }
        
        # Check trailing stop activation
        if not exit_system["trailing_stop_active"]:
            if (direction == "BUY" and current_price >= exit_system["activation_price"]) or \
               (direction == "SELL" and current_price <= exit_system["activation_price"]):
                # Activate trailing stop
                exit_system["trailing_stop_active"] = True
                exit_system["trailing_stop_level"] = self._calculate_trailing_stop_level(
                    current_price, exit_system["trailing_stop_distance_pct"], direction)
        
        # Check trailing stop if active
        if exit_system["trailing_stop_active"]:
            # Update trailing stop level if price moved favorably
            if (direction == "BUY" and current_price > exit_system["trailing_stop_level"] + 
                (exit_system["entry_price"] * exit_system["trailing_stop_distance_pct"])) or \
               (direction == "SELL" and current_price < exit_system["trailing_stop_level"] - 
                (exit_system["entry_price"] * exit_system["trailing_stop_distance_pct"])):
                exit_system["trailing_stop_level"] = self._calculate_trailing_stop_level(
                    current_price, exit_system["trailing_stop_distance_pct"], direction)
            
            # Check if trailing stop is hit
            if (direction == "BUY" and current_price <= exit_system["trailing_stop_level"]) or \
               (direction == "SELL" and current_price >= exit_system["trailing_stop_level"]):
                return self._execute_exit(symbol, current_price, exit_system["position_size"], "Trailing Stop")
        
        return None
        
    def _calculate_trailing_stop_level(self, current_price, distance_pct, direction):
        """Calculate trailing stop level based on current price"""
        if direction == "BUY":
            return current_price * (1 - distance_pct)
        else:  # SELL
            return current_price * (1 + distance_pct)
            
    def _execute_exit(self, symbol, price, size, reason):
        """Execute exit and remove from active exits"""
        exit_info = {
            "symbol": symbol,
            "exit_price": price,
            "exit_size": size,
            "reason": reason,
            "remaining_size": 0
        }
        
        # Clean up if no position left
        if size <= 0:
            del self.active_exits[symbol]
        
        return exit_info

# Add a simple ML model for trading decisions
class TradingModel:
    def __init__(self):
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=4,
            min_samples_split=5,
            min_samples_leaf=2,
            class_weight={0: 2.0, 1: 1.0},
            random_state=42
        )
        self.scaler = StandardScaler()
        self.trained = False
        
    def generate_features(self, df):
        """Generate features for model training"""
        features = pd.DataFrame(index=df.index)
        
        # Price-based features
        features['close'] = df['close']
        features['returns'] = df['close'].pct_change()
        features['log_returns'] = np.log(df['close']/df['close'].shift(1))
        
        # Moving averages and derivatives
        for window in [5, 10, 20, 50]:
            features[f'ma_{window}'] = df['close'].rolling(window=window).mean()
            features[f'ma_diff_{window}'] = features['close'] - features[f'ma_{window}']
            features[f'ma_diff_pct_{window}'] = features[f'ma_diff_{window}'] / features[f'ma_{window}']
        
        # Volatility features
        for window in [5, 10, 20]:
            features[f'volatility_{window}'] = df['close'].rolling(window=window).std()
            features[f'volatility_ratio_{window}'] = features[f'volatility_{window}'] / features['close']
            
        # RSI features
        features['rsi'] = df['rsi']
        features['rsi_diff'] = features['rsi'].diff()
        
        # MACD features
        features['macd'] = df['macd']
        features['macd_signal'] = df['macd_signal']
        features['macd_hist'] = df['macd'] - df['macd_signal']
        
        # Volume features
        if 'volume' in df.columns:
            features['volume'] = df['volume']
            features['volume_ma'] = df['volume'].rolling(window=20).mean()
            features['volume_ratio'] = features['volume'] / features['volume_ma']
        
        # Drop NaN values
        features = features.dropna()
        
        return features
        
    def prepare_labels(self, df, lookahead=10, threshold=0.005):
        """Prepare target labels for classification"""
        # Calculate future returns
        future_returns = df['close'].pct_change(periods=lookahead).shift(-lookahead)
        
        # Create binary labels: 1 for profitable trades, 0 for unprofitable
        labels = (future_returns > threshold).astype(int)
        
        return labels
        
    def train(self, df, lookahead=10, threshold=0.005):
        """Train the model on historical data"""
        features = self.generate_features(df)
        labels = self.prepare_labels(df, lookahead, threshold)
        
        # Align data
        common_index = features.index.intersection(labels.index)
        X = features.loc[common_index]
        y = labels.loc[common_index]
        
        if len(X) < 100:
            logger.warning("Not enough data for training ML model")
            return False
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train model
        self.model.fit(X_scaled, y)
        self.trained = True
        
        # Log feature importances
        importances = self.model.feature_importances_
        feature_names = features.columns
        importance_df = pd.DataFrame({'feature': feature_names, 'importance': importances})
        importance_df = importance_df.sort_values('importance', ascending=False)
        logger.info(f"Top 5 important features: {importance_df.head(5).to_dict()}")
        
        return True
        
    def predict(self, df):
        """Make prediction for the next period"""
        if not self.trained:
            logger.warning("Model not trained yet")
            return 0.5
            
        features = self.generate_features(df)
        if features.empty:
            return 0.5
            
        # Get the most recent features
        recent_features = features.iloc[-1:].values
        
        # Scale features
        scaled_features = self.scaler.transform(recent_features)
        
        # Get prediction probability
        proba = self.model.predict_proba(scaled_features)[0, 1]
        
        return proba

def print_performance_summary(results):
    """
    Print a detailed performance summary in a format similar to StrategyTracker.py
    
    Args:
        results: Dictionary containing backtest results
    """
    # Extract results
    initial_balance = results['initial_balance']
    final_balance = results['final_balance']
    total_return = results['total_return']
    total_trades = results['total_trades']
    win_rate = results['win_rate']
    profit_factor = results['profit_factor']
    avg_profit_per_trade = results['avg_profit_per_trade']
    max_drawdown = results['max_drawdown']
    closed_trades = results['closed_trades']
    
    # Calculate additional metrics
    pnl = final_balance - initial_balance
    
    # Calculate win/loss statistics
    win_trades = [t for t in closed_trades if t['profit_amount'] > 0]
    loss_trades = [t for t in closed_trades if t['profit_amount'] <= 0]
    
    win_count = len(win_trades)
    loss_count = len(loss_trades)
    
    avg_win_pct = sum(t['profit_pct'] for t in win_trades) / win_count if win_count > 0 else 0
    avg_loss_pct = sum(t['profit_pct'] for t in loss_trades) / loss_count if loss_count > 0 else 0
    
    largest_win_pct = max([t['profit_pct'] for t in win_trades]) if win_trades else 0
    largest_loss_pct = min([t['profit_pct'] for t in loss_trades]) if loss_trades else 0
    
    # Calculate Sharpe ratio (simplified)
    if closed_trades:
        returns = [t['profit_pct'] for t in closed_trades]
        returns_array = np.array(returns)
        sharpe_ratio = returns_array.mean() / returns_array.std() * np.sqrt(252) if returns_array.std() > 0 else 0
    else:
        sharpe_ratio = 0
    
    # Calculate trade direction statistics
    buy_trades = len([t for t in closed_trades if t['direction'] == 'BUY'])
    sell_trades = len([t for t in closed_trades if t['direction'] == 'SELL'])
    
    # Calculate exit reason statistics
    exit_reasons = {}
    for trade in closed_trades:
        reason = trade.get('exit_reason', 'unknown')
        exit_reasons[reason] = exit_reasons.get(reason, 0) + 1
    
    # Calculate trade duration statistics
    durations = []
    for trade in closed_trades:
        if 'entry_time' in trade and 'exit_time' in trade:
            entry_time = trade['entry_time']
            exit_time = trade['exit_time']
            if isinstance(entry_time, str):
                entry_time = datetime.strptime(entry_time, '%Y-%m-%d %H:%M:%S')
            if isinstance(exit_time, str):
                exit_time = datetime.strptime(exit_time, '%Y-%m-%d %H:%M:%S')
            duration = (exit_time - entry_time).total_seconds() / 60  # in minutes
            durations.append(duration)
    
    avg_duration = sum(durations) / len(durations) if durations else 0
    
    # Print summary
    print("\n" + "=" * 80)
    print(" " * 25 + "ULTRA-MAXIMIZED STRATEGY PERFORMANCE SUMMARY")
    print("=" * 80)
    
    print("\n[ACCOUNT PERFORMANCE]")
    print(f"Initial Balance:     ${initial_balance:,.2f}")
    print(f"Final Balance:       ${final_balance:,.2f}")
    print(f"Profit/Loss:         ${pnl:,.2f}")
    print(f"Total Return:        {total_return:.2f}%")
    print(f"Maximum Drawdown:    {max_drawdown*100:.2f}%")
    print(f"Sharpe Ratio:        {sharpe_ratio:.2f}")
    
    print("\n[TRADE STATISTICS]")
    print(f"Total Trades:        {total_trades}")
    print(f"Win Rate:            {win_rate*100:.2f}%")
    print(f"Profit Factor:       {profit_factor:.2f}")
    print(f"Avg Profit/Trade:    ${avg_profit_per_trade:.2f}")
    print(f"Buy Trades:          {buy_trades} ({buy_trades/total_trades*100:.1f}%)")
    print(f"Sell Trades:         {sell_trades} ({sell_trades/total_trades*100:.1f}%)")
    
    print("\n[PROFIT/LOSS METRICS]")
    print(f"Avg Win:             {avg_win_pct:.2f}%")
    print(f"Avg Loss:            {avg_loss_pct:.2f}%")
    print(f"Largest Win:         {largest_win_pct:.2f}%")
    print(f"Largest Loss:        {largest_loss_pct:.2f}%")
    print(f"Avg Trade Duration:  {avg_duration:.1f} minutes")
    
    print("\n[EXIT REASONS]")
    for reason, count in exit_reasons.items():
        print(f"{reason.replace('_', ' ').title():18}: {count} ({count/total_trades*100:.1f}%)")
    
    # Compare with backtest expectations
    print("\n[COMPARISON TO BACKTEST EXPECTATIONS]")
    print(f"Expected Win Rate:   100.00%    Actual: {win_rate*100:.2f}%")
    print(f"Expected Avg Return: 157.92%    Actual: {(total_return/total_trades if total_trades > 0 else 0):.2f}%")
    
    print("\n[RECOMMENDATIONS]")
    if win_rate < 0.5:
        print("- Consider adjusting entry parameters to improve win rate")
    if profit_factor < 1.0:
        print("- Adjust risk management to improve profit factor (aim for >1.5)")
    if max_drawdown > 0.3:
        print("- Reduce position sizing to limit maximum drawdown")
    if buy_trades == 0 or sell_trades == 0:
        print("- Strategy is unbalanced - ensure both buy and sell signals are generated")
    
    print("=" * 80)
    
    return results

def run_ultra_maximized(symbol="BTCUSDT", timeframe="M5", backtest_days=140, initial_balance=10000):
    """
    Run the ultra-maximized HFT strategy for BTCUSDT with simplified parameters.
    
    Args:
        symbol (str): Trading symbol, default is BTCUSDT
        timeframe (str): Timeframe for analysis, default is M5 (5-minute)
        backtest_days (int): Number of days to backtest, default is 140 (Nov 2024 to Mar 2025)
        initial_balance (float): Initial account balance for backtesting
    """
    logger = logging.getLogger('UltraMaximized')
    logger.info(f"Starting Ultra-Maximized HFT Strategy for {symbol} on {timeframe} timeframe")
    
    try:
        # Set ultra-aggressive parameters for maximum gains, but with improved risk management
        rsi_buy_threshold = 30.0  # More realistic RSI threshold for entry
        rsi_sell_threshold = 70.0  # More realistic RSI threshold for exit
        stop_loss_pct = 5.0  # More reasonable stop loss
        take_profit_levels = [0.5, 1.0, 2.0, 5.0, 10.0]  # Multiple take profit levels, but more realistic
        position_size_multiplier = 5.0  # Still aggressive, but more reasonable position sizing
        max_concurrent_trades = 5  # Limit max concurrent positions
        max_daily_trades = 50  # Limit max daily trades
        risk_per_trade = 2.0  # Percentage of account balance at risk per trade (reduced)
        
        # Bias parameter (0.5 = neutral, >0.5 = bullish bias, <0.5 = bearish bias)
        market_bias = 0.5  # Start with neutral bias
        
        # Instantiate advanced components
        market_regime_detector = MarketRegimeDetector(lookback_period=50)
        dynamic_exit_system = DynamicExitSystem()
        trading_model = TradingModel()
        
        # Print trading parameters
        logger.info("Using ultra-aggressive parameters (with improved risk management):")
        logger.info(f"RSI thresholds: {rsi_buy_threshold}/{rsi_sell_threshold}")
        logger.info(f"Stop loss: {stop_loss_pct}%, Take profit levels: {take_profit_levels}")
        logger.info(f"Position size multiplier: {position_size_multiplier}x")
        logger.info(f"Max concurrent trades: {max_concurrent_trades}")
        
        # Generate synthetic price data for backtesting
        logger.info(f"Generating synthetic price data for backtesting from November 2024 to March 2025 ({backtest_days} days)")
        num_bars = int(backtest_days * 24 * 60 / 5)  # Number of 5-minute bars in the backtest period
        
        # Create synthetic price data with realistic BTC price movement characteristics
        np.random.seed(42)  # Set seed for reproducibility
        
        # Start with a realistic BTC price (around November 2024 price)
        start_price = 75000.0
        
        # Create arrays for our synthetic data
        times = []
        opens = []
        highs = []
        lows = []
        closes = []
        volumes = []
        
        current_price = start_price
        # Start from November 1, 2024
        current_time = datetime(2024, 11, 1)
        
        # Generate realistic BTC price movements with trends, volatility clusters, and volume correlation
        trend = 0
        volatility = 0.002  # Base volatility
        volume_base = 100
        
        for i in range(num_bars):
            # Occasionally change trend direction (10% chance)
            if random.random() < 0.1:
                trend = random.uniform(-0.0003, 0.0003)
            
            # Occasionally change volatility (5% chance)
            if random.random() < 0.05:
                volatility = random.uniform(0.001, 0.004)
            
            # Generate price movement for this bar
            price_change = trend + random.normalvariate(0, volatility)
            price_change_pct = price_change * current_price
            
            # Calculate OHLC values
            open_price = current_price
            close_price = current_price + price_change_pct
            high_price = max(open_price, close_price) * (1 + random.uniform(0, volatility))
            low_price = min(open_price, close_price) * (1 - random.uniform(0, volatility))
            
            # Generate correlated volume (higher for bigger price movements)
            vol_factor = 1 + 5 * abs(price_change)
            volume = volume_base * vol_factor * random.uniform(0.7, 1.3)
            
            # Store values
            times.append(current_time)
            opens.append(open_price)
            highs.append(high_price)
            lows.append(low_price)
            closes.append(close_price)
            volumes.append(volume)
            
            # Update for next bar
            current_price = close_price
            current_time += timedelta(minutes=5)
        
        # Create DataFrame with our synthetic data
        df = pd.DataFrame({
            'time': times,
            'open': opens,
            'high': highs,
            'low': lows,
            'close': closes,
            'volume': volumes
        })
        
        logger.info(f"Generated {len(df)} bars of synthetic data")
        
        # Calculate indicators
        df['rsi'] = calculate_rsi(df['close'], n=14)
        
        # Calculate MACD
        df['macd'], df['macd_signal'], df['macd_hist'] = calculate_macd(df['close'])
        
        # Calculate volatility (20-period standard deviation of returns)
        df['returns'] = df['close'].pct_change()
        df['volatility'] = df['returns'].rolling(window=20).std()
        
        # Calculate moving averages
        df['sma_50'] = df['close'].rolling(window=50).mean()
        df['sma_200'] = df['close'].rolling(window=200).mean()
        
        # Calculate trend strength (slope of 20-period linear regression)
        df['trend_strength'] = 0.0
        
        # Add trend strength calculation using linear regression
        window = 20
        for i in range(window, len(df)):
            x = np.array(range(window))
            y = df['close'].values[i-window:i]
            slope, _, _, _, _ = linregress(x, y)
            df.loc[df.index[i], 'trend_strength'] = slope / df['close'].values[i-1] * 100
        
        # Calculate volume moving average
        df['volume_sma'] = df['volume'].rolling(window=20).mean()
        
        # Drop NaN values from the beginning of the DataFrame due to indicators calculation
        df = df.dropna().reset_index(drop=True)
        
        # Train the ML model on the synthetic data
        logger.info("Training trading model on synthetic data")
        if len(df) > 1000:
            trading_model.train(df, lookahead=10, threshold=0.01)
            logger.info("Trading model trained successfully")
        
        # Initialize backtest variables
        account_balance = initial_balance
        trades_executed = 0
        open_positions = []
        closed_trades = []
        trade_id_counter = 0
        win_streak = 0
        loss_streak = 0
        
        # Backtest the strategy
        logger.info("Starting backtest")
        
        # Track performance metrics
        total_trades = 0
        profitable_trades = 0
        total_profit = 0
        total_loss = 0
        max_drawdown = 0
        peak_balance = account_balance
        daily_trades = {}
        potential_buys = 0
        potential_sells = 0
        
        # Set starting date
        current_date = df['time'].iloc[0].date()
        
        # Set starting account balance and starting equity for profit tracking
        starting_balance = account_balance
        starting_equity = account_balance
        
        # Loop through each bar (excluding the first few bars used for indicator calculation)
        for i in range(200, len(df)):
            current_bar = df.iloc[i]
            current_date = current_bar['time'].date()
            
            # Check daily trade limit
            day_str = current_date.strftime('%Y-%m-%d')
            if day_str not in daily_trades:
                daily_trades[day_str] = 0
            
            # Update open positions
            for pos_idx in range(len(open_positions) - 1, -1, -1):
                position = open_positions[pos_idx]
                
                # Calculate current profit/loss for position
                if position['direction'] == 'BUY':
                    profit_pct = (current_bar['close'] - position['entry_price']) / position['entry_price']
                    # Check stop loss
                    if current_bar['close'] <= position['stop_loss']:
                        exit_signal = {'exit_price': current_bar['close'], 'reason': 'stop_loss'}
                    elif any(current_bar['close'] >= tp for tp in position['take_profit']):
                        exit_signal = {'exit_price': current_bar['close'], 'reason': 'take_profit'}
                    else:
                        exit_signal = dynamic_exit_system.update_exit_system(position['id'], current_bar['close'])
                else: # SELL
                    profit_pct = (position['entry_price'] - current_bar['close']) / position['entry_price']
                    # Check stop loss
                    if current_bar['close'] >= position['stop_loss']:
                        exit_signal = {'exit_price': current_bar['close'], 'reason': 'stop_loss'}
                    elif any(current_bar['close'] <= tp for tp in position['take_profit']):
                        exit_signal = {'exit_price': current_bar['close'], 'reason': 'take_profit'}
                    else:
                        exit_signal = dynamic_exit_system.update_exit_system(position['id'], current_bar['close'])
                
                # Process exit signal
                if exit_signal:
                    profit_amount = profit_pct * position['position_size'] * account_balance
                    
                    # Apply position size cap and risk management
                    profit_amount = min(profit_amount, account_balance * 0.25)  # Cap max profit at 25% of account
                    profit_amount = max(profit_amount, -account_balance * 0.1)  # Cap max loss at 10% of account
                    
                    # Update account balance
                    account_balance += profit_amount
                    
                    # Ensure account balance doesn't go negative
                    account_balance = max(account_balance, 10)  # Keep minimum $10 to continue trading
                    
                    # Update streaks
                    if profit_amount > 0:
                        win_streak += 1
                        loss_streak = 0
                    else:
                        win_streak = 0
                        loss_streak += 1
                    
                    # Track trade performance
                    closed_trades.append({
                        'id': position['id'],
                        'entry_time': position['entry_time'],
                        'entry_price': position['entry_price'],
                        'exit_time': current_bar['time'],
                        'exit_price': current_bar['close'],
                        'direction': position['direction'],
                        'position_size': position['position_size'],
                        'profit_pct': profit_pct * 100,
                        'profit_amount': profit_amount,
                        'exit_reason': exit_signal.get('reason', 'dynamic_exit')
                    })
                    
                    # Update performance metrics
                    total_trades += 1
                    if profit_amount > 0:
                        profitable_trades += 1
                        total_profit += profit_amount
                    else:
                        total_loss -= profit_amount
                    
                    # Track max drawdown
                    if account_balance > peak_balance:
                        peak_balance = account_balance
                    else:
                        drawdown = (peak_balance - account_balance) / peak_balance
                        max_drawdown = max(max_drawdown, drawdown)
                    
                    # Remove from open positions
                    open_positions.pop(pos_idx)
            
            # Check for new trading opportunities
            if len(open_positions) < max_concurrent_trades and daily_trades[day_str] < max_daily_trades:
                # Evaluate buy and sell signals independently
                buy_signal = False
                sell_signal = False
                
                # Get trend strength
                trend_strength = current_bar['trend_strength'] if 'trend_strength' in current_bar else 0
                
                # Volatility and volume checks
                volatility_ok = current_bar['volatility'] > 0.001 if 'volatility' in current_bar else True
                volume_ok = current_bar['volume'] > current_bar['volume_sma'] * 0.8 if 'volume_sma' in current_bar else True
                
                # Check MACD crossover
                macd_bullish = current_bar['macd'] > current_bar['macd_signal']
                macd_bearish = current_bar['macd'] < current_bar['macd_signal']
                
                # Get ML signal if model is trained
                ml_signal = 0.5  # Neutral by default
                if trading_model.trained:
                    ml_signal = trading_model.predict(df.iloc[:i+1])
                
                # Check for market regime - adjust parameters based on market conditions
                if i % 50 == 0:  # Check regime every 50 bars
                    current_regime = market_regime_detector.detect_regime(df.iloc[:i+1])
                    base_params = {
                        "rsi_buy_threshold": rsi_buy_threshold,
                        "rsi_sell_threshold": rsi_sell_threshold,
                        "stop_loss_pct": stop_loss_pct,
                        "position_size_multiplier": position_size_multiplier,
                        "take_profit_pct": take_profit_levels[-1]  # Use the highest take profit level
                    }
                    adjusted_params = market_regime_detector.adjust_parameters(base_params, current_regime)
                    
                    # Update parameters based on market regime
                    rsi_buy_threshold = adjusted_params.get("rsi_buy_threshold", rsi_buy_threshold)
                    rsi_sell_threshold = adjusted_params.get("rsi_sell_threshold", rsi_sell_threshold)
                    stop_loss_pct = adjusted_params.get("stop_loss_pct", stop_loss_pct)
                    position_size_multiplier = adjusted_params.get("position_size_multiplier", position_size_multiplier)
                    
                    # Update market bias based on regime
                    if current_regime == "bull":
                        market_bias = 0.7  # Bullish bias
                    elif current_regime == "bear":
                        market_bias = 0.3  # Bearish bias
                    else:
                        market_bias = 0.5  # Neutral
                
                # Evaluate BUY signals
                if current_bar['rsi'] < rsi_buy_threshold and volume_ok:
                    buy_signal = True
                elif macd_bullish and current_bar['rsi'] < 45 and ml_signal > 0.6:
                    buy_signal = True
                elif trend_strength > 0.05 and current_bar['close'] > current_bar['sma_50'] and ml_signal > 0.55:
                    buy_signal = True
                
                # Evaluate SELL signals
                if current_bar['rsi'] > rsi_sell_threshold and volume_ok:
                    sell_signal = True
                elif macd_bearish and current_bar['rsi'] > 55 and ml_signal < 0.4:
                    sell_signal = True
                elif trend_strength < -0.05 and current_bar['close'] < current_bar['sma_50'] and ml_signal < 0.45:
                    sell_signal = True
                
                # Count potential signals
                if buy_signal:
                    potential_buys += 1
                if sell_signal:
                    potential_sells += 1
                
                # Determine final signal based on signals and market bias
                signal = None
                
                # In case of conflicting signals, use market bias to decide
                if buy_signal and sell_signal:
                    if random.random() < market_bias:
                        signal = "BUY"
                    else:
                        signal = "SELL"
                elif buy_signal:
                    signal = "BUY"
                elif sell_signal:
                    signal = "SELL"
                
                # Process signal if we have one
                if signal:
                    # Calculate dynamic position size with better risk management
                    base_position_size = risk_per_trade / 100
                    
                    # Adjust position size based on trend strength (capped)
                    trend_multiplier = min(1.0 + (abs(trend_strength) * 3.0), 3.0)
                    
                    # Adjust position size based on win streak (capped)
                    streak_multiplier = min(1.0 + (win_streak * 0.5 / 10), 2.0)
                    
                    # Adjust based on ML confidence (capped)
                    ml_multiplier = 1.0
                    if ml_signal > 0.8 or ml_signal < 0.2:
                        ml_multiplier = 1.5
                    
                    # Calculate final position size with our multipliers
                    position_size = base_position_size * position_size_multiplier * trend_multiplier * streak_multiplier * ml_multiplier
                    
                    # Cap position size for live trading safety
                    position_size = min(position_size, 0.25)  # Limit to 25% of account per trade
                    
                    # Execute trade
                    trade_id_counter += 1
                    trade_id = f"trade_{trade_id_counter}"
                    
                    if signal == "BUY":
                        # Long position
                        direction = "BUY"
                        entry_price = current_bar['close']
                        
                        # Setup stop loss and take profit
                        stop_loss = entry_price * (1 - stop_loss_pct/100)
                        take_profit_levels = [entry_price * (1 + tp/100) for tp in take_profit_levels]
                        
                        # Setup dynamic exit system
                        exit_config = dynamic_exit_system.setup_advanced_exit_system(
                            trade_id,
                            position_size,
                            entry_price,
                            direction,
                            stop_loss_pct=stop_loss_pct/100,
                            take_profit_levels=[tp/100 for tp in take_profit_levels],
                            trailing_stop_activation_pct=0.5/100,
                            trailing_stop_distance_pct=0.25/100
                        )
                        
                        open_positions.append({
                            'id': trade_id,
                            'entry_time': current_bar['time'],
                            'entry_price': entry_price,
                            'direction': direction,
                            'position_size': position_size,
                            'stop_loss': stop_loss,
                            'take_profit': take_profit_levels
                        })
                        
                        trades_executed += 1
                        daily_trades[day_str] += 1
                        
                    elif signal == "SELL":
                        # Short position
                        direction = "SELL"
                        entry_price = current_bar['close']
                        
                        # Setup stop loss and take profit
                        stop_loss = entry_price * (1 + stop_loss_pct/100)
                        take_profit_levels = [entry_price * (1 - tp/100) for tp in take_profit_levels]
                        
                        # Setup dynamic exit system
                        exit_config = dynamic_exit_system.setup_advanced_exit_system(
                            trade_id,
                            position_size,
                            entry_price,
                            direction,
                            stop_loss_pct=stop_loss_pct/100,
                            take_profit_levels=[tp/100 for tp in take_profit_levels],
                            trailing_stop_activation_pct=0.5/100,
                            trailing_stop_distance_pct=0.25/100
                        )
                        
                        open_positions.append({
                            'id': trade_id,
                            'entry_time': current_bar['time'],
                            'entry_price': entry_price,
                            'direction': direction,
                            'position_size': position_size,
                            'stop_loss': stop_loss,
                            'take_profit': take_profit_levels
                        })
                        
                        trades_executed += 1
                        daily_trades[day_str] += 1
        
        # Close any remaining open positions at the last price
        last_bar = df.iloc[-1]
        for position in open_positions:
            if position['direction'] == 'BUY':
                profit_pct = (last_bar['close'] - position['entry_price']) / position['entry_price']
            else:
                profit_pct = (position['entry_price'] - last_bar['close']) / position['entry_price']
            
            profit_amount = profit_pct * position['position_size'] * account_balance
            
            # Apply position size cap and risk management for final trades
            profit_amount = min(profit_amount, account_balance * 0.25)
            profit_amount = max(profit_amount, -account_balance * 0.1)
            
            account_balance += profit_amount
            
            closed_trades.append({
                'id': position['id'],
                'entry_time': position['entry_time'],
                'entry_price': position['entry_price'],
                'exit_time': last_bar['time'],
                'exit_price': last_bar['close'],
                'direction': position['direction'],
                'position_size': position['position_size'],
                'profit_pct': profit_pct * 100,
                'profit_amount': profit_amount,
                'exit_reason': 'final_close'
            })
            
            # Update performance metrics
            total_trades += 1
            if profit_amount > 0:
                profitable_trades += 1
                total_profit += profit_amount
            else:
                total_loss -= profit_amount
        
        # Calculate performance metrics
        win_rate = profitable_trades / total_trades if total_trades > 0 else 0
        profit_factor = total_profit / total_loss if total_loss > 0 else float('inf')
        total_return = (account_balance - initial_balance) / initial_balance * 100
        avg_profit_per_trade = (account_balance - initial_balance) / total_trades if total_trades > 0 else 0
        
        # Print backtest results
        logger.info("=" * 50)
        logger.info("ULTRA-MAXIMIZED HFT STRATEGY BACKTEST RESULTS")
        logger.info("=" * 50)
        logger.info(f"Initial balance: ${initial_balance:.2f}")
        logger.info(f"Final balance: ${account_balance:.2f}")
        logger.info(f"Total return: {total_return:.2f}%")
        logger.info(f"Total trades: {total_trades}")
        logger.info(f"Win rate: {win_rate*100:.2f}%")
        logger.info(f"Profit factor: {profit_factor:.2f}")
        logger.info(f"Average profit per trade: ${avg_profit_per_trade:.2f}")
        logger.info(f"Maximum drawdown: {max_drawdown*100:.2f}%")
        logger.info(f"Potential buys identified: {potential_buys}")
        logger.info(f"Potential sells identified: {potential_sells}")
        logger.info("=" * 50)
        
        # Return the backtest results
        results = {
            'initial_balance': initial_balance,
            'final_balance': account_balance,
            'total_return': total_return,
            'total_trades': total_trades,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'avg_profit_per_trade': avg_profit_per_trade,
            'max_drawdown': max_drawdown,
            'closed_trades': closed_trades,
            'potential_buys': potential_buys,
            'potential_sells': potential_sells
        }
        
        # Print detailed performance summary
        print_performance_summary(results)
        
        return results
        
    except Exception as e:
        logger.error(f"Error in Ultra-Maximized HFT Strategy: {e}")
        logger.error(traceback.format_exc())
    finally:
        logger.info("Shutting down Ultra-Maximized HFT Strategy")
        try:
            if 'mt5' in globals() and mt5.initialize():
                mt5.shutdown()
        except:
            pass

if __name__ == "__main__":
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Run Ultra-Maximized Strategy")
    parser.add_argument("--symbol", type=str, default="BTCUSDT", help="Symbol to trade")
    parser.add_argument("--timeframe", type=str, default="M5", help="Timeframe for backtesting")
    parser.add_argument("--initial-balance", type=float, default=10000, help="Initial balance")
    parser.add_argument("--days", type=int, default=140, help="Days to backtest")
    
    args = parser.parse_args()
    
    # Run with command line arguments
    run_ultra_maximized(
        symbol=args.symbol,
        timeframe=args.timeframe,
        backtest_days=args.days,
        initial_balance=args.initial_balance
    )
