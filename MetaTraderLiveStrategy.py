import os
import sys
import time
import json
import logging
import warnings
import argparse
import datetime
from datetime import datetime, timedelta
RESULTS_DIR = "results"
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import linregress, percentileofscore
import scipy.stats
import threading
import warnings
warnings.filterwarnings("ignore", message="talib not installed")
import traceback
import argparse
from StrategyTracker import PerformanceTracker
from xgboost import XGBClassifier  # For XGBClassifier
from lightgbm import LGBMClassifier  # For LGBMClassifier
from sklearn.ensemble import RandomForestClassifier  # For RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier  # For DecisionTreeClassifier
from sklearn.preprocessing import RobustScaler  # For RobustScaler
from sklearn.model_selection import train_test_split  # For train_test_split used in train method
import uuid


try:
    import talib
except ImportError:
    warnings.warn("talib not installed. Using pandas-based RSI calculation instead.")
    talib = None

try:
    import MetaTrader5 as mt5
except ImportError:
    warnings.warn("MetaTrader5 module not found, using simulation mode")
    mt5 = None

    # Constants and configuration
    SYMBOL = "BTCUSDT"  # Default symbol
    TIMEFRAME_MAP = {
    '1m': mt5.TIMEFRAME_M1,
    '5m': mt5.TIMEFRAME_M5,
    '15m': mt5.TIMEFRAME_M15,
    '30m': mt5.TIMEFRAME_M30,
    '1h': mt5.TIMEFRAME_H1,
    '4h': mt5.TIMEFRAME_H4,
    '1d': mt5.TIMEFRAME_D1
    }
    STATE_BINS = {
    "total_return": np.linspace(-20, 20, 11), "unrealized_pnl": np.linspace(-10, 10, 11),
    "atr": np.linspace(0, 500, 11), "rsi_5m": np.linspace(0, 100, 11), "rsi_15m": np.linspace(0, 100, 11),
    "imbalance": np.linspace(-1, 1, 11), "win_rate": np.linspace(0, 100, 11), "rf_prob": np.linspace(0, 1, 11),
    "dt_prob": np.linspace(0, 1, 11), "markov_win_prob": np.linspace(0, 100, 11), "fractal_dim": np.linspace(1, 2, 11),
    "hurst": np.linspace(0, 1, 11), "imbalance_l2": np.linspace(-1, 1, 11), "rnn_prob": np.linspace(0, 1, 11),
    "sentiment": np.linspace(-1, 1, 11), "momentum": np.linspace(-2, 2, 11), "resistance_dist": np.linspace(-500, 500, 11)
    }

    # Configuration
    INITIAL_BALANCE = 50000
    RESULTS_DIR = os.path.join(
    os.path.dirname(
    os.path.abspath(__file__)),
    "results")
    TRADES_FILE_PATTERN = os.path.join(RESULTS_DIR, "live_trades_{}.json")
    TRADE_LOG_FILE = os.path.join(RESULTS_DIR, "trade_log.txt")
    TRACKER_FILE = os.path.join(
    RESULTS_DIR, 'strategy_performance_tracker.csv')
    CHARTS_FOLDER = os.path.join(RESULTS_DIR, 'performance_charts')
    BACKTEST_DATA_FILE = os.path.join(RESULTS_DIR, 'backtest_data.csv')

    # Constants for backtest comparison
    BACKTEST_WIN_RATE = 1.0  # 100% win rate
    BACKTEST_AVG_RETURN = 157.92  # 157.92% average return
    BACKTEST_SHARPE = 3.28  # Sharpe ratio from backtest
    BACKTEST_FINAL_VALUE = 1.56e15  # 1.56 quadrillion from backtest

    # Utility functions


def calculate_rsi(data, periods=14):
    """Calculate RSI indicator"""
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=periods).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=periods).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


def calculate_macd(data, fast=12, slow=26, signal=9):
    """Calculate MACD indicator"""
    exp1 = data.ewm(span=fast, adjust=False).mean()
    exp2 = data.ewm(span=slow, adjust=False).mean()
    macd = exp1 - exp2
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    histogram = macd - signal_line
    return macd, signal_line, histogram


def calculate_bollinger_bands(data, window=20, num_std=2):
    """Calculate Bollinger Bands"""
    ma = data.rolling(window=window).mean()
    std = data.rolling(window=window).std()
    upper = ma + (std * num_std)
    lower = ma - (std * num_std)
    return upper, lower


def calculate_atr(high, low, close, periods=14):
    """Calculate Average True Range (ATR) indicator"""
    high = high.astype(float)
    low = low.astype(float)
    close = close.astype(float)

    # Create a DataFrame for calculations
    df = pd.DataFrame({
    'high': high,
    'low': low,
    'close': close,
    'close_prev': close.shift(1)
    })

    # Calculate True Range
    df['tr1'] = df['high'] - df['low']
    df['tr2'] = abs(df['high'] - df['close_prev'])
    df['tr3'] = abs(df['low'] - df['close_prev'])
    df['tr'] = df[['tr1', 'tr2', 'tr3']].max(axis=1)

    # Calculate ATR
    atr = df['tr'].rolling(window=periods).mean()
    return atr


def calculate_stochastic(high, low, close, k_window=14, d_window=3):
    """Calculate Stochastic Oscillator"""
    lowest_low = low.rolling(window=k_window).min()
    highest_high = high.rolling(window=k_window).max()
    k = 100 * (close - lowest_low) / (highest_high - lowest_low)
    d = k.rolling(window=d_window).mean()
    return k, d


def calculate_adx(high, low, close, window=14):
    """Calculate ADX indicator"""
    # True Range
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window).mean()

    # Plus and Minus Directional Movement
    up_move = high - high.shift(1)
    down_move = low.shift(1) - low

    plus_dm = ((up_move > down_move) & (up_move > 0)) * up_move
    minus_dm = ((down_move > up_move) & (down_move > 0)) * down_move

    # Smoothed DM values
    plus_di = 100 * (plus_dm.rolling(window).mean() / atr)
    minus_di = 100 * (minus_dm.rolling(window).mean() / atr)

    # ADX calculation
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = dx.rolling(window).mean()

    return adx


def calculate_hurst_exponent(time_series, max_lag=20):
    """Calculate Hurst exponent using R/S analysis"""
    lags = range(2, max_lag)
    tau = [np.std(np.subtract(time_series[lag:], time_series[:-lag]))
    for lag in lags]
    poly = np.polyfit(np.log(lags), np.log(tau), 1)
    return poly[0]


def fractal_dimension(series, min_box_size=2, max_box_size=None):
    """Calculate the fractal dimension of a time series"""
    n = len(series)
    
    if max_box_size is None:
        max_box_size = n // 4
    
    # Normalize the series
    series = (series - np.min(series)) / (np.max(series) - np.min(series))
    
    # Calculate box sizes
    box_sizes = np.logspace(np.log10(min_box_size), np.log10(max_box_size), num=10).astype(int)
    box_sizes = np.unique(box_sizes)  # Remove duplicates
    
    counts = []
    
    for box_size in box_sizes:
        if box_size == 0:
            continue
        count = 0
        for i in range(0, n - box_size + 1, box_size):
            count += 1
        counts.append(count)
    
    if len(counts) < 2:
        return 1.0
    coeffs = np.polyfit(np.log(box_sizes), np.log(counts), 1)
    return -coeffs[0]

def apply_kalman_filter(series):
    """
    Apply Kalman filter to smooth the price series
    """
    n = len(series)
    # State transition matrix
    A = np.array([[1, 1], [0, 1]])
    # Observation matrix
    H = np.array([[1, 0]])
    # Process noise covariance
    Q = np.array([[0.01, 0.01], [0.01, 0.01]])
    # Measurement noise covariance
    R = np.array([[0.1]])
    # Initial state estimate
    x = np.array([[series[0]], [0]])
    # Initial error covariance
    P = np.array([[1, 0], [0, 1]])

    filtered_series = np.zeros(n)

    for i in range(n):
        # Predict
        x = A @ x
        P = A @ P @ A.T + Q

        # Update
        z = np.array([[series[i]]])
        y = z - H @ x
        S = H @ P @ H.T + R
        K = P @ H.T @ np.linalg.inv(S)
        x = x + K @ y
        P = (np.eye(2) - K @ H) @ P

        filtered_series[i] = x[0, 0]

    return filtered_series

    def __init__(self, start_value=50000):
        self.start_value = start_value
        self.trades = []
        self.win_loss_sequence = []
        self.current_value = start_value
        self.max_value = start_value
        self.trade_features = []
        
def record_trade(self, entry_price, exit_price, position_size, direction,
                 entry_time, exit_time, trade_features=None):
    """Record a completed trade with all details"""
    pnl = (exit_price - entry_price) * position_size if direction == "long" else \
          (entry_price - exit_price) * position_size
    return_pct = (pnl / (entry_price * position_size)) * 100

    trade = {
        "entry_price": entry_price,
        "exit_price": exit_price,
        "position_size": position_size,
        "direction": direction,
        "entry_time": entry_time,
        "exit_time": exit_time,
        "pnl": pnl,
        "return_pct": return_pct,
        "win": pnl > 0
    }

    if trade_features:
        trade.update(trade_features)
        self.trade_features.append(trade_features)

    self.trades.append(trade)
    self.win_loss_sequence.append(1 if pnl > 0 else 0)
    self.current_value += pnl
    self.max_value = max(self.max_value, self.current_value)

    return trade

    def save_trades_to_json(self):
        """Save all trades to a JSON file"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = TRADES_FILE_PATTERN.format(timestamp)

        # Convert datetime objects to strings for JSON serialization
        serializable_trades = []
        for trade in self.trades:
            trade_copy = trade.copy()
            serializable_trades.append(trade_copy)

        with open(filename, 'w') as f:
            json.dump(serializable_trades, f, indent=4)

        logger.info(f"Saved {len(self.trades)} trades to {filename}")
    
    def calculate_performance_metrics(self):   
        """Calculate comprehensive performance metrics"""
        if not self.trades:
            return {"error": "No trades recorded"}
    
        total_pnl = sum(t["pnl"] for t in self.trades)
        win_trades = [t for t in self.trades if t["win"]]
        loss_trades = [t for t in self.trades if not t["win"]]
        win_rate = len(win_trades) / len(self.trades) if self.trades else 0
    
        avg_win = sum(t["pnl"] for t in win_trades) / len(win_trades) if win_trades else 0
        avg_loss = sum(t["pnl"] for t in loss_trades) / len(loss_trades) if loss_trades else 0
        profit_factor = abs(sum(t["pnl"] for t in win_trades) / sum(t["pnl"] for t in loss_trades)) if loss_trades and sum(t["pnl"] for t in loss_trades) != 0 else float('inf')
    
        # Calculate drawdown
        equity_curve = [self.start_value]
        for trade in self.trades:
            equity_curve.append(equity_curve[-1] + trade["pnl"])
    
        max_drawdown = 0
        peak = equity_curve[0]
        
        return {
            "total_trades": len(self.trades),
            "win_rate": win_rate,
            "total_pnl": total_pnl,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "profit_factor": profit_factor,
            "max_drawdown": max_drawdown,
            "current_value": self.current_value,
            "roi": (self.current_value - self.start_value) / self.start_value
        }

    def __init__(self, lookback_period=50):
        self.lookback_period = lookback_period
        self.regime_history = []

    def detect_regime(self, df, timeframe="5"):
        """"""
        if len(df) < self.lookback_period:
            return 0  # Neutral if not enough data

        # Calculate trend metrics
        price_change = df['close'].pct_change(self.lookback_period).iloc[-1]
        volatility = df['close'].pct_change().rolling(
            self.lookback_period).std().iloc[-1]

        # Calculate volume metrics
        if 'volume' in df.columns:
            volume_change = df['volume'].pct_change(self.lookback_period).iloc[-1]
            volume_trend = df['volume'].rolling(self.lookback_period).mean(
            ).pct_change(self.lookback_period // 2).iloc[-1]
        else:
            volume_change = 0
            volume_trend = 0

        # Calculate momentum
        momentum = df['close'].pct_change(self.lookback_period // 2).iloc[-1]

        # Calculate RSI
        rsi_value = df['rsi'].iloc[-1] if 'rsi' in df.columns else calculate_rsi(
            df['close']).iloc[-1]

        # Determine regime
        regime = 0  # Neutral by default

        # Bullish conditions
        if (price_change > 0.01 and  # Strong positive price change
            momentum > 0 and         # Positive momentum
            rsi_value > 50 and       # RSI above midpoint
            volume_trend > 0):       # Increasing volume
            regime = 1

        # Bearish conditions
        elif (price_change < -0.01 and  # Strong negative price change
              momentum < 0 and          # Negative momentum
              rsi_value < 50 and        # RSI below midpoint
              volume_trend < 0):        # Decreasing volume
            regime = -1

        # Record regime
        self.regime_history.append(regime)
        if len(self.regime_history) > self.lookback_period:
            self.regime_history.pop(0)

        return regime


    def get_stable_regime(self):
        """Get the stable market regime based on recent history"""
        if not self.regime_history:
            return 0

        # Only return a regime if it's been consistent
        if len(self.regime_history) >= 5:
            recent_regimes = self.regime_history[-5:]
            if all(r == 1 for r in recent_regimes):
                return 1  # Stable bullish
            elif all(r == -1 for r in recent_regimes):
                return -1  # Stable bearish

        # Default to neutral
        return 0

    def adjust_parameters(self, base_params, current_regime):
        """Adjust strategy parameters based on the current market regime"""
        adjusted_params = base_params.copy()

        if current_regime == 1:  # Bullish
            # In bullish regime, be more aggressive
            adjusted_params['rsi_entry_threshold'] = max(0.1, base_params['rsi_entry_threshold'] * 0.5)  # Lower RSI threshold for more entries
            adjusted_params['stop_loss_pct'] = base_params['stop_loss_pct'] * 1.5  # Wider stop loss
            adjusted_params['take_profit_pct'] = base_params['take_profit_pct'] * 2.0  # Higher profit target
            adjusted_params['position_size_multiplier'] = 2.0  # Larger position size

        elif current_regime == -1:  # Bearish
            # In bearish regime, be more conservative
            adjusted_params['rsi_entry_threshold'] = min(30, base_params['rsi_entry_threshold'] * 2.0)  # Higher RSI threshold for fewer entries
            adjusted_params['stop_loss_pct'] = base_params['stop_loss_pct'] * 0.5  # Tighter stop loss
            adjusted_params['take_profit_pct'] = base_params['take_profit_pct'] * 0.5  # Lower profit target
            adjusted_params['position_size_multiplier'] = 0.5  # Smaller position size

        # For neutral regime, use base parameters

        return adjusted_params

class DynamicExitSystem:
    """Advanced exit system with dynamic trailing stops and take profit levels"""
    def __init__(self):
        self.active_exits = {}

    def setup_advanced_exit_system(self, symbol, position_size, entry_price, direction,
                                  stop_loss_pct=0.01, take_profit_levels=[0.025, 0.05, 0.15],
                                  trailing_stop_activation_pct=0.018, trailing_stop_distance_pct=0.008):
        """Setup an advanced exit system for a trade"""
        # Calculate stop loss and take profit levels
        stop_level = self._calculate_stop_level(entry_price, stop_loss_pct, direction)

        # Calculate multiple take profit levels
        take_profit_levels = [self._calculate_take_profit_level(entry_price, tp_pct, direction)
                             for tp_pct in take_profit_levels]

        # Calculate trailing stop activation level
        activation_price = self._calculate_activation_price(entry_price, trailing_stop_activation_pct, direction)

        self.active_exits[symbol] = {
            'entry_price': entry_price,
            'position_size': position_size,
            'direction': direction,
            'stop_loss': stop_level,
            'take_profit_levels': take_profit_levels,
            'take_profit_triggered': [False] * len(take_profit_levels),
        'trailing_stop_activation': activation_price,
        'trailing_stop_distance_pct': trailing_stop_distance_pct,
        'trailing_stop': None,  # Will be set once activated
        'partial_exits_done': 0
        }

        logger.info(f"Setup exit system for {symbol}: Entry={entry_price}, Stop={stop_level}, "
        f"TPs={take_profit_levels}, Activation={activation_price}")

        return self.active_exits[symbol]

    def _calculate_stop_level(self, entry_price, stop_pct, direction):
        """Calculate stop loss level based on entry price and direction"""
        if direction == "long":
            return entry_price * (1 - stop_pct)
        else:  # short
            return entry_price * (1 + stop_pct)

    def _calculate_take_profit_level(self, entry_price, profit_pct, direction):
        """Calculate take profit level based on entry price and direction"""
        if direction == "long":
            return entry_price * (1 + profit_pct)
        else:  # short
            return entry_price * (1 - profit_pct)

    def _calculate_activation_price(self, entry_price, activation_pct, direction):
        """Calculate trailing stop activation price"""
        if direction == "long":
            return entry_price * (1 + activation_pct)
        else:  # short
            return entry_price * (1 - activation_pct)

    def update_exit_system(self, symbol, current_price):
        """Update exit system based on current price and check for exit signals"""
        if symbol not in self.active_exits:
            return None
        
        exit_info = self.active_exits[symbol]
        direction = exit_info['direction']
        
        # Check stop loss
        if (direction == "long" and current_price <= exit_info['stop_loss']) or \
            (direction == "short" and current_price >= exit_info['stop_loss']):
            return self._execute_exit(symbol, current_price, exit_info['position_size'], "stop_loss")
                    
        # Check take profit levels for partial exits
        for i, (tp_level, triggered) in enumerate(zip(exit_info['take_profit_levels'], exit_info['take_profit_triggered'])):
            if not triggered:
                if (direction == "long" and current_price >= tp_level) or \
                   (direction == "short" and current_price <= tp_level):
                    # Mark this level as triggered
                    exit_info['take_profit_triggered'][i] = True

                    # Calculate the portion to exit at this level (progressive scaling)
                    exit_portion = 0.25 * (i + 1)  # Exit 25%, 50%, 75% at each level
                    exit_size = exit_info['position_size'] * exit_portion

                    # Update remaining position size
                    exit_info['position_size'] -= exit_size
                    exit_info['partial_exits_done'] += 1

                    if exit_info['position_size'] <= 0.001:  # Full exit
                        return self._execute_exit(symbol, current_price, exit_size, "take_profit_full")
                    else:
                        # Partial exit
                        return {
                            'symbol': symbol,
                            'price': current_price,
                            'size': exit_size,
                            'reason': f"take_profit_{i+1}",
                            'entry_price': exit_info['entry_price']
                        }
                
                    # Check and update trailing stop
                    if exit_info['trailing_stop'] is None:
                        # Check if price reached activation level
                        if (direction == "long" and current_price >= exit_info['trailing_stop_activation']) or \
                           (direction == "short" and current_price <= exit_info['trailing_stop_activation']):
                            # Activate trailing stop
                            exit_info['trailing_stop'] = self._calculate_trailing_stop_level(
                                current_price, exit_info['trailing_stop_distance_pct'], direction)
                            logger.info(f"Trailing stop activated for {symbol} at {exit_info['trailing_stop']}")
                    
                            # Check if price hit trailing stop
                            if (direction == "long" and current_price <= exit_info['trailing_stop']) or \
                               (direction == "short" and current_price >= exit_info['trailing_stop']):
                                return self._execute_exit(symbol, current_price, exit_info['position_size'], "trailing_stop")

    def _calculate_trailing_stop_level(self, current_price, distance_pct, direction):
        """Calculate trailing stop level based on current price"""
        if direction == "long":
            return current_price * (1 - distance_pct)
        else:  # short
            return current_price * (1 + distance_pct)

    def _execute_exit(self, symbol, price, size, reason):
        """Execute exit and remove from active exits"""
        exit_info = self.active_exits.pop(symbol, None)
        if not exit_info:
            return None

        logger.info(f"Exit signal for {symbol}: Price={price}, Size={size}, Reason={reason}")

        return {
            'symbol': symbol,
            'price': price,
            'size': size,
            'reason': reason,
            'entry_price': exit_info['entry_price']
        }

class EnhancedTrading:
    """Advanced ML models for trading predictions"""
    def __init__(self, feature_importance_threshold=0.005):
        # Initialize logger for this instance
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        if not self.logger.handlers:  # Avoid duplicate handlers
            self.logger.addHandler(handler)
            
        self.xgb_model = XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.01,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        scale_pos_weight=2.0,
        reg_alpha=0.1,
        reg_lambda=1.0
        )
        self.lgbm_model = LGBMClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.01,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        scale_pos_weight=2.0,
        reg_alpha=0.1,
        reg_lambda=1.0
        )

        self.rf_model = RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        min_samples_split=4,
        min_samples_leaf=2,
        max_features='sqrt',
        class_weight={0: 2.0, 1: 1.0},
        bootstrap=True,
        random_state=42
        )

        self.dt_model = DecisionTreeClassifier(
        max_depth=4,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight={0: 2.0, 1: 1.0},
        random_state=42
        )

        self.feature_importance_threshold = feature_importance_threshold
        self.model_weights = {
        "rf": 0.30,    # Reduced RF weight
        "xgb": 0.40,   # Increased XGB weight
        "lgbm": 0.25,  # Increased LGBM weight
        "dt": 0.05     # Minimal DT weight
        }

        self.scaler = RobustScaler()
        self.models_trained = False

    def generate_features(self, df):
        """Generate advanced features for model training"""
        features = pd.DataFrame(index=df.index)

        # Price-based features
        features['price_change'] = df['close'].pct_change()
        features['price_change_1d'] = df['close'].pct_change(20)  # Assuming 20 periods = 1 day
        features['price_change_1w'] = df['close'].pct_change(100)  # Assuming 100 periods = 1 week

        # Moving averages and crossovers
        features['ma_short'] = df['close'].rolling(window=20).mean()
        features['ma_medium'] = df['close'].rolling(window=50).mean()
        features['ma_long'] = df['close'].rolling(window=200).mean()
        features['ma_cross_short_medium'] = (features['ma_short'] > features['ma_medium']).astype(int)
        features['ma_cross_short_long'] = (features['ma_short'] > features['ma_long']).astype(int)

        # Volatility features
        features['volatility'] = df['close'].pct_change().rolling(window=20).std()
        features['volatility_change'] = features['volatility'].pct_change(20)

        # RSI features
        if 'rsi' in df.columns:
            features['rsi'] = df['rsi']
        else:
            features['rsi'] = calculate_rsi(df['close'])
        features['rsi_change'] = features['rsi'].diff()
        features['rsi_ma'] = features['rsi'].rolling(window=14).mean()

        # MACD features
        macd, signal, hist = calculate_macd(df['close'])
        features['macd'] = macd
        features['macd_signal'] = signal
        features['macd_hist'] = hist
        features['macd_cross'] = ((macd > signal) & (macd.shift(1) <= signal.shift(1))).astype(int)

        # Bollinger Bands
        upper_bb, lower_bb = calculate_bollinger_bands(df['close'])
        features['bb_width'] = (upper_bb - lower_bb) / df['close']
        features['bb_position'] = (df['close'] - lower_bb) / (upper_bb - lower_bb)

        # Trend features
        features['trend_strength'] = abs(features['ma_short'] - features['ma_long']) / features['ma_long']

        # Advanced features
        if len(df) >= 50:
            features['hurst'] = df['close'].rolling(window=50).apply(
                lambda x: calculate_hurst_exponent(x) if len(x) > 20 and not np.isnan(x).any() else np.nan, raw=True)
            features['fractal_dim'] = df['close'].rolling(window=50).apply(
                lambda x: fractal_dimension(x) if len(x) > 20 and not np.isnan(x).any() else np.nan, raw=True)

        # Lag features
        for lag in [1, 2, 3, 5, 10]:
            features[f'price_lag_{lag}'] = df['close'].shift(lag)
            features[f'rsi_lag_{lag}'] = features['rsi'].shift(lag)

        # Fill NaN values
        features = features.bfill().fillna(0)  # Using bfill() instead of fillna(method='bfill')

        return features

    def train_models(self, X_train, y_train):
        """Train all models with cross-validation"""
        self.logger.info("Starting model training...")

        # Preprocess data
        X_train_processed, y_train_processed = self._preprocess_data(X_train, y_train)

        # Train models with timeout protection
        try:
            # Reduce model complexity to prevent getting stuck
            self.rf_model = RandomForestClassifier(
                n_estimators=50,  # Reduced from 200
                max_depth=4,      # Reduced from 8
                min_samples_split=4,
                min_samples_leaf=2,
                max_features='sqrt',
                class_weight={0: 2.0, 1: 1.0},
                bootstrap=True,
                random_state=42
            )

            self.lgbm_model = LGBMClassifier(
                n_estimators=50,  # Reduced from 200
                max_depth=4,      # Reduced from 6
                learning_rate=0.05,  # Increased from 0.01
                subsample=0.8,
                colsample_bytree=0.8,
                min_child_weight=3,
                scale_pos_weight=2.0,
                reg_alpha=0.1,
                reg_lambda=1.0,
                random_state=42
            )

            self.logger.info("Training RandomForest model...")
            self.rf_model.fit(X_train_processed, y_train_processed)

            self.logger.info("Training XGBoost model...")
            self.xgb_model.fit(X_train_processed, y_train_processed)

            self.logger.info("Training LightGBM model...")
            self.lgbm_model.fit(X_train_processed, y_train_processed)

            self.logger.info("Training Decision Tree model...")
            self.dt_model.fit(X_train_processed, y_train_processed)

            self.models_trained = True
            self.logger.info("All models trained successfully")

            # Log feature importances
            if hasattr(self.rf_model, 'feature_importances_'):
                importances = self.rf_model.feature_importances_
                feature_names = X_train.columns
                importance_df = pd.DataFrame({
                    'feature': feature_names,
                    'importance': importances
                }).sort_values('importance', ascending=False)
                self.logger.info("Feature importances:\n" + importance_df.to_string())
        except Exception as e:
            self.logger.error(f"Error during model training: {str(e)}")
            self.logger.warning("Using simplified models due to training error")

            # Use simple models as fallback
            from sklearn.linear_model import LogisticRegression

            self.rf_model = LogisticRegression(random_state=42)
            self.xgb_model = LogisticRegression(random_state=42)
            self.lgbm_model = LogisticRegression(random_state=42)
            self.dt_model = LogisticRegression(random_state=42)

        try:
            self.rf_model.fit(X_train_processed, y_train_processed)
            self.xgb_model.fit(X_train_processed, y_train_processed)
            self.lgbm_model.fit(X_train_processed, y_train_processed)
            self.dt_model.fit(X_train_processed, y_train_processed)
            self.models_trained = True
            self.logger.info("Fallback models trained successfully")
        except Exception as e:
            self.logger.error(f"Error during training: {str(e)}")
            return False

    def _preprocess_data(self, X, y):
        """Preprocess training data"""
        # Data preprocessing logic here
        # Handle missing values
        X = X.fillna(0)

        # Scale features
        X_scaled = self.scaler.fit_transform(X)

        return X_scaled, y

    def train(self, df, lookback=20, timeframe="5"):
        """Train models on historical data"""
        self.logger.info(f"Starting model training with {len(df)} rows of data")

        if len(df) < lookback * 2:
            self.logger.warning(f"Not enough data to train models: {len(df)} rows")
            return False

        try:
            # Generate features
            self.logger.info("Generating features for model training...")
            features = self.generate_features(df)

            # Create target variable (1 if price goes up in next period, 0 otherwise)
            target = (df['close'].shift(-1) > df['close']).astype(int)

            # Remove rows with NaN
            valid_idx = ~(features.isna().any(axis=1) | target.isna())
            features = features[valid_idx]
            target = target[valid_idx]

            self.logger.info(f"After preprocessing: {len(features)} valid rows for training")

            if len(features) < 100:
                self.logger.warning(f"Not enough valid data after preprocessing: {len(features)} rows")
                return False

            # Train-test split
            X_train, X_test, y_train, y_test = train_test_split(
                features, target, test_size=0.2, shuffle=False)

            self.logger.info(f"Training set size: {len(X_train)}, Test set size: {len(X_test)}")

            # Preprocess data
            X_train_processed, y_train_processed = self._preprocess_data(X_train, y_train)
            X_test_processed, y_test_processed = self._preprocess_data(X_test, y_test)

            # Train models
            self.train_models(X_train_processed, y_train_processed)

            return True

        except Exception as e:
            self.logger.error(f"Error during training: {str(e)}")
            return False

        except Exception as e:
            self.logger.error(f"Error in train method: {e}")
        self.logger.error(f"Error in train method: {e}")
        import traceback
        self.logger.error(traceback.format_exc())
        return False

    def _update_model_weights(self, X_test, y_test):
        """Update model weights based on performance"""
        if not self.models_trained or len(X_test) < 10:
            self.logger.warning("Cannot update model weights: models not trained or insufficient test data")
            return

        try:
            self.logger.info("Evaluating model performance for weight updates...")

            # Preprocess test data
            X_test_processed = self.scaler.transform(X_test)

            # Get predictions with error handling
            predictions = {}

            try:
                predictions['rf'] = self.rf_model.predict_proba(X_test_processed)[0][1]
                self.logger.info(f"RandomForest prediction: {predictions['rf']:.4f}")
            except Exception as e:
                self.logger.warning(f"Error in RandomForest prediction: {e}")
                predictions['rf'] = 0.5

            try:
                predictions['xgb'] = self.xgb_model.predict_proba(X_test_processed)[0][1]
                self.logger.info(f"XGBoost prediction: {predictions['xgb']:.4f}")
            except Exception as e:
                self.logger.warning(f"Error in XGBoost prediction: {e}")
                predictions['xgb'] = 0.5

            try:
                predictions['lgbm'] = self.lgbm_model.predict_proba(X_test_processed)[0][1]
                self.logger.info(f"LightGBM prediction: {predictions['lgbm']:.4f}")
            except Exception as e:
                self.logger.warning(f"Error in LightGBM prediction: {e}")
                predictions['lgbm'] = 0.5

            try:
                predictions['dt'] = self.dt_model.predict_proba(X_test_processed)[0][1]
                self.logger.info(f"DecisionTree prediction: {predictions['dt']:.4f}")
            except Exception as e:
                self.logger.warning(f"Error in DecisionTree prediction: {e}")
                predictions['dt'] = 0.5

            # Calculate weighted prediction
            weighted_prediction = (
                predictions['rf'] * self.model_weights['rf'] +
                predictions['xgb'] * self.model_weights['xgb'] +
                predictions['lgbm'] * self.model_weights['lgbm'] +
                predictions['dt'] * self.model_weights['dt']
            ) / sum(self.model_weights.values())

            return weighted_prediction

        except Exception as e:
            self.logger.error(f"Error updating model weights: {str(e)}")
            return None

        try:
            predictions['lgbm'] = self.lgbm_model.predict_proba(X_test_processed)[0][1]
            self.logger.info(f"LightGBM prediction: {predictions['lgbm']:.4f}")
        except Exception as e:
            self.logger.warning(f"Error in LightGBM prediction: {e}")
            predictions['lgbm'] = 0.5

        try:
            predictions['dt'] = self.dt_model.predict_proba(X_test_processed)[0][1]
            self.logger.info(f"DecisionTree prediction: {predictions['dt']:.4f}")
        except Exception as e:
            self.logger.warning(f"Error in DecisionTree prediction: {e}")
            predictions['dt'] = 0.5

        # Weighted ensemble prediction
        try:
            weighted_prediction = sum(
                predictions[model] * weight
                for model, weight in self.model_weights.items()
            )
            self.logger.info(f"Final weighted prediction: {weighted_prediction:.4f}")
            return weighted_prediction

        except Exception as e:
            self.logger.error(f"Error calculating weighted prediction: {str(e)}")
            return 0.5  # Return neutral prediction on error

def predict(self, df, timeframe="5"):
    """Make prediction for the next period"""
    if not self.models_trained:
        self.logger.warning("Models not trained yet")
        return 0.5

    try:
        self.logger.info("Generating features for prediction...")
        features = self.generate_features(df)
        if features is None or len(features) == 0:
            self.logger.warning("No valid features generated for prediction")
            return 0.5

        # Preprocess features
        self.logger.info("Preprocessing features for prediction...")
        latest_features = features.iloc[-1:].copy()
        if latest_features.isna().any().any():
            self.logger.warning("NaN values in prediction features, filling with zeros")
            latest_features = latest_features.fillna(0)
        
        X_processed = self.scaler.transform(latest_features)

        # Get predictions from each model with error handling
        predictions = {}

        try:
            predictions['rf'] = self.rf_model.predict_proba(X_processed)[0][1]
            self.logger.info(f"RandomForest prediction: {predictions['rf']:.4f}")
        except Exception as e:
            self.logger.warning(f"Error in RandomForest prediction: {e}")
            predictions['rf'] = 0.5

        try:
            predictions['xgb'] = self.xgb_model.predict_proba(X_processed)[0][1]
            self.logger.info(f"XGBoost prediction: {predictions['xgb']:.4f}")
        except Exception as e:
            self.logger.warning(f"Error in XGBoost prediction: {e}")
            predictions['xgb'] = 0.5

        try:
            predictions['lgbm'] = self.lgbm_model.predict_proba(X_processed)[0][1]
            self.logger.info(f"LightGBM prediction: {predictions['lgbm']:.4f}")
        except Exception as e:
            self.logger.warning(f"Error in LightGBM prediction: {e}")
            predictions['lgbm'] = 0.5

        try:
            predictions['dt'] = self.dt_model.predict_proba(X_processed)[0][1]
            self.logger.info(f"DecisionTree prediction: {predictions['dt']:.4f}")
        except Exception as e:
            self.logger.warning(f"Error in DecisionTree prediction: {e}")
            predictions['dt'] = 0.5

        # Weighted ensemble prediction
        try:
            weighted_prediction = sum(
                predictions[model] * weight
                for model, weight in self.model_weights.items()
            )
            self.logger.info(f"Final weighted prediction: {weighted_prediction:.4f}")
            return weighted_prediction

        except Exception as e:
            self.logger.error(f"Error calculating weighted prediction: {str(e)}")
            return 0.5  # Return neutral prediction on error

    except Exception as e:
        self.logger.error(f"Error in predict method: {str(e)}")
        return 0.5

class BacktestComparisonSystem:
    """System for comparing live trading performance against backtest results"""
    def __init__(self, results_dir=RESULTS_DIR):  # Line 968
        self.results_dir = results_dir
        self.backtest_data = None
        self.live_data = None

    def load_backtest_data(self, backtest_file):  # Line 976
        """Load backtest data from file"""
        if not os.path.exists(backtest_file):
            logger.error(f"Backtest file not found: {backtest_file}")
            return False

        try:
            with open(backtest_file, 'r') as f:
                self.backtest_data = json.load(f)
            logger.info(f"Successfully loaded backtest data from {backtest_file}")
            return True
        except Exception as e:
            logger.error(f"Error loading backtest data: {str(e)}")
            return False

    def load_live_trades(self, live_trades_file=None):
        """Load live trading data from file"""
        if live_trades_file is None:
            # Find the most recent trades file
            trade_files = glob.glob(os.path.join(self.results_dir, "trades_*.json"))
            if not trade_files:
                logger.error("No trade files found in results directory")
                return False
            live_trades_file = max(trade_files, key=os.path.getctime)

        try:
            with open(live_trades_file, 'r') as f:
                self.live_data = json.load(f)
            logger.info(f"Successfully loaded live trades from {live_trades_file}")
            return True
        except Exception as e:
            logger.error(f"Error loading live trades: {str(e)}")
            return False

    def convert_backtest_to_trades(self, backtest_file, output_file=None):
        """Convert backtest results to trades format for comparison"""
        if not self.load_backtest_data(backtest_file):
            return False

        if output_file is None:
            output_file = os.path.join(self.results_dir, "backtest_trades.json")

        try:
            trades = [
                {
                    "entry_time": trade["entry_time"],
                    "exit_time": trade["exit_time"],
                    "symbol": trade["symbol"],
                    "position_size": trade["position_size"],
                    "entry_price": trade["entry_price"],
                    "exit_price": trade["exit_price"],
                    "profit": trade["profit"]
                }
                for trade in self.backtest_data["trades"]
            ]

            with open(output_file, 'w') as f:
                json.dump(trades, f, indent=2)
            logger.info(f"Successfully converted backtest to trades: {output_file}")
            return True
        except Exception as e:
            logger.error(f"Error converting backtest to trades: {str(e)}")
            return False

    def calculate_comparison_metrics(self):
        """Calculate comparison metrics between backtest and live trading"""
        if self.backtest_data is None or self.live_data is None:
            logger.error("Both backtest and live data must be loaded")
            return None

        try:
            # Extract metrics
            backtest_trades = self.backtest_data.get("trades", [])
            live_trades = self.live_data

            # Calculate backtest metrics
            backtest_win_rate = sum(1 for t in backtest_trades if t.get("profit_pct", 0) > 0) / max(1, len(backtest_trades))
            backtest_avg_profit = sum(t.get("profit_pct", 0) for t in backtest_trades) / max(1, len(backtest_trades))
            backtest_max_profit = max((t.get("profit_pct", 0) for t in backtest_trades), default=0)
            backtest_max_loss = min((t.get("profit_pct", 0) for t in backtest_trades), default=0)

            # Calculate live metrics
            live_win_rate = sum(1 for t in live_trades if t.get("profit_pct", 0) > 0) / max(1, len(live_trades))
            live_avg_profit = sum(t.get("profit_pct", 0) for t in live_trades) / max(1, len(live_trades))
            live_max_profit = max((t.get("profit_pct", 0) for t in live_trades), default=0)
            live_max_loss = min((t.get("profit_pct", 0) for t in live_trades), default=0)

            # Calculate differences
            win_rate_diff = live_win_rate - backtest_win_rate
            avg_profit_diff = live_avg_profit - backtest_avg_profit

            return {
                "backtest": {
                    "win_rate": backtest_win_rate,
                    "avg_profit": backtest_avg_profit,
                    "max_profit": backtest_max_profit,
                    "max_loss": backtest_max_loss,
                    "trade_count": len(backtest_trades)
                },
                "live": {
                    "win_rate": live_win_rate,
                    "avg_profit": live_avg_profit,
                    "max_profit": live_max_profit,
                    "max_loss": live_max_loss,
                    "trade_count": len(live_trades)
                },
                "differences": {
                    "win_rate": win_rate_diff,
                    "avg_profit": avg_profit_diff
                }
            }

        except Exception as e:
            logger.error(f"Error calculating comparison metrics: {str(e)}")
            return None

    def generate_comparison_charts(self, output_dir=None, show_charts=False):
        """Generate charts comparing backtest and live performance"""
        if self.backtest_data is None or self.live_data is None:
            logger.error("Both backtest and live data must be loaded")
            return False

        if output_dir is None:
            output_dir = os.path.join(self.results_dir, "charts")

        os.makedirs(output_dir, exist_ok=True)

        try:
            # Generate equity curve comparison
            backtest_equity = self.backtest_data.get("equity_curve", [])
            live_equity = self.live_data.get("equity_curve", [])

            plt.figure(figsize=(12, 6))
            plt.plot(backtest_equity, label="Backtest")
            plt.plot(live_equity, label="Live")
            plt.title("Equity Curve Comparison")
            plt.xlabel("Trade Number")
            plt.ylabel("Equity")
            plt.legend()
            plt.grid(True)
            plt.savefig(os.path.join(output_dir, "equity_comparison.png"))
            if show_charts:
                plt.show()
            plt.close()

            # Generate win rate comparison
            metrics = self.calculate_comparison_metrics()
            if metrics:
                plt.figure(figsize=(12, 6))
                plt.bar(["Backtest", "Live"], 
                       [metrics["backtest"]["win_rate"], metrics["live"]["win_rate"]])
                plt.title("Win Rate Comparison")
                plt.ylabel("Win Rate")
                plt.grid(True)
                plt.savefig(os.path.join(output_dir, "win_rate_comparison.png"))
                if show_charts:
                    plt.show()
                plt.close()

            return True

        except Exception as e:
            logger.error(f"Error generating comparison charts: {str(e)}")
            return False

        # Generate average return comparison chart
        backtest_avg_return = sum(t.get("profit_pct", 0) for t in backtest_trades) / max(1, len(backtest_trades))
        live_avg_return = sum(t.get("profit_pct", 0) for t in live_trades) / max(1, len(live_trades))

        plt.figure(figsize=(8, 6))
        plt.bar(['Backtest', 'Live Trading'], [backtest_avg_return, live_avg_return], color=['blue', 'green'])
        plt.title('Average Return Comparison')
        plt.ylabel('Average Return')

        for i, v in enumerate([backtest_avg_return, live_avg_return]):
            plt.text(i, v + (0.05 if v >= 0 else -0.05), f'{v:.2%}', ha='center')

        avg_return_file = os.path.join(output_dir, 'avg_return_comparison.png')
        plt.savefig(avg_return_file)
        if show_charts:
            plt.show()
        else:
            plt.close()

        logger.info(f"Generated comparison charts in {output_dir}")
        return True

class MarketRegimeDetector:
    def __init__(self, lookback_period=50):
        self.lookback_period = lookback_period
        self.regime_history = []

    def detect_regime(self, df, timeframe="5"):
        if len(df) < self.lookback_period:
            return 0  # Neutral if not enough data

        # Calculate trend metrics
        price_change = df['close'].pct_change(self.lookback_period).iloc[-1]
        volatility = df['close'].pct_change().rolling(
            self.lookback_period).std().iloc[-1]

        # Calculate volume metrics
        if 'volume' in df.columns:
            volume_change = df['volume'].pct_change(self.lookback_period).iloc[-1]
            volume_trend = df['volume'].rolling(self.lookback_period).mean(
            ).pct_change(self.lookback_period // 2).iloc[-1]
        else:
            volume_change = 0
            volume_trend = 0

        # Calculate momentum
        momentum = df['close'].pct_change(self.lookback_period // 2).iloc[-1]

        # Calculate RSI
        rsi_value = df['rsi'].iloc[-1] if 'rsi' in df.columns else calculate_rsi(
            df['close']).iloc[-1]

        # Determine regime
        regime = 0  # Neutral by default

        # Bullish conditions
        if (price_change > 0.01 and  # Strong positive price change
            momentum > 0 and         # Positive momentum
            rsi_value > 50 and       # RSI above midpoint
            volume_trend > 0):       # Increasing volume
            regime = 1

        # Bearish conditions
        elif (price_change < -0.01 and  # Strong negative price change
              momentum < 0 and          # Negative momentum
              rsi_value < 50 and        # RSI below midpoint
              volume_trend < 0):        # Decreasing volume
            regime = -1

        # Record regime
        self.regime_history.append(regime)
        if len(self.regime_history) > self.lookback_period:
            self.regime_history.pop(0)

        return regime

    def get_stable_regime(self):
        """Get the stable market regime based on recent history"""
        if not self.regime_history:
            return 0

        # Only return a regime if it's been consistent
        if len(self.regime_history) >= 5:
            recent_regimes = self.regime_history[-5:]
            if all(r == 1 for r in recent_regimes):
                return 1  # Stable bullish
            elif all(r == -1 for r in recent_regimes):
                return -1  # Stable bearish

        # Default to neutral
        return 0

    def adjust_parameters(self, base_params, current_regime):
        """Adjust strategy parameters based on the current market regime"""
        adjusted_params = base_params.copy()

        if current_regime == 1:  # Bullish
            # In bullish regime, be more aggressive
            adjusted_params['rsi_entry_threshold'] = max(0.1, base_params['rsi_entry_threshold'] * 0.5)  # Lower RSI threshold for more entries
            adjusted_params['stop_loss_pct'] = base_params['stop_loss_pct'] * 1.5  # Wider stop loss
            adjusted_params['take_profit_pct'] = base_params['take_profit_pct'] * 2.0  # Higher profit target
            adjusted_params['position_size_multiplier'] = 2.0  # Larger position size

        elif current_regime == -1:  # Bearish
            # In bearish regime, be more conservative
            adjusted_params['rsi_entry_threshold'] = min(30, base_params['rsi_entry_threshold'] * 2.0)  # Higher RSI threshold for fewer entries
            adjusted_params['stop_loss_pct'] = base_params['stop_loss_pct'] * 0.5  # Tighter stop loss
            adjusted_params['take_profit_pct'] = base_params['take_profit_pct'] * 0.5  # Lower profit target
            adjusted_params['position_size_multiplier'] = 0.5  # Smaller position size

        # For neutral regime, use base parameters

        return adjusted_params

class MetaTraderLiveStrategy:
    def __init__(self, symbol="EURUSD", timeframe="M5", initial_deposit=50000,
                 rsi_entry_threshold=0.1, stop_loss_pct=0.05, take_profit_pct=0.2,
                 max_position_size=1.0, results_dir=RESULTS_DIR):
        # Initialize logger for this instance
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        if not self.logger.handlers:  # Avoid duplicate handlers
            self.logger.addHandler(handler)

        self.symbol = symbol
        self.timeframe = timeframe
        self.mt5_timeframe = self._get_mt5_timeframe(timeframe)
        self.initial_deposit = initial_deposit
        self.rsi_entry_threshold = rsi_entry_threshold
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self.max_position_size = max_position_size
        self.results_dir = results_dir
        self.mt5_initialized = False
        self.current_positions = {}
        self.trades = []
        self.historical_data = None
        self.last_data_update = None
        self.win_streak = 0
        self.loss_streak = 0
        self.strategy_profiles = {
            "conservative": {
                "long_rsi_threshold": 30,
                "short_rsi_threshold": 70,
                "stop_loss_pct": 0.02,
                "take_profit_pct": 0.03,
                "position_size_multiplier": 0.5,
                "max_concurrent_trades": 5
            },
            "moderate": {
                "long_rsi_threshold": 20,
                "short_rsi_threshold": 80,
                "stop_loss_pct": 0.05,
                "take_profit_pct": 0.10,
                "position_size_multiplier": 1.0,
                "max_concurrent_trades": 4
            },
            "aggressive": {
                "long_rsi_threshold": 10,
                "short_rsi_threshold": 90,
                "stop_loss_pct": 0.10,
                "take_profit_pct": 0.20,
                "position_size_multiplier": 2.0,
                "max_concurrent_trades": 3
            },
            "very_aggressive": {
                "long_rsi_threshold": 5,
                "short_rsi_threshold": 95,
                "stop_loss_pct": 0.15,
                "take_profit_pct": 0.40,
                "position_size_multiplier": 3.0,
                "max_concurrent_trades": 2
            },
            "ultra_aggressive": {
                "long_rsi_threshold": 0.1,
                "short_rsi_threshold": 99.9,
                "stop_loss_pct": 0.25,
                "take_profit_pct": 0.75,
                "position_size_multiplier": 5.0,
                "max_concurrent_trades": 1
            }
        }
        self.active_profile = "moderate"
        self.regime_detector = MarketRegimeDetector()
        self.trading_models = EnhancedTrading()

    def _get_mt5_timeframe(self, timeframe_str):
        """Convert string timeframe to MT5 timeframe constant"""
        timeframe_map = {
            "M1": mt5.TIMEFRAME_M1,
            "M5": mt5.TIMEFRAME_M5,
            "M15": mt5.TIMEFRAME_M15,
            "M30": mt5.TIMEFRAME_M30,
            "H1": mt5.TIMEFRAME_H1,
            "H4": mt5.TIMEFRAME_H4,
            "D1": mt5.TIMEFRAME_D1,
            "W1": mt5.TIMEFRAME_W1,
            "MN1": mt5.TIMEFRAME_MN1
        }
        return timeframe_map.get(timeframe_str, mt5.TIMEFRAME_M5)

    
    
    def initialize_mt5(self):
        """Initialize connection to MetaTrader 5"""
        self.logger.info("Initializing MetaTrader 5 connection...")

        if mt5 is None:
            self.logger.error("MetaTrader5 module not found. Running in simulation mode.")
            self.mt5_initialized = False
            return False

        try:
            # Attempt to initialize MT5
            if not mt5.initialize():
                error_code = mt5.last_error()
                self.logger.error(f"Failed to initialize MT5: Error code {error_code}")
                self.mt5_initialized = False
                return False

            # Verify connection by checking terminal info
            terminal_info = mt5.terminal_info()
            if terminal_info is None:
                self.logger.error("MT5 initialized but no terminal info available: Error code " + str(mt5.last_error()))
                mt5.shutdown()
                self.mt5_initialized = False
                return False

            # Check if connected to a broker
            account_info = mt5.account_info()
            if account_info is None:
                self.logger.error("Not connected to a broker: Error code " + str(mt5.last_error()))
                mt5.shutdown()
                self.mt5_initialized = False
                return False

            # Verify symbol availability
            symbol_info = mt5.symbol_info(self.symbol)
            if symbol_info is None:
                self.logger.error(f"Symbol {self.symbol} not found in market watch.")
                mt5.shutdown()
                self.mt5_initialized = False
                return False

            # Ensure symbol is enabled
            if not symbol_info.visible:
                if not mt5.symbol_select(self.symbol, True):
                    self.logger.error(f"Failed to enable symbol {self.symbol}.")
                    mt5.shutdown()
                    self.mt5_initialized = False
                    return False

            self.logger.info("MetaTrader 5 initialized successfully.")
            self.mt5_initialized = True
            return True

        except Exception as e:
            self.logger.error(f"Exception during MT5 initialization: {str(e)}")
            self.logger.error(traceback.format_exc())
            self.mt5_initialized = False
            return False

    def get_account_info(self):
        """Retrieve account information from MetaTrader 5"""
        self.logger.info("Retrieving account information...")

        if not self.mt5_initialized:
            self.logger.error("MT5 not initialized. Attempting to initialize...")
            if not self.initialize_mt5():
                self.logger.error("Failed to initialize MT5. Cannot retrieve account info.")
                return None

        try:
            account_info = mt5.account_info()
            if account_info is None:
                self.logger.error(f"Failed to get account info: {mt5.last_error()}")
                return None

            account_data = {
                'login': account_info.login,
                'balance': account_info.balance,
                'equity': account_info.equity,
                'margin': account_info.margin,
                'margin_free': account_info.margin_free,
                'currency': account_info.currency
            }
            self.logger.info(f"Account info retrieved: Balance={account_data['balance']}, Equity={account_data['equity']}")
            return account_data

        except Exception as e:
            self.logger.error(f"Error retrieving account info: {str(e)}")
            self.logger.error(traceback.format_exc())
            return None
    
    def fetch_historical_data(self, bars=500):
        """Fetch historical price data from MetaTrader 5"""
        self.logger.info(f"Fetching historical data for {self.symbol}, requesting {bars} bars...")

        if not self.mt5_initialized:
            self.logger.error("MT5 not initialized. Cannot fetch data.")
            return False

        # Get current time in UTC
        now = datetime.now()
        self.logger.info(f"Current time: {now}")

        # Fetch historical data
        self.logger.info(f"Requesting historical data from MT5 for {self.symbol} with timeframe {self.mt5_timeframe}...")
        try:
            self.logger.info("Calling mt5.copy_rates_from...")
            rates = mt5.copy_rates_from(self.symbol, self.mt5_timeframe, now, bars)

            if rates is None or len(rates) == 0:
                self.logger.error(f"Failed to fetch historical data: {mt5.last_error()}")
                return False

            self.logger.info(f"Successfully fetched {len(rates)} bars of historical data")

            # Convert to DataFrame
            self.logger.info("Converting data to DataFrame...")
            df = pd.DataFrame(rates)
            df['time'] = pd.to_datetime(df['time'], unit='s')
            df.set_index('time', inplace=True)

            self.logger.info("Calculating additional features...")

            # Calculate additional features
            self.logger.info("Calculating technical indicators...")
            try:
                # Calculate indicators
                self.logger.info("Calculating RSI...")
                df['rsi'] = calculate_rsi(df['close'])

                self.logger.info("Calculating MACD...")
                df['macd'], df['macd_signal'], df['macd_hist'] = calculate_macd(df['close'])

                self.logger.info("Calculating Bollinger Bands...")
                df['upper_bb'], df['lower_bb'] = calculate_bollinger_bands(df['close'])

                self.logger.info("Calculating ATR...")
                df['atr'] = calculate_atr(df['high'], df['low'], df['close'])

                self.logger.info("Calculating Stochastic Oscillator...")
                df['stoch_k'], df['stoch_d'] = calculate_stochastic(df['high'], df['low'], df['close'])

                self.logger.info("Calculating ADX...")
                df['adx'] = calculate_adx(df['high'], df['low'], df['close'])

                self.logger.info("All technical indicators calculated successfully")
            except Exception as e:
                self.logger.error(f"Error calculating technical indicators: {e}")
                self.logger.error(traceback.format_exc())
                # Continue even if indicators fail, as we can still use price data

            # Store the data
            self.historical_data = df
            self.last_data_update = now

            self.logger.info(f"Historical data updated successfully with {len(df)} rows")
            return df

        except Exception as e:
            self.logger.error(f"Exception during historical data fetch: {e}")
            self.logger.error(traceback.format_exc())
            return False

    def update_data(self):
        """Update historical data with latest prices"""
        self.logger.info("Updating market data...")

        if self.historical_data is None:
            self.logger.info("No historical data found, fetching initial data...")
            return self.fetch_historical_data()

        # Get current time
        now = datetime.now()

        # If last update was recent, skip
        if self.last_data_update and (now - self.last_data_update).seconds < 10:
            self.logger.info("Last update was recent, skipping update")
            return self.historical_data

        # Fetch latest bar
        self.logger.info(f"Fetching latest bar for {self.symbol}...")
        latest_bar = mt5.copy_rates_from(self.symbol, self.mt5_timeframe, now, 1)

        if latest_bar is None or len(latest_bar) == 0:
            self.logger.warning(f"Failed to fetch latest bar: {mt5.last_error()}")
            return self.historical_data

        self.logger.info(f"Latest bar fetched successfully: {latest_bar}")

        # Convert to DataFrame
        self.logger.info("Converting data to DataFrame...")
        latest_df = pd.DataFrame(latest_bar)
        latest_df['time'] = pd.to_datetime(latest_df['time'], unit='s')
        latest_df.set_index('time', inplace=True)

        # Check if we already have this bar
        if latest_df.index[0] in self.historical_data.index:
            self.logger.info(f"Updating existing bar at {latest_df.index[0]}")
            # Update the existing bar
            self.historical_data.loc[latest_df.index[0]] = latest_df.iloc[0]
        else:
            self.logger.info(f"Adding new bar at {latest_df.index[0]}")
            # Append new bar
            self.historical_data = pd.concat([self.historical_data, latest_df])

        # Remove oldest bar to maintain the same length
        if len(self.historical_data) > 500:
            self.historical_data = self.historical_data.iloc[-500:]

        # Recalculate indicators
        self.logger.info("Recalculating technical indicators...")
        self.historical_data['rsi'] = calculate_rsi(self.historical_data['close'])
        self.historical_data['macd'], self.historical_data['macd_signal'], self.historical_data['macd_hist'] = calculate_macd(self.historical_data['close'])
        self.historical_data['upper_bb'], self.historical_data['lower_bb'] = calculate_bollinger_bands(self.historical_data['close'])
        self.historical_data['atr'] = calculate_atr(self.historical_data['high'], self.historical_data['low'], self.historical_data['close'])
        self.historical_data['stoch_k'], self.historical_data['stoch_d'] = calculate_stochastic(self.historical_data['high'], self.historical_data['low'], self.historical_data['close'])
        self.historical_data['adx'] = calculate_adx(self.historical_data['high'], self.historical_data['low'], self.historical_data['close'])

        self.last_data_update = now

        return self.historical_data

    def select_strategy_profile(self, df, model_confidence=None):
        """
        Select the appropriate strategy profile based on market conditions and ML confidence

        Args:
            df: DataFrame with market data
            model_confidence: Confidence score from ML model (0-1)

        Returns:
            dict: Selected strategy profile parameters
        """
        self.logger.info("Selecting strategy profile based on market conditions and ML confidence")

        # Default to moderate if no ML confidence provided
        if model_confidence is None:
            model_confidence = 0.5  # Default to moderate confidence
            self.logger.info("No ML confidence provided, using default value of 0.5")

        # Get market regime
        try:
            regime = self.regime_detector.detect_regime(df, self.timeframe)
            self.logger.info(f"Detected market regime: {regime}")
        except Exception as e:
            self.logger.error(f"Error detecting market regime: {e}")
            regime = 0  # Default regime

        # Calculate volatility percentile
        try:
            if 'atr' in df.columns and len(df) > 20:
                atr_history = df['atr'].dropna()
                if len(atr_history) > 0:
                    atr_percentile = int(pd.Series(atr_history).rank(pct=True).iloc[-1] * 100)
                    self.logger.info(f"Current volatility percentile: {atr_percentile}")
                else:
                    atr_percentile = 50
        except Exception as e:
            self.logger.error(f"Error calculating volatility percentile: {e}")
            atr_percentile = 50

        # Calculate win rate factor (0-1)
        win_rate_factor = 0.5  # Default
        if hasattr(self, 'trades') and len(self.trades) >= 5:
            recent_trades = self.trades[-min(len(self.trades), 20):]  # Last 20 trades or all if fewer
            wins = sum(1 for trade in recent_trades if trade.get('profit_pct', 0) > 0)
            win_rate = wins / len(recent_trades) if recent_trades else 0.5
            win_rate_factor = min(win_rate, 1.0)  # Cap at 1.0
            self.logger.info(f"Recent win rate: {win_rate:.2f}, factor: {win_rate_factor:.2f}")

        # Calculate current market strength (0-1)
        market_strength = 0.5  # Default
        try:
            if 'adx' in df.columns and not df['adx'].isnull().all():
                adx_value = df['adx'].iloc[-1]
                # Normalize ADX (0-100) to 0-1 scale
                market_strength = min(adx_value / 50.0, 1.0)
                self.logger.info(f"Market strength factor: {market_strength:.2f} (ADX: {adx_value:.2f})")
        except Exception as e:
            self.logger.error(f"Error calculating market strength: {e}")

        # Calculate volatility factor (0-1, higher means lower volatility is better for aggressive profiles)
        volatility_factor = 1.0 - (min(atr_percentile, 100) / 100.0)
        self.logger.info(f"Volatility factor: {volatility_factor:.2f} (ATR percentile: {atr_percentile})")

        # Calculate overall confidence score combining all factors
        # Weights: ML confidence (40%), win rate (25%), market strength (20%), volatility (15%)
        overall_score = (
            (model_confidence * 0.40) +
            (win_rate_factor * 0.25) +
            (market_strength * 0.20) +
            (volatility_factor * 0.15)
        )
        
        self.logger.info(f"Overall confidence score: {overall_score:.2f}")

        # Select profile based on overall score
        if overall_score >= 0.85:
            self.logger.info("Ultra high confidence score - using ultra_aggressive profile")
            self.active_profile = "ultra_aggressive"
            return self.strategy_profiles["ultra_aggressive"]
        elif overall_score >= 0.70:
            self.logger.info("Very high confidence score - using very_aggressive profile")
            self.active_profile = "very_aggressive"
            return self.strategy_profiles["very_aggressive"]
        elif overall_score >= 0.55:
            self.logger.info("High confidence score - using aggressive profile")
            self.active_profile = "aggressive"
            return self.strategy_profiles["aggressive"]
        elif overall_score >= 0.40:
            self.logger.info("Moderate confidence score - using moderate profile")
            self.active_profile = "moderate"
            return self.strategy_profiles["moderate"]
        else:
            self.logger.info("Low confidence score - using conservative profile")
            self.active_profile = "conservative"
            return self.strategy_profiles["conservative"]

    def adjust_parameters(self, base_params, current_regime):
        """Adjust strategy parameters based on the current market regime"""
        adjusted_params = base_params.copy()

        if current_regime == 1:  # Bullish
            # In bullish regime, be more aggressive
            adjusted_params['rsi_entry_threshold'] = max(0.1, base_params['rsi_entry_threshold'] * 0.5)  # Lower RSI threshold for more entries
            adjusted_params['stop_loss_pct'] = base_params['stop_loss_pct'] * 1.5  # Wider stop loss
            adjusted_params['take_profit_pct'] = base_params['take_profit_pct'] * 2.0  # Higher profit target
            adjusted_params['position_size_multiplier'] = 2.0  # Larger position size

        elif current_regime == -1:  # Bearish
            # In bearish regime, be more conservative
            adjusted_params['rsi_entry_threshold'] = min(30, base_params['rsi_entry_threshold'] * 2.0)  # Higher RSI threshold for fewer entries
            adjusted_params['stop_loss_pct'] = base_params['stop_loss_pct'] * 0.5  # Tighter stop loss
            adjusted_params['take_profit_pct'] = base_params['take_profit_pct'] * 0.5  # Lower profit target
            adjusted_params['position_size_multiplier'] = 0.5  # Smaller position size

        # For neutral regime, use base parameters

        return adjusted_params

class DynamicExitSystem:
    """Advanced exit system with dynamic trailing stops and take profit levels"""
    def __init__(self):
        self.active_exits = {}

    def setup_advanced_exit_system(self, symbol, position_size, entry_price, direction,
                                  stop_loss_pct=0.01, take_profit_levels=[0.025, 0.05, 0.15],
                                  trailing_stop_activation_pct=0.018, trailing_stop_distance_pct=0.008):
        """Setup an advanced exit system for a trade"""
        # Calculate stop loss and take profit levels
        stop_level = self._calculate_stop_level(entry_price, stop_loss_pct, direction)

        # Calculate multiple take profit levels
        take_profit_levels = [self._calculate_take_profit_level(entry_price, tp_pct, direction)
                             for tp_pct in take_profit_levels]

        # Calculate trailing stop activation level
        activation_price = self._calculate_activation_price(entry_price, trailing_stop_activation_pct, direction)

        self.active_exits[symbol] = {
            'entry_price': entry_price,
            'position_size': position_size,
            'direction': direction,
            'stop_loss': stop_level,
            'take_profit_levels': take_profit_levels,
            'take_profit_triggered': [False] * len(take_profit_levels),
        'trailing_stop_activation': activation_price,
        'trailing_stop_distance_pct': trailing_stop_distance_pct,
        'trailing_stop': None,  # Will be set once activated
        'partial_exits_done': 0
        }

        logger.info(f"Setup exit system for {symbol}: Entry={entry_price}, Stop={stop_level}, "
        f"TPs={take_profit_levels}, Activation={activation_price}")

        return self.active_exits[symbol]

    def _calculate_stop_level(self, entry_price, stop_pct, direction):
        """Calculate stop loss level based on entry price and direction"""
        if direction == "long":
            return entry_price * (1 - stop_pct)
        else:  # short
            return entry_price * (1 + stop_pct)

    def _calculate_take_profit_level(self, entry_price, profit_pct, direction):
        """Calculate take profit level based on entry price and direction"""
        if direction == "long":
            return entry_price * (1 + profit_pct)
        else:  # short
            return entry_price * (1 - profit_pct)

    def _calculate_activation_price(self, entry_price, activation_pct, direction):
        """Calculate trailing stop activation price"""
        if direction == "long":
            return entry_price * (1 + activation_pct)
        else:  # short
            return entry_price * (1 - activation_pct)

    def update_exit_system(self, symbol, current_price):
        """Update exit system based on current price and check for exit signals"""
        if symbol not in self.active_exits:
            return None
        
        exit_info = self.active_exits[symbol]
        direction = exit_info['direction']
        
        # Check stop loss
        if (direction == "long" and current_price <= exit_info['stop_loss']) or \
            (direction == "short" and current_price >= exit_info['stop_loss']):
            return self._execute_exit(symbol, current_price, exit_info['position_size'], "stop_loss")
                    
        # Check take profit levels for partial exits
        for i, (tp_level, triggered) in enumerate(zip(exit_info['take_profit_levels'], exit_info['take_profit_triggered'])):
            if not triggered:
                if (direction == "long" and current_price >= tp_level) or \
                   (direction == "short" and current_price <= tp_level):
                    # Mark this level as triggered
                    exit_info['take_profit_triggered'][i] = True

                    # Calculate the portion to exit at this level (progressive scaling)
                    exit_portion = 0.25 * (i + 1)  # Exit 25%, 50%, 75% at each level
                    exit_size = exit_info['position_size'] * exit_portion

                    # Update remaining position size
                    exit_info['position_size'] -= exit_size
                    exit_info['partial_exits_done'] += 1

                    if exit_info['position_size'] <= 0.001:  # Full exit
                        return self._execute_exit(symbol, current_price, exit_size, "take_profit_full")
                    else:
                        # Partial exit
                        return {
                            'symbol': symbol,
                            'price': current_price,
                            'size': exit_size,
                            'reason': f"take_profit_{i+1}",
                            'entry_price': exit_info['entry_price']
                        }
                
                    # Check and update trailing stop
                    if exit_info['trailing_stop'] is None:
                        # Check if price reached activation level
                        if (direction == "long" and current_price >= exit_info['trailing_stop_activation']) or \
                           (direction == "short" and current_price <= exit_info['trailing_stop_activation']):
                            # Activate trailing stop
                            exit_info['trailing_stop'] = self._calculate_trailing_stop_level(
                                current_price, exit_info['trailing_stop_distance_pct'], direction)
                            logger.info(f"Trailing stop activated for {symbol} at {exit_info['trailing_stop']}")
                    
                            # Check if price hit trailing stop
                            if (direction == "long" and current_price <= exit_info['trailing_stop']) or \
                               (direction == "short" and current_price >= exit_info['trailing_stop']):
                                return self._execute_exit(symbol, current_price, exit_info['position_size'], "trailing_stop")

    def _calculate_trailing_stop_level(self, current_price, distance_pct, direction):
        """Calculate trailing stop level based on current price"""
        if direction == "long":
            return current_price * (1 - distance_pct)
        else:  # short
            return current_price * (1 + distance_pct)

    def _execute_exit(self, symbol, price, size, reason):
        """Execute exit and remove from active exits"""
        exit_info = self.active_exits.pop(symbol, None)
        if not exit_info:
            return None

        logger.info(f"Exit signal for {symbol}: Price={price}, Size={size}, Reason={reason}")

        return {
            'symbol': symbol,
            'price': price,
            'size': size,
            'reason': reason,
            'entry_price': exit_info['entry_price']
        }

class MetaTraderLiveStrategy:
    def __init__(self, symbol="EURUSD", timeframe="M5", initial_deposit=50000,
                 rsi_entry_threshold=0.1, stop_loss_pct=0.05, take_profit_pct=0.2,
                 max_position_size=1.0, results_dir=RESULTS_DIR):
        # Initialize logger for this instance
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        if not self.logger.handlers:  # Avoid duplicate handlers
            self.logger.addHandler(handler)

        self.symbol = symbol
        self.timeframe = timeframe
        self.mt5_timeframe = self._get_mt5_timeframe(timeframe)
        self.initial_deposit = initial_deposit
        self.rsi_entry_threshold = rsi_entry_threshold
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self.max_position_size = max_position_size
        self.results_dir = results_dir
        self.mt5_initialized = False
        self.current_positions = {}
        self.trades = []
        self.historical_data = None
        self.last_data_update = None
        self.win_streak = 0
        self.loss_streak = 0
        self.strategy_profiles = {
            "conservative": {
                "long_rsi_threshold": 30,
                "short_rsi_threshold": 70,
                "stop_loss_pct": 0.02,
                "take_profit_pct": 0.03,
                "position_size_multiplier": 0.5,
                "max_concurrent_trades": 5
            },
            "moderate": {
                "long_rsi_threshold": 20,
                "short_rsi_threshold": 80,
                "stop_loss_pct": 0.05,
                "take_profit_pct": 0.10,
                "position_size_multiplier": 1.0,
                "max_concurrent_trades": 4
            },
            "aggressive": {
                "long_rsi_threshold": 10,
                "short_rsi_threshold": 90,
                "stop_loss_pct": 0.10,
                "take_profit_pct": 0.20,
                "position_size_multiplier": 2.0,
                "max_concurrent_trades": 3
            },
            "very_aggressive": {
                "long_rsi_threshold": 5,
                "short_rsi_threshold": 95,
                "stop_loss_pct": 0.15,
                "take_profit_pct": 0.40,
                "position_size_multiplier": 3.0,
                "max_concurrent_trades": 2
            },
            "ultra_aggressive": {
                "long_rsi_threshold": 0.1,
                "short_rsi_threshold": 99.9,
                "stop_loss_pct": 0.25,
                "take_profit_pct": 0.75,
                "position_size_multiplier": 5.0,
                "max_concurrent_trades": 1
            }
        }
        self.active_profile = "moderate"
        self.regime_detector = MarketRegimeDetector()
        self.trading_models = EnhancedTrading()

    def _get_mt5_timeframe(self, timeframe_str):
        """Convert string timeframe to MT5 timeframe constant"""
        timeframe_map = {
            "M1": mt5.TIMEFRAME_M1,
            "M5": mt5.TIMEFRAME_M5,
            "M15": mt5.TIMEFRAME_M15,
            "M30": mt5.TIMEFRAME_M30,
            "H1": mt5.TIMEFRAME_H1,
            "H4": mt5.TIMEFRAME_H4,
            "D1": mt5.TIMEFRAME_D1,
            "W1": mt5.TIMEFRAME_W1,
            "MN1": mt5.TIMEFRAME_MN1
        }
        return timeframe_map.get(timeframe_str, mt5.TIMEFRAME_M5)

    
    
    def initialize_mt5(self):
        """Initialize connection to MetaTrader 5"""
        self.logger.info("Initializing MetaTrader 5 connection...")

        if mt5 is None:
            self.logger.error("MetaTrader5 module not found. Running in simulation mode.")
            self.mt5_initialized = False
            return False

        try:
            # Attempt to initialize MT5
            if not mt5.initialize():
                error_code = mt5.last_error()
                self.logger.error(f"Failed to initialize MT5: Error code {error_code}")
                self.mt5_initialized = False
                return False

            # Verify connection by checking terminal info
            terminal_info = mt5.terminal_info()
            if terminal_info is None:
                self.logger.error("MT5 initialized but no terminal info available: Error code " + str(mt5.last_error()))
                mt5.shutdown()
                self.mt5_initialized = False
                return False

            # Check if connected to a broker
            account_info = mt5.account_info()
            if account_info is None:
                self.logger.error("Not connected to a broker: Error code " + str(mt5.last_error()))
                mt5.shutdown()
                self.mt5_initialized = False
                return False

            # Verify symbol availability
            symbol_info = mt5.symbol_info(self.symbol)
            if symbol_info is None:
                self.logger.error(f"Symbol {self.symbol} not found in market watch.")
                mt5.shutdown()
                self.mt5_initialized = False
                return False

            # Ensure symbol is enabled
            if not symbol_info.visible:
                if not mt5.symbol_select(self.symbol, True):
                    self.logger.error(f"Failed to enable symbol {self.symbol}.")
                    mt5.shutdown()
                    self.mt5_initialized = False
                    return False

            self.logger.info("MetaTrader 5 initialized successfully.")
            self.mt5_initialized = True
            return True

        except Exception as e:
            self.logger.error(f"Exception during MT5 initialization: {str(e)}")
            self.logger.error(traceback.format_exc())
            self.mt5_initialized = False
            return False

    def get_account_info(self):
        """Retrieve account information from MetaTrader 5"""
        self.logger.info("Retrieving account information...")

        if not self.mt5_initialized:
            self.logger.error("MT5 not initialized. Attempting to initialize...")
            if not self.initialize_mt5():
                self.logger.error("Failed to initialize MT5. Cannot retrieve account info.")
                return None

        try:
            account_info = mt5.account_info()
            if account_info is None:
                self.logger.error(f"Failed to get account info: {mt5.last_error()}")
                return None

            account_data = {
                'login': account_info.login,
                'balance': account_info.balance,
                'equity': account_info.equity,
                'margin': account_info.margin,
                'margin_free': account_info.margin_free,
                'currency': account_info.currency
            }
            self.logger.info(f"Account info retrieved: Balance={account_data['balance']}, Equity={account_data['equity']}")
            return account_data

        except Exception as e:
            self.logger.error(f"Error retrieving account info: {str(e)}")
            self.logger.error(traceback.format_exc())
            return None
    
    def fetch_historical_data(self, bars=500):
        """Fetch historical price data from MetaTrader 5"""
        self.logger.info(f"Fetching historical data for {self.symbol}, requesting {bars} bars...")

        if not self.mt5_initialized:
            self.logger.error("MT5 not initialized. Cannot fetch data.")
            return False

        # Get current time in UTC
        now = datetime.now()
        self.logger.info(f"Current time: {now}")

        # Fetch historical data
        self.logger.info(f"Requesting historical data from MT5 for {self.symbol} with timeframe {self.mt5_timeframe}...")
        try:
            self.logger.info("Calling mt5.copy_rates_from...")
            rates = mt5.copy_rates_from(self.symbol, self.mt5_timeframe, now, bars)

            if rates is None or len(rates) == 0:
                self.logger.error(f"Failed to fetch historical data: {mt5.last_error()}")
                return False

            self.logger.info(f"Successfully fetched {len(rates)} bars of historical data")

            # Convert to DataFrame
            self.logger.info("Converting data to DataFrame...")
            df = pd.DataFrame(rates)
            df['time'] = pd.to_datetime(df['time'], unit='s')
            df.set_index('time', inplace=True)

            self.logger.info("Calculating additional features...")

            # Calculate additional features
            self.logger.info("Calculating technical indicators...")
            try:
                # Calculate indicators
                self.logger.info("Calculating RSI...")
                df['rsi'] = calculate_rsi(df['close'])

                self.logger.info("Calculating MACD...")
                df['macd'], df['macd_signal'], df['macd_hist'] = calculate_macd(df['close'])

                self.logger.info("Calculating Bollinger Bands...")
                df['upper_bb'], df['lower_bb'] = calculate_bollinger_bands(df['close'])

                self.logger.info("Calculating ATR...")
                df['atr'] = calculate_atr(df['high'], df['low'], df['close'])

                self.logger.info("Calculating Stochastic Oscillator...")
                df['stoch_k'], df['stoch_d'] = calculate_stochastic(df['high'], df['low'], df['close'])

                self.logger.info("Calculating ADX...")
                df['adx'] = calculate_adx(df['high'], df['low'], df['close'])

                self.logger.info("All technical indicators calculated successfully")
            except Exception as e:
                self.logger.error(f"Error calculating technical indicators: {e}")
                self.logger.error(traceback.format_exc())
                # Continue even if indicators fail, as we can still use price data

            # Store the data
            self.historical_data = df
            self.last_data_update = now

            self.logger.info(f"Historical data updated successfully with {len(df)} rows")
            return df

        except Exception as e:
            self.logger.error(f"Exception during historical data fetch: {e}")
            self.logger.error(traceback.format_exc())
            return False

    def update_data(self):
        """Update historical data with latest prices"""
        self.logger.info("Updating market data...")

        if self.historical_data is None:
            self.logger.info("No historical data found, fetching initial data...")
            return self.fetch_historical_data()

        # Get current time
        now = datetime.now()

        # If last update was recent, skip
        if self.last_data_update and (now - self.last_data_update).seconds < 10:
            self.logger.info("Last update was recent, skipping update")
            return self.historical_data

        # Fetch latest bar
        self.logger.info(f"Fetching latest bar for {self.symbol}...")
        latest_bar = mt5.copy_rates_from(self.symbol, self.mt5_timeframe, now, 1)

        if latest_bar is None or len(latest_bar) == 0:
            self.logger.warning(f"Failed to fetch latest bar: {mt5.last_error()}")
            return self.historical_data

        self.logger.info(f"Latest bar fetched successfully: {latest_bar}")

        # Convert to DataFrame
        self.logger.info("Converting data to DataFrame...")
        latest_df = pd.DataFrame(latest_bar)
        latest_df['time'] = pd.to_datetime(latest_df['time'], unit='s')
        latest_df.set_index('time', inplace=True)

        # Check if we already have this bar
        if latest_df.index[0] in self.historical_data.index:
            self.logger.info(f"Updating existing bar at {latest_df.index[0]}")
            # Update the existing bar
            self.historical_data.loc[latest_df.index[0]] = latest_df.iloc[0]
        else:
            self.logger.info(f"Adding new bar at {latest_df.index[0]}")
            # Append new bar
            self.historical_data = pd.concat([self.historical_data, latest_df])

        # Remove oldest bar to maintain the same length
        if len(self.historical_data) > 500:
            self.historical_data = self.historical_data.iloc[-500:]

        # Recalculate indicators
        self.logger.info("Recalculating technical indicators...")
        self.historical_data['rsi'] = calculate_rsi(self.historical_data['close'])
        self.historical_data['macd'], self.historical_data['macd_signal'], self.historical_data['macd_hist'] = calculate_macd(self.historical_data['close'])
        self.historical_data['upper_bb'], self.historical_data['lower_bb'] = calculate_bollinger_bands(self.historical_data['close'])
        self.historical_data['atr'] = calculate_atr(self.historical_data['high'], self.historical_data['low'], self.historical_data['close'])
        self.historical_data['stoch_k'], self.historical_data['stoch_d'] = calculate_stochastic(self.historical_data['high'], self.historical_data['low'], self.historical_data['close'])
        self.historical_data['adx'] = calculate_adx(self.historical_data['high'], self.historical_data['low'], self.historical_data['close'])

        self.last_data_update = now

        return self.historical_data

    def select_strategy_profile(self, df, model_confidence=None):
        """
        Select the appropriate strategy profile based on market conditions and ML confidence

        Args:
            df: DataFrame with market data
            model_confidence: Confidence score from ML model (0-1)

        Returns:
            dict: Selected strategy profile parameters
        """
        self.logger.info("Selecting strategy profile based on market conditions and ML confidence")

        # Default to moderate if no ML confidence provided
        if model_confidence is None:
            model_confidence = 0.5  # Default to moderate confidence
            self.logger.info("No ML confidence provided, using default value of 0.5")

        # Get market regime
        try:
            regime = self.regime_detector.detect_regime(df, self.timeframe)
            self.logger.info(f"Detected market regime: {regime}")
        except Exception as e:
            self.logger.error(f"Error detecting market regime: {e}")
            regime = 0  # Default regime

        # Calculate volatility percentile
        try:
            if 'atr' in df.columns and len(df) > 20:
                atr_history = df['atr'].dropna()
                if len(atr_history) > 0:
                    atr_percentile = int(pd.Series(atr_history).rank(pct=True).iloc[-1] * 100)
                    self.logger.info(f"Current volatility percentile: {atr_percentile}")
                else:
                    atr_percentile = 50
        except Exception as e:
            self.logger.error(f"Error calculating volatility percentile: {e}")
            atr_percentile = 50

        # Calculate win rate factor (0-1)
        win_rate_factor = 0.5  # Default
        if hasattr(self, 'trades') and len(self.trades) >= 5:
            recent_trades = self.trades[-min(len(self.trades), 20):]  # Last 20 trades or all if fewer
            wins = sum(1 for trade in recent_trades if trade.get('profit_pct', 0) > 0)
            win_rate = wins / len(recent_trades) if recent_trades else 0.5
            win_rate_factor = min(win_rate, 1.0)  # Cap at 1.0
            self.logger.info(f"Recent win rate: {win_rate:.2f}, factor: {win_rate_factor:.2f}")

        # Calculate current market strength (0-1)
        market_strength = 0.5  # Default
        try:
            if 'adx' in df.columns and not df['adx'].isnull().all():
                adx_value = df['adx'].iloc[-1]
                # Normalize ADX (0-100) to 0-1 scale
                market_strength = min(adx_value / 50.0, 1.0)
                self.logger.info(f"Market strength factor: {market_strength:.2f} (ADX: {adx_value:.2f})")
        except Exception as e:
            self.logger.error(f"Error calculating market strength: {e}")

        # Calculate volatility factor (0-1, higher means lower volatility is better for aggressive profiles)
        volatility_factor = 1.0 - (min(atr_percentile, 100) / 100.0)
        self.logger.info(f"Volatility factor: {volatility_factor:.2f} (ATR percentile: {atr_percentile})")

        # Calculate overall confidence score combining all factors
        # Weights: ML confidence (40%), win rate (25%), market strength (20%), volatility (15%)
        overall_score = (
            (model_confidence * 0.40) +
            (win_rate_factor * 0.25) +
            (market_strength * 0.20) +
            (volatility_factor * 0.15)
        )
        
        self.logger.info(f"Overall confidence score: {overall_score:.2f}")

        # Select profile based on overall score
        if overall_score >= 0.85:
            self.logger.info("Ultra high confidence score - using ultra_aggressive profile")
            self.active_profile = "ultra_aggressive"
            return self.strategy_profiles["ultra_aggressive"]
        elif overall_score >= 0.70:
            self.logger.info("Very high confidence score - using very_aggressive profile")
            self.active_profile = "very_aggressive"
            return self.strategy_profiles["very_aggressive"]
        elif overall_score >= 0.55:
            self.logger.info("High confidence score - using aggressive profile")
            self.active_profile = "aggressive"
            return self.strategy_profiles["aggressive"]
        elif overall_score >= 0.40:
            self.logger.info("Moderate confidence score - using moderate profile")
            self.active_profile = "moderate"
            return self.strategy_profiles["moderate"]
        else:
            self.logger.info("Low confidence score - using conservative profile")
            self.active_profile = "conservative"
            return self.strategy_profiles["conservative"]

    def adjust_parameters(self, base_params, current_regime):
        """Adjust strategy parameters based on the current market regime"""
        adjusted_params = base_params.copy()

        if current_regime == 1:  # Bullish
            # In bullish regime, be more aggressive
            adjusted_params['rsi_entry_threshold'] = max(0.1, base_params['rsi_entry_threshold'] * 0.5)  # Lower RSI threshold for more entries
            adjusted_params['stop_loss_pct'] = base_params['stop_loss_pct'] * 1.5  # Wider stop loss
            adjusted_params['take_profit_pct'] = base_params['take_profit_pct'] * 2.0  # Higher profit target
            adjusted_params['position_size_multiplier'] = 2.0  # Larger position size

        elif current_regime == -1:  # Bearish
            # In bearish regime, be more conservative
            adjusted_params['rsi_entry_threshold'] = min(30, base_params['rsi_entry_threshold'] * 2.0)  # Higher RSI threshold for fewer entries
            adjusted_params['stop_loss_pct'] = base_params['stop_loss_pct'] * 0.5  # Tighter stop loss
            adjusted_params['take_profit_pct'] = base_params['take_profit_pct'] * 0.5  # Lower profit target
            adjusted_params['position_size_multiplier'] = 0.5  # Smaller position size

        # For neutral regime, use base parameters

        return adjusted_params

class DynamicExitSystem:
    """Advanced exit system with dynamic trailing stops and take profit levels"""
    def __init__(self):
        self.active_exits = {}

    def setup_advanced_exit_system(self, symbol, position_size, entry_price, direction,
                                  stop_loss_pct=0.01, take_profit_levels=[0.025, 0.05, 0.15],
                                  trailing_stop_activation_pct=0.018, trailing_stop_distance_pct=0.008):
        """Setup an advanced exit system for a trade"""
        # Calculate stop loss and take profit levels
        stop_level = self._calculate_stop_level(entry_price, stop_loss_pct, direction)

        # Calculate multiple take profit levels
        take_profit_levels = [self._calculate_take_profit_level(entry_price, tp_pct, direction)
                             for tp_pct in take_profit_levels]

        # Calculate trailing stop activation level
        activation_price = self._calculate_activation_price(entry_price, trailing_stop_activation_pct, direction)

        self.active_exits[symbol] = {
            'entry_price': entry_price,
            'position_size': position_size,
            'direction': direction,
            'stop_loss': stop_level,
            'take_profit_levels': take_profit_levels,
            'take_profit_triggered': [False] * len(take_profit_levels),
        'trailing_stop_activation': activation_price,
        'trailing_stop_distance_pct': trailing_stop_distance_pct,
        'trailing_stop': None,  # Will be set once activated
        'partial_exits_done': 0
        }

        logger.info(f"Setup exit system for {symbol}: Entry={entry_price}, Stop={stop_level}, "
        f"TPs={take_profit_levels}, Activation={activation_price}")

        return self.active_exits[symbol]

    def _calculate_stop_level(self, entry_price, stop_pct, direction):
        """Calculate stop loss level based on entry price and direction"""
        if direction == "long":
            return entry_price * (1 - stop_pct)
        else:  # short
            return entry_price * (1 + stop_pct)

    def _calculate_take_profit_level(self, entry_price, profit_pct, direction):
        """Calculate take profit level based on entry price and direction"""
        if direction == "long":
            return entry_price * (1 + profit_pct)
        else:  # short
            return entry_price * (1 - profit_pct)

    def _calculate_activation_price(self, entry_price, activation_pct, direction):
        """Calculate trailing stop activation price"""
        if direction == "long":
            return entry_price * (1 + activation_pct)
        else:  # short
            return entry_price * (1 - activation_pct)

    def update_exit_system(self, symbol, current_price):
        """Update exit system based on current price and check for exit signals"""
        if symbol not in self.active_exits:
            return None
        
        exit_info = self.active_exits[symbol]
        direction = exit_info['direction']
        
        # Check stop loss
        if (direction == "long" and current_price <= exit_info['stop_loss']) or \
            (direction == "short" and current_price >= exit_info['stop_loss']):
            return self._execute_exit(symbol, current_price, exit_info['position_size'], "stop_loss")
                    
        # Check take profit levels for partial exits
        for i, (tp_level, triggered) in enumerate(zip(exit_info['take_profit_levels'], exit_info['take_profit_triggered'])):
            if not triggered:
                if (direction == "long" and current_price >= tp_level) or \
                   (direction == "short" and current_price <= tp_level):
                    # Mark this level as triggered
                    exit_info['take_profit_triggered'][i] = True

                    # Calculate the portion to exit at this level (progressive scaling)
                    exit_portion = 0.25 * (i + 1)  # Exit 25%, 50%, 75% at each level
                    exit_size = exit_info['position_size'] * exit_portion

                    # Update remaining position size
                    exit_info['position_size'] -= exit_size
                    exit_info['partial_exits_done'] += 1

                    if exit_info['position_size'] <= 0.001:  # Full exit
                        return self._execute_exit(symbol, current_price, exit_size, "take_profit_full")
                    else:
                        # Partial exit
                        return {
                            'symbol': symbol,
                            'price': current_price,
                            'size': exit_size,
                            'reason': f"take_profit_{i+1}",
                            'entry_price': exit_info['entry_price']
                        }
                
                    # Check and update trailing stop
                    if exit_info['trailing_stop'] is None:
                        # Check if price reached activation level
                        if (direction == "long" and current_price >= exit_info['trailing_stop_activation']) or \
                           (direction == "short" and current_price <= exit_info['trailing_stop_activation']):
                            # Activate trailing stop
                            exit_info['trailing_stop'] = self._calculate_trailing_stop_level(
                                current_price, exit_info['trailing_stop_distance_pct'], direction)
                            logger.info(f"Trailing stop activated for {symbol} at {exit_info['trailing_stop']}")
                    
                            # Check if price hit trailing stop
                            if (direction == "long" and current_price <= exit_info['trailing_stop']) or \
                               (direction == "short" and current_price >= exit_info['trailing_stop']):
                                return self._execute_exit(symbol, current_price, exit_info['position_size'], "trailing_stop")

    def _calculate_trailing_stop_level(self, current_price, distance_pct, direction):
        """Calculate trailing stop level based on current price"""
        if direction == "long":
            return current_price * (1 - distance_pct)
        else:  # short
            return current_price * (1 + distance_pct)

    def _execute_exit(self, symbol, price, size, reason):
        """Execute exit and remove from active exits"""
        exit_info = self.active_exits.pop(symbol, None)
        if not exit_info:
            return None

        logger.info(f"Exit signal for {symbol}: Price={price}, Size={size}, Reason={reason}")

        return {
            'symbol': symbol,
            'price': price,
            'size': size,
            'reason': reason,
            'entry_price': exit_info['entry_price']
        }

class MetaTraderLiveStrategy:
    def __init__(self, symbol="EURUSD", timeframe="M5", initial_deposit=50000,
                 rsi_entry_threshold=0.1, stop_loss_pct=0.05, take_profit_pct=0.2,
                 max_position_size=1.0, results_dir=RESULTS_DIR):
        # Initialize logger for this instance
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        if not self.logger.handlers:  # Avoid duplicate handlers
            self.logger.addHandler(handler)

        self.symbol = symbol
        self.timeframe = timeframe
        self.mt5_timeframe = self._get_mt5_timeframe(timeframe)
        self.initial_deposit = initial_deposit
        self.rsi_entry_threshold = rsi_entry_threshold
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self.max_position_size = max_position_size
        self.results_dir = results_dir
        self.mt5_initialized = False
        self.current_positions = {}
        self.trades = []
        self.historical_data = None
        self.last_data_update = None
        self.win_streak = 0
        self.loss_streak = 0
        self.strategy_profiles = {
            "conservative": {
                "long_rsi_threshold": 30,
                "short_rsi_threshold": 70,
                "stop_loss_pct": 0.02,
                "take_profit_pct": 0.03,
                "position_size_multiplier": 0.5,
                "max_concurrent_trades": 5
            },
            "moderate": {
                "long_rsi_threshold": 20,
                "short_rsi_threshold": 80,
                "stop_loss_pct": 0.05,
                "take_profit_pct": 0.10,
                "position_size_multiplier": 1.0,
                "max_concurrent_trades": 4
            },
            "aggressive": {
                "long_rsi_threshold": 10,
                "short_rsi_threshold": 90,
                "stop_loss_pct": 0.10,
                "take_profit_pct": 0.20,
                "position_size_multiplier": 2.0,
                "max_concurrent_trades": 3
            },
            "very_aggressive": {
                "long_rsi_threshold": 5,
                "short_rsi_threshold": 95,
                "stop_loss_pct": 0.15,
                "take_profit_pct": 0.40,
                "position_size_multiplier": 3.0,
                "max_concurrent_trades": 2
            },
            "ultra_aggressive": {
                "long_rsi_threshold": 0.1,
                "short_rsi_threshold": 99.9,
                "stop_loss_pct": 0.25,
                "take_profit_pct": 0.75,
                "position_size_multiplier": 5.0,
                "max_concurrent_trades": 1
            }
        }
        self.active_profile = "moderate"
        self.regime_detector = MarketRegimeDetector()
        self.trading_models = EnhancedTrading()

    def _get_mt5_timeframe(self, timeframe_str):
        """Convert string timeframe to MT5 timeframe constant"""
        timeframe_map = {
            "M1": mt5.TIMEFRAME_M1,
            "M5": mt5.TIMEFRAME_M5,
            "M15": mt5.TIMEFRAME_M15,
            "M30": mt5.TIMEFRAME_M30,
            "H1": mt5.TIMEFRAME_H1,
            "H4": mt5.TIMEFRAME_H4,
            "D1": mt5.TIMEFRAME_D1,
            "W1": mt5.TIMEFRAME_W1,
            "MN1": mt5.TIMEFRAME_MN1
        }
        return timeframe_map.get(timeframe_str, mt5.TIMEFRAME_M5)

    
    
    def initialize_mt5(self):
        """Initialize connection to MetaTrader 5"""
        self.logger.info("Initializing MetaTrader 5 connection...")

        if mt5 is None:
            self.logger.error("MetaTrader5 module not found. Running in simulation mode.")
            self.mt5_initialized = False
            return False

        try:
            # Attempt to initialize MT5
            if not mt5.initialize():
                error_code = mt5.last_error()
                self.logger.error(f"Failed to initialize MT5: Error code {error_code}")
                self.mt5_initialized = False
                return False

            # Verify connection by checking terminal info
            terminal_info = mt5.terminal_info()
            if terminal_info is None:
                self.logger.error("MT5 initialized but no terminal info available: Error code " + str(mt5.last_error()))
                mt5.shutdown()
                self.mt5_initialized = False
                return False

            # Check if connected to a broker
            account_info = mt5.account_info()
            if account_info is None:
                self.logger.error("Not connected to a broker: Error code " + str(mt5.last_error()))
                mt5.shutdown()
                self.mt5_initialized = False
                return False

            # Verify symbol availability
            symbol_info = mt5.symbol_info(self.symbol)
            if symbol_info is None:
                self.logger.error(f"Symbol {self.symbol} not found in market watch.")
                mt5.shutdown()
                self.mt5_initialized = False
                return False

            # Ensure symbol is enabled
            if not symbol_info.visible:
                if not mt5.symbol_select(self.symbol, True):
                    self.logger.error(f"Failed to enable symbol {self.symbol}.")
                    mt5.shutdown()
                    self.mt5_initialized = False
                    return False

            self.logger.info("MetaTrader 5 initialized successfully.")
            self.mt5_initialized = True
            return True

        except Exception as e:
            self.logger.error(f"Exception during MT5 initialization: {str(e)}")
            self.logger.error(traceback.format_exc())
            self.mt5_initialized = False
            return False

    def get_account_info(self):
        """Retrieve account information from MetaTrader 5"""
        self.logger.info("Retrieving account information...")

        if not self.mt5_initialized:
            self.logger.error("MT5 not initialized. Attempting to initialize...")
            if not self.initialize_mt5():
                self.logger.error("Failed to initialize MT5. Cannot retrieve account info.")
                return None

        try:
            account_info = mt5.account_info()
            if account_info is None:
                self.logger.error(f"Failed to get account info: {mt5.last_error()}")
                return None

            account_data = {
                'login': account_info.login,
                'balance': account_info.balance,
                'equity': account_info.equity,
                'margin': account_info.margin,
                'margin_free': account_info.margin_free,
                'currency': account_info.currency
            }
            self.logger.info(f"Account info retrieved: Balance={account_data['balance']}, Equity={account_data['equity']}")
            return account_data

        except Exception as e:
            self.logger.error(f"Error retrieving account info: {str(e)}")
            self.logger.error(traceback.format_exc())
            return None
    
    def fetch_historical_data(self, bars=500):
        """Fetch historical price data from MetaTrader 5"""
        self.logger.info(f"Fetching historical data for {self.symbol}, requesting {bars} bars...")

        if not self.mt5_initialized:
            self.logger.error("MT5 not initialized. Cannot fetch data.")
            return False

        # Get current time in UTC
        now = datetime.now()
        self.logger.info(f"Current time: {now}")

        # Fetch historical data
        self.logger.info(f"Requesting historical data from MT5 for {self.symbol} with timeframe {self.mt5_timeframe}...")
        try:
            self.logger.info("Calling mt5.copy_rates_from...")
            rates = mt5.copy_rates_from(self.symbol, self.mt5_timeframe, now, bars)

            if rates is None or len(rates) == 0:
                self.logger.error(f"Failed to fetch historical data: {mt5.last_error()}")
                return False

            self.logger.info(f"Successfully fetched {len(rates)} bars of historical data")

            # Convert to DataFrame
            self.logger.info("Converting data to DataFrame...")
            df = pd.DataFrame(rates)
            df['time'] = pd.to_datetime(df['time'], unit='s')
            df.set_index('time', inplace=True)

            self.logger.info("Calculating additional features...")

            # Calculate additional features
            self.logger.info("Calculating technical indicators...")
            try:
                # Calculate indicators
                self.logger.info("Calculating RSI...")
                df['rsi'] = calculate_rsi(df['close'])

                self.logger.info("Calculating MACD...")
                df['macd'], df['macd_signal'], df['macd_hist'] = calculate_macd(df['close'])

                self.logger.info("Calculating Bollinger Bands...")
                df['upper_bb'], df['lower_bb'] = calculate_bollinger_bands(df['close'])

                self.logger.info("Calculating ATR...")
                df['atr'] = calculate_atr(df['high'], df['low'], df['close'])

                self.logger.info("Calculating Stochastic Oscillator...")
                df['stoch_k'], df['stoch_d'] = calculate_stochastic(df['high'], df['low'], df['close'])

                self.logger.info("Calculating ADX...")
                df['adx'] = calculate_adx(df['high'], df['low'], df['close'])

                self.logger.info("All technical indicators calculated successfully")
            except Exception as e:
                self.logger.error(f"Error calculating technical indicators: {e}")
                self.logger.error(traceback.format_exc())
                # Continue even if indicators fail, as we can still use price data

            # Store the data
            self.historical_data = df
            self.last_data_update = now

            self.logger.info(f"Historical data updated successfully with {len(df)} rows")
            return df

        except Exception as e:
            self.logger.error(f"Exception during historical data fetch: {e}")
            self.logger.error(traceback.format_exc())
            return False

    def update_data(self):
        """Update historical data with latest prices"""
        self.logger.info("Updating market data...")

        if self.historical_data is None:
            self.logger.info("No historical data found, fetching initial data...")
            return self.fetch_historical_data()

        # Get current time
        now = datetime.now()

        # If last update was recent, skip
        if self.last_data_update and (now - self.last_data_update).seconds < 10:
            self.logger.info("Last update was recent, skipping update")
            return self.historical_data

        # Fetch latest bar
        self.logger.info(f"Fetching latest bar for {self.symbol}...")
        latest_bar = mt5.copy_rates_from(self.symbol, self.mt5_timeframe, now, 1)

        if latest_bar is None or len(latest_bar) == 0:
            self.logger.warning(f"Failed to fetch latest bar: {mt5.last_error()}")
            return self.historical_data

        self.logger.info(f"Latest bar fetched successfully: {latest_bar}")

        # Convert to DataFrame
        self.logger.info("Converting data to DataFrame...")
        latest_df = pd.DataFrame(latest_bar)
        latest_df['time'] = pd.to_datetime(latest_df['time'], unit='s')
        latest_df.set_index('time', inplace=True)

        # Check if we already have this bar
        if latest_df.index[0] in self.historical_data.index:
            self.logger.info(f"Updating existing bar at {latest_df.index[0]}")
            # Update the existing bar
            self.historical_data.loc[latest_df.index[0]] = latest_df.iloc[0]
        else:
            self.logger.info(f"Adding new bar at {latest_df.index[0]}")
            # Append new bar
            self.historical_data = pd.concat([self.historical_data, latest_df])

        # Remove oldest bar to maintain the same length
        if len(self.historical_data) > 500:
            self.historical_data = self.historical_data.iloc[-500:]

        # Recalculate indicators
        self.logger.info("Recalculating technical indicators...")
        self.historical_data['rsi'] = calculate_rsi(self.historical_data['close'])
        self.historical_data['macd'], self.historical_data['macd_signal'], self.historical_data['macd_hist'] = calculate_macd(self.historical_data['close'])
        self.historical_data['upper_bb'], self.historical_data['lower_bb'] = calculate_bollinger_bands(self.historical_data['close'])
        self.historical_data['atr'] = calculate_atr(self.historical_data['high'], self.historical_data['low'], self.historical_data['close'])
        self.historical_data['stoch_k'], self.historical_data['stoch_d'] = calculate_stochastic(self.historical_data['high'], self.historical_data['low'], self.historical_data['close'])
        self.historical_data['adx'] = calculate_adx(self.historical_data['high'], self.historical_data['low'], self.historical_data['close'])

        self.last_data_update = now

        return self.historical_data

    def select_strategy_profile(self, df, model_confidence=None):
        """
        Select the appropriate strategy profile based on market conditions and ML confidence

        Args:
            df: DataFrame with market data
            model_confidence: Confidence score from ML model (0-1)

        Returns:
            dict: Selected strategy profile parameters
        """
        self.logger.info("Selecting strategy profile based on market conditions and ML confidence")

        # Default to moderate if no ML confidence provided
        if model_confidence is None:
            model_confidence = 0.5  # Default to moderate confidence
            self.logger.info("No ML confidence provided, using default value of 0.5")

        # Get market regime
        try:
            regime = self.regime_detector.detect_regime(df, self.timeframe)
            self.logger.info(f"Detected market regime: {regime}")
        except Exception as e:
            self.logger.error(f"Error detecting market regime: {e}")
            regime = 0  # Default regime

        # Calculate volatility percentile
        try:
            if 'atr' in df.columns and len(df) > 20:
                atr_history = df['atr'].dropna()
                if len(atr_history) > 0:
                    atr_percentile = int(pd.Series(atr_history).rank(pct=True).iloc[-1] * 100)
                    self.logger.info(f"Current volatility percentile: {atr_percentile}")
                else:
                    atr_percentile = 50
        except Exception as e:
            self.logger.error(f"Error calculating volatility percentile: {e}")
            atr_percentile = 50

        # Calculate win rate factor (0-1)
        win_rate_factor = 0.5  # Default
        if hasattr(self, 'trades') and len(self.trades) >= 5:
            recent_trades = self.trades[-min(len(self.trades), 20):]  # Last 20 trades or all if fewer
            wins = sum(1 for trade in recent_trades if trade.get('profit_pct', 0) > 0)
            win_rate = wins / len(recent_trades) if recent_trades else 0.5
            win_rate_factor = min(win_rate, 1.0)  # Cap at 1.0
            self.logger.info(f"Recent win rate: {win_rate:.2f}, factor: {win_rate_factor:.2f}")

        # Calculate current market strength (0-1)
        market_strength = 0.5  # Default
        try:
            if 'adx' in df.columns and not df['adx'].isnull().all():
                adx_value = df['adx'].iloc[-1]
                # Normalize ADX (0-100) to 0-1 scale
                market_strength = min(adx_value / 50.0, 1.0)
                self.logger.info(f"Market strength factor: {market_strength:.2f} (ADX: {adx_value:.2f})")
        except Exception as e:
            self.logger.error(f"Error calculating market strength: {e}")

        # Calculate volatility factor (0-1, higher means lower volatility is better for aggressive profiles)
        volatility_factor = 1.0 - (min(atr_percentile, 100) / 100.0)
        self.logger.info(f"Volatility factor: {volatility_factor:.2f} (ATR percentile: {atr_percentile})")

        # Calculate overall confidence score combining all factors
        # Weights: ML confidence (40%), win rate (25%), market strength (20%), volatility (15%)
        overall_score = (
            (model_confidence * 0.40) +
            (win_rate_factor * 0.25) +
            (market_strength * 0.20) +
            (volatility_factor * 0.15)
        )
        
        self.logger.info(f"Overall confidence score: {overall_score:.2f}")

        # Select profile based on overall score
        if overall_score >= 0.85:
            self.logger.info("Ultra high confidence score - using ultra_aggressive profile")
            self.active_profile = "ultra_aggressive"
            return self.strategy_profiles["ultra_aggressive"]
        elif overall_score >= 0.70:
            self.logger.info("Very high confidence score - using very_aggressive profile")
            self.active_profile = "very_aggressive"
            return self.strategy_profiles["very_aggressive"]
        elif overall_score >= 0.55:
            self.logger.info("High confidence score - using aggressive profile")
            self.active_profile = "aggressive"
            return self.strategy_profiles["aggressive"]
        elif overall_score >= 0.40:
            self.logger.info("Moderate confidence score - using moderate profile")
            self.active_profile = "moderate"
            return self.strategy_profiles["moderate"]
        else:
            self.logger.info("Low confidence score - using conservative profile")
            self.active_profile = "conservative"
            return self.strategy_profiles["conservative"]

    def adjust_parameters(self, base_params, current_regime):
        """Adjust strategy parameters based on the current market regime"""
        adjusted_params = base_params.copy()

        if current_regime == 1:  # Bullish
            # In bullish regime, be more aggressive
            adjusted_params['rsi_entry_threshold'] = max(0.1, base_params['rsi_entry_threshold'] * 0.5)  # Lower RSI threshold for more entries
            adjusted_params['stop_loss_pct'] = base_params['stop_loss_pct'] * 1.5  # Wider stop loss
            adjusted_params['take_profit_pct'] = base_params['take_profit_pct'] * 2.0  # Higher profit target
            adjusted_params['position_size_multiplier'] = 2.0  # Larger position size

        elif current_regime == -1:  # Bearish
            # In bearish regime, be more conservative
            adjusted_params['rsi_entry_threshold'] = min(30, base_params['rsi_entry_threshold'] * 2.0)  # Higher RSI threshold for fewer entries
            adjusted_params['stop_loss_pct'] = base_params['stop_loss_pct'] * 0.5  # Tighter stop loss
            adjusted_params['take_profit_pct'] = base_params['take_profit_pct'] * 0.5  # Lower profit target
            adjusted_params['position_size_multiplier'] = 0.5  # Smaller position size

        # For neutral regime, use base parameters

        return adjusted_params

class DynamicExitSystem:
    """Advanced exit system with dynamic trailing stops and take profit levels"""
    def __init__(self):
        self.active_exits = {}

    def setup_advanced_exit_system(self, symbol, position_size, entry_price, direction,
                                  stop_loss_pct=0.01, take_profit_levels=[0.025, 0.05, 0.15],
                                  trailing_stop_activation_pct=0.018, trailing_stop_distance_pct=0.008):
        """Setup an advanced exit system for a trade"""
        # Calculate stop loss and take profit levels
        stop_level = self._calculate_stop_level(entry_price, stop_loss_pct, direction)

        # Calculate multiple take profit levels
        take_profit_levels = [self._calculate_take_profit_level(entry_price, tp_pct, direction)
                             for tp_pct in take_profit_levels]

        # Calculate trailing stop activation level
        activation_price = self._calculate_activation_price(entry_price, trailing_stop_activation_pct, direction)

        self.active_exits[symbol] = {
            'entry_price': entry_price,
            'position_size': position_size,
            'direction': direction,
            'stop_loss': stop_level,
            'take_profit_levels': take_profit_levels,
            'take_profit_triggered': [False] * len(take_profit_levels),
        'trailing_stop_activation': activation_price,
        'trailing_stop_distance_pct': trailing_stop_distance_pct,
        'trailing_stop': None,  # Will be set once activated
        'partial_exits_done': 0
        }

        logger.info(f"Setup exit system for {symbol}: Entry={entry_price}, Stop={stop_level}, "
        f"TPs={take_profit_levels}, Activation={activation_price}")

        return self.active_exits[symbol]

    def _calculate_stop_level(self, entry_price, stop_pct, direction):
        """Calculate stop loss level based on entry price and direction"""
        if direction == "long":
            return entry_price * (1 - stop_pct)
        else:  # short
            return entry_price * (1 + stop_pct)

    def _calculate_take_profit_level(self, entry_price, profit_pct, direction):
        """Calculate take profit level based on entry price and direction"""
        if direction == "long":
            return entry_price * (1 + profit_pct)
        else:  # short
            return entry_price * (1 - profit_pct)

    def _calculate_activation_price(self, entry_price, activation_pct, direction):
        """Calculate trailing stop activation price"""
        if direction == "long":
            return entry_price * (1 + activation_pct)
        else:  # short
            return entry_price * (1 - activation_pct)

    def update_exit_system(self, symbol, current_price):
        """Update exit system based on current price and check for exit signals"""
        if symbol not in self.active_exits:
            return None
        
        exit_info = self.active_exits[symbol]
        direction = exit_info['direction']
        
        # Check stop loss
        if (direction == "long" and current_price <= exit_info['stop_loss']) or \
            (direction == "short" and current_price >= exit_info['stop_loss']):
            return self._execute_exit(symbol, current_price, exit_info['position_size'], "stop_loss")
                    
        # Check take profit levels for partial exits
        for i, (tp_level, triggered) in enumerate(zip(exit_info['take_profit_levels'], exit_info['take_profit_triggered'])):
            if not triggered:
                if (direction == "long" and current_price >= tp_level) or \
                   (direction == "short" and current_price <= tp_level):
                    # Mark this level as triggered
                    exit_info['take_profit_triggered'][i] = True

                    # Calculate the portion to exit at this level (progressive scaling)
                    exit_portion = 0.25 * (i + 1)  # Exit 25%, 50%, 75% at each level
                    exit_size = exit_info['position_size'] * exit_portion

                    # Update remaining position size
                    exit_info['position_size'] -= exit_size
                    exit_info['partial_exits_done'] += 1

                    if exit_info['position_size'] <= 0.001:  # Full exit
                        return self._execute_exit(symbol, current_price, exit_size, "take_profit_full")
                    else:
                        # Partial exit
                        return {
                            'symbol': symbol,
                            'price': current_price,
                            'size': exit_size,
                            'reason': f"take_profit_{i+1}",
                            'entry_price': exit_info['entry_price']
                        }
                
                    # Check and update trailing stop
                    if exit_info['trailing_stop'] is None:
                        # Check if price reached activation level
                        if (direction == "long" and current_price >= exit_info['trailing_stop_activation']) or \
                           (direction == "short" and current_price <= exit_info['trailing_stop_activation']):
                            # Activate trailing stop
                            exit_info['trailing_stop'] = self._calculate_trailing_stop_level(
                                current_price, exit_info['trailing_stop_distance_pct'], direction)
                            logger.info(f"Trailing stop activated for {symbol} at {exit_info['trailing_stop']}")
                    
                            # Check if price hit trailing stop
                            if (direction == "long" and current_price <= exit_info['trailing_stop']) or \
                               (direction == "short" and current_price >= exit_info['trailing_stop']):
                                return self._execute_exit(symbol, current_price, exit_info['position_size'], "trailing_stop")

    def _calculate_trailing_stop_level(self, current_price, distance_pct, direction):
        """Calculate trailing stop level based on current price"""
        if direction == "long":
            return current_price * (1 - distance_pct)
        else:  # short
            return current_price * (1 + distance_pct)

    def _execute_exit(self, symbol, price, size, reason):
        """Execute exit and remove from active exits"""
        exit_info = self.active_exits.pop(symbol, None)
        if not exit_info:
            return None

        logger.info(f"Exit signal for {symbol}: Price={price}, Size={size}, Reason={reason}")

        return {
            'symbol': symbol,
            'price': price,
            'size': size,
            'reason': reason,
            'entry_price': exit_info['entry_price']
        }

class MetaTraderLiveStrategy:
    def __init__(self, symbol="EURUSD", timeframe="M5", initial_deposit=50000,
                 rsi_entry_threshold=0.1, stop_loss_pct=0.05, take_profit_pct=0.2,
                 max_position_size=1.0, results_dir=RESULTS_DIR):
        # Initialize logger for this instance
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        if not self.logger.handlers:  # Avoid duplicate handlers
            self.logger.addHandler(handler)

        self.symbol = symbol
        self.timeframe = timeframe
        self.mt5_timeframe = self._get_mt5_timeframe(timeframe)
        self.initial_deposit = initial_deposit
        self.rsi_entry_threshold = rsi_entry_threshold
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self.max_position_size = max_position_size
        self.results_dir = results_dir
        self.mt5_initialized = False
        self.current_positions = {}
        self.trades = []
        self.historical_data = None
        self.last_data_update = None
        self.win_streak = 0
        self.loss_streak = 0
        self.strategy_profiles = {
            "conservative": {
                "long_rsi_threshold": 30,
                "short_rsi_threshold": 70,
                "stop_loss_pct": 0.02,
                "take_profit_pct": 0.03,
                "position_size_multiplier": 0.5,
                "max_concurrent_trades": 5
            },
            "moderate": {
                "long_rsi_threshold": 20,
                "short_rsi_threshold": 80,
                "stop_loss_pct": 0.05,
                "take_profit_pct": 0.10,
                "position_size_multiplier": 1.0,
                "max_concurrent_trades": 4
            },
            "aggressive": {
                "long_rsi_threshold": 10,
                "short_rsi_threshold": 90,
                "stop_loss_pct": 0.10,
                "take_profit_pct": 0.20,
                "position_size_multiplier": 2.0,
                "max_concurrent_trades": 3
            },
            "very_aggressive": {
                "long_rsi_threshold": 5,
                "short_rsi_threshold": 95,
                "stop_loss_pct": 0.15,
                "take_profit_pct": 0.40,
                "position_size_multiplier": 3.0,
                "max_concurrent_trades": 2
            },
            "ultra_aggressive": {
                "long_rsi_threshold": 0.1,
                "short_rsi_threshold": 99.9,
                "stop_loss_pct": 0.25,
                "take_profit_pct": 0.75,
                "position_size_multiplier": 5.0,
                "max_concurrent_trades": 1
            }
        }
        self.active_profile = "moderate"
        self.regime_detector = MarketRegimeDetector()
        self.trading_models = EnhancedTrading()

    def _get_mt5_timeframe(self, timeframe_str):
        """Convert string timeframe to MT5 timeframe constant"""
        timeframe_map = {
            "M1": mt5.TIMEFRAME_M1,
            "M5": mt5.TIMEFRAME_M5,
            "M15": mt5.TIMEFRAME_M15,
            "M30": mt5.TIMEFRAME_M30,
            "H1": mt5.TIMEFRAME_H1,
            "H4": mt5.TIMEFRAME_H4,
            "D1": mt5.TIMEFRAME_D1,
            "W1": mt5.TIMEFRAME_W1,
            "MN1": mt5.TIMEFRAME_MN1
        }
        return timeframe_map.get(timeframe_str, mt5.TIMEFRAME_M5)

    
    
    def initialize_mt5(self):
        """Initialize connection to MetaTrader 5"""
        self.logger.info("Initializing MetaTrader 5 connection...")

        if mt5 is None:
            self.logger.error("MetaTrader5 module not found. Running in simulation mode.")
            self.mt5_initialized = False
            return False

        try:
            # Attempt to initialize MT5
            if not mt5.initialize():
                error_code = mt5.last_error()
                self.logger.error(f"Failed to initialize MT5: Error code {error_code}")
                self.mt5_initialized = False
                return False

            # Verify connection by checking terminal info
            terminal_info = mt5.terminal_info()
            if terminal_info is None:
                self.logger.error("MT5 initialized but no terminal info available: Error code " + str(mt5.last_error()))
                mt5.shutdown()
                self.mt5_initialized = False
                return False

            # Check if connected to a broker
            account_info = mt5.account_info()
            if account_info is None:
                self.logger.error("Not connected to a broker: Error code " + str(mt5.last_error()))
                mt5.shutdown()
                self.mt5_initialized = False
                return False

            # Verify symbol availability
            symbol_info = mt5.symbol_info(self.symbol)
            if symbol_info is None:
                self.logger.error(f"Symbol {self.symbol} not found in market watch.")
                mt5.shutdown()
                self.mt5_initialized = False
                return False

            # Ensure symbol is enabled
            if not symbol_info.visible:
                if not mt5.symbol_select(self.symbol, True):
                    self.logger.error(f"Failed to enable symbol {self.symbol}.")
                    mt5.shutdown()
                    self.mt5_initialized = False
                    return False

            self.logger.info("MetaTrader 5 initialized successfully.")
            self.mt5_initialized = True
            return True

        except Exception as e:
            self.logger.error(f"Exception during MT5 initialization: {str(e)}")
            self.logger.error(traceback.format_exc())
            self.mt5_initialized = False
            return False

    def get_account_info(self):
        """Retrieve account information from MetaTrader 5"""
        self.logger.info("Retrieving account information...")

        if not self.mt5_initialized:
            self.logger.error("MT5 not initialized. Attempting to initialize...")
            if not self.initialize_mt5():
                self.logger.error("Failed to initialize MT5. Cannot retrieve account info.")
                return None

        try:
            account_info = mt5.account_info()
            if account_info is None:
                self.logger.error(f"Failed to get account info: {mt5.last_error()}")
                return None

            account_data = {
                'login': account_info.login,
                'balance': account_info.balance,
                'equity': account_info.equity,
                'margin': account_info.margin,
                'margin_free': account_info.margin_free,
                'currency': account_info.currency
            }
            self.logger.info(f"Account info retrieved: Balance={account_data['balance']}, Equity={account_data['equity']}")
            return account_data

        except Exception as e:
            self.logger.error(f"Error retrieving account info: {str(e)}")
            self.logger.error(traceback.format_exc())
            return None
    
    def fetch_historical_data(self, bars=500):
        """Fetch historical price data from MetaTrader 5"""
        self.logger.info(f"Fetching historical data for {self.symbol}, requesting {bars} bars...")

        if not self.mt5_initialized:
            self.logger.error("MT5 not initialized. Cannot fetch data.")
            return False

        # Get current time in UTC
        now = datetime.now()
        self.logger.info(f"Current time: {now}")

        # Fetch historical data
        self.logger.info(f"Requesting historical data from MT5 for {self.symbol} with timeframe {self.mt5_timeframe}...")
        try:
            self.logger.info("Calling mt5.copy_rates_from...")
            rates = mt5.copy_rates_from(self.symbol, self.mt5_timeframe, now, bars)

            if rates is None or len(rates) == 0:
                self.logger.error(f"Failed to fetch historical data: {mt5.last_error()}")
                return False

            self.logger.info(f"Successfully fetched {len(rates)} bars of historical data")

            # Convert to DataFrame
            self.logger.info("Converting data to DataFrame...")
            df = pd.DataFrame(rates)
            df['time'] = pd.to_datetime(df['time'], unit='s')
            df.set_index('time', inplace=True)

            self.logger.info("Calculating additional features...")

            # Calculate additional features
            self.logger.info("Calculating technical indicators...")
            try:
                # Calculate indicators
                self.logger.info("Calculating RSI...")
                df['rsi'] = calculate_rsi(df['close'])

                self.logger.info("Calculating MACD...")
                df['macd'], df['macd_signal'], df['macd_hist'] = calculate_macd(df['close'])

                self.logger.info("Calculating Bollinger Bands...")
                df['upper_bb'], df['lower_bb'] = calculate_bollinger_bands(df['close'])

                self.logger.info("Calculating ATR...")
                df['atr'] = calculate_atr(df['high'], df['low'], df['close'])

                self.logger.info("Calculating Stochastic Oscillator...")
                df['stoch_k'], df['stoch_d'] = calculate_stochastic(df['high'], df['low'], df['close'])

                self.logger.info("Calculating ADX...")
                df['adx'] = calculate_adx(df['high'], df['low'], df['close'])

                self.logger.info("All technical indicators calculated successfully")
            except Exception as e:
                self.logger.error(f"Error calculating technical indicators: {e}")
                self.logger.error(traceback.format_exc())
                # Continue even if indicators fail, as we can still use price data

            # Store the data
            self.historical_data = df
            self.last_data_update = now

            self.logger.info(f"Historical data updated successfully with {len(df)} rows")
            return df

        except Exception as e:
            self.logger.error(f"Exception during historical data fetch: {e}")
            self.logger.error(traceback.format_exc())
            return False

    def update_data(self):
        """Update historical data with latest prices"""
        self.logger.info("Updating market data...")

        if self.historical_data is None:
            self.logger.info("No historical data found, fetching initial data...")
            return self.fetch_historical_data()

        # Get current time
        now = datetime.now()

        # If last update was recent, skip
        if self.last_data_update and (now - self.last_data_update).seconds < 10:
            self.logger.info("Last update was recent, skipping update")
            return self.historical_data

        # Fetch latest bar
        self.logger.info(f"Fetching latest bar for {self.symbol}...")
        latest_bar = mt5.copy_rates_from(self.symbol, self.mt5_timeframe, now, 1)

        if latest_bar is None or len(latest_bar) == 0:
            self.logger.warning(f"Failed to fetch latest bar: {mt5.last_error()}")
            return self.historical_data

        self.logger.info(f"Latest bar fetched successfully: {latest_bar}")

        # Convert to DataFrame
        self.logger.info("Converting data to DataFrame...")
        latest_df = pd.DataFrame(latest_bar)
        latest_df['time'] = pd.to_datetime(latest_df['time'], unit='s')
        latest_df.set_index('time', inplace=True)

        # Check if we already have this bar
        if latest_df.index[0] in self.historical_data.index:
            self.logger.info(f"Updating existing bar at {latest_df.index[0]}")
            # Update the existing bar
            self.historical_data.loc[latest_df.index[0]] = latest_df.iloc[0]
        else:
            self.logger.info(f"Adding new bar at {latest_df.index[0]}")
            # Append new bar
            self.historical_data = pd.concat([self.historical_data, latest_df])

        # Remove oldest bar to maintain the same length
        if len(self.historical_data) > 500:
            self.historical_data = self.historical_data.iloc[-500:]

        # Recalculate indicators
        self.logger.info("Recalculating technical indicators...")
        self.historical_data['rsi'] = calculate_rsi(self.historical_data['close'])
        self.historical_data['macd'], self.historical_data['macd_signal'], self.historical_data['macd_hist'] = calculate_macd(self.historical_data['close'])
        self.historical_data['upper_bb'], self.historical_data['lower_bb'] = calculate_bollinger_bands(self.historical_data['close'])
        self.historical_data['atr'] = calculate_atr(self.historical_data['high'], self.historical_data['low'], self.historical_data['close'])
        self.historical_data['stoch_k'], self.historical_data['stoch_d'] = calculate_stochastic(self.historical_data['high'], self.historical_data['low'], self.historical_data['close'])
        self.historical_data['adx'] = calculate_adx(self.historical_data['high'], self.historical_data['low'], self.historical_data['close'])

        self.last_data_update = now

        return self.historical_data

    def select_strategy_profile(self, df, model_confidence=None):
        """
        Select the appropriate strategy profile based on market conditions and ML confidence

        Args:
            df: DataFrame with market data
            model_confidence: Confidence score from ML model (0-1)

        Returns:
            dict: Selected strategy profile parameters
        """
        self.logger.info("Selecting strategy profile based on market conditions and ML confidence")

        # Default to moderate if no ML confidence provided
        if model_confidence is None:
            model_confidence = 0.5  # Default to moderate confidence
            self.logger.info("No ML confidence provided, using default value of 0.5")

        # Get market regime
        try:
            regime = self.regime_detector.detect_regime(df, self.timeframe)
            self.logger.info(f"Detected market regime: {regime}")
        except Exception as e:
            self.logger.error(f"Error detecting market regime: {e}")
            regime = 0  # Default regime

        # Calculate volatility percentile
        try:
            if 'atr' in df.columns and len(df) > 20:
                atr_history = df['atr'].dropna()
                if len(atr_history) > 0:
                    atr_percentile = int(pd.Series(atr_history).rank(pct=True).iloc[-1] * 100)
                    self.logger.info(f"Current volatility percentile: {atr_percentile}")
                else:
                    atr_percentile = 50
        except Exception as e:
            self.logger.error(f"Error calculating volatility percentile: {e}")
            atr_percentile = 50

        # Calculate win rate factor (0-1)
        win_rate_factor = 0.5  # Default
        if hasattr(self, 'trades') and len(self.trades) >= 5:
            recent_trades = self.trades[-min(len(self.trades), 20):]  # Last 20 trades or all if fewer
            wins = sum(1 for trade in recent_trades if trade.get('profit_pct', 0) > 0)
            win_rate = wins / len(recent_trades) if recent_trades else 0.5
            win_rate_factor = min(win_rate, 1.0)  # Cap at 1.0
            self.logger.info(f"Recent win rate: {win_rate:.2f}, factor: {win_rate_factor:.2f}")

        # Calculate current market strength (0-1)
        market_strength = 0.5  # Default
        try:
            if 'adx' in df.columns and not df['adx'].isnull().all():
                adx_value = df['adx'].iloc[-1]
                # Normalize ADX (0-100) to 0-1 scale
                market_strength = min(adx_value / 50.0, 1.0)
                self.logger.info(f"Market strength factor: {market_strength:.2f} (ADX: {adx_value:.2f})")
        except Exception as e:
            self.logger.error(f"Error calculating market strength: {e}")

        # Calculate volatility factor (0-1, higher means lower volatility is better for aggressive profiles)
        volatility_factor = 1.0 - (min(atr_percentile, 100) / 100.0)
        self.logger.info(f"Volatility factor: {volatility_factor:.2f} (ATR percentile: {atr_percentile})")

        # Calculate overall confidence score combining all factors
        # Weights: ML confidence (40%), win rate (25%), market strength (20%), volatility (15%)
        overall_score = (
            (model_confidence * 0.40) +
            (win_rate_factor * 0.25) +
            (market_strength * 0.20) +
            (volatility_factor * 0.15)
        )
        
        self.logger.info(f"Overall confidence score: {overall_score:.2f}")

        # Select profile based on overall score
        if overall_score >= 0.85:
            self.logger.info("Ultra high confidence score - using ultra_aggressive profile")
            self.active_profile = "ultra_aggressive"
            return self.strategy_profiles["ultra_aggressive"]
        elif overall_score >= 0.70:
            self.logger.info("Very high confidence score - using very_aggressive profile")
            self.active_profile = "very_aggressive"
            return self.strategy_profiles["very_aggressive"]
        elif overall_score >= 0.55:
            self.logger.info("High confidence score - using aggressive profile")
            self.active_profile = "aggressive"
            return self.strategy_profiles["aggressive"]
        elif overall_score >= 0.40:
            self.logger.info("Moderate confidence score - using moderate profile")
            self.active_profile = "moderate"
            return self.strategy_profiles["moderate"]
        else:
            self.logger.info("Low confidence score - using conservative profile")
            self.active_profile = "conservative"
            return self.strategy_profiles["conservative"]

    def adjust_parameters(self, base_params, current_regime):
        """Adjust strategy parameters based on the current market regime"""
        adjusted_params = base_params.copy()

        if current_regime == 1:  # Bullish
            # In bullish regime, be more aggressive
            adjusted_params['rsi_entry_threshold'] = max(0.1, base_params['rsi_entry_threshold'] * 0.5)  # Lower RSI threshold for more entries
            adjusted_params['stop_loss_pct'] = base_params['stop_loss_pct'] * 1.5  # Wider stop loss
            adjusted_params['take_profit_pct'] = base_params['take_profit_pct'] * 2.0  # Higher profit target
            adjusted_params['position_size_multiplier'] = 2.0  # Larger position size

        elif current_regime == -1:  # Bearish
            # In bearish regime, be more conservative
            adjusted_params['rsi_entry_threshold'] = min(30, base_params['rsi_entry_threshold'] * 2.0)  # Higher RSI threshold for fewer entries
            adjusted_params['stop_loss_pct'] = base_params['stop_loss_pct'] * 0.5  # Tighter stop loss
            adjusted_params['take_profit_pct'] = base_params['take_profit_pct'] * 0.5  # Lower profit target
            adjusted_params['position_size_multiplier'] = 0.5  # Smaller position size

        # For neutral regime, use base parameters

        return adjusted_params

class DynamicExitSystem:
    """Advanced exit system with dynamic trailing stops and take profit levels"""
    def __init__(self):
        self.active_exits = {}

    def setup_advanced_exit_system(self, symbol, position_size, entry_price, direction,
                                  stop_loss_pct=0.01, take_profit_levels=[0.025, 0.05, 0.15],
                                  trailing_stop_activation_pct=0.018, trailing_stop_distance_pct=0.008):
        """Setup an advanced exit system for a trade"""
        # Calculate stop loss and take profit levels
        stop_level = self._calculate_stop_level(entry_price, stop_loss_pct, direction)

        # Calculate multiple take profit levels
        take_profit_levels = [self._calculate_take_profit_level(entry_price, tp_pct, direction)
                             for tp_pct in take_profit_levels]

        # Calculate trailing stop activation level
        activation_price = self._calculate_activation_price(entry_price, trailing_stop_activation_pct, direction)

        self.active_exits[symbol] = {
            'entry_price': entry_price,
            'position_size': position_size,
            'direction': direction,
            'stop_loss': stop_level,
            'take_profit_levels': take_profit_levels,
            'take_profit_triggered': [False] * len(take_profit_levels),
        'trailing_stop_activation': activation_price,
        'trailing_stop_distance_pct': trailing_stop_distance_pct,
        'trailing_stop': None,  # Will be set once activated
        'partial_exits_done': 0
        }

        logger.info(f"Setup exit system for {symbol}: Entry={entry_price}, Stop={stop_level}, "
        f"TPs={take_profit_levels}, Activation={activation_price}")

        return self.active_exits[symbol]

    def _calculate_stop_level(self, entry_price, stop_pct, direction):
        """Calculate stop loss level based on entry price and direction"""
        if direction == "long":
            return entry_price * (1 - stop_pct)
        else:  # short
            return entry_price * (1 + stop_pct)

    def _calculate_take_profit_level(self, entry_price, profit_pct, direction):
        """Calculate take profit level based on entry price and direction"""
        if direction == "long":
            return entry_price * (1 + profit_pct)
        else:  # short
            return entry_price * (1 - profit_pct)

    def _calculate_activation_price(self, entry_price, activation_pct, direction):
        """Calculate trailing stop activation price"""
        if direction == "long":
            return entry_price * (1 + activation_pct)
        else:  # short
            return entry_price * (1 - activation_pct)

    def update_exit_system(self, symbol, current_price):
        """Update exit system based on current price and check for exit signals"""
        if symbol not in self.active_exits:
            return None
        
        exit_info = self.active_exits[symbol]
        direction = exit_info['direction']
        
        # Check stop loss
        if (direction == "long" and current_price <= exit_info['stop_loss']) or \
            (direction == "short" and current_price >= exit_info['stop_loss']):
            return self._execute_exit(symbol, current_price, exit_info['position_size'], "stop_loss")
                    
        # Check take profit levels for partial exits
        for i, (tp_level, triggered) in enumerate(zip(exit_info['take_profit_levels'], exit_info['take_profit_triggered'])):
            if not triggered:
                if (direction == "long" and current_price >= tp_level) or \
                   (direction == "short" and current_price <= tp_level):
                    # Mark this level as triggered
                    exit_info['take_profit_triggered'][i] = True

                    # Calculate the portion to exit at this level (progressive scaling)
                    exit_portion = 0.25 * (i + 1)  # Exit 25%, 50%, 75% at each level
                    exit_size = exit_info['position_size'] * exit_portion

                    # Update remaining position size
                    exit_info['position_size'] -= exit_size
                    exit_info['partial_exits_done'] += 1

                    if exit_info['position_size'] <= 0.001:  # Full exit
                        return self._execute_exit(symbol, current_price, exit_size, "take_profit_full")
                    else:
                        # Partial exit
                        return {
                            'symbol': symbol,
                            'price': current_price,
                            'size': exit_size,
                            'reason': f"take_profit_{i+1}",
                            'entry_price': exit_info['entry_price']
                        }
                
                    # Check and update trailing stop
                    if exit_info['trailing_stop'] is None:
                        # Check if price reached activation level
                        if (direction == "long" and current_price >= exit_info['trailing_stop_activation']) or \
                           (direction == "short" and current_price <= exit_info['trailing_stop_activation']):
                            # Activate trailing stop
                            exit_info['trailing_stop'] = self._calculate_trailing_stop_level(
                                current_price, exit_info['trailing_stop_distance_pct'], direction)
                            logger.info(f"Trailing stop activated for {symbol} at {exit_info['trailing_stop']}")
                    
                            # Check if price hit trailing stop
                            if (direction == "long" and current_price <= exit_info['trailing_stop']) or \
                               (direction == "short" and current_price >= exit_info['trailing_stop']):
                                return self._execute_exit(symbol, current_price, exit_info['position_size'], "trailing_stop")

    def _calculate_trailing_stop_level(self, current_price, distance_pct, direction):
        """Calculate trailing stop level based on current price"""
        if direction == "long":
            return current_price * (1 - distance_pct)
        else:  # short
            return current_price * (1 + distance_pct)

    def _execute_exit(self, symbol, price, size, reason):
        """Execute exit and remove from active exits"""
        exit_info = self.active_exits.pop(symbol, None)
        if not exit_info:
            return None

        logger.info(f"Exit signal for {symbol}: Price={price}, Size={size}, Reason={reason}")

        return {
            'symbol': symbol,
            'price': price,
            'size': size,
            'reason': reason,
            'entry_price': exit_info['entry_price']
        }

    def check_entry_signal(self, df, direction_check=None):
        """
        Check for entry signals using multiple strategy profiles simultaneously.
        Returns tuple of (signal, params, ultra_aggressive_flag)
        signal: "long", "short", or None
        params: dict of parameters for the trade
        ultra_aggressive_flag: boolean indicating if this is an ultra-aggressive trade
        """
        if df is None or len(df) < 100:
            self.logger.warning("Not enough data for signal detection")
            return None, {}, False
        
        # Count current active trades by profile
        profile_trade_counts = {profile: 0 for profile in self.strategy_profiles.keys()}
        for position in self.current_positions.values():
            if position['profile'] in profile_trade_counts:
                profile_trade_counts[position['profile']] += 1
        
        # Calculate indicators
        rsi = talib.RSI(df['close'].values, timeperiod=14)
        current_rsi = rsi[-1]
        
        macd, macd_signal, macd_hist = talib.MACD(
            df['close'].values, fastperiod=12, slowperiod=26, signalperiod=9
        )
        
        # Get ML predictions if available
        ml_prediction = 0.5  # Neutral default
        if hasattr(self, 'trading_models') and self.trading_models is not None:
            try:
                ml_prediction = self.trading_models.predict(df)
                self.logger.info(f"ML prediction: {ml_prediction:.4f}")
            except Exception as e:
                self.logger.error(f"Error getting ML prediction: {e}")
        
        # Detect current market regime
        market_regime = self.detect_market_regime(df)
        self.logger.info(f"Current market regime: {market_regime}")
        
        # Detect current market volatility
        volatility = self.calculate_volatility(df)
        self.logger.info(f"Current volatility: {volatility:.4f}")
        
        # Store available signals for different profiles
        valid_signals = []
        
        # Check for signals across all profiles
        for profile_name, profile in self.strategy_profiles.items():
            # Skip if we've reached the max concurrent trades for this profile
            if profile_trade_counts[profile_name] >= profile['max_concurrent_trades']:
                continue
                
            # Check for long entry signals
            long_signal = False
            short_signal = False
            
            # Extract profile parameters
            long_rsi_threshold = profile['long_rsi_threshold']
            short_rsi_threshold = profile['short_rsi_threshold']
            stop_loss_pct = profile['stop_loss_pct']
            take_profit_pct = profile['take_profit_pct']
            position_size_multiplier = profile['position_size_multiplier']
            
            # Determine signal strength based on how far RSI is from thresholds
            long_strength = 0
            if current_rsi <= long_rsi_threshold:
                # Calculate strength as a percentage of distance from threshold to 0
                long_strength = (long_rsi_threshold - current_rsi) / long_rsi_threshold
                long_signal = True
                
            short_strength = 0
            if current_rsi >= short_rsi_threshold:
                # Calculate strength as a percentage of distance from threshold to 100
                short_strength = (current_rsi - short_rsi_threshold) / (100 - short_rsi_threshold)
                short_signal = True
            
            # MACD trend-following signals for additional confirmation
            macd_long_signal = macd_hist[-1] > 0 and macd_hist[-2] <= 0  # Bullish crossover
            macd_short_signal = macd_hist[-1] < 0 and macd_hist[-2] >= 0  # Bearish crossover
            
            # Adjust signal strength based on ML prediction if available
            if long_signal and ml_prediction is not None:
                long_strength *= (ml_prediction * 1.5)  # Boost strength if ML agrees
                
            if short_signal and ml_prediction is not None:
                short_strength *= ((1 - ml_prediction) * 1.5)  # Boost strength if ML agrees
                
            # Check direction constraints if specified
            if direction_check == "long" and not long_signal:
                continue
            if direction_check == "short" and not short_signal:
                continue
                
            # Create parameters for the potential trade
            if long_signal:
                params = {
                    'profile': profile_name,
                    'signal_strength': long_strength,
                    'stop_loss_pct': stop_loss_pct,
                    'take_profit_pct': take_profit_pct,
                    'position_size_multiplier': position_size_multiplier,
                    'market_regime': market_regime,
                    'volatility': volatility,
                    'ml_prediction': ml_prediction
                }
                # Check if this is an ultra-aggressive entry
                is_ultra = profile_name == "ultra_aggressive"
                valid_signals.append(("long", params, is_ultra, long_strength))
                
            if short_signal:
                params = {
                    'profile': profile_name,
                    'signal_strength': short_strength,
                    'stop_loss_pct': stop_loss_pct,
                    'take_profit_pct': take_profit_pct,
                    'position_size_multiplier': position_size_multiplier,
                    'market_regime': market_regime,
                    'volatility': volatility,
                    'ml_prediction': ml_prediction
                }
                # Check if this is an ultra-aggressive entry
                is_ultra = profile_name == "ultra_aggressive"
                valid_signals.append(("short", params, is_ultra, short_strength))
        
        # If we have valid signals, select the strongest one
        if valid_signals:
            # Sort by signal strength (highest first)
            valid_signals.sort(key=lambda x: x[3], reverse=True)
            signal, params, is_ultra, _ = valid_signals[0]
            
            # Log the signal details
            self.logger.info(f"Selected {signal.upper()} signal with {params['profile']} profile. Strength: {params['signal_strength']:.4f}")
            
            # Return the selected signal
            return signal, params, is_ultra
        
        # No valid signals found
        return None, {}, False

    def calculate_position_size(self, df, direction, params, ultra_aggressive_entry=False):
        """Calculate position size based on multiple factors"""
        self.logger.info(f"Calculating position size for {direction} position")
        
        try:
            # Get account balance
            balance = self.get_account_balance()
            self.logger.info(f"Current balance: {balance}")
            
            # Base position size as percentage of balance (e.g., 2%)
            base_size = balance * 0.02
            
            # Get position size multiplier from params
            multiplier = params.get('position_size_multiplier', 1.0)
            
            # Apply win streak multiplier if enabled
            if self.win_streak > 0 and hasattr(self, 'use_win_streak_sizing') and self.use_win_streak_sizing:
                # Increase position size by 5% for each win in streak, cap at 5x
                win_streak_multiplier = min(1 + (self.win_streak * 0.05), 5.0)
                self.logger.info(f"Win streak multiplier: {win_streak_multiplier} (streak: {self.win_streak})")
                multiplier *= win_streak_multiplier
            
            # Apply trend strength multiplier if available
            if hasattr(self, 'trend_strength') and self.trend_strength is not None:
                # Scale between 1.0 and 5.0 based on trend strength
                trend_multiplier = 1.0 + (self.trend_strength * 4.0)
                self.logger.info(f"Trend strength multiplier: {trend_multiplier} (strength: {self.trend_strength})")
                multiplier *= trend_multiplier
            
            # Apply volatility adjustment if available
            atr_percentile = 50  # Default value
            if 'atr' in df.columns:
                # Get current ATR as percentage of price
                current_price = df['close'].iloc[-1]
                atr_value = df['atr'].iloc[-1]
                atr_pct = atr_value / current_price
                
                # Calculate ATR percentile if we have history
                try:
                    # Create a history of ATR values if not already available
                    if not hasattr(self, 'atr_history'):
                        self.atr_history = []
                    
                    # Add current ATR to history
                    self.atr_history.append(atr_value)
                    
                    # Keep history to a reasonable size
                    if len(self.atr_history) > 100:
                        self.atr_history = self.atr_history[-100:]
                    
                    # Calculate percentile
                    if len(self.atr_history) > 0:
                        atr_percentile = int(pd.Series(self.atr_history).rank(pct=True).iloc[-1] * 100)
                        self.logger.info(f"Current volatility percentile: {atr_percentile}")
                except Exception as e:
                    self.logger.error(f"Error calculating volatility percentile: {e}")
                    atr_percentile = 50
                
                # If volatility is high, reduce position size
                if atr_pct > 0.02 or atr_percentile > 80:  # If ATR > 2% of price or in top 20% of history
                    vol_multiplier = 0.5  # Reduce by 50%
                    self.logger.info(f"High volatility detected, reducing position size by 50% (ATR: {atr_pct:.2%}, Percentile: {atr_percentile})")
                    multiplier *= vol_multiplier
            
            # Apply model confidence multiplier
            model_confidence = params.get('model_confidence', 0.5)
            if model_confidence > 0.5:
                # Scale between 1.0 and 5.0 based on model confidence
                confidence_multiplier = 1.0 + ((model_confidence - 0.5) * 8.0)  # 0.5->1.0, 0.75->3.0, 1.0->5.0
                self.logger.info(f"Model confidence multiplier: {confidence_multiplier:.2f} (confidence: {model_confidence:.2f})")
                multiplier *= confidence_multiplier
            
            # Apply market regime multiplier
            market_regime = params.get('market_regime', 0)
            if market_regime == 1:  # Bullish regime
                regime_multiplier = 2.0
                self.logger.info(f"Bullish market regime detected, increasing position size by 2x")
                multiplier *= regime_multiplier
            elif market_regime == -1:  # Bearish regime
                regime_multiplier = 0.5
                self.logger.info(f"Bearish market regime detected, reducing position size by 50%")
                multiplier *= regime_multiplier
            
            # Ultra-aggressive entry gets an additional boost
            if ultra_aggressive_entry:
                # For ultra-aggressive profile, use the extraordinary position sizing from the backtest
                if params.get('profile') == 'ultra_aggressive':
                    ultra_multiplier = 10.0  # Scaled down from 5000x for live testing
                    self.logger.info(f"Ultra-maximized entry detected, applying extraordinary multiplier: {ultra_multiplier}")
                else:
                    ultra_multiplier = 2.0
                    self.logger.info(f"Ultra-aggressive entry detected, applying additional multiplier: {ultra_multiplier}")
                multiplier *= ultra_multiplier
            
            # Calculate final position size
            position_size = base_size * multiplier
            
            # Cap maximum position size at 10.0 for safety in live trading
            position_size = min(position_size, 10.0)
            
            self.logger.info(f"Final position size: {position_size} (base: {base_size}, multiplier: {multiplier})")
            
            return position_size
        
        except Exception as e:
            self.logger.error(f"Error calculating position size: {e}")
            traceback.print_exc()
            return 0.1  # Return minimal position size on error

    def place_trade(self, order_type, price, position_size, stop_loss=None, take_profit=None, ultra_aggressive=False, params=None):
        """Place a trade with the specified parameters"""
        self.logger.info(f"Placing {order_type} trade for {self.symbol} at {price} with size {position_size}")

        if not self.mt5_initialized:
            success = self.initialize_mt5()
            if not success:
                self.logger.error("Failed to initialize MT5, cannot place trade")
                return False

        # Get symbol information
        symbol_info = mt5.symbol_info(self.symbol)
        if symbol_info is None:
            self.logger.error(f"Failed to get symbol info for {self.symbol}")
            return False

        # Get minimum stop level in points
        min_stop_level = symbol_info.trade_stops_level

        # Get point value
        point_value = symbol_info.point

        self.logger.info(f"Symbol info: min_stop_level={min_stop_level} points, point_value={point_value}")

        # Determine MT5 order type
        if order_type == "long":
            mt5_order_type = mt5.ORDER_TYPE_BUY
        elif order_type == "short":
            mt5_order_type = mt5.ORDER_TYPE_SELL
        else:
            self.logger.error(f"Invalid order type: {order_type}")
            return False

        # Calculate stop loss and take profit levels if not provided
        if stop_loss is None or take_profit is None:
            # Get current profile's stop loss and take profit percentages
            profile = self.strategy_profiles.get(self.active_profile, self.strategy_profiles["moderate"])
            stop_loss_pct = profile["stop_loss_pct"]
            take_profit_pct = profile["take_profit_pct"]
            
            # If params are provided, use those values instead
            if params:
                stop_loss_pct = params.get('stop_loss_pct', stop_loss_pct)
                take_profit_pct = params.get('take_profit_pct', take_profit_pct)
        
        # For ultra aggressive entries, use more aggressive parameters
        if ultra_aggressive:
            # Use the extraordinary parameters from the successful backtest
            stop_loss_pct = self.strategy_profiles["ultra_aggressive"]["stop_loss_pct"]
            take_profit_pct = self.strategy_profiles["ultra_aggressive"]["take_profit_pct"]
            self.logger.info(f"Using ULTRA-MAXIMIZED parameters: SL=-1000%, TP=10000%")

        # Calculate stop loss and take profit prices
        if order_type == "long":
            if stop_loss is None:
                stop_loss = price * (1 - stop_loss_pct)
            if take_profit is None:
                take_profit = price * (1 + take_profit_pct)
        else:  # short
            if stop_loss is None:
                stop_loss = price * (1 + stop_loss_pct)
            if take_profit is None:
                take_profit = price * (1 - take_profit_pct)

        # Calculate minimum price difference for stop levels
        min_stop_price_diff = min_stop_level * point_value
        
        # Ensure stop loss and take profit are within broker's minimum distance requirements
        if order_type == "long":
            # For long positions, stop loss must be below entry price, take profit above
            min_stop_loss = price - min_stop_price_diff
            min_take_profit = price + min_stop_price_diff

            # Adjust if needed
            if stop_loss > min_stop_loss:
                self.logger.warning(f"Stop loss too close to entry price. Adjusting from {stop_loss} to {min_stop_loss}")
                stop_loss = min_stop_loss

            if take_profit < min_take_profit:
                self.logger.warning(f"Take profit too close to entry price. Adjusting from {take_profit} to {min_take_profit}")
                take_profit = min_take_profit

        else:  # short
            # For short positions, stop loss must be above entry price, take profit below
            min_stop_loss = price + min_stop_price_diff
            min_take_profit = price - min_stop_price_diff

            # Adjust if needed
            if stop_loss < min_stop_loss:
                self.logger.warning(f"Stop loss too close to entry price. Adjusting from {stop_loss} to {min_stop_loss}")
                stop_loss = min_stop_loss

            if take_profit > min_take_profit:
                self.logger.warning(f"Take profit too close to entry price. Adjusting from {take_profit} to {min_take_profit}")
                take_profit = min_take_profit

        # Round values to appropriate number of digits
        digits = symbol_info.digits
        price = round(price, digits)
        stop_loss = round(stop_loss, digits)
        take_profit = round(take_profit, digits)

        # Prepare the trade request
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": self.symbol,
            "volume": float(position_size),
            "type": mt5_order_type,
            "price": price,
            "sl": stop_loss,
            "tp": take_profit,
            "deviation": 10,  # Allow price deviation in points
            "magic": 234000,  # Magic number for identification
            "comment": f"Ultra-Maximized {order_type} trade",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }

        # Log the trade request
        self.logger.info(f"Trade request: {request}")

        # Send the trade request with retries
        max_retries = 3
        retry_count = 0
        success = False
        order_id = None
        stop_loss_set = True
        take_profit_set = True

        while retry_count < max_retries and not success:
            self.logger.info(f"Sending trade request (attempt {retry_count+1}): {request}")
            result = mt5.order_send(request)

            # Check if the trade was successful
            if result is None:
                error_code = mt5.last_error()
                self.logger.error(f"Order send returned None. Error code: {error_code}")

                # Try to reconnect if it seems like a connection issue
                if error_code in [10018, 10019, 10021]:  # Connection issues
                    self.logger.info("Attempting to reconnect to broker...")
                    mt5.shutdown()
                    time.sleep(2)
                    init_result = mt5.initialize()
                    if not init_result:
                        self.logger.error(f"Failed to reconnect to MT5: Error code {mt5.last_error()}")

                # Wait before retrying
                time.sleep(1)
            elif result.retcode != mt5.TRADE_RETCODE_DONE:
                self.logger.error(f"Trade request failed with error code: {result.retcode}")
                self.logger.error(f"Error message: {result.comment}")

                if result.retcode in [10009, 10013, 10018]:  # Position not found or invalid
                    self.logger.warning("Position not found. Trying market order instead.")

                    # Create a market order to close the position
                    market_request = {
                        "action": mt5.TRADE_ACTION_DEAL,
                        "symbol": self.symbol,
                        "volume": position_size,
                        "type": mt5_order_type,
                        "price": mt5.symbol_info_tick(self.symbol).ask if mt5_order_type == mt5.ORDER_TYPE_BUY else mt5.symbol_info_tick(self.symbol).bid,
                        "deviation": 10,
                        "magic": 123456,
                        "comment": f"Market order: {order_type}",
                        "type_time": mt5.ORDER_TIME_GTC,
                        "type_filling": mt5.ORDER_FILLING_IOC,
                        "sl": stop_loss,
                        "tp": take_profit
                    }

                    self.logger.info(f"Sending market order request: {market_request}")
                    market_result = mt5.order_send(market_request)

                    if market_result is not None and market_result.retcode == mt5.TRADE_RETCODE_DONE:
                        self.logger.info(f"Market order placed successfully! Order ID: {market_result.order}")
                        result = market_result
                        success = True
                        order_id = result.order
                        break

                # Handle invalid stops error
                if result.retcode == 10016:  # Invalid stops
                    self.logger.error("Invalid stops error. Trying with adjusted stops...")

                    # Get current price
                    tick = mt5.symbol_info_tick(self.symbol)
                    if tick is None:
                        self.logger.error(f"Failed to get tick data for {self.symbol}")
                        time.sleep(1)
                        continue

                    current_bid = tick.bid
                    current_ask = tick.ask

                    # Try with more conservative stops
                    if order_type == "long":
                        stop_loss = current_bid - (min_stop_price_diff * 2)  # Double the minimum distance
                        take_profit = current_ask + (min_stop_price_diff * 2)  # Double the minimum distance
                    else:  # short
                        stop_loss = current_ask + (min_stop_price_diff * 2)  # Double the minimum distance
                        take_profit = current_bid - (min_stop_price_diff * 2)  # Double the minimum distance

                    # Round values
                    stop_loss = round(stop_loss, digits)
                    take_profit = round(take_profit, digits)

                    # Update request
                    request["sl"] = stop_loss
                    request["tp"] = take_profit

                    self.logger.info(f"Retrying with adjusted stops: SL={stop_loss}, TP={take_profit}")
                else:
                    # Wait before retrying
                    time.sleep(1)
            else:
                # Trade was successful
                self.logger.info(f"Order placed successfully! Order ID: {result.order}")
                success = True
                order_id = result.order
                break

            retry_count += 1
            time.sleep(1)

        if not success:
            self.logger.error(f"Failed to place trade after {max_retries} attempts")

            # Try one last time without stops
            self.logger.warning("Trying one final attempt without stops...")
            request_no_stops = request.copy()
            request_no_stops.pop("sl", None)
            request_no_stops.pop("tp", None)

            try:
                result = mt5.order_send(request_no_stops)
                if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
                    retry_error = mt5.last_error() if result is None else result.retcode
                    self.logger.error(f"Retry order also failed with error: {retry_error}")
                    return False
                else:
                    self.logger.info(f"Order placed without stops! Order ID: {result.order}")
                    order_id = result.order
                    # We'll need to set the stops separately
                    stop_loss_set = False
                    take_profit_set = False
            except Exception as e:
                self.logger.error(f"Final attempt failed: {e}")
                return False

        # Verify the order and check if stops were set correctly
        time.sleep(0.5)  # Wait for the order to be processed
        
        # Get the position from MT5
        positions = mt5.positions_get(symbol=self.symbol)
        if positions is None:
            self.logger.warning(f"Could not verify position - no positions found for {self.symbol}")
        else:
            # Find our position by order ID
            position_found = False
            for pos in positions:
                if pos.ticket == order_id or abs(pos.price_open - price) < 0.0001:
                    position_found = True
                    self.logger.info(f"Verified position: Entry={pos.price_open}, SL={pos.sl}, TP={pos.tp}")

                    # Check if stops were set correctly
                    if not stop_loss_set or abs(pos.sl - stop_loss) > 0.0001:
                        self.logger.warning(f"Stop loss was not set correctly. Expected: {stop_loss}, Actual: {pos.sl}")

                        # Try to set the stop loss separately
                        if not self.update_position_stops(position_id, new_sl=stop_loss, new_tp=None):
                            self.logger.error("Failed to set stop loss separately")

                    if not take_profit_set or abs(pos.tp - take_profit) > 0.0001:
                        self.logger.warning(f"Take profit was not set correctly. Expected: {take_profit}, Actual: {pos.tp}")

                        # Try to set the take profit separately
                        if not self.update_position_stops(position_id, new_sl=None, new_tp=take_profit):
                            self.logger.error("Failed to set take profit separately")

                    break

        if not position_found:
            self.logger.warning("Could not find the newly opened position in MT5")

        # Generate a unique position ID
        position_id = str(uuid.uuid4())

        # Set up trailing stop parameters
        trailing_stop_activation_pct = 0.018  # Default 1.8%
        trailing_stop_distance_pct = 0.008    # Default 0.8%

        # For ultra aggressive trades, use more aggressive trailing parameters
        if ultra_aggressive:
            trailing_stop_activation_pct = 0.01  # Activate sooner at 1%
            trailing_stop_distance_pct = 0.005   # Tighter trailing at 0.5%

        # Record the position with trailing stop parameters
        position_info = {
            'position_id': position_id,
            'order_id': order_id,
            'symbol': self.symbol,
            'direction': order_type,
            'entry_price': price,
            'volume': position_size,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'entry_time': datetime.now(),
            'profile': self.active_profile,
            'ultra_aggressive': ultra_aggressive,
            'trailing_stop_activation_pct': trailing_stop_activation_pct,
            'trailing_stop_distance_pct': trailing_stop_distance_pct,
            'trailing_stop_active': False
        }

        # Add to current positions
        self.current_positions[position_id] = position_info

        # Save state to persist the position information
        self.save_state()

        return True

    def close_specific_position(self, position_id, reason="manual", exit_price=None):
        """
        Attempt to close a specific position by its ID
        
        Args:
            position_id: Unique identifier for the position
            reason: Reason for closing the position
            exit_price: Optional price at which the position was closed
        
        Returns:
            bool: True if position was closed successfully, False otherwise
        """
        self.logger.info(f"Attempting to close position {position_id} due to {reason}")
        
        if not self.mt5_initialized:
            success = self.initialize_mt5()
            if not success:
                self.logger.error("Failed to initialize MT5, cannot close position")
                return False
        
        # Check if position exists in our records
        if position_id not in self.current_positions:
            self.logger.error(f"Position {position_id} not found in current positions")
            return False
        
        position_info = self.current_positions[position_id]
        symbol = position_info['symbol']
        direction = position_info['direction']
        volume = position_info['volume']
        entry_price = position_info['entry_price']
        entry_time = position_info['entry_time']
        order_id = position_info.get('order_id')
        
        # Find the position in MT5
        mt5_position = None
        positions = mt5.positions_get(symbol=symbol)
        
        if positions is None:
            self.logger.warning(f"No positions found for {symbol} in MT5")
        else:
            for pos in positions:
                if pos.ticket == order_id or (abs(pos.price_open - entry_price) < 0.0001 and pos.volume == volume):
                    mt5_position = pos
                    break
        
        if mt5_position is None:
            self.logger.warning(f"Position {position_id} not found in MT5, removing from local records")
            del self.current_positions[position_id]
            self.save_state()
            return False
        
        # Determine the close order type
        if direction == "long":
            close_type = mt5.ORDER_TYPE_SELL
        else:  # short
            close_type = mt5.ORDER_TYPE_BUY
        
        # Get current price
        current_price = None
        if exit_price is not None:
            current_price = exit_price
        else:
            # Get current bid/ask
            symbol_info = mt5.symbol_info(symbol)
            if symbol_info is None:
                self.logger.error(f"Failed to get symbol info for {symbol}")
                return False
            
            if direction == "long":
                current_price = symbol_info.bid  # Sell at bid when closing a long
            else:
                current_price = symbol_info.ask  # Buy at ask when closing a short
        
        # Prepare the request
        request = {
            'action': mt5.TRADE_ACTION_DEAL,
            'position': mt5_position.ticket,
            'symbol': symbol,
            'volume': volume,
            'type': close_type,
            'price': current_price,
            'deviation': 20,
            'magic': 123456,
            'comment': f"Close {reason}",
            'type_time': mt5.ORDER_TIME_GTC,
            'type_filling': mt5.ORDER_FILLING_IOC,
        }
        
        # Send the request
        self.logger.info(f"Sending close request for position {position_id}")
        result = mt5.order_send(request)
        
        if result is None:
            error = mt5.last_error()
            self.logger.error(f"Failed to close position {position_id}: {error}")
            return False
        
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            self.logger.error(f"Failed to close position {position_id}: {result.retcode}")
            return False
        
        self.logger.info(f"Successfully closed position {position_id} at {current_price}")
        
        # Record the trade
        exit_time = datetime.now()
        trade_duration = (exit_time - entry_time).total_seconds() / 60  # Duration in minutes
        
        # Calculate profit/loss
        if direction == "long":
            profit_pct = (current_price - entry_price) / entry_price
        else:  # short
            profit_pct = (entry_price - current_price) / entry_price
        
        profit_amount = profit_pct * volume * entry_price
        
        # Update win/loss streak
        if profit_pct > 0:
            self.win_streak += 1
            self.loss_streak = 0
        else:
            self.win_streak = 0
            self.loss_streak += 1
        
        # Record the trade details
        self.record_trade(
            entry_price=entry_price,
            exit_price=current_price,
            position_size=volume,
            direction=direction,
            entry_time=entry_time,
            exit_time=exit_time,
            trade_features={
                'profit_pct': profit_pct,
                'profit_amount': profit_amount,
                'duration_minutes': trade_duration,
                'exit_reason': reason,
                'profile': position_info.get('profile', 'unknown'),
                'ultra_aggressive': position_info.get('ultra_aggressive', False)
            }
        )
        
        # Remove from current positions
        del self.current_positions[position_id]
        
        # Save state to persist the position information
        self.save_state()
        
        return True

    def modify_position_sl_tp(self, position_id, new_sl=None, new_tp=None):
        """
        Modify the stop loss and/or take profit levels for a specific position
        
        Args:
            position_id: ID of the position to update
            new_sl: New stop loss level
            new_tp: New take profit level
            
        Returns:
            bool: True if modification was successful, False otherwise
        """
        self.logger.info(f"Modifying SL/TP for position {position_id}: SL={new_sl}, TP={new_tp}")
        
        if not self.mt5_initialized:
            success = self.initialize_mt5()
            if not success:
                self.logger.error("Failed to initialize MT5, cannot modify position")
                return False
        
        # Check if position exists in our records
        if position_id not in self.current_positions:
            self.logger.error(f"Position {position_id} not found in current positions")
            return False
        
        position_info = self.current_positions[position_id]
        symbol = position_info['symbol']
        order_id = position_info.get('order_id')
        
        # Find the position in MT5
        mt5_position = None
        positions = mt5.positions_get(symbol=symbol)
        
        if positions is None:
            self.logger.warning(f"No positions found for {symbol} in MT5")
            return False
        
        for pos in positions:
            if pos.ticket == order_id:
                mt5_position = pos
                break
        
        if mt5_position is None:
            self.logger.warning(f"Position {position_id} not found in MT5")
            return False
        
        # Prepare the request
        request = {
            "action": mt5.TRADE_ACTION_SLTP,
            "position": mt5_position.ticket,
            "symbol": symbol,
        }
        
        # Only include parameters that are being modified
        if new_sl is not None:
            request["sl"] = new_sl
        else:
            request["sl"] = mt5_position.sl
            
        if new_tp is not None:
            request["tp"] = new_tp
        else:
            request["tp"] = mt5_position.tp
        
        # Send the request
        self.logger.info(f"Sending modify request for position {position_id}")
        result = mt5.order_send(request)
        
        if result is None:
            error = mt5.last_error()
            self.logger.error(f"Failed to modify position {position_id}: {error}")
            return False
        
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            self.logger.error(f"Failed to modify position {position_id}: {result.retcode}")
            return False
        
        self.logger.info(f"Successfully modified position {position_id}: SL={new_sl}, TP={new_tp}")
        
        # Update our position record
        if new_sl is not None:
            self.current_positions[position_id]['stop_loss'] = new_sl
        if new_tp is not None:
            self.current_positions[position_id]['take_profit'] = new_tp
        
        # Save state to persist the position information
        self.save_state()
        
        return True

    def update_position_stops(self, position_id, new_sl=None, new_tp=None):
        """
        Update the stop loss and take profit levels for a position
        
        Args:
            position_id: ID of the position to update
            new_sl: New stop loss level
            new_tp: New take profit level
            
        Returns:
            bool: True if update was successful, False otherwise
        """
        return self.modify_position_sl_tp(position_id, new_sl, new_tp)

    def save_state(self):
        """
        Save the current state of the strategy to a file
        This includes current positions, trades, and other important state information
        """
        self.logger.info("Saving strategy state to file")
        
        try:
            # Create state directory if it doesn't exist
            state_dir = os.path.join(self.results_dir, "state")
            os.makedirs(state_dir, exist_ok=True)
            
            # Prepare state data
            state = {
                'timestamp': datetime.now().isoformat(),
                'symbol': self.symbol,
                'timeframe': self.timeframe,
                'win_streak': self.win_streak,
                'loss_streak': self.loss_streak,
                'active_profile': self.active_profile,
                'positions': {}
            }
            
            # Convert position data to serializable format
            for pos_id, pos_info in self.current_positions.items():
                serializable_pos = pos_info.copy()
                # Convert datetime to string
                if 'entry_time' in serializable_pos and isinstance(serializable_pos['entry_time'], datetime):
                    serializable_pos['entry_time'] = serializable_pos['entry_time'].isoformat()
                state['positions'][pos_id] = serializable_pos
            
            # Save to file
            state_file = os.path.join(state_dir, f"{self.symbol}_{self.timeframe}_state.json")
            with open(state_file, 'w') as f:
                json.dump(state, f, indent=4)
            
            self.logger.info(f"State saved to {state_file}")
            
            # Also save trades to a separate file for analysis
            trades_file = os.path.join(self.results_dir, f"{self.symbol}_{self.timeframe}_trades.json")
            serializable_trades = []
            for trade in self.trades:
                serializable_trade = trade.copy()
                # Convert datetime objects to strings
                for key in ['entry_time', 'exit_time']:
                    if key in serializable_trade and isinstance(serializable_trade[key], datetime):
                        serializable_trade[key] = serializable_trade[key].isoformat()
                serializable_trades.append(serializable_trade)
            
            with open(trades_file, 'w') as f:
                json.dump(serializable_trades, f, indent=4)
            
            self.logger.info(f"Trades saved to {trades_file}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error saving state: {e}")
            traceback.print_exc()
            return False

    def load_state(self):
        """
        Load the strategy state from a file
        This includes current positions, trades, and other important state information
        
        Returns:
            bool: True if state was loaded successfully, False otherwise
        """
        self.logger.info("Loading strategy state from file")
        
        try:
            # Check if state file exists
            state_dir = os.path.join(self.results_dir, "state")
            state_file = os.path.join(state_dir, f"{self.symbol}_{self.timeframe}_state.json")
            
            if not os.path.exists(state_file):
                self.logger.info(f"No state file found at {state_file}")
                return False
            
            # Load state from file
            with open(state_file, 'r') as f:
                state = json.load(f)
            
            # Update strategy state
            self.win_streak = state.get('win_streak', 0)
            self.loss_streak = state.get('loss_streak', 0)
            self.active_profile = state.get('active_profile', 'moderate')
            
            # Convert positions from serializable format
            self.current_positions = {}
            for pos_id, pos_info in state.get('positions', {}).items():
                # Convert string timestamps back to datetime
                if 'entry_time' in pos_info and isinstance(pos_info['entry_time'], str):
                    pos_info['entry_time'] = datetime.fromisoformat(pos_info['entry_time'])
                self.current_positions[pos_id] = pos_info
            
            self.logger.info(f"Loaded {len(self.current_positions)} positions from state file")
            
            # Also load trades if available
            trades_file = os.path.join(self.results_dir, f"{self.symbol}_{self.timeframe}_trades.json")
            if os.path.exists(trades_file):
                with open(trades_file, 'r') as f:
                    serializable_trades = json.load(f)
                
                self.trades = []
                for trade in serializable_trades:
                    # Convert string timestamps back to datetime
                    for key in ['entry_time', 'exit_time']:
                        if key in trade and isinstance(trade[key], str):
                            trade[key] = datetime.fromisoformat(trade[key])
                    self.trades.append(trade)
                
                self.logger.info(f"Loaded {len(self.trades)} trades from trades file")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error loading state: {e}")
            traceback.print_exc()
            return False

    def sync_mt5_positions(self):
        """
        Synchronize positions with MT5 to ensure consistency
        This checks MT5 positions and updates our local records accordingly
        
        Returns:
            bool: True if synchronization was successful, False otherwise
        """
        self.logger.info("Synchronizing positions with MT5")
        
        if not self.mt5_initialized:
            success = self.initialize_mt5()
            if not success:
                self.logger.error("Failed to initialize MT5, cannot sync positions")
                return False
        
        # Get all positions from MT5
        positions = mt5.positions_get(symbol=self.symbol)
        
        if positions is None:
            self.logger.info(f"No positions found in MT5 for {self.symbol}")
            # If no positions in MT5, clear our local records
            if self.current_positions:
                self.logger.warning("Clearing local position records as no positions found in MT5")
                self.current_positions = {}
                self.save_state()
            return True
        
        # Create a set of MT5 position tickets
        mt5_tickets = set()
        for pos in positions:
            mt5_tickets.add(pos.ticket)
            
            # Check if this position is in our records
            found = False
            for pos_id, pos_info in self.current_positions.items():
                if pos_info.get('order_id') == pos.ticket:
                    found = True
                    # Update position info if needed
                    if pos_info['stop_loss'] != pos.sl or pos_info['take_profit'] != pos.tp:
                        self.logger.info(f"Updating position {pos_id} with current MT5 stops: SL={pos.sl}, TP={pos.tp}")
                        pos_info['stop_loss'] = pos.sl
                        pos_info['take_profit'] = pos.tp
                    break
            
            # If position not found in our records, add it
            if not found:
                self.logger.warning(f"Found position in MT5 not in local records: {pos.ticket}. Adding to records.")
                
                # Determine direction
                if pos.type == mt5.POSITION_TYPE_BUY:
                    direction = "long"
                else:
                    direction = "short"
                
                # Generate a unique position ID
                position_id = str(uuid.uuid4())
                
                # Add to our records
                self.current_positions[position_id] = {
                    'position_id': position_id,
                    'order_id': pos.ticket,
                    'symbol': pos.symbol,
                    'direction': direction,
                    'entry_price': pos.price_open,
                    'volume': pos.volume,
                    'stop_loss': pos.sl,
                    'take_profit': pos.tp,
                    'entry_time': datetime.fromtimestamp(pos.time),
                    'profile': self.active_profile,
                    'ultra_aggressive': False,  # Default
                    'trailing_stop_activation_pct': 0.018,  # Default
                    'trailing_stop_distance_pct': 0.008,  # Default
                    'trailing_stop_active': False
                }
        
        # Check for positions in our records that no longer exist in MT5
        positions_to_remove = []
        for pos_id, pos_info in self.current_positions.items():
            order_id = pos_info.get('order_id')
            if order_id is not None and order_id not in mt5_tickets:
                self.logger.warning(f"Position {pos_id} (order {order_id}) not found in MT5. Removing from records.")
                positions_to_remove.append(pos_id)
        
        # Remove positions that no longer exist
        for pos_id in positions_to_remove:
            del self.current_positions[pos_id]
        
        # Save state if any changes were made
        if positions_to_remove or len(mt5_tickets) != len(self.current_positions):
            self.save_state()
        
        self.logger.info(f"Sync complete. {len(self.current_positions)} positions in local records.")
        return True

def setup_logging(log_level=logging.INFO, log_file=None):
    """Set up logging configuration for the application"""
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    
    # Create console handler with formatting
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # Add file handler if specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
    
    return root_logger

if __name__ == "__main__":
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='MetaTrader Live Strategy')
    parser.add_argument('mode', choices=['live', 'backtest'], help='Trading mode')
    parser.add_argument('--symbol', default='BTCUSDT', help='Trading symbol')
    parser.add_argument('--timeframe', default='M5', help='Trading timeframe')
    parser.add_argument('--max_position_size', type=float, default=10.0, help='Maximum position size')
    parser.add_argument('--log_file', help='Log file path')
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging(log_level=logging.INFO, log_file=args.log_file)
    logger.info(f"Starting MetaTrader Live Strategy in {args.mode} mode for {args.symbol}")
    
    try:
        # Initialize and run strategy
        strategy = MetaTraderLiveStrategy(
            symbol=args.symbol,
            timeframe=args.timeframe,
            max_position_size=args.max_position_size,
            rsi_entry_threshold=0.1,  # Ultra-aggressive threshold from backtest
            stop_loss_pct=5.0,        # Wide stop loss (500%)
            take_profit_pct=50.0      # Massive profit target (5000%)
        )
        
        # Initialize MT5 connection
        if not strategy.initialize_mt5():
            logger.error("Failed to initialize MT5 connection. Exiting.")
            sys.exit(1)
        
        logger.info("MT5 connection initialized successfully")
        
        # Load previous state if available
        strategy.load_state()
        
        # Sync with MT5 positions
        strategy.sync_mt5_positions()
        
        # Fetch initial historical data
        df = strategy.fetch_historical_data(bars=500)
        if df is None or len(df) < 100:
            logger.error("Failed to fetch sufficient historical data. Exiting.")
            sys.exit(1)
        
        logger.info(f"Fetched {len(df)} historical bars for {args.symbol}")
        
        # Main trading loop
        if args.mode == 'live':
            logger.info("Starting live trading loop")
            try:
                while True:
                    # Update data with latest prices
                    df = strategy.update_data()
                    if df is None or not isinstance(df, pd.DataFrame):
                        logger.warning("Failed to get valid DataFrame, retrying in 60 seconds")
                        time.sleep(60)
                        continue
                    
                    # Check for entry signals
                    try:
                        entry_signal, params, ultra_aggressive_entry = strategy.check_entry_signal(df)
                        
                        if entry_signal == "long":
                            logger.info("LONG signal detected")
                            # Get current price
                            current_price = df['close'].iloc[-1]
                            # Calculate position size
                            position_size = strategy.calculate_position_size(df, 'long', params, ultra_aggressive_entry=ultra_aggressive_entry)
                            # Place trade with ultra-aggressive parameters
                            strategy.place_trade('buy', current_price, position_size, 
                                               stop_loss=current_price * (1 - 5.0),  # 500% stop loss
                                               take_profit=current_price * (1 + 50.0),  # 5000% profit target
                                               ultra_aggressive=True)
                        
                        elif entry_signal == "short":
                            logger.info("SHORT signal detected")
                            # Get current price
                            current_price = df['close'].iloc[-1]
                            # Calculate position size
                            position_size = strategy.calculate_position_size(df, 'short', params, ultra_aggressive_entry=ultra_aggressive_entry)
                            # Place trade with ultra-aggressive parameters
                            strategy.place_trade('sell', current_price, position_size,
                                               stop_loss=current_price * (1 + 5.0),  # 500% stop loss
                                               take_profit=current_price * (1 - 50.0),  # 5000% profit target
                                               ultra_aggressive=True)
                        
                        else:
                            logger.info("No entry signals detected")
                    except Exception as e:
                        logger.exception(f"Error in signal detection or trade placement: {e}")
                    
                    # Sleep for a while before checking again
                    time.sleep(60)  # Check every minute
                    
            except KeyboardInterrupt:
                logger.info("Trading loop interrupted by user")
            except Exception as e:
                logger.exception(f"Error in trading loop: {e}")
            finally:
                # Save final state
                strategy.save_state()
                logger.info("Trading loop stopped, final state saved")
        
        else:  # Backtest mode
            logger.info("Backtest mode not implemented in this script")
    
    except Exception as e:
        logger.exception(f"Unhandled exception: {e}")

