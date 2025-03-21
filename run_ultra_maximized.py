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
from scipy import stats
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
    """
    Enhanced machine learning model for trading decisions with improved feature engineering
    and prediction quality.
    """
    def __init__(self):
        self.model = RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=42,
            class_weight='balanced'  # Address class imbalance
        )
        self.scaler = StandardScaler()
        self.feature_names = None
        self.importance = None
        self.trained = False  # Add trained attribute to track if model has been trained
        
    def prepare_data(self, data):
        """Prepare data with enhanced feature engineering"""
        # Create enhanced feature set
        features = []
        targets = []
        
        for i in range(50, len(data)):
            # Current close price normalized to recent range
            recent_high = data['high'][i-20:i].max()
            recent_low = data['low'][i-20:i].min()
            price_position = (data['close'][i] - recent_low) / (recent_high - recent_low) if recent_high > recent_low else 0.5
            
            # Enhanced momentum features
            rsi = data['rsi'][i]
            rsi_diff = data['rsi'][i] - data['rsi'][i-5]
            
            # Enhanced trend features
            trend_10 = (data['close'][i] / data['close'][i-10] - 1) * 100
            trend_20 = (data['close'][i] / data['close'][i-20] - 1) * 100
            trend_50 = (data['close'][i] / data['close'][i-50] - 1) * 100
            
            # Volatility features
            volatility_5 = data['close'][i-5:i].pct_change().std() * 100
            volatility_10 = data['close'][i-10:i].pct_change().std() * 100
            volatility_20 = data['close'][i-20:i].pct_change().std() * 100
            
            # Volume features (using synthetic volume)
            volume_change = (data['volume'][i] / data['volume'][i-5] - 1) * 100 if data['volume'][i-5] > 0 else 0
            
            # Price pattern features
            upper_wick = (data['high'][i] - max(data['open'][i], data['close'][i])) / (data['high'][i] - data['low'][i]) if (data['high'][i] - data['low'][i]) > 0 else 0
            lower_wick = (min(data['open'][i], data['close'][i]) - data['low'][i]) / (data['high'][i] - data['low'][i]) if (data['high'][i] - data['low'][i]) > 0 else 0
            
            # Candle size and direction
            candle_size = abs(data['close'][i] - data['open'][i]) / (data['high'][i] - data['low'][i]) if (data['high'][i] - data['low'][i]) > 0 else 0
            is_bullish = 1 if data['close'][i] > data['open'][i] else 0
            
            # Support/resistance proximity
            levels = find_support_resistance(data[:i+1], lookback=50)
            proximity_to_support = min([abs(data['close'][i] / level - 1) * 100 for level in levels['support']], default=100)
            proximity_to_resistance = min([abs(data['close'][i] / level - 1) * 100 for level in levels['resistance']], default=100)
            
            # Create feature vector with feature names
            feature_row = {
                'price_position': price_position,
                'rsi': rsi,
                'rsi_diff': rsi_diff,
                'trend_10': trend_10,
                'trend_20': trend_20,
                'trend_50': trend_50,
                'volatility_5': volatility_5,
                'volatility_10': volatility_10,
                'volatility_20': volatility_20,
                'volume_change': volume_change,
                'upper_wick': upper_wick,
                'lower_wick': lower_wick,
                'candle_size': candle_size,
                'is_bullish': is_bullish,
                'proximity_to_support': proximity_to_support,
                'proximity_to_resistance': proximity_to_resistance
            }
            
            # Add feature row
            features.append(list(feature_row.values()))
            
            # Create target: 1 if price goes up in next 3 bars, 0 otherwise
            future_return = (data['close'][i+3] / data['close'][i] - 1) * 100 if i+3 < len(data) else 0
            target = 1 if future_return > 0.2 else 0  # 0.2% threshold for significant move
            targets.append(target)
            
            # Store feature names for the first iteration
            if i == 50:
                self.feature_names = list(feature_row.keys())
        
        return np.array(features), np.array(targets)
    
    def train(self, data):
        """Train the model with enhanced data preparation and cross-validation"""
        features, targets = self.prepare_data(data)
        
        # Scale features
        self.scaler.fit(features)
        features_scaled = self.scaler.transform(features)
        
        # Split data for validation
        from sklearn.model_selection import train_test_split
        X_train, X_val, y_train, y_val = train_test_split(features_scaled, targets, test_size=0.3, random_state=42)
        
        # Train model
        self.model.fit(X_train, y_train)
        
        # Calculate feature importance
        self.importance = {
            'feature': {i: name for i, name in enumerate(self.feature_names)},
            'importance': {i: imp for i, imp in enumerate(self.model.feature_importances_)}
        }
        
        # Get top 5 important features
        importance_sorted = sorted(range(len(self.model.feature_importances_)), 
                                  key=lambda i: self.model.feature_importances_[i], 
                                  reverse=True)[:5]
        
        top_features = {
            'feature': {i: self.feature_names[idx] for i, idx in enumerate(importance_sorted)},
            'importance': {i: self.model.feature_importances_[idx] for i, idx in enumerate(importance_sorted)}
        }
        
        # Evaluate model performance
        val_accuracy = self.model.score(X_val, y_val)
        logger.info(f"Model validation accuracy: {val_accuracy:.2f}")
        
        self.trained = True  # Set trained flag to True after successful training
        
        return top_features

    def predict_proba(self, features):
        """Predict probability of price increase with enhanced confidence"""
        if self.feature_names is None:
            raise ValueError("Model has not been trained yet")
        
        # Ensure feature names match
        features_df = pd.DataFrame([features])
        
        # Scale features
        features_scaled = self.scaler.transform(features_df)
        
        # Predict probability of price increase
        return self.model.predict_proba(features_scaled)

# Enhanced Trading Models (component 1) with XGBoost, LightGBM and RandomForest ensemble
class EnhancedTrading:
    """Advanced ML models for trading predictions"""
    def __init__(self, use_ml=True):
        # Initialize logger for this instance
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        if not self.logger.handlers:  # Avoid duplicate handlers
            self.logger.addHandler(handler)
            
        self.use_ml = use_ml
        self.ml_available = False
        
        try:
            import xgboost as xgb
            import lightgbm as lgb
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.tree import DecisionTreeClassifier
            from sklearn.preprocessing import RobustScaler
            from sklearn.model_selection import train_test_split
            self.ml_available = True
            self.train_test_split = train_test_split
            self.xgb = xgb
            self.lgb = lgb
        except ImportError:
            warnings.warn("Machine learning libraries not installed. Using basic signal generation.")
            
        if self.ml_available:
            self.xgb_model = xgb.XGBClassifier(
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
            self.lgbm_model = lgb.LGBMClassifier(
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
        else:
            self.xgb_model = None
            self.lgbm_model = None
            self.rf_model = None
            self.dt_model = None

        self.model_weights = {
            "rf": 0.30,    # Reduced RF weight
            "xgb": 0.40,   # Increased XGB weight
            "lgbm": 0.25,  # Increased LGBM weight
            "dt": 0.05     # Minimal DT weight
        }

        self.scaler = RobustScaler() if self.ml_available else None
        self.models_trained = False
        self.trained = False  # Flag to track if model has been trained

    def generate_features(self, df):
        """Generate advanced features for model training"""
        features = pd.DataFrame(index=df.index)
        
        # Price-based features
        features['close'] = df['close']
        features['open'] = df['open']
        features['high'] = df['high']
        features['low'] = df['low']
        features['volume'] = df['volume'] if 'volume' in df.columns else 0
        features['price_change'] = df['close'].pct_change()
        features['range'] = (df['high'] - df['low']) / df['close']
        
        # Moving averages
        for window in [5, 10, 20, 50, 100]:
            features[f'sma_{window}'] = df['close'].rolling(window=window).mean()
            features[f'ema_{window}'] = df['close'].ewm(span=window, adjust=False).mean()
            features[f'volume_sma_{window}'] = df['volume'].rolling(window=window).mean() if 'volume' in df.columns else 0
        
        # Price distance from moving averages (normalized)
        for window in [5, 10, 20, 50, 100]:
            features[f'dist_from_sma_{window}'] = (df['close'] - features[f'sma_{window}']) / features[f'sma_{window}']
            features[f'dist_from_ema_{window}'] = (df['close'] - features[f'ema_{window}']) / features[f'ema_{window}']
        
        # Trend strength indicators
        for window in [10, 20, 50]:
            price_data = df['close'].values
            x = np.array(range(window))
            for i in range(window, len(df) + 1):
                if i < window:
                    features.loc[features.index[i-1], f'trend_strength_{window}'] = 0
                else:
                    prices = price_data[i-window:i]
                    slope, _, r_value, _, _ = linregress(x, prices)
                    features.loc[features.index[i-1], f'trend_strength_{window}'] = slope * r_value**2
                    
        # Momentum indicators
        features['momentum_1d'] = df['close'].pct_change(1)
        features['momentum_2d'] = df['close'].pct_change(2)
        features['momentum_5d'] = df['close'].pct_change(5)
        features['momentum_10d'] = df['close'].pct_change(10)
        
        # Volatility indicators
        features['volatility_5d'] = df['close'].pct_change().rolling(window=5).std()
        features['volatility_10d'] = df['close'].pct_change().rolling(window=10).std()
        features['volatility_20d'] = df['close'].pct_change().rolling(window=20).std()
        
        # Price patterns
        features['higher_high'] = (df['high'] > df['high'].shift(1)) & (df['high'].shift(1) > df['high'].shift(2))
        features['lower_low'] = (df['low'] < df['low'].shift(1)) & (df['low'].shift(1) < df['low'].shift(2))
        features['inside_day'] = (df['high'] < df['high'].shift(1)) & (df['low'] > df['low'].shift(1))
        features['outside_day'] = (df['high'] > df['high'].shift(1)) & (df['low'] < df['low'].shift(1))
        
        # RSI and related
        if 'rsi' in df.columns:
            features['rsi'] = df['rsi']
            features['rsi_ma_5'] = df['rsi'].rolling(window=5).mean()
            features['rsi_ma_10'] = df['rsi'].rolling(window=10).mean()
            features['rsi_slope'] = df['rsi'].diff()
        
        # MACD related
        if all(x in df.columns for x in ['macd', 'macd_signal', 'macd_hist']):
            features['macd'] = df['macd']
            features['macd_signal'] = df['macd_signal']
            features['macd_hist'] = df['macd_hist']
            features['macd_hist_slope'] = df['macd_hist'].diff()
        
        # Convert boolean to int
        for col in features.select_dtypes(include=['bool']).columns:
            features[col] = features[col].astype(int)
        
        # Fill NaN values
        features = features.fillna(0)
        
        return features

    def train_models(self, X_train, y_train):
        """Train all models with cross-validation"""
        if not self.ml_available:
            self.logger.warning("ML libraries not available. Skipping model training.")
            return False
            
        try:
            self.logger.info("Training ML models...")
            
            # Train XGBoost
            self.xgb_model.fit(X_train, y_train)
            
            # Train LightGBM
            self.lgbm_model.fit(X_train, y_train)
            
            # Train Random Forest
            self.rf_model.fit(X_train, y_train)
            
            # Train Decision Tree
            self.dt_model.fit(X_train, y_train)
            
            self.models_trained = True
            self.trained = True
            self.logger.info("ML models trained successfully")
            return True
        except Exception as e:
            self.logger.error(f"Error training models: {str(e)}")
            self.logger.error(traceback.format_exc())
            return False

    def _preprocess_data(self, X, y=None):
        """Preprocess training data"""
        if not self.ml_available:
            return None
            
        # Handle categorical features if any
        X_numeric = X.select_dtypes(include=['number'])
        
        # Scale the data
        X_scaled = self.scaler.fit_transform(X_numeric) if y is not None else self.scaler.transform(X_numeric)
        
        return pd.DataFrame(X_scaled, index=X.index, columns=X_numeric.columns)

    def train(self, df, lookback=20, target_column='close'):
        """Train models on historical data"""
        if not self.ml_available:
            self.logger.warning("ML libraries not available. Skipping model training.")
            return False
            
        try:
            self.logger.info("Preparing training data...")
            features = self.generate_features(df)
            
            # Create target variable (1 if price goes up in next period, 0 otherwise)
            future_returns = df[target_column].shift(-1) > df[target_column]
            y = future_returns.iloc[lookback:-1].astype(int)  # Remove last row as we don't know future
            
            # Use only complete rows of features
            X = features.iloc[lookback:-1]
            
            # Split into training and testing
            X_train, X_test, y_train, y_test = self.train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)
            
            # Preprocess data
            X_train_processed = self._preprocess_data(X_train, y_train)
            
            # Train the models
            success = self.train_models(X_train_processed, y_train)
            
            if success:
                # Update model weights based on performance
                X_test_processed = self._preprocess_data(X_test)
                self._update_model_weights(X_test_processed, y_test)
                
            return success
        except Exception as e:
            self.logger.error(f"Error in training: {str(e)}")
            self.logger.error(traceback.format_exc())
            return False

    def _update_model_weights(self, X_test, y_test):
        """Update model weights based on performance"""
        if not self.ml_available or X_test is None:
            return
            
        try:
            # Get predictions from each model
            rf_preds = self.rf_model.predict_proba(X_test)[:, 1]
            xgb_preds = self.xgb_model.predict_proba(X_test)[:, 1]
            lgbm_preds = self.lgbm_model.predict_proba(X_test)[:, 1]
            dt_preds = self.dt_model.predict_proba(X_test)[:, 1]
            
            # Calculate performance metrics (using log loss)
            from sklearn.metrics import log_loss
            
            rf_loss = log_loss(y_test, rf_preds)
            xgb_loss = log_loss(y_test, xgb_preds)
            lgbm_loss = log_loss(y_test, lgbm_preds)
            dt_loss = log_loss(y_test, dt_preds)
            
            # Inverse (lower loss is better)
            total_inverse = (1/rf_loss) + (1/xgb_loss) + (1/lgbm_loss) + (1/dt_loss)
            
            # Update weights (normalize to sum to 1)
            self.model_weights = {
                "rf": (1/rf_loss) / total_inverse,
                "xgb": (1/xgb_loss) / total_inverse,
                "lgbm": (1/lgbm_loss) / total_inverse,
                "dt": (1/dt_loss) / total_inverse
            }
            
            self.logger.info(f"Updated model weights: {self.model_weights}")
            
        except Exception as e:
            self.logger.error(f"Error updating model weights: {str(e)}")
            # Keep default weights
            pass

    def predict(self, features):
        """Make prediction for the next period"""
        if not self.ml_available or not self.models_trained:
            # Return neutral signal if ML not available
            return 0.5
            
        try:
            # Ensure features is a DataFrame
            if isinstance(features, list):
                # Convert list to DataFrame
                if len(features) > 0 and isinstance(features[0], dict):
                    # List of dictionaries
                    features_df = pd.DataFrame(features)
                else:
                    # Just a list of values, assume it's a single row
                    if hasattr(self, 'feature_names') and len(self.feature_names) == len(features):
                        features_df = pd.DataFrame([features], columns=self.feature_names)
                    else:
                        # Can't properly convert, return neutral
                        return 0.5
            elif isinstance(features, pd.Series):
                # Convert Series to DataFrame
                features_df = features.to_frame().T
            elif not isinstance(features, pd.DataFrame):
                # Not a supported type
                return 0.5
            else:
                features_df = features
                
            # Preprocess features
            features_processed = self._preprocess_data(features_df)
            if features_processed is None:
                return 0.5
                
            # Make predictions with each model
            predictions = {}
            if hasattr(self, 'rf_model') and self.rf_model is not None:
                predictions['rf'] = self.rf_model.predict(features_processed)[0]
            if hasattr(self, 'xgb_model') and self.xgb_model is not None:
                predictions['xgb'] = self.xgb_model.predict(features_processed)[0]
            if hasattr(self, 'lgb_model') and self.lgb_model is not None:
                predictions['lgbm'] = self.lgb_model.predict(features_processed)[0]
            if hasattr(self, 'dt_model') and self.dt_model is not None:
                predictions['dt'] = self.dt_model.predict(features_processed)[0]
                
            # Weighted ensemble prediction
            if predictions:
                weighted_pred = sum(predictions[model] * self.model_weights.get(model, 0.25) 
                                   for model in predictions)
                return weighted_pred
            else:
                return 0.5
                
        except Exception as e:
            self.logger.error(f"Error in prediction: {str(e)}")
            return 0.5

    def predict_proba(self, features):
        """Return probability of price increase"""
        if not self.ml_available or not self.models_trained:
            # Return neutral probabilities if ML not available
            return np.array([[0.5, 0.5]])
            
        prob_up = self.predict(features)
        return np.array([[1 - prob_up, prob_up]])
        
    def get_feature_importance(self):
        """Get feature importance across models"""
        if not self.ml_available or not self.models_trained:
            return {}
            
        importance_dict = {}
        
        # XGBoost feature importance
        xgb_importance = self.xgb_model.feature_importances_
        
        # LightGBM feature importance
        lgbm_importance = self.lgbm_model.feature_importances_
        
        # Random Forest feature importance
        rf_importance = self.rf_model.feature_importances_
        
        # Decision Tree feature importance
        dt_importance = self.dt_model.feature_importances_
        
        # Weighted average of importances
        feature_names = self.xgb_model.feature_names_in_
        for i, feature in enumerate(feature_names):
            importance_dict[feature] = (
                xgb_importance[i] * self.model_weights["xgb"] +
                lgbm_importance[i] * self.model_weights["lgbm"] +
                rf_importance[i] * self.model_weights["rf"] +
                dt_importance[i] * self.model_weights["dt"]
            )
        
        return importance_dict

# Advanced feature engineering functions
def calculate_hurst_exponent(time_series, max_lag=20):
    """Calculate Hurst exponent using R/S analysis"""
    time_series = np.array(time_series)
    lags = range(2, min(max_lag, len(time_series) // 4))
    if not lags:  # Handle short series
        return 0.5
        
    # Calculate the array of the variances of the lagged differences
    tau = [np.std(np.subtract(time_series[lag:], time_series[:-lag])) for lag in lags]
    
    # Use a linear fit to estimate the Hurst Exponent
    poly = np.polyfit(np.log(lags), np.log(tau), 1)
    
    # Return the Hurst exponent
    return poly[0]

def fractal_dimension(series, min_box_size=2, max_box_size=None):
    """Calculate the fractal dimension of a time series"""
    series = np.array(series)
    n = len(series)
    
    if max_box_size is None:
        max_box_size = n // 4
    
    if n <= min_box_size or min_box_size >= max_box_size:
        return 1.0  # Default for short series
    
    # Normalize the series
    series = (series - np.min(series)) / (np.max(series) - np.min(series))
    
    # Calculate box sizes
    box_sizes = np.logspace(np.log10(min_box_size), np.log10(max_box_size), num=10).astype(int)
    box_sizes = np.unique(box_sizes)  # Remove duplicates
    
    counts = []
    
    for box_size in box_sizes:
        if box_size == 0:
            continue
            
        # Calculate box count for this size
        count = 0
        for i in range(0, n - box_size + 1, box_size):
            min_val = np.min(series[i:i + box_size])
            max_val = np.max(series[i:i + box_size])
            count += (max_val - min_val) / box_size
            
        counts.append(count)
    
    # If we have fewer than 2 valid counts, return default dimension
    if len(counts) < 2:
        return 1.0
        
    # Use a linear fit to estimate the fractal dimension
    poly = np.polyfit(np.log(box_sizes), np.log(counts), 1)
    
    # Return the fractal dimension
    return -poly[0]

def apply_kalman_filter(series, process_variance=1e-5, measurement_variance=1e-3):
    """Apply Kalman filter to smooth the price series"""
    n = len(series)
    
    # Initialize Kalman filter variables
    filtered_values = np.zeros(n)
    prediction = series[0]
    prediction_variance = 1.0
    
    for i in range(n):
        # Prediction update
        prediction_variance += process_variance
        
        # Measurement update
        kalman_gain = prediction_variance / (prediction_variance + measurement_variance)
        filtered_values[i] = prediction + kalman_gain * (series[i] - prediction)
        prediction = filtered_values[i]
        prediction_variance = (1 - kalman_gain) * prediction_variance
        
    return filtered_values

def butter_lowpass_filter(data, cutoff, fs, order=5):
    """Apply Butterworth low-pass filter to smooth the data"""
    nyq = 0.5 * fs  # Nyquist frequency
    normal_cutoff = cutoff / nyq
    # Get the filter coefficients 
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a, data)
    return y

def detect_support_resistance(df, window=20, threshold=0.01):
    """Detect support and resistance levels from price data"""
    highs = df['high'].values
    lows = df['low'].values
    n = len(df)
    
    support_levels = []
    resistance_levels = []
    
    for i in range(window, n - window):
        # Check for local minima (support)
        if all(lows[i] <= lows[i-window:i]) and all(lows[i] <= lows[i+1:i+window+1]):
            # Found a potential support level
            support_levels.append((i, lows[i]))
            
        # Check for local maxima (resistance)
        if all(highs[i] >= highs[i-window:i]) and all(highs[i] >= highs[i+1:i+window+1]):
            # Found a potential resistance level
            resistance_levels.append((i, highs[i]))
    
    # Cluster levels that are close to each other
    clustered_support = cluster_price_levels(support_levels, threshold)
    clustered_resistance = cluster_price_levels(resistance_levels, threshold)
    
    return clustered_support, clustered_resistance

def cluster_price_levels(price_levels, threshold_pct=0.2):
    """Group price levels that are within threshold percentage of each other"""
    if not price_levels:
        return []
    
    # Sort by price
    sorted_levels = sorted(price_levels, key=lambda x: x[1])
    clusters = []
    current_cluster = [sorted_levels[0]]
    
    for i in range(1, len(sorted_levels)):
        current_price = sorted_levels[i][1]
        prev_price = sorted_levels[i-1][1]
        
        price_diff_pct = abs(current_price - prev_price) / prev_price
        
        if price_diff_pct <= threshold_pct:
            # Add to current cluster
            current_cluster.append(sorted_levels[i])
        else:
            # Start a new cluster
            if current_cluster:
                avg_price = sum(level[1] for level in current_cluster) / len(current_cluster)
                avg_idx = int(sum(level[0] for level in current_cluster) / len(current_cluster))
                clusters.append((avg_idx, avg_price))
                
            current_cluster = [sorted_levels[i]]
    
    # Handle the last cluster
    if current_cluster:
        avg_price = sum(level[1] for level in current_cluster) / len(current_cluster)
        avg_idx = int(sum(level[0] for level in current_cluster) / len(current_cluster))
        clusters.append((avg_idx, avg_price))
    
    # Sort final clusters by price
    clusters = sorted(clusters, key=lambda x: x[1])
    
    return clusters

def find_support_resistance(data, lookback=50):
    """Find support and resistance levels in price data"""
    support = []
    resistance = []
    
    # Use only a subset of data for efficiency
    data_subset = data.iloc[-lookback:] if len(data) > lookback else data
    
    # Find local minima and maxima
    for i in range(2, len(data_subset) - 2):
        # Check for local minimum (support)
        if (data_subset['low'].iloc[i] < data_subset['low'].iloc[i-1] and 
            data_subset['low'].iloc[i] < data_subset['low'].iloc[i-2] and
            data_subset['low'].iloc[i] < data_subset['low'].iloc[i+1] and
            data_subset['low'].iloc[i] < data_subset['low'].iloc[i+2]):
            support.append(data_subset['low'].iloc[i])
            
        # Check for local maximum (resistance)
        if (data_subset['high'].iloc[i] > data_subset['high'].iloc[i-1] and 
            data_subset['high'].iloc[i] > data_subset['high'].iloc[i-2] and
            data_subset['high'].iloc[i] > data_subset['high'].iloc[i+1] and
            data_subset['high'].iloc[i] > data_subset['high'].iloc[i+2]):
            resistance.append(data_subset['high'].iloc[i])
    
    # Group close levels together
    support = cluster_levels(support, 0.2)
    resistance = cluster_levels(resistance, 0.2)
    
    return {'support': support, 'resistance': resistance}

def cluster_levels(levels, threshold_pct=0.2):
    """Group price levels that are within threshold_pct of each other"""
    if not levels:
        return []
    
    # Sort levels
    sorted_levels = sorted(levels)
    clusters = []
    current_cluster = [sorted_levels[0]]
    
    for level in sorted_levels[1:]:
        # If level is within threshold of the current cluster average
        if abs(level / np.mean(current_cluster) - 1) * 100 < threshold_pct:
            current_cluster.append(level)
        else:
            # Add current cluster average to clusters and start a new cluster
            clusters.append(np.mean(current_cluster))
            current_cluster = [level]
            
    # Add the last cluster
    if current_cluster:
        clusters.append(np.mean(current_cluster))
    
    return clusters

def prepare_features(data, bar_index):
    """Prepare features for model prediction with enhanced feature set"""
    i = bar_index
    
    # Current close price normalized to recent range
    recent_high = data['high'][i-20:i].max()
    recent_low = data['low'][i-20:i].min()
    price_position = (data['close'][i] - recent_low) / (recent_high - recent_low) if recent_high > recent_low else 0.5
    
    # Enhanced momentum features
    rsi = data['rsi'][i]
    rsi_diff = data['rsi'][i] - data['rsi'][i-5]
    
    # Enhanced trend features
    trend_10 = (data['close'][i] / data['close'][i-10] - 1) * 100
    trend_20 = (data['close'][i] / data['close'][i-20] - 1) * 100
    trend_50 = (data['close'][i] / data['close'][i-50] - 1) * 100
    
    # Volatility features
    volatility_5 = data['close'][i-5:i].pct_change().std() * 100
    volatility_10 = data['close'][i-10:i].pct_change().std() * 100
    volatility_20 = data['close'][i-20:i].pct_change().std() * 100
    
    # Volume features (using synthetic volume)
    volume_change = (data['volume'][i] / data['volume'][i-5] - 1) * 100 if data['volume'][i-5] > 0 else 0
    
    # Price pattern features
    upper_wick = (data['high'][i] - max(data['open'][i], data['close'][i])) / (data['high'][i] - data['low'][i]) if (data['high'][i] - data['low'][i]) > 0 else 0
    lower_wick = (min(data['open'][i], data['close'][i]) - data['low'][i]) / (data['high'][i] - data['low'][i]) if (data['high'][i] - data['low'][i]) > 0 else 0
    
    # Candle size and direction
    candle_size = abs(data['close'][i] - data['open'][i]) / (data['high'][i] - data['low'][i]) if (data['high'][i] - data['low'][i]) > 0 else 0
    is_bullish = 1 if data['close'][i] > data['open'][i] else 0
    
    # Support/resistance proximity
    levels = find_support_resistance(data[:i+1], lookback=50)
    proximity_to_support = min([abs(data['close'][i] / level - 1) * 100 for level in levels['support']], default=100)
    proximity_to_resistance = min([abs(data['close'][i] / level - 1) * 100 for level in levels['resistance']], default=100)
    
    # Create feature vector
    return [
        price_position,
        rsi,
        rsi_diff,
        trend_10,
        trend_20,
        trend_50,
        volatility_5,
        volatility_10,
        volatility_20,
        volume_change,
        upper_wick,
        lower_wick,
        candle_size,
        is_bullish,
        proximity_to_support,
        proximity_to_resistance
    ]

def run_ultra_maximized(symbol="BTCUSDT", timeframe="M5", backtest_days=30, initial_balance=200000):
    """
    Run the ultra-maximized HFT strategy for BTCUSDT with ultra-aggressive parameters.
    
    Args:
        symbol (str): Trading symbol, default is BTCUSDT
        timeframe (str): Timeframe for analysis, default is M5 (5-minute)
        backtest_days (int): Number of days to backtest, default is 30
        initial_balance (float): Initial account balance for backtesting, default is 200,000 USDT
    """
    logger = logging.getLogger('UltraMaximized')
    logger.info(f"Starting Ultra-Maximized HFT Strategy for {symbol} on {timeframe} timeframe")
    
    try:
        # Set ultra-aggressive parameters
        rsi_buy_threshold = 0.05  # Even more aggressive RSI threshold for buys
        rsi_sell_threshold = 99.95  # More aggressive RSI threshold for sells
        stop_loss_pct = -2000.0  # Wider stop loss for more room
        take_profit_levels = [10.0, 50.0, 100.0, 500.0, 1000.0]  # Higher profit targets
        position_size_multiplier = 8000.0  # More aggressive position sizing
        max_concurrent_trades = 30  # Allow more concurrent trades
        max_leverage = 200.0  # Set to match the starting leverage for BTCUSDT
        
        # Enhanced parameters
        confirmation_period = 2  # Reduced confirmation period for faster entries
        min_signal_strength = 0.80  # Slightly more selective signals
        
        # BTCUSDT specific parameters
        tick_value = 0.1
        tick_size = 0.1
        stop_level = 100
        min_order_size = 0.01
        max_order_size = 100.0
        
        # Safety parameters for live trading (if applicable)
        mt5_max_order_size = 10.0  # Cap position size for MetaTrader
        
        logger.info("Using ultra-aggressive parameters:")
        logger.info(f"RSI thresholds: {rsi_buy_threshold}/{rsi_sell_threshold}")
        logger.info(f"Stop loss: {stop_loss_pct}%, Take profit levels: {take_profit_levels}")
        logger.info(f"Position size multiplier: {position_size_multiplier}x")
        logger.info(f"Max concurrent trades: {max_concurrent_trades}")
        logger.info(f"Dynamic leverage: up to {max_leverage}x")
        logger.info(f"MetaTrader max order size: {mt5_max_order_size}")
        logger.info(f"BTCUSDT parameters: Tick value={tick_value}, Tick size={tick_size}, Stop level={stop_level}")
        logger.info(f"Order size limits: Min={min_order_size}, Max={max_order_size}")
        logger.info(f"Enhanced settings: confirmation_period={confirmation_period}, min_signal_strength={min_signal_strength}")
        
        # Fetch historical data
        logger.info(f"Fetching historical data for {symbol} ({backtest_days} days)...")
        
        # Determine the start and end dates for backtesting
        end_date = datetime.now()
        start_date = end_date - timedelta(days=backtest_days)
        
        # Convert timeframe to minutes for data fetching
        timeframe_minutes = {
            'M1': 1,
            'M5': 5,
            'M15': 15,
            'M30': 30,
            'H1': 60,
            'H4': 240,
            'D1': 1440
        }.get(timeframe, 5)
        
        # Generate synthetic data for testing
        df = generate_synthetic_data(symbol, start_date, end_date, timeframe_minutes)
        
        # Add indicators
        df = add_indicators(df)
        
        # Initialize market regime detector
        market_regime_detector = MarketRegimeDetector(lookback_period=50)
        
        # Initialize multiple timeframe analysis
        timeframes = [1, 5, 15, 30, 60, 240, 1440]  # 1min, 5min, 15min, 30min, 1h, 4h, daily
        mtf_analysis = MultipleTimeframeAnalysis(symbol, timeframes)
        
        # Update MTF data with our synthetic data
        for tf in timeframes:
            # Resample the data to the current timeframe
            tf_df = df.copy()
            if tf != timeframe_minutes:
                # Resample to the current timeframe
                tf_df = resample_data(df, tf)
            
            # Store in the MTF analysis object
            mtf_analysis.data[tf] = tf_df
            mtf_analysis._calculate_indicators(tf)
        
        # Initialize position sizing system
        position_sizing = AdvancedPositionSizing(
            max_position_size=position_size_multiplier,
            initial_risk_per_trade=0.02,
            max_risk_per_trade=0.05,
            max_open_risk=0.25,
            volatility_adjustment=True,
            win_streak_adjustment=True,
            trend_adjustment=True,
            account_growth_adjustment=True,
            min_order_size=min_order_size,
            max_order_size=max_order_size
        )
        
        # Initialize enhanced trading model
        enhanced_trading = EnhancedTrading(use_ml=True)
        
        # Train the model if data is sufficient
        if len(df) > 1000:
            logger.info("Training ML models...")
            enhanced_trading.train(df)
        
        # Initialize the ultra-maximized strategy
        strategy = UltraMaximizedStrategy(
            symbol=symbol,
            timeframe=timeframe,
            rsi_buy_threshold=rsi_buy_threshold,
            rsi_sell_threshold=rsi_sell_threshold,
            stop_loss_pct=stop_loss_pct,
            take_profit_levels=take_profit_levels,
            position_size_multiplier=position_size_multiplier,
            max_concurrent_trades=max_concurrent_trades,
            enhanced_trading=enhanced_trading,
            market_regime_detector=market_regime_detector,
            position_sizing=position_sizing,
            mtf_analysis=mtf_analysis,
            mt5_max_order_size=mt5_max_order_size
        )
        
        # Run backtest
        logger.info("Running backtest...")
        results = strategy.backtest(df, initial_balance=initial_balance)
        
        # Print performance summary
        print_performance_summary(results)
        
        # Return results for further analysis if needed
        return results
        
    except Exception as e:
        logger.error(f"Error running Ultra-Maximized strategy: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return None

def print_performance_summary(results):
    """
    Print a detailed performance summary in a format similar to StrategyTracker.py
    
    Args:
        results: Dictionary containing backtest results
    """
    # Extract results
    initial_balance = results['initial_balance']
    final_balance = results['final_balance']
    total_return = results['total_return_pct']
    total_trades = results['total_trades']
    win_rate = results['win_rate']
    profit_factor = results['profit_factor']
    avg_profit_per_trade = results['avg_profit_per_trade']
    max_drawdown = results['max_drawdown_pct']
    closed_trades = results['closed_trades']
    
    # Calculate additional metrics
    pnl = final_balance - initial_balance
    
    # Calculate win/loss statistics
    win_trades = [t for t in closed_trades if t['profit_loss'] > 0]
    loss_trades = [t for t in closed_trades if t['profit_loss'] <= 0]
    
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
    print(f"Maximum Drawdown:    {max_drawdown:.2f}%")
    print(f"Sharpe Ratio:        {sharpe_ratio:.2f}")
    
    print("\n[TRADE STATISTICS]")
    print(f"Total Trades:        {total_trades}")
    print(f"Win Rate:            {win_rate:.2f}%")
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
    print(f"Expected Win Rate:   100.00%    Actual: {win_rate:.2f}%")
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

if __name__ == "__main__":
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Run Ultra-Maximized Strategy")
    parser.add_argument("--symbol", type=str, default="BTCUSDT", help="Symbol to trade")
    parser.add_argument("--timeframe", type=str, default="M5", help="Timeframe for backtesting")
    parser.add_argument("--initial-balance", type=float, default=200000, help="Initial balance")
    parser.add_argument("--days", type=int, default=30, help="Days to backtest")
    parser.add_argument("--live", action="store_true", help="Run in live trading mode with MetaTrader constraints")
    
    args = parser.parse_args()
    
    # Run with command line arguments
    run_ultra_maximized(
        symbol=args.symbol,
        timeframe=args.timeframe,
        backtest_days=args.days,
        initial_balance=args.initial_balance
    )

class AdvancedPositionSizing:
    """Advanced position sizing system that dynamically adjusts position sizes 
    based on various market conditions, win rates, and account metrics"""
    
    def __init__(self, max_position_size=100.0, initial_risk_per_trade=0.02, 
                 max_risk_per_trade=0.05, max_open_risk=0.25, 
                 volatility_adjustment=True, win_streak_adjustment=True,
                 trend_adjustment=True, account_growth_adjustment=True):
        
        self.max_position_size = max_position_size  # Maximum position size for safety
        self.initial_risk_per_trade = initial_risk_per_trade  # Initial risk per trade (2%)
        self.max_risk_per_trade = max_risk_per_trade  # Maximum risk per trade (5%)
        self.max_open_risk = max_open_risk  # Maximum open risk exposure (25%)
        
        # Feature toggles
        self.volatility_adjustment = volatility_adjustment
        self.win_streak_adjustment = win_streak_adjustment
        self.trend_adjustment = trend_adjustment
        self.account_growth_adjustment = account_growth_adjustment
        
        # Performance metrics
        self.win_rate = 0.5  # Default value
        self.win_streak = 0
        self.loss_streak = 0
        self.trades_history = []
        self.recent_trades = []  # Last 20 trades
        self.profit_factor = 1.0  # Default value
        
        # Additional risk parameters
        self.risk_multiplier = 1.0
        self.current_risk_per_trade = initial_risk_per_trade
        self.total_open_risk = 0.0
        
        # MT5 exchange limits for BTCUSDT
        self.min_order_size = 0.01
        self.max_order_size = 100.0
        
        # Logger setup
        self.logger = logging.getLogger(__name__)
        
    def update_metrics(self, trade_result, profit_loss, risk_amount):
        """Update performance metrics based on trade results"""
        # Add to trades history
        self.trades_history.append({
            'result': trade_result,  # 'win' or 'loss'
            'pnl': profit_loss,
            'risk': risk_amount
        })
        
        # Update recent trades list (keep last 20)
        self.recent_trades.append({
            'result': trade_result,
            'pnl': profit_loss,
            'risk': risk_amount
        })
        if len(self.recent_trades) > 20:
            self.recent_trades.pop(0)
        
        # Update win rate
        total_trades = len(self.trades_history)
        wins = sum(1 for trade in self.trades_history if trade['result'] == 'win')
        self.win_rate = wins / total_trades if total_trades > 0 else 0.5
        
        # Update profit factor
        gross_profit = sum(trade['pnl'] for trade in self.trades_history if trade['pnl'] > 0)
        gross_loss = abs(sum(trade['pnl'] for trade in self.trades_history if trade['pnl'] < 0))
        self.profit_factor = gross_profit / gross_loss if gross_loss > 0 else 1.0
        
        # Update win/loss streaks
        if trade_result == 'win':
            self.win_streak += 1
            self.loss_streak = 0
        else:
            self.loss_streak += 1
            self.win_streak = 0
            
        # Update risk multiplier based on performance
        self._update_risk_multiplier()
        
    def _update_risk_multiplier(self):
        """Update the risk multiplier based on recent performance"""
        # Base multiplier on win rate
        if self.win_rate >= 0.8:  # Exceptional win rate
            base_multiplier = 2.0
        elif self.win_rate >= 0.7:  # Very good win rate
            base_multiplier = 1.5
        elif self.win_rate >= 0.6:  # Good win rate
            base_multiplier = 1.2
        elif self.win_rate >= 0.5:  # Average win rate
            base_multiplier = 1.0
        elif self.win_rate >= 0.4:  # Below average win rate
            base_multiplier = 0.8
        else:  # Poor win rate
            base_multiplier = 0.5
            
        # Adjust for profit factor
        if self.profit_factor >= 5.0:
            pf_multiplier = 2.0
        elif self.profit_factor >= 3.0:
            pf_multiplier = 1.5
        elif self.profit_factor >= 2.0:
            pf_multiplier = 1.2
        elif self.profit_factor >= 1.5:
            pf_multiplier = 1.0
        else:
            pf_multiplier = 0.8
            
        # Combine multipliers
        self.risk_multiplier = base_multiplier * pf_multiplier
        
        # Cap at maximum risk
        self.current_risk_per_trade = min(
            self.initial_risk_per_trade * self.risk_multiplier,
            self.max_risk_per_trade
        )
        
    def calculate_position_size(self, market_regime, trend_strength, volatility, 
                               buy_probability, win_streak=None, loss_streak=None):
        """
        Calculate the optimal position size based on various factors
        
        Parameters:
        -----------
        market_regime: str - Current market regime ('trend', 'range', 'volatile')
        trend_strength: float - Strength of the current trend (0.0 to 1.0)
        volatility: float - Current market volatility (standard deviation)
        buy_probability: float - Probability of price increase (0.0 to 1.0)
        win_streak: int - Optional current win streak to override internal tracking
        loss_streak: int - Optional current loss streak to override internal tracking
        
        Returns:
        --------
        position_size: float - Calculated position size
        """
        # Use provided streak values if given
        win_streak = win_streak if win_streak is not None else self.win_streak
        loss_streak = loss_streak if loss_streak is not None else self.loss_streak
        
        # Base position sizing using fixed percent risk
        risk_amount = 200000.0 * self.current_risk_per_trade
        
        # Calculate stop loss distance
        stop_distance = abs(75000.0 * (1 - (-2000.0/100)))  # Using base price and stop loss
        
        # Base position size from risk and stop distance
        base_position_size = risk_amount / stop_distance
            
        # Apply multipliers based on enabled features
        multiplier = 1.0
        
        # 1. Market regime adjustment
        if market_regime == 'trend':
            regime_multiplier = 1.5  # Increase size in trending markets
        elif market_regime == 'range':
            regime_multiplier = 1.0  # Normal size in range markets
        else:  # 'volatile'
            regime_multiplier = 0.7  # Reduce size in volatile markets
        
        multiplier *= regime_multiplier
        
        # 2. Trend strength adjustment (if enabled)
        if self.trend_adjustment:
            # Scale from 0.5 (weak trend) to 2.0 (strong trend)
            trend_multiplier = 0.5 + (1.5 * trend_strength)
            multiplier *= trend_multiplier
        
        # 3. Volatility-based adjustment (if enabled)
        if self.volatility_adjustment:
            # Inverse relationship - higher volatility means smaller position
            # Assuming volatility is normalized between 0 and 1
            vol_multiplier = 1.0 - (volatility * 0.5)  # At max volatility, reduce by 50%
            multiplier *= max(0.5, vol_multiplier)  # Ensure at least 50% of base size
        
        # 4. Win streak adjustment (if enabled)
        if self.win_streak_adjustment:
            if win_streak >= 10:
                streak_multiplier = 2.5  # Maximum multiplier
            elif win_streak >= 7:
                streak_multiplier = 2.0
            elif win_streak >= 5:
                streak_multiplier = 1.7
            elif win_streak >= 3:
                streak_multiplier = 1.3
            elif win_streak >= 1:
                streak_multiplier = 1.1
            elif loss_streak >= 5:
                streak_multiplier = 0.5  # Significant reduction after several losses
            elif loss_streak >= 3:
                streak_multiplier = 0.7
            elif loss_streak >= 1:
                streak_multiplier = 0.9
            else:
                streak_multiplier = 1.0
                
            multiplier *= streak_multiplier
        
        # 5. Account growth adjustment (if enabled)
        # This is typically handled through the risk_amount calculation using current balance
        
        # 6. Signal strength adjustment
        signal_multiplier = 0.5 + (buy_probability * 1.5)  # Scale from 0.5 to 2.0
        multiplier *= signal_multiplier
        
        # Apply the final multiplier to the base position size
        position_size = base_position_size * multiplier
        
        # Safety checks
        # 1. Cap at maximum position size
        position_size = min(position_size, self.max_position_size)
        
        # 2. Make sure we don't risk more than max_risk_per_trade
        actual_risk = (stop_distance * position_size) / 200000.0
        if actual_risk > self.max_risk_per_trade:
            position_size = (self.max_risk_per_trade * 200000.0) / stop_distance
        
        # 3. Respect maximum open risk across all positions
        if (self.total_open_risk + actual_risk) > self.max_open_risk:
            available_risk = max(0, self.max_open_risk - self.total_open_risk)
            position_size = min(position_size, (available_risk * 200000.0) / stop_distance)
        
        # 4. For live trading, apply special cap from memory settings
        if hasattr(self, 'live_trading') and self.live_trading:
            position_size = min(position_size, 10.0)  # Hard cap for live trading
            
        # Ensure position size is within exchange limits
        position_size = max(self.min_order_size, min(position_size, self.max_order_size))
        
        # Round to 2 decimal places for proper lot sizing
        position_size = round(position_size, 2)
        
        self.logger.debug(f"Position sizing: base={base_position_size:.2f}, multiplier={multiplier:.2f}, final={position_size:.2f}")
        
        return position_size
    
    def update_open_risk(self, operation, risk_amount):
        """Update the total open risk"""
        if operation == 'add':
            self.total_open_risk += risk_amount
        elif operation == 'remove':
            self.total_open_risk -= risk_amount
            self.total_open_risk = max(0, self.total_open_risk)  # Ensure no negative values
    
    def set_live_trading(self, is_live=True):
        """Set whether this is being used for live trading (applies additional safeguards)"""
        self.live_trading = is_live
        
        # Apply special safety limits for live trading
        if is_live:
            self.max_position_size = min(self.max_position_size, 10.0)  # Reduce max position size
            self.max_risk_per_trade = min(self.max_risk_per_trade, 0.02)  # Max 2% risk per trade
            self.max_open_risk = min(self.max_open_risk, 0.10)  # Max 10% open risk

class MultipleTimeframeAnalysis:
    """Analyze price action across multiple timeframes for more robust signals"""
    
    def __init__(self, symbol, timeframes=None, weighting_scheme=None):
        """
        Initialize the multiple timeframe analysis system
        
        Parameters:
        -----------
        symbol: str - Trading symbol
        timeframes: list - List of timeframes to analyze (in minutes)
        weighting_scheme: dict - Weighting scheme for each timeframe (sum should be 1.0)
        """
        self.symbol = symbol
        
        # Default timeframes if none provided (in minutes)
        self.timeframes = timeframes or [1, 5, 15, 30, 60, 240, 1440]
        
        # Default weighting scheme if none provided
        if weighting_scheme is None:
            # Higher weight to medium timeframes
            self.weighting_scheme = {
                1: 0.05,    # 1 minute (noise filtering)
                5: 0.10,    # 5 minutes (short-term entry/exit)
                15: 0.20,   # 15 minutes (primary trading timeframe)
                30: 0.25,   # 30 minutes (primary confirmation)
                60: 0.20,   # 1 hour (medium trend)
                240: 0.15,  # 4 hours (longer trend)
                1440: 0.05  # Daily (market regime)
            }
        else:
            self.weighting_scheme = weighting_scheme
            
        # Data storage for each timeframe
        self.data = {}
        self.indicators = {}
        
        # Last update time for each timeframe
        self.last_update = {}
        
        # MT5 timeframe mapping
        self.mt5_timeframes = {
            1: mt5.TIMEFRAME_M1 if mt5 else 1,
            5: mt5.TIMEFRAME_M5 if mt5 else 5,
            15: mt5.TIMEFRAME_M15 if mt5 else 15,
            30: mt5.TIMEFRAME_M30 if mt5 else 30,
            60: mt5.TIMEFRAME_H1 if mt5 else 60,
            240: mt5.TIMEFRAME_H4 if mt5 else 240,
            1440: mt5.TIMEFRAME_D1 if mt5 else 1440
        }
        
        # Logger setup
        self.logger = logging.getLogger(__name__)
        
    def update_data(self, current_time=None, bars_count=200, use_mt5=False):
        """
        Update data for all timeframes
        
        Parameters:
        -----------
        current_time: datetime - Current time (if None, use current time)
        bars_count: int - Number of bars to fetch for each timeframe
        use_mt5: bool - Whether to use MetaTrader5 to fetch data
        
        Returns:
        --------
        bool - Success or failure
        """
        current_time = current_time or datetime.now()
        
        try:
            for tf in self.timeframes:
                # Skip if data is recent enough (within half of the timeframe)
                if tf in self.last_update and (current_time - self.last_update[tf]).total_seconds() < (tf * 60 * 0.5):
                    continue
                    
                if use_mt5 and mt5:
                    # Fetch data from MetaTrader 5
                    mt5_tf = self.mt5_timeframes[tf]
                    from_date = current_time - timedelta(minutes=tf * bars_count)
                    bars = mt5.copy_rates_range(self.symbol, mt5_tf, from_date, current_time)
                    if bars is not None:
                        df = pd.DataFrame(bars)
                        # Convert time to datetime
                        df['time'] = pd.to_datetime(df['time'], unit='s')
                        self.data[tf] = df
                        self.last_update[tf] = current_time
                else:
                    # Generate synthetic data for testing
                    if tf not in self.data:
                        # Create new synthetic data
                        self.data[tf] = self._generate_synthetic_data(tf, bars_count, current_time)
                    else:
                        # Add new bars to existing data
                        last_bar_time = self.data[tf]['time'].iloc[-1]
                        new_bars_needed = (current_time - last_bar_time).total_seconds() / (tf * 60)
                        if new_bars_needed >= 1:
                            new_bars = int(new_bars_needed)
                            new_data = self._generate_synthetic_data(tf, new_bars, last_bar_time + timedelta(minutes=tf))
                            self.data[tf] = pd.concat([self.data[tf], new_data]).reset_index(drop=True)
                            
                    self.last_update[tf] = current_time
                
                # Calculate indicators for this timeframe
                self._calculate_indicators(tf)
                
            return True
        except Exception as e:
            self.logger.error(f"Error updating data: {str(e)}")
            return False
    
    def _generate_synthetic_data(self, timeframe, bars_count, end_time):
        """Generate synthetic price data for testing"""
        # Start time is end_time minus (timeframe * bars_count) minutes
        start_time = end_time - timedelta(minutes=timeframe * bars_count)
        
        # Create time index
        times = [start_time + timedelta(minutes=timeframe * i) for i in range(bars_count)]
        
        # Generate random price movements (more volatility for lower timeframes)
        volatility = 0.001 * (60 / timeframe) ** 0.5  # Higher volatility for lower timeframes
        
        # Start with a base price
        base_price = 50000  # Starting price (e.g., for BTC)
        
        # Create arrays for synthetic data
        opens = []
        highs = []
        lows = []
        closes = []
        volumes = []
        
        current_price = base_price
        for i in range(bars_count):
            # Random walk with slight upward bias
            close_change = np.random.normal(0.0001, volatility)
            new_close = current_price * (1 + close_change)
            closes.append(new_close)
            
            # Generate open, high, low based on close
            opens.append(closes[-1])
            
            high_range = opens[-1] * (1 + np.random.uniform(0, volatility * 2))
            low_range = opens[-1] * (1 - np.random.uniform(0, volatility * 2))
            
            highs.append(max(opens[-1], closes[-1], high_range))
            lows.append(min(opens[-1], closes[-1], low_range))
            
            # Generate random volume (higher for higher timeframes)
            volume_base = 100 * timeframe
            volumes.append(max(1, int(np.random.gamma(2, volume_base))))
            
            # Update for next bar
            current_price = closes[-1]
        
        # Create DataFrame
        df = pd.DataFrame({
            'time': times,
            'open': opens,
            'high': highs,
            'low': lows,
            'close': closes,
            'volume': volumes
        })
        
        return df
    
    def _calculate_indicators(self, timeframe):
        """Calculate indicators for a specific timeframe"""
        if timeframe not in self.data or self.data[timeframe].empty:
            return
            
        df = self.data[timeframe].copy()
        
        # Calculate RSI
        if 'close' in df.columns:
            if talib:
                df['rsi'] = talib.RSI(df['close'].values, timeperiod=14)
            else:
                # Calculate RSI using pandas
                delta = df['close'].diff()
                gain = delta.where(delta > 0, 0)
                loss = -delta.where(delta < 0, 0)
                
                avg_gain = gain.rolling(window=14).mean()
                avg_loss = loss.rolling(window=14).mean()
                
                rs = avg_gain / avg_loss
                df['rsi'] = 100 - (100 / (1 + rs))
                
        # Calculate MACD
        if 'close' in df.columns:
            if talib:
                df['macd'], df['macd_signal'], df['macd_hist'] = talib.MACD(
                    df['close'].values, fastperiod=12, slowperiod=26, signalperiod=9
                )
            else:
                # Calculate MACD using pandas
                ema12 = df['close'].ewm(span=12, adjust=False).mean()
                ema26 = df['close'].ewm(span=26, adjust=False).mean()
                df['macd'] = ema12 - ema26
                df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
                df['macd_hist'] = df['macd'] - df['macd_signal']
                
        # Calculate Bollinger Bands
        if 'close' in df.columns:
            if talib:
                df['bb_upper'], df['bb_middle'], df['bb_lower'] = talib.BBANDS(
                    df['close'].values, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0
                )
            else:
                # Calculate Bollinger Bands using pandas
                df['bb_middle'] = df['close'].rolling(window=20).mean()
                df['bb_std'] = df['close'].rolling(window=20).std()
                df['bb_upper'] = df['bb_middle'] + (df['bb_std'] * 2)
                df['bb_lower'] = df['bb_middle'] - (df['bb_std'] * 2)
                
        # Calculate ATR for volatility
        if all(col in df.columns for col in ['high', 'low', 'close']):
            if talib:
                df['atr'] = talib.ATR(df['high'].values, df['low'].values, df['close'].values, timeperiod=14)
            else:
                # Calculate ATR using pandas
                tr1 = df['high'] - df['low']
                tr2 = abs(df['high'] - df['close'].shift())
                tr3 = abs(df['low'] - df['close'].shift())
                tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
                df['atr'] = tr.rolling(window=14).mean()
                
        # Calculate trend strength
        df['ema50'] = df['close'].ewm(span=50, adjust=False).mean()
        df['ema200'] = df['close'].ewm(span=200, adjust=False).mean()
        
        # Trend direction (1 = up, -1 = down, 0 = neutral)
        df['trend'] = np.where(df['ema50'] > df['ema200'], 1, 
                     np.where(df['ema50'] < df['ema200'], -1, 0))
        
        # Price relative to moving averages
        df['price_vs_ema50'] = (df['close'] - df['ema50']) / df['ema50']
        df['price_vs_ema200'] = (df['close'] - df['ema200']) / df['ema200']
        
        # Slope of EMA (trend strength)
        df['ema50_slope'] = df['ema50'].diff(5) / df['ema50'].shift(5)
        df['ema200_slope'] = df['ema200'].diff(20) / df['ema200'].shift(20)
        
        # Store calculated indicators
        self.indicators[timeframe] = df
    
    def get_timeframe_signal(self, timeframe, lookback=1):
        """
        Get trading signal for a specific timeframe
        
        Parameters:
        -----------
        timeframe: int - Timeframe in minutes
        lookback: int - Number of bars to look back
        
        Returns:
        --------
        signal: float - Signal strength (-1.0 to 1.0, where -1.0 is strong sell, 1.0 is strong buy)
        """
        if timeframe not in self.indicators or self.indicators[timeframe].empty:
            return 0.0  # Neutral if no data
            
        df = self.indicators[timeframe]
        
        # Get the most recent bars for analysis
        recent_bars = df.iloc[-lookback:].copy()
        
        if recent_bars.empty:
            return 0.0  # Neutral if no data
            
        # Calculate various signal components
        
        # 1. RSI signal (-1.0 to 1.0)
        if 'rsi' in recent_bars.columns:
            last_rsi = recent_bars['rsi'].iloc[-1]
            # Scale RSI from 0-100 to -1.0 to 1.0
            # RSI below 30 -> strong buy, above 70 -> strong sell
            rsi_signal = -1.0 + (last_rsi / 50.0)  # 0 -> -1.0, 100 -> 1.0
            rsi_signal = max(-1.0, min(1.0, rsi_signal))  # Clip to [-1.0, 1.0]
        else:
            rsi_signal = 0.0
            
        # 2. MACD signal (-1.0 to 1.0)
        if all(col in recent_bars.columns for col in ['macd', 'macd_signal']):
            last_macd = recent_bars['macd'].iloc[-1]
            last_signal = recent_bars['macd_signal'].iloc[-1]
            
            # MACD line position relative to signal line
            macd_cross = last_macd - last_signal
            
            # Normalize based on typical MACD values (depends on the instrument)
            avg_price = recent_bars['close'].mean()
            normalized_cross = macd_cross / (avg_price * 0.001)  # Scale factor can be adjusted
            
            # Clip to reasonable range
            macd_signal = max(-1.0, min(1.0, normalized_cross))
        else:
            macd_signal = 0.0
            
        # 3. Bollinger Band signal (-1.0 to 1.0)
        if all(col in recent_bars.columns for col in ['close', 'bb_upper', 'bb_lower', 'bb_middle']):
            last_close = recent_bars['close'].iloc[-1]
            last_upper = recent_bars['bb_upper'].iloc[-1]
            last_lower = recent_bars['bb_lower'].iloc[-1]
            last_middle = recent_bars['bb_middle'].iloc[-1]
            
            # Calculate price position within bands
            band_width = last_upper - last_lower
            if band_width > 0:
                position = (last_close - last_middle) / (band_width / 2)
                # Position is now between -1.0 (at lower band) and 1.0 (at upper band)
                bb_signal = -position  # Reverse for trading signal (near upper band = sell)
            else:
                bb_signal = 0.0
        else:
            bb_signal = 0.0
            
        # 4. Trend signal (-1.0 to 1.0)
        if 'trend' in recent_bars.columns:
            trend_signal = recent_bars['trend'].iloc[-1]  # Already -1, 0, or 1
            
            # Enhance with slope information
            if 'ema50_slope' in recent_bars.columns:
                slope = recent_bars['ema50_slope'].iloc[-1]
                # Stronger trend signal if slope is steep
                trend_signal *= (1.0 + min(1.0, abs(slope) * 50))
                trend_signal = max(-1.0, min(1.0, trend_signal))
        else:
            trend_signal = 0.0
            
        # 5. Price relative to moving averages (-1.0 to 1.0)
        if all(col in recent_bars.columns for col in ['price_vs_ema50', 'price_vs_ema200']):
            # How far price is from EMAs
            price_ema50 = recent_bars['price_vs_ema50'].iloc[-1]
            price_ema200 = recent_bars['price_vs_ema200'].iloc[-1]
            
            # Average the signals (stronger weight to ema50)
            ema_signal = (price_ema50 * 0.7 + price_ema200 * 0.3) * -10.0  # Scale and invert
            ema_signal = max(-1.0, min(1.0, ema_signal))
        else:
            ema_signal = 0.0
            
        # Combine all signals with simple weighting
        combined_signal = (
            rsi_signal * 0.3 +
            macd_signal * 0.2 +
            bb_signal * 0.2 +
            trend_signal * 0.2 +
            ema_signal * 0.1
        )
        
        return combined_signal
    
    def get_combined_signal(self):
        """
        Get combined signal across all timeframes
        
        Returns:
        --------
        dict: Dictionary with combined signal and confidence
        """
        if not self.indicators:
            return {'signal': 0.0, 'confidence': 0.0, 'buy_probability': 0.5, 'sell_probability': 0.5}
            
        signals = {}
        missing_timeframes = []
        
        # Get signal for each timeframe
        for tf in self.timeframes:
            if tf in self.indicators:
                signal = self.get_timeframe_signal(tf)
                signals[tf] = signal
            else:
                missing_timeframes.append(tf)
                
        if missing_timeframes:
            self.logger.warning(f"Missing data for timeframes: {missing_timeframes}")
            
        if not signals:
            return {'signal': 0.0, 'confidence': 0.0, 'buy_probability': 0.5, 'sell_probability': 0.5}
            
        # Calculate weighted average signal
        weighted_signal = 0.0
        total_weight = 0.0
        
        for tf, signal in signals.items():
            weight = self.weighting_scheme.get(tf, 0.0)
            weighted_signal += signal * weight
            total_weight += weight
            
        if total_weight > 0:
            final_signal = weighted_signal / total_weight
        else:
            final_signal = 0.0
            
        # Calculate signal confidence (higher when timeframes agree)
        signal_values = list(signals.values())
        signal_std = np.std(signal_values)
        
        # More agreement (lower std) means higher confidence
        confidence = max(0.0, 1.0 - signal_std)
        
        # Calculate probability of price movement
        # Convert signal from [-1, 1] to probability [0, 1]
        buy_probability = (final_signal + 1.0) / 2.0
        sell_probability = 1.0 - buy_probability
        
        return {
            'signal': final_signal,
            'confidence': confidence,
            'buy_probability': buy_probability,
            'sell_probability': sell_probability,
            'timeframe_signals': signals
        }
    
    def get_market_structure(self):
        """
        Analyze market structure across timeframes
        
        Returns:
        --------
        dict: Dictionary with market structure analysis
        """
        if not self.indicators:
            return {'trend': 'unknown', 'volatility': 'unknown', 'strength': 0.0}
            
        # Analyze trends across timeframes
        trend_votes = {'up': 0, 'down': 0, 'neutral': 0}
        trend_strength = 0.0
        volatility = []
        
        for tf in self.timeframes:
            if tf not in self.indicators or self.indicators[tf].empty:
                continue
                
            df = self.indicators[tf]
            last_row = df.iloc[-1]
            
            # Determine trend for this timeframe
            if 'trend' in last_row:
                if last_row['trend'] > 0:
                    trend_votes['up'] += self.weighting_scheme.get(tf, 0.0)
                elif last_row['trend'] < 0:
                    trend_votes['down'] += self.weighting_scheme.get(tf, 0.0)
                else:
                    trend_votes['neutral'] += self.weighting_scheme.get(tf, 0.0)
                    
            # Collect trend strength
            if 'ema50_slope' in last_row:
                trend_strength += abs(last_row['ema50_slope']) * self.weighting_scheme.get(tf, 0.0)
                
            # Collect volatility data
            if 'atr' in last_row and last_row['close'] > 0:
                # Normalize ATR as percentage of price
                norm_atr = last_row['atr'] / last_row['close']
                # Weight by timeframe
                volatility.append(norm_atr * self.weighting_scheme.get(tf, 0.0))
                
        # Determine overall trend
        if trend_votes['up'] > trend_votes['down'] + trend_votes['neutral']:
            trend = 'up'
        elif trend_votes['down'] > trend_votes['up'] + trend_votes['neutral']:
            trend = 'down'
        elif trend_votes['neutral'] > trend_votes['up'] + trend_votes['down']:
            trend = 'neutral'
        elif trend_votes['up'] > trend_votes['down']:
            trend = 'weak_up'
        elif trend_votes['down'] > trend_votes['up']:
            trend = 'weak_down'
        else:
            trend = 'neutral'
            
        # Calculate average volatility
        avg_volatility = sum(volatility) / len(volatility) if volatility else 0.0
        
        # Categorize volatility
        if avg_volatility < 0.005:  # Less than 0.5%
            volatility_category = 'low'
        elif avg_volatility < 0.015:  # Less than 1.5%
            volatility_category = 'normal'
        elif avg_volatility < 0.03:  # Less than 3%
            volatility_category = 'high'
        else:
            volatility_category = 'extreme'
            
        return {
            'trend': trend,
            'trend_strength': trend_strength,
            'volatility': volatility_category,
            'volatility_value': avg_volatility,
            'details': {
                'trend_votes': trend_votes
            }
        }
    
    def get_market_regime(self):
        """
        Determine the overall market regime by combining trend and volatility
        
        Returns:
        --------
        str: Market regime description
        """
        structure = self.get_market_structure()
        
        trend = structure['trend']
        volatility = structure['volatility']
        
        # Combine trend and volatility for regime
        if trend == 'up':
            if volatility == 'low':
                regime = 'bullish_calm'
            elif volatility == 'normal':
                regime = 'bullish_normal'
            elif volatility == 'high':
                regime = 'bullish_volatile'
            else:
                regime = 'bullish_extreme'
        elif trend == 'weak_up':
            if volatility == 'low':
                regime = 'weakly_bullish_calm'
            elif volatility == 'normal':
                regime = 'weakly_bullish_normal'
            else:
                regime = 'weakly_bullish_volatile'
        elif trend == 'down':
            if volatility == 'low':
                regime = 'bearish_calm'
            elif volatility == 'normal':
                regime = 'bearish_normal'
            elif volatility == 'high':
                regime = 'bearish_volatile'
            else:
                regime = 'bearish_extreme'
        elif trend == 'weak_down':
            if volatility == 'low':
                regime = 'weakly_bearish_calm'
            elif volatility == 'normal':
                regime = 'weakly_bearish_normal'
            else:
                regime = 'weakly_bearish_volatile'
        else:  # neutral
            if volatility == 'low':
                regime = 'ranging_tight'
            elif volatility == 'normal':
                regime = 'ranging_normal'
            elif volatility == 'high':
                regime = 'ranging_wide'
            else:
                regime = 'choppy_extreme'
                
        return regime
        
    def get_support_resistance_levels(self):
        """
        Identify support and resistance levels across timeframes
        
        Returns:
        --------
        dict: Support and resistance levels with confidence
        """
        if not self.indicators:
            return {'support': [], 'resistance': []}
            
        # For key timeframes, detect support and resistance
        support_levels = []
        resistance_levels = []
        
        # Focus on higher timeframes for S/R levels
        for tf in [tf for tf in self.timeframes if tf >= 60]:  # 1h and above
            if tf not in self.indicators or self.indicators[tf].empty:
                continue
                
            df = self.indicators[tf]
            
            # Find support and resistance using price action
            support, resistance = detect_support_resistance(df, window=10)
            
            # Weight by timeframe (higher timeframes have more significance)
            weight = self.weighting_scheme.get(tf, 0.0) * 2.0  # Double the importance
            
            # Add levels with timeframe information
            for idx, level in support:
                support_levels.append({
                    'price': level,
                    'timeframe': tf,
                    'weight': weight,
                    'age': len(df) - idx  # How many bars ago
                })
                
            for idx, level in resistance:
                resistance_levels.append({
                    'price': level,
                    'timeframe': tf,
                    'weight': weight,
                    'age': len(df) - idx  # How many bars ago
                })
                
        # Cluster nearby levels
        clustered_support = self._cluster_levels(support_levels)
        clustered_resistance = self._cluster_levels(resistance_levels)
        
        return {
            'support': clustered_support,
            'resistance': clustered_resistance
        }
        
    def _cluster_levels(self, levels, threshold_pct=0.005):
        """Cluster nearby price levels"""
        if not levels:
            return []
            
        # Sort by price
        sorted_levels = sorted(levels, key=lambda x: x['price'])
        
        clusters = []
        current_cluster = [sorted_levels[0]]
        
        for i in range(1, len(sorted_levels)):
            current_price = sorted_levels[i]['price']
            prev_price = sorted_levels[i-1]['price']
            
            price_diff_pct = abs(current_price - prev_price) / prev_price
            
            if price_diff_pct <= threshold_pct:
                # Add to current cluster
                current_cluster.append(sorted_levels[i])
            else:
                # Start a new cluster
                if current_cluster:
                    # Calculate weighted average price
                    total_weight = sum(level['weight'] for level in current_cluster)
                    avg_price = sum(level['price'] * level['weight'] for level in current_cluster) / total_weight
                    
                    # Get the average age
                    avg_age = sum(level['age'] for level in current_cluster) / len(current_cluster)
                    
                    # Create a confidence score (based on number of levels, weights, and age)
                    confidence = min(1.0, (len(current_cluster) * total_weight) / (1.0 + avg_age / 10.0))
                    
                    # Add clustered level
                    clusters.append({
                        'price': avg_price,
                        'count': len(current_cluster),
                        'confidence': confidence
                    })
                    
                current_cluster = [sorted_levels[i]]
        
        # Add the last cluster
        if current_cluster:
            total_weight = sum(level['weight'] for level in current_cluster)
            avg_price = sum(level['price'] * level['weight'] for level in current_cluster) / total_weight
            avg_age = sum(level['age'] for level in current_cluster) / len(current_cluster)
            confidence = min(1.0, (len(current_cluster) * total_weight) / (1.0 + avg_age / 10.0))
            
            clusters.append({
                'price': avg_price,
                'count': len(current_cluster),
                'confidence': confidence
            })
        
        # Sort by confidence (descending)
        return sorted(clusters, key=lambda x: x['confidence'], reverse=True)

class UltraMaximizedStrategy:
    """Ultra-aggressive trading strategy with dynamic position sizing and advanced exit management"""
    
    def __init__(self, symbol="BTCUSDT", timeframe=None, ultra_aggressive=True,
                 rsi_buy_threshold=0.05, rsi_sell_threshold=99.95, stop_loss_pct=-2000.0, 
                 take_profit_levels=None, position_size_multiplier=8000.0, max_concurrent_trades=30,
                 trading_model=None, enhanced_trading=None, market_regime_detector=None,
                 position_sizing=None, support_resistance=None, mtf_analysis=None,
                 mt5_max_order_size=10.0):
        self.symbol = symbol
        self.timeframe = timeframe
        self.ultra_aggressive = ultra_aggressive
        
        self.rsi_buy_threshold = rsi_buy_threshold
        self.rsi_sell_threshold = rsi_sell_threshold
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_levels = take_profit_levels or [10.0, 50.0, 100.0, 500.0, 1000.0]
        self.position_size_multiplier = position_size_multiplier
        self.max_concurrent_trades = max_concurrent_trades
        self.mt5_max_order_size = mt5_max_order_size
        
        # BTCUSDT specific trading parameters
        self.tick_value = 0.1
        self.tick_size = 0.1
        self.stop_level = 100  # Minimum stop level in points
        self.starting_leverage = 200
        self.min_order_size = 0.01
        self.max_order_size = 100.0
        
        # Import MetaTrader5 if available
        try:
            import MetaTrader5 as mt5
            self.mt5_available = True
            self.mt5 = mt5
        except ImportError:
            self.mt5_available = False
            self.mt5 = None
        
        # Strategy components
        self.trading_model = trading_model or TradingModel()
        self.enhanced_trading = enhanced_trading or EnhancedTrading(use_ml=True)
        self.market_regime_detector = market_regime_detector or MarketRegimeDetector(lookback_period=50)
        self.position_sizing = position_sizing or AdvancedPositionSizing(
            max_position_size=position_size_multiplier,
            min_order_size=self.min_order_size,
            max_order_size=self.max_order_size
        )
        self.support_resistance = support_resistance or SupportResistanceEnhancement()
        self.mtf_analysis = mtf_analysis
        
        # Performance tracking
        self.trades = []
        self.balance_history = []
        self.equity_history = []
        self.drawdowns = []
        
        # Logger
        self.logger = logging.getLogger(__name__)
        
    def backtest(self, df, initial_balance=200000):
        """
        Run backtest on historical data
        
        Args:
            df (DataFrame): Historical price data with OHLCV
            initial_balance (float): Initial account balance
            
        Returns:
            dict: Backtest results
        """
        # Ensure we have all required indicators
        df = add_indicators(df)
        
        # Initialize backtest variables
        balance = initial_balance
        equity = initial_balance
        open_positions = []
        closed_trades = []
        max_equity = initial_balance
        max_drawdown = 0
        win_streak = 0
        loss_streak = 0
        daily_trades = defaultdict(int)
        
        # Track buy/sell signals
        potential_buys = 0
        potential_sells = 0
        
        # Process each bar
        for i in range(100, len(df)):
            current_time = df.index[i]
            current_price = df['close'].iloc[i]
            day_str = current_time.strftime('%Y-%m-%d')
            
            # Update support and resistance levels
            self.support_resistance.update_from_price_action(df.iloc[:i+1])
            
            # Check for closed positions first
            for pos_idx in reversed(range(len(open_positions))):
                position = open_positions[pos_idx]
                
                # Calculate current P&L
                if position['type'] == 'buy':
                    current_pl_pct = (current_price / position['entry_price'] - 1) * 100
                else:  # sell
                    current_pl_pct = (1 - current_price / position['entry_price']) * 100
                
                # Check for stop loss
                if (position['type'] == 'buy' and current_price <= position['stop_loss']) or \
                   (position['type'] == 'sell' and current_price >= position['stop_loss']):
                    # Close at stop loss
                    position['exit_time'] = current_time
                    position['exit_price'] = position['stop_loss']
                    position['profit_loss'] = position['position_size'] * (position['exit_price'] / position['entry_price'] - 1) \
                                             if position['type'] == 'buy' else \
                                             position['position_size'] * (1 - position['exit_price'] / position['entry_price'])
                    position['status'] = 'closed'
                    position['exit_reason'] = 'Stop Loss'
                    
                    # Update balance
                    balance += position['profit_loss']
                    
                    # Update win/loss streak
                    if position['profit_loss'] > 0:
                        win_streak += 1
                        loss_streak = 0
                    else:
                        win_streak = 0
                        loss_streak += 1
                    
                    # Add to closed trades
                    closed_trades.append(position)
                    
                    # Remove from open positions
                    open_positions.pop(pos_idx)
                    continue
                
                # Check for take profit
                for tp_price in position['take_profit']:
                    if (position['type'] == 'buy' and current_price >= tp_price) or \
                       (position['type'] == 'sell' and current_price <= tp_price):
                        # Close at take profit
                        position['exit_time'] = current_time
                        position['exit_price'] = tp_price
                        position['profit_loss'] = position['position_size'] * (position['exit_price'] / position['entry_price'] - 1) \
                                                 if position['type'] == 'buy' else \
                                                 position['position_size'] * (1 - position['exit_price'] / position['entry_price'])
                        position['status'] = 'closed'
                        position['exit_reason'] = 'Take Profit'
                        
                        # Update balance
                        balance += position['profit_loss']
                        
                        # Update win/loss streak
                        if position['profit_loss'] > 0:
                            win_streak += 1
                            loss_streak = 0
                        else:
                            win_streak = 0
                            loss_streak += 1
                        
                        # Add to closed trades
                        closed_trades.append(position)
                        
                        # Remove from open positions
                        open_positions.pop(pos_idx)
                        break
            
            # Calculate current equity
            open_equity = sum(
                position['position_size'] * (current_price / position['entry_price'] - 1) if position['type'] == 'buy' else
                position['position_size'] * (1 - current_price / position['entry_price'])
                for position in open_positions
            )
            equity = balance + open_equity
            
            # Track maximum equity and drawdown
            if equity > max_equity:
                max_equity = equity
            
            current_drawdown = (max_equity - equity) / max_equity * 100 if max_equity > 0 else 0
            max_drawdown = max(max_drawdown, current_drawdown)
            
            # Track equity history
            self.equity_history.append((current_time, equity))
            
            # Check for new trading opportunities
            if len(open_positions) < self.max_concurrent_trades and daily_trades[day_str] < 20:
                # Get current indicators
                rsi = df['rsi'].iloc[i]
                
                # Get market regime
                market_regime = self.market_regime_detector.detect_regime(df.iloc[:i+1])
                
                # Calculate trend strength
                trend_strength = self.market_regime_detector.get_trend_strength(df.iloc[:i+1])
                
                # Calculate volatility
                volatility = self.market_regime_detector.get_volatility(df.iloc[:i+1])
                
                # Get buy probability from ML model
                buy_probability = self.enhanced_trading.predict(df.iloc[i-20:i+1])
                sell_probability = 1 - buy_probability
                
                # Check for buy signal
                at_support, support_level = self.support_resistance.is_at_support(current_price)
                buy_signal = (
                    rsi <= self.rsi_buy_threshold or 
                    (rsi <= 30 and at_support) or 
                    buy_probability >= 0.8
                )
                
                # Check for sell signal
                at_resistance, resistance_level = self.support_resistance.is_at_resistance(current_price)
                sell_signal = (
                    rsi >= self.rsi_sell_threshold or 
                    (rsi >= 70 and at_resistance) or 
                    sell_probability >= 0.8
                )
                
                # Track potential signals
                if buy_signal:
                    potential_buys += 1
                if sell_signal:
                    potential_sells += 1
                
                # Execute trades if signals are valid
                if buy_signal:
                    # Calculate position size
                    position_size = self.position_sizing.calculate_position_size(
                        market_regime, trend_strength, volatility, buy_probability, win_streak, loss_streak
                    )
                    
                    # Cap position size for safety
                    position_size = min(position_size, self.mt5_max_order_size)
                    
                    # Execute buy trade
                    trade = self.execute_trade(
                        df, i, 'buy', current_price, self.stop_loss_pct, 
                        self.take_profit_levels, position_size
                    )
                    
                    # Add to open positions
                    open_positions.append(trade)
                    
                    # Track daily trades
                    daily_trades[day_str] += 1
                
                elif sell_signal:
                    # Calculate position size
                    position_size = self.position_sizing.calculate_position_size(
                        market_regime, trend_strength, volatility, sell_probability, win_streak, loss_streak
                    )
                    
                    # Cap position size for safety
                    position_size = min(position_size, self.mt5_max_order_size)
                    
                    # Execute sell trade
                    trade = self.execute_trade(
                        df, i, 'sell', current_price, self.stop_loss_pct, 
                        self.take_profit_levels, position_size
                    )
                    
                    # Add to open positions
                    open_positions.append(trade)
                    
                    # Track daily trades
                    daily_trades[day_str] += 1
        
        # Close any remaining open positions at the last price
        last_price = df['close'].iloc[-1]
        last_time = df.index[-1]
        
        for position in open_positions:
            position['exit_time'] = last_time
            position['exit_price'] = last_price
            position['profit_loss'] = position['position_size'] * (position['exit_price'] / position['entry_price'] - 1) \
                                     if position['type'] == 'buy' else \
                                     position['position_size'] * (1 - position['exit_price'] / position['entry_price'])
            position['status'] = 'closed'
            position['exit_reason'] = 'End of Backtest'
            
            # Update balance
            balance += position['profit_loss']
            
            closed_trades.append(position)
        
        # Calculate performance metrics
        total_trades = len(closed_trades)
        winning_trades = sum(1 for trade in closed_trades if trade['profit_loss'] > 0)
        losing_trades = total_trades - winning_trades
        
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
        
        total_profit = sum(trade['profit_loss'] for trade in closed_trades if trade['profit_loss'] > 0)
        total_loss = abs(sum(trade['profit_loss'] for trade in closed_trades if trade['profit_loss'] < 0))
        
        profit_factor = total_profit / total_loss if total_loss > 0 else float('inf')
        
        avg_profit_per_trade = sum(trade['profit_loss'] for trade in closed_trades) / total_trades if total_trades > 0 else 0
        
        # Calculate average win/loss percentages
        if winning_trades > 0:
            avg_win_pct = sum((trade['exit_price'] / trade['entry_price'] - 1) * 100 if trade['type'] == 'buy' else
                             (1 - trade['exit_price'] / trade['entry_price']) * 100
                             for trade in closed_trades if trade['profit_loss'] > 0) / winning_trades
            largest_win_pct = max((trade['exit_price'] / trade['entry_price'] - 1) * 100 if trade['type'] == 'buy' else
                                (1 - trade['exit_price'] / trade['entry_price']) * 100
                                for trade in closed_trades if trade['profit_loss'] > 0)
        else:
            avg_win_pct = 0
            largest_win_pct = 0
            
        if losing_trades > 0:
            avg_loss_pct = sum((trade['exit_price'] / trade['entry_price'] - 1) * 100 if trade['type'] == 'buy' else
                              (1 - trade['exit_price'] / trade['entry_price']) * 100
                              for trade in closed_trades if trade['profit_loss'] <= 0) / losing_trades
            largest_loss_pct = min((trade['exit_price'] / trade['entry_price'] - 1) * 100 if trade['type'] == 'buy' else
                                 (1 - trade['exit_price'] / trade['entry_price']) * 100
                                 for trade in closed_trades if trade['profit_loss'] <= 0)
        else:
            avg_loss_pct = 0
            largest_loss_pct = 0
            
        # Calculate average trade duration
        trade_durations = [(trade['exit_time'] - trade['entry_time']).total_seconds() / 60 for trade in closed_trades]
        avg_trade_duration = sum(trade_durations) / len(trade_durations) if trade_durations else 0
        
        # Count exit reasons
        exit_reasons = {}
        for trade in closed_trades:
            reason = trade['exit_reason']
            exit_reasons[reason] = exit_reasons.get(reason, 0) + 1
            
        # Count buy/sell trades
        buy_trades = sum(1 for trade in closed_trades if trade['type'] == 'buy')
        sell_trades = sum(1 for trade in closed_trades if trade['type'] == 'sell')
        
        # Calculate Sharpe ratio (simplified)
        if len(self.equity_history) > 1:
            equity_values = [eq[1] for eq in self.equity_history]
            returns = [(equity_values[i] / equity_values[i-1] - 1) for i in range(1, len(equity_values))]
            sharpe_ratio = (sum(returns) / len(returns)) / (statistics.stdev(returns) if len(returns) > 1 else 0.01) * math.sqrt(252 * 24 * 60 / 5)  # Annualized
        else:
            sharpe_ratio = 0
            
        # Store all trades
        self.trades = closed_trades
        
        # Return results
        return {
            'initial_balance': initial_balance,
            'final_balance': balance,
            'total_return_pct': (balance / initial_balance - 1) * 100,
            'total_trades': total_trades,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'avg_profit_per_trade': avg_profit_per_trade,
            'max_drawdown_pct': max_drawdown,
            'potential_buys': potential_buys,
            'potential_sells': potential_sells,
            'buy_trades': buy_trades,
            'sell_trades': sell_trades,
            'avg_win_pct': avg_win_pct,
            'avg_loss_pct': avg_loss_pct,
            'largest_win_pct': largest_win_pct,
            'largest_loss_pct': largest_loss_pct,
            'avg_trade_duration': avg_trade_duration,
            'exit_reasons': exit_reasons,
            'sharpe_ratio': sharpe_ratio
        }

    def execute_trade(self, df, i, signal_type, entry_price, stop_loss_pct, take_profit_levels, position_size):
        """
        Execute a trade with the given parameters, adjusting stop loss and take profit based on support and resistance
        
        Args:
            df: DataFrame with price data and indicators
            i: Current index in the dataframe
            signal_type: 'buy' or 'sell'
            entry_price: Entry price for the trade
            stop_loss_pct: Stop loss percentage (negative for buy, positive for sell)
            take_profit_levels: List of take profit percentages
            position_size: Position size to trade
            
        Returns:
            dict: Trade details
        """
        # Get current price data
        current_row = df.iloc[i]
        
        # Ensure position size respects MT5 limits for live trading
        position_size = min(position_size, self.max_order_size)
        position_size = max(position_size, self.min_order_size)
        
        # Round position size to 2 decimal places (0.01 precision for BTCUSDT)
        position_size = round(position_size, 2)
        
        # Calculate base stop loss price
        if signal_type == 'buy':
            # For buy trades, stop loss is below entry price
            stop_loss_price = entry_price * (1 + stop_loss_pct / 100)
        else:
            # For sell trades, stop loss is above entry price
            stop_loss_price = entry_price * (1 + stop_loss_pct / 100)
        
        # Calculate take profit prices
        take_profit_prices = []
        for tp_level in take_profit_levels:
            if signal_type == 'buy':
                # For buy trades, take profit is above entry price
                tp_price = entry_price * (1 + tp_level / 100)
            else:
                # For sell trades, take profit is below entry price
                tp_price = entry_price * (1 - tp_level / 100)
            take_profit_prices.append(tp_price)
        
        # Get support and resistance levels if available
        support_resistance_levels = None
        if self.mtf_analysis:
            support_resistance_levels = self.mtf_analysis.get_support_resistance_levels()
        
        # Adjust stop loss based on support and resistance levels
        if support_resistance_levels:
            if signal_type == 'buy':
                # For buy trades, find the closest support level below entry price
                supports = support_resistance_levels.get('support', [])
                closest_support = None
                closest_distance = float('inf')
                
                for support in supports:
                    if support < entry_price:
                        distance = entry_price - support
                        if distance < closest_distance:
                            closest_distance = distance
                            closest_support = support
                
                # Only adjust if the support level is higher than the calculated stop loss
                if closest_support and closest_support > stop_loss_price:
                    # Add a small buffer below the support level (0.5%)
                    adjusted_stop = closest_support * 0.995
                    
                    # Ensure the stop loss respects the minimum stop level
                    min_stop_distance = self.stop_level * self.tick_value
                    if entry_price - adjusted_stop >= min_stop_distance:
                        stop_loss_price = adjusted_stop
            else:
                # For sell trades, find the closest resistance level above entry price
                resistances = support_resistance_levels.get('resistance', [])
                closest_resistance = None
                closest_distance = float('inf')
                
                for resistance in resistances:
                    if resistance > entry_price:
                        distance = resistance - entry_price
                        if distance < closest_distance:
                            closest_distance = distance
                            closest_resistance = resistance
                
                # Only adjust if the resistance level is lower than the calculated stop loss
                if closest_resistance and closest_resistance < stop_loss_price:
                    # Add a small buffer above the resistance level (0.5%)
                    adjusted_stop = closest_resistance * 1.005
                    
                    # Ensure the stop loss respects the minimum stop level
                    min_stop_distance = self.stop_level * self.tick_value
                    if adjusted_stop - entry_price >= min_stop_distance:
                        stop_loss_price = adjusted_stop
        
        # Adjust take profit targets based on support and resistance
        if support_resistance_levels:
            adjusted_take_profits = []
            
            if signal_type == 'buy':
                # For buy trades, use resistance levels as additional take profit targets
                resistances = support_resistance_levels.get('resistance', [])
                
                # Add resistance levels that are above entry price as potential take profit targets
                for resistance in resistances:
                    if resistance > entry_price:
                        adjusted_take_profits.append(resistance)
            else:
                # For sell trades, use support levels as additional take profit targets
                supports = support_resistance_levels.get('support', [])
                
                # Add support levels that are below entry price as potential take profit targets
                for support in supports:
                    if support < entry_price:
                        adjusted_take_profits.append(support)
            
            # Combine original take profit levels with support/resistance levels
            take_profit_prices.extend(adjusted_take_profits)
            
            # Remove duplicates and sort
            if signal_type == 'buy':
                take_profit_prices = sorted(list(set(take_profit_prices)))
            else:
                take_profit_prices = sorted(list(set(take_profit_prices)), reverse=True)
        
        # Create trade object
        trade = {
            'entry_time': df.index[i],
            'entry_price': entry_price,
            'type': signal_type,
            'stop_loss': stop_loss_price,
            'take_profit': take_profit_prices,
            'position_size': position_size,
            'status': 'open',
            'exit_time': None,
            'exit_price': None,
            'profit_loss': 0,
            'exit_reason': None
        }
        
        return trade

def generate_synthetic_data(symbol, start_date, end_date, timeframe_minutes):
    """
    Generate synthetic price data for backtesting
    
    Args:
        symbol: Trading symbol
        start_date: Start date for data generation
        end_date: End date for data generation
        timeframe_minutes: Timeframe in minutes
        
    Returns:
        DataFrame with synthetic OHLCV data
    """
    logger = logging.getLogger('UltraMaximized')
    logger.info(f"Generating synthetic price data for {symbol} from {start_date} to {end_date}")
    
    # Calculate number of bars based on timeframe
    total_minutes = int((end_date - start_date).total_seconds() / 60)
    num_bars = total_minutes // timeframe_minutes
    
    # Set seed for reproducibility
    np.random.seed(42)
    
    # Start with a realistic BTC price (around $75,000)
    start_price = 75000.0
    
    # Create arrays for synthetic data
    times = []
    opens = []
    highs = []
    lows = []
    closes = []
    volumes = []
    
    current_price = start_price
    current_time = start_date
    
    # Generate realistic price movements with trends, volatility clusters
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
        high_price = max(open_price, close_price) * (1 + random.uniform(0, volatility * 2))
        low_price = min(open_price, close_price) * (1 - random.uniform(0, volatility * 2))
        
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
        current_time += timedelta(minutes=timeframe_minutes)
    
    # Create DataFrame
    df = pd.DataFrame({
        'time': times,
        'open': opens,
        'high': highs,
        'low': lows,
        'close': closes,
        'volume': volumes
    })
    
    # Set time as index
    df.set_index('time', inplace=True)
    
    logger.info(f"Generated {len(df)} bars of synthetic data")
    return df

def resample_data(df, target_timeframe_minutes):
    """
    Resample data to a different timeframe
    
    Args:
        df: DataFrame with OHLCV data
        target_timeframe_minutes: Target timeframe in minutes
        
    Returns:
        Resampled DataFrame
    """
    # Make sure the dataframe has a datetime index
    if not isinstance(df.index, pd.DatetimeIndex):
        if 'time' in df.columns:
            df = df.set_index('time')
        else:
            raise ValueError("DataFrame must have a datetime index or a 'time' column")
    
    # Define the resampling rule
    rule = f'{target_timeframe_minutes}T'
    
    # Resample the data
    resampled = df.resample(rule).agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    })
    
    # Forward fill any missing values
    resampled = resampled.ffill()
    
    return resampled

def add_indicators(df):
    """
    Add technical indicators to the dataframe
    
    Args:
        df: DataFrame with OHLCV data
        
    Returns:
        DataFrame with added indicators
    """
    # Make a copy to avoid modifying the original
    df = df.copy()
    
    # Calculate RSI
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
    df = df.dropna().reset_index(drop=False)
    
    return df
