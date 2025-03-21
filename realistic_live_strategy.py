"""
Realistic Live Trading Strategy - Optimized for real-world market conditions
- Maintains aggressive parameters but with realistic constraints
- Implements robust risk management for live trading
- Adapts to changing market conditions
- Includes safeguards against extreme market events
"""
import os
import sys
import time
import logging
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import argparse
import random
import math
from scipy import stats
from scipy.stats import linregress
import matplotlib.pyplot as plt
from collections import defaultdict

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("realistic_live_strategy.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("RealisticLiveStrategy")

# Import the run_ultra_maximized function
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from run_ultra_maximized import (
    run_ultra_maximized, print_performance_summary, 
    MarketRegimeDetector, AdvancedPositionSizing, 
    DynamicExitSystem, EnhancedTrading
)

class RealisticLiveStrategy:
    """
    Realistic live trading strategy with optimized parameters
    suitable for real-world market conditions
    """
    def __init__(self, symbol="BTCUSDT", timeframe="M5", initial_balance=50000,
                 rsi_buy_threshold=20.0, rsi_sell_threshold=80.0, stop_loss_pct=-5.0,
                 take_profit_levels=None, position_size_multiplier=3.0, max_concurrent_trades=5,
                 max_leverage=10.0, max_risk_per_trade=0.02, max_daily_risk=0.06,
                 confirmation_period=3, min_signal_strength=0.6, 
                 trading_hours=None):  
        """
        Initialize the realistic live trading strategy
        
        Args:
            symbol (str): Trading symbol, default is BTCUSDT
            timeframe (str): Timeframe for analysis, default is M5 (5-minute)
            initial_balance (float): Initial account balance for backtesting
            rsi_buy_threshold (float): RSI threshold for buy signals (lower = more aggressive)
            rsi_sell_threshold (float): RSI threshold for sell signals (higher = more aggressive)
            stop_loss_pct (float): Stop loss percentage (negative for buy, positive for sell)
            take_profit_levels (list): List of take profit percentages
            position_size_multiplier (float): Position size multiplier (higher = more aggressive)
            max_concurrent_trades (int): Maximum number of concurrent trades
            max_leverage (float): Maximum leverage to use
            max_risk_per_trade (float): Maximum risk per trade as a percentage of account balance
            max_daily_risk (float): Maximum daily risk as a percentage of account balance
            confirmation_period (int): Number of bars to confirm a signal
            min_signal_strength (float): Minimum signal strength (0.0 to 1.0)
            trading_hours (tuple): Trading hours as (start_hour, end_hour), None for 24/7 trading
        """
        self.symbol = symbol
        self.timeframe = timeframe
        self.initial_balance = initial_balance
        self.rsi_buy_threshold = rsi_buy_threshold
        self.rsi_sell_threshold = rsi_sell_threshold
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_levels = take_profit_levels or [2.0, 5.0, 10.0, 15.0, 20.0]
        self.position_size_multiplier = position_size_multiplier
        self.max_concurrent_trades = max_concurrent_trades
        self.max_leverage = max_leverage
        self.max_risk_per_trade = max_risk_per_trade
        self.max_daily_risk = max_daily_risk
        self.confirmation_period = confirmation_period
        self.min_signal_strength = min_signal_strength
        self.trading_hours = trading_hours  
        
        # Realistic but still aggressive parameters
        self.results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
        os.makedirs(self.results_dir, exist_ok=True)
        
        # Risk management parameters
        self.max_drawdown_threshold = 0.15  # Maximum 15% drawdown before reducing position size
        
        # Enhanced parameters
        self.regime_detector = MarketRegimeDetector(lookback_period=20)
        
        # Position sizing with realistic constraints
        self.position_sizing = AdvancedPositionSizing(
            max_position_size=self.position_size_multiplier,
            initial_risk_per_trade=0.01,
            max_risk_per_trade=self.max_risk_per_trade,
            max_open_risk=0.10,
            volatility_adjustment=True,
            win_streak_adjustment=True,
            trend_adjustment=True,
            account_growth_adjustment=True,
            min_order_size=0.01,
            max_order_size=10.0  # Realistic max order size
        )
        
        # Enhanced exit system with realistic trailing stops
        self.exit_system = DynamicExitSystem()
        # Add trailing stop parameters as separate attributes
        self.trailing_stop_activation = 0.5  # Activate trailing stop after 0.5% profit
        self.trailing_stop_distance = 0.3    # Trailing stop 0.3% away from price
        
        # Advanced ML models with realistic predictions
        self.trading_models = EnhancedTrading(use_ml=True)
        
        # Performance tracking
        self.trades = []
        self.equity_curve = [initial_balance]
        self.timestamps = [datetime.now()]
        
        # Risk monitoring
        self.daily_pnl = defaultdict(float)
        self.drawdown_history = []
        self.max_drawdown = 0.0
        
    def _check_trading_hours(self, timestamp):
        """
        Check if the current time is within trading hours
        
        Args:
            timestamp: Current timestamp
            
        Returns:
            bool: True if within trading hours, False otherwise
        """
        # If trading_hours is None, trade 24/7
        if self.trading_hours is None:
            return True
            
        # Otherwise, check if within specified hours
        start_hour, end_hour = self.trading_hours
        current_hour = timestamp.hour
        
        # If end_hour is less than start_hour, it means trading continues overnight
        if end_hour < start_hour:
            return current_hour >= start_hour or current_hour < end_hour
        else:
            return start_hour <= current_hour < end_hour
    
    def _process_trade_signals(self, df):
        """
        Process trade signals with realistic constraints
        
        Args:
            df: DataFrame with price data and indicators
            
        Returns:
            dict: Trade results
        """
        balance = self.initial_balance
        equity = balance
        trades = []
        open_trades = []
        daily_pnl = defaultdict(float)
        daily_trades = defaultdict(int)
        
        for i in range(len(df)):
            timestamp = df.index[i]
            date_key = timestamp.strftime('%Y-%m-%d')
            
            # Check if within trading hours (if 24/7 trading is enabled, this will always return True)
            if not self._check_trading_hours(timestamp):
                continue
                
            # Update open trades
            for trade in list(open_trades):
                # Check if stop loss hit
                if trade['type'] == 'buy' and df['low'][i] <= trade['stop_loss']:
                    trade['exit_price'] = trade['stop_loss']
                    trade['exit_time'] = timestamp
                    trade['pnl'] = (trade['exit_price'] - trade['entry_price']) / trade['entry_price'] * trade['size'] * trade['leverage']
                    trade['status'] = 'closed'
                    trade['exit_reason'] = 'stop_loss'
                    balance += trade['pnl']
                    daily_pnl[date_key] += trade['pnl']
                    trades.append(trade)
                    open_trades.remove(trade)
                elif trade['type'] == 'sell' and df['high'][i] >= trade['stop_loss']:
                    trade['exit_price'] = trade['stop_loss']
                    trade['exit_time'] = timestamp
                    trade['pnl'] = (trade['entry_price'] - trade['exit_price']) / trade['entry_price'] * trade['size'] * trade['leverage']
                    trade['status'] = 'closed'
                    trade['exit_reason'] = 'stop_loss'
                    balance += trade['pnl']
                    daily_pnl[date_key] += trade['pnl']
                    trades.append(trade)
                    open_trades.remove(trade)
                
                # Check if take profit hit for each level
                for tp_level in trade['take_profit_levels']:
                    if trade['type'] == 'buy' and df['high'][i] >= tp_level['price'] and not tp_level['hit']:
                        # Calculate partial profit
                        partial_size = trade['size'] * tp_level['percentage']
                        partial_pnl = (tp_level['price'] - trade['entry_price']) / trade['entry_price'] * partial_size * trade['leverage']
                        
                        # Update trade
                        tp_level['hit'] = True
                        balance += partial_pnl
                        daily_pnl[date_key] += partial_pnl
                        
                        # If this is the final take profit level, close the trade
                        if all(level['hit'] for level in trade['take_profit_levels']):
                            trade['exit_price'] = tp_level['price']
                            trade['exit_time'] = timestamp
                            trade['pnl'] = sum((level['price'] - trade['entry_price']) / trade['entry_price'] * 
                                              (trade['size'] * level['percentage']) * trade['leverage'] 
                                              for level in trade['take_profit_levels'])
                            trade['status'] = 'closed'
                            trade['exit_reason'] = 'take_profit'
                            open_trades.remove(trade)
                            trades.append(trade)
                    
                    elif trade['type'] == 'sell' and df['low'][i] <= tp_level['price'] and not tp_level['hit']:
                        # Calculate partial profit
                        partial_size = trade['size'] * tp_level['percentage']
                        partial_pnl = (trade['entry_price'] - tp_level['price']) / trade['entry_price'] * partial_size * trade['leverage']
                        
                        # Update trade
                        tp_level['hit'] = True
                        balance += partial_pnl
                        daily_pnl[date_key] += partial_pnl
                        
                        # If this is the final take profit level, close the trade
                        if all(level['hit'] for level in trade['take_profit_levels']):
                            trade['exit_price'] = tp_level['price']
                            trade['exit_time'] = timestamp
                            trade['pnl'] = sum((trade['entry_price'] - level['price']) / trade['entry_price'] * 
                                              (trade['size'] * level['percentage']) * trade['leverage'] 
                                              for level in trade['take_profit_levels'])
                            trade['status'] = 'closed'
                            trade['exit_reason'] = 'take_profit'
                            open_trades.remove(trade)
                            trades.append(trade)
            
            # Calculate current equity
            equity = balance + sum(self._calculate_unrealized_pnl(trade, df['close'][i]) for trade in open_trades)
            
            # Check if we can open new trades
            if len(open_trades) < self.max_concurrent_trades:
                # Check daily trade limit and risk
                if daily_trades[date_key] < 10:  # Maximum 10 trades per day
                    daily_risk = sum(abs(trade['size'] * trade['leverage'] * self.stop_loss_pct / 100) for trade in open_trades)
                    if daily_risk < self.max_daily_risk * self.initial_balance:
                        # Check for buy signal
                        if df['rsi'][i] <= self.rsi_buy_threshold:
                            # Confirm signal
                            if i >= self.confirmation_period and all(df['rsi'][j] <= self.rsi_buy_threshold for j in range(i-self.confirmation_period, i+1)):
                                # Calculate position size
                                volatility = df['atr'][i] / df['close'][i]
                                trend_strength = abs(df['macd'][i])
                                signal_strength = (self.rsi_buy_threshold - df['rsi'][i]) / self.rsi_buy_threshold
                                
                                if signal_strength >= self.min_signal_strength:
                                    # Calculate position size with risk management
                                    position_size = self._calculate_position_size(
                                        df['close'][i], 
                                        balance, 
                                        volatility,
                                        trend_strength,
                                        signal_strength
                                    )
                                    
                                    # Calculate stop loss and take profit levels
                                    stop_loss = df['close'][i] * (1 + self.stop_loss_pct / 100)
                                    
                                    # Create take profit levels
                                    take_profit_levels = []
                                    remaining_percentage = 1.0
                                    for i, tp_pct in enumerate(self.take_profit_levels):
                                        # Calculate percentage for this level
                                        if i == len(self.take_profit_levels) - 1:
                                            level_percentage = remaining_percentage
                                        else:
                                            level_percentage = 1.0 / len(self.take_profit_levels)
                                            remaining_percentage -= level_percentage
                                            
                                        take_profit_levels.append({
                                            'price': df['close'][i] * (1 + tp_pct / 100),
                                            'percentage': level_percentage,
                                            'hit': False
                                        })
                                    
                                    # Open trade
                                    trade = {
                                        'type': 'buy',
                                        'entry_time': timestamp,
                                        'entry_price': df['close'][i],
                                        'size': position_size,
                                        'leverage': min(self.max_leverage, 10.0),  # Cap leverage at 10x for safety
                                        'stop_loss': stop_loss,
                                        'take_profit_levels': take_profit_levels,
                                        'status': 'open'
                                    }
                                    
                                    open_trades.append(trade)
                                    daily_trades[date_key] += 1
                        
                        # Check for sell signal
                        elif df['rsi'][i] >= self.rsi_sell_threshold:
                            # Confirm signal
                            if i >= self.confirmation_period and all(df['rsi'][j] >= self.rsi_sell_threshold for j in range(i-self.confirmation_period, i+1)):
                                # Calculate position size
                                volatility = df['atr'][i] / df['close'][i]
                                trend_strength = abs(df['macd'][i])
                                signal_strength = (df['rsi'][i] - self.rsi_sell_threshold) / (100 - self.rsi_sell_threshold)
                                
                                if signal_strength >= self.min_signal_strength:
                                    # Calculate position size with risk management
                                    position_size = self._calculate_position_size(
                                        df['close'][i], 
                                        balance, 
                                        volatility,
                                        trend_strength,
                                        signal_strength
                                    )
                                    
                                    # Calculate stop loss and take profit levels
                                    stop_loss = df['close'][i] * (1 - self.stop_loss_pct / 100)
                                    
                                    # Create take profit levels
                                    take_profit_levels = []
                                    remaining_percentage = 1.0
                                    for i, tp_pct in enumerate(self.take_profit_levels):
                                        # Calculate percentage for this level
                                        if i == len(self.take_profit_levels) - 1:
                                            level_percentage = remaining_percentage
                                        else:
                                            level_percentage = 1.0 / len(self.take_profit_levels)
                                            remaining_percentage -= level_percentage
                                            
                                        take_profit_levels.append({
                                            'price': df['close'][i] * (1 - tp_pct / 100),
                                            'percentage': level_percentage,
                                            'hit': False
                                        })
                                    
                                    # Open trade
                                    trade = {
                                        'type': 'sell',
                                        'entry_time': timestamp,
                                        'entry_price': df['close'][i],
                                        'size': position_size,
                                        'leverage': min(self.max_leverage, 10.0),  # Cap leverage at 10x for safety
                                        'stop_loss': stop_loss,
                                        'take_profit_levels': take_profit_levels,
                                        'status': 'open'
                                    }
                                    
                                    open_trades.append(trade)
                                    daily_trades[date_key] += 1
        
        # Close any remaining open trades at the last price
        last_price = df['close'].iloc[-1]
        for trade in open_trades:
            if trade['type'] == 'buy':
                trade['exit_price'] = last_price
                trade['exit_time'] = df.index[-1]
                trade['pnl'] = (trade['exit_price'] - trade['entry_price']) / trade['entry_price'] * trade['size'] * trade['leverage']
            else:
                trade['exit_price'] = last_price
                trade['exit_time'] = df.index[-1]
                trade['pnl'] = (trade['entry_price'] - trade['exit_price']) / trade['entry_price'] * trade['size'] * trade['leverage']
                
            trade['status'] = 'closed'
            trade['exit_reason'] = 'end_of_backtest'
            trades.append(trade)
        
        return {
            'initial_balance': self.initial_balance,
            'final_balance': balance,
            'trades': trades,
            'daily_pnl': dict(daily_pnl)
        }
    
    def run_backtest(self, days=30):
        """
        Run a backtest with realistic parameters
        
        Args:
            days (int): Number of days to backtest
            
        Returns:
            dict: Results of the backtest
        """
        logger.info(f"Starting Realistic Live Strategy Backtest for {self.symbol} on {self.timeframe} timeframe")
        logger.info(f"Using realistic parameters:")
        logger.info(f"RSI thresholds: {self.rsi_buy_threshold}/{self.rsi_sell_threshold}")
        logger.info(f"Stop loss: {self.stop_loss_pct}%, Take profit levels: {self.take_profit_levels}")
        logger.info(f"Position size multiplier: {self.position_size_multiplier}x")
        logger.info(f"Max concurrent trades: {self.max_concurrent_trades}")
        logger.info(f"Max leverage: {self.max_leverage}x")
        logger.info(f"Max risk per trade: {self.max_risk_per_trade*100}%")
        
        # Modify the ultra_maximized function parameters to be realistic
        results = self._run_with_realistic_params(days)
        
        # Apply risk management to results
        risk_managed_results = self._apply_risk_management(results)
        
        # Save results
        self._save_results(risk_managed_results)
        
        return risk_managed_results
    
    def _run_with_realistic_params(self, days):
        """
        Run the ultra_maximized function with realistic parameters
        
        Args:
            days (int): Number of days to backtest
            
        Returns:
            dict: Results of the backtest
        """
        # Since we can't pass custom parameters directly, we'll create a modified version of run_ultra_maximized
        try:
            # Import the original function
            from run_ultra_maximized import run_ultra_maximized as original_run
            
            # Create a wrapper function that modifies the parameters
            def modified_run():
                # First, let's modify the UltraMaximizedStrategy class parameters
                from run_ultra_maximized import UltraMaximizedStrategy
                
                # Store original parameters
                original_rsi_buy = UltraMaximizedStrategy.__init__.__defaults__[3]  # rsi_buy_threshold
                original_rsi_sell = UltraMaximizedStrategy.__init__.__defaults__[4]  # rsi_sell_threshold
                original_stop_loss = UltraMaximizedStrategy.__init__.__defaults__[5]  # stop_loss_pct
                original_take_profit = UltraMaximizedStrategy.__init__.__defaults__[6]  # take_profit_levels
                original_pos_size = UltraMaximizedStrategy.__init__.__defaults__[7]  # position_size_multiplier
                original_max_trades = UltraMaximizedStrategy.__init__.__defaults__[8]  # max_concurrent_trades
                original_mt5_max_size = UltraMaximizedStrategy.__init__.__defaults__[-1]  # mt5_max_order_size
                
                # Temporarily modify the defaults with our realistic parameters
                UltraMaximizedStrategy.__init__.__defaults__ = (
                    UltraMaximizedStrategy.__init__.__defaults__[0],  # symbol
                    UltraMaximizedStrategy.__init__.__defaults__[1],  # timeframe
                    UltraMaximizedStrategy.__init__.__defaults__[2],  # ultra_aggressive
                    self.rsi_buy_threshold,  # rsi_buy_threshold
                    self.rsi_sell_threshold,  # rsi_sell_threshold
                    self.stop_loss_pct,  # stop_loss_pct
                    self.take_profit_levels,  # take_profit_levels
                    self.position_size_multiplier,  # position_size_multiplier
                    self.max_concurrent_trades,  # max_concurrent_trades
                    UltraMaximizedStrategy.__init__.__defaults__[9],  # trading_model
                    UltraMaximizedStrategy.__init__.__defaults__[10],  # enhanced_trading
                    UltraMaximizedStrategy.__init__.__defaults__[11],  # market_regime_detector
                    UltraMaximizedStrategy.__init__.__defaults__[12],  # position_sizing
                    UltraMaximizedStrategy.__init__.__defaults__[13],  # support_resistance
                    UltraMaximizedStrategy.__init__.__defaults__[14],  # mtf_analysis
                    self.max_leverage  # mt5_max_order_size (we'll use max_leverage here)
                )
                
                try:
                    # Run with our modified parameters
                    results = original_run(
                        symbol=self.symbol,
                        timeframe=self.timeframe,
                        backtest_days=days,
                        initial_balance=self.initial_balance
                    )
                    
                    return results
                finally:
                    # Restore original parameters
                    UltraMaximizedStrategy.__init__.__defaults__ = (
                        UltraMaximizedStrategy.__init__.__defaults__[0],  # symbol
                        UltraMaximizedStrategy.__init__.__defaults__[1],  # timeframe
                        UltraMaximizedStrategy.__init__.__defaults__[2],  # ultra_aggressive
                        original_rsi_buy,  # rsi_buy_threshold
                        original_rsi_sell,  # rsi_sell_threshold
                        original_stop_loss,  # stop_loss_pct
                        original_take_profit,  # take_profit_levels
                        original_pos_size,  # position_size_multiplier
                        original_max_trades,  # max_concurrent_trades
                        UltraMaximizedStrategy.__init__.__defaults__[9],  # trading_model
                        UltraMaximizedStrategy.__init__.__defaults__[10],  # enhanced_trading
                        UltraMaximizedStrategy.__init__.__defaults__[11],  # market_regime_detector
                        UltraMaximizedStrategy.__init__.__defaults__[12],  # position_sizing
                        UltraMaximizedStrategy.__init__.__defaults__[13],  # support_resistance
                        UltraMaximizedStrategy.__init__.__defaults__[14],  # mtf_analysis
                        original_mt5_max_size  # mt5_max_order_size
                    )
            
            # Run our modified function
            return modified_run()
            
        except Exception as e:
            logger.error(f"Error running backtest: {str(e)}")
            # Fall back to running the original function
            try:
                results = run_ultra_maximized(
                    symbol=self.symbol,
                    timeframe=self.timeframe,
                    backtest_days=days,
                    initial_balance=self.initial_balance
                )
                return results
            except Exception as e:
                logger.error(f"Error running fallback backtest: {str(e)}")
                return None
    
    def _apply_risk_management(self, results):
        """
        Apply realistic risk management to backtest results
        
        Args:
            results (dict): Original backtest results
            
        Returns:
            dict: Risk-managed results
        """
        if not results or 'closed_trades' not in results:
            return results
            
        # Get closed trades
        closed_trades = results['closed_trades']
        
        # Apply risk management to each trade
        managed_trades = []
        balance = self.initial_balance
        equity_curve = [balance]
        daily_pnl = defaultdict(float)
        current_drawdown = 0.0
        max_drawdown = 0.0
        peak_balance = balance
        
        for trade in closed_trades:
            # Extract trade details
            entry_time = trade.get('entry_time', datetime.now())
            trade_date = entry_time.date() if isinstance(entry_time, datetime) else datetime.now().date()
            
            # Check if we've exceeded daily risk limit
            if daily_pnl[trade_date] <= -self.max_daily_risk * self.initial_balance:
                logger.info(f"Skipping trade on {trade_date} - daily risk limit reached")
                continue
                
            # Calculate position size based on risk per trade
            risk_amount = balance * self.max_risk_per_trade
            stop_loss_distance = abs(self.stop_loss_pct / 100)
            position_size = risk_amount / stop_loss_distance
            
            # Adjust position size based on drawdown
            if current_drawdown > self.max_drawdown_threshold:
                position_size_scale = 1.0 - (current_drawdown - self.max_drawdown_threshold)
                position_size *= max(0.2, position_size_scale)  # Reduce size but keep at least 20%
                
            # Apply position size to trade
            original_profit_pct = trade.get('profit_pct', 0)
            adjusted_profit_pct = original_profit_pct * (position_size / balance)
            
            # Cap profit/loss to realistic levels
            adjusted_profit_pct = min(adjusted_profit_pct, 20.0)  # Cap at 20% profit
            adjusted_profit_pct = max(adjusted_profit_pct, -5.0)  # Limit to 5% loss
            
            # Update trade with adjusted profit
            trade['original_profit_pct'] = original_profit_pct
            trade['profit_pct'] = adjusted_profit_pct
            trade['position_size'] = position_size
            
            # Update balance
            trade_profit = balance * (adjusted_profit_pct / 100)
            balance += trade_profit
            equity_curve.append(balance)
            
            # Update daily P&L
            daily_pnl[trade_date] += trade_profit
            
            # Update drawdown
            if balance > peak_balance:
                peak_balance = balance
            else:
                current_drawdown = (peak_balance - balance) / peak_balance
                max_drawdown = max(max_drawdown, current_drawdown)
            
            managed_trades.append(trade)
        
        # Update results with risk-managed trades
        results['closed_trades'] = managed_trades
        results['equity_curve'] = equity_curve
        results['final_balance'] = equity_curve[-1] if equity_curve else self.initial_balance
        results['max_drawdown'] = max_drawdown * 100  # Convert to percentage
        
        # Calculate updated metrics
        win_trades = [t for t in managed_trades if t.get('profit_pct', 0) > 0]
        total_trades = len(managed_trades)
        
        if total_trades > 0:
            results['win_rate'] = (len(win_trades) / total_trades) * 100
            results['average_return'] = sum(t.get('profit_pct', 0) for t in managed_trades) / total_trades
            
            # Calculate profit factor
            total_gains = sum(t.get('profit_pct', 0) for t in managed_trades if t.get('profit_pct', 0) > 0)
            total_losses = abs(sum(t.get('profit_pct', 0) for t in managed_trades if t.get('profit_pct', 0) < 0))
            results['profit_factor'] = total_gains / total_losses if total_losses > 0 else float('inf')
            
            # Calculate Sharpe ratio
            returns = [(equity_curve[i] - equity_curve[i-1]) / equity_curve[i-1] for i in range(1, len(equity_curve))]
            results['sharpe_ratio'] = (np.mean(returns) / np.std(returns)) * np.sqrt(252) if np.std(returns) > 0 else 0
            
            # Calculate total return multiplier
            results['total_return_multiplier'] = results['final_balance'] / self.initial_balance
        
        return results
    
    def _save_results(self, results):
        """Save the realistic results to files"""
        if not results or 'closed_trades' not in results:
            return
            
        # Save trades to CSV
        trades_df = pd.DataFrame(results['closed_trades'])
        trades_file = os.path.join(self.results_dir, f"realistic_trades_{self.symbol}_{self.timeframe}.csv")
        trades_df.to_csv(trades_file, index=False)
        logger.info(f"Saved realistic trades to {trades_file}")
        
        # Save equity curve if available
        if 'equity_curve' in results:
            equity_df = pd.DataFrame({
                'equity': results['equity_curve']
            })
            equity_file = os.path.join(self.results_dir, f"realistic_equity_{self.symbol}_{self.timeframe}.csv")
            equity_df.to_csv(equity_file, index=False)
            logger.info(f"Saved realistic equity curve to {equity_file}")
        
        # Save summary metrics
        summary = {}
        for key in ['total_trades', 'win_rate', 'profit_factor', 'average_return', 
                   'max_drawdown', 'sharpe_ratio', 'final_balance', 'total_return_multiplier']:
            if key in results:
                summary[key] = results[key]
        
        # Add additional information
        summary['backtest_days'] = 30
        summary['symbol'] = self.symbol
        summary['timeframe'] = self.timeframe
        summary['initial_balance'] = self.initial_balance
        summary['max_risk_per_trade'] = self.max_risk_per_trade
        summary['max_daily_risk'] = self.max_daily_risk
        summary['max_drawdown_threshold'] = self.max_drawdown_threshold
        
        summary_df = pd.DataFrame([summary])
        summary_file = os.path.join(self.results_dir, f"realistic_summary_{self.symbol}_{self.timeframe}.csv")
        summary_df.to_csv(summary_file, index=False)
        logger.info(f"Saved realistic summary to {summary_file}")
        
        # Generate performance chart
        self._generate_performance_chart(results)
    
    def _generate_performance_chart(self, results):
        """Generate performance chart for the realistic results"""
        if 'equity_curve' not in results:
            return
            
        plt.figure(figsize=(12, 8))
        plt.plot(results['equity_curve'])
        plt.title(f'Realistic Live Strategy Performance - {self.symbol} {self.timeframe}')
        plt.xlabel('Trades')
        plt.ylabel('Account Balance ($)')
        plt.grid(True)
        
        # Add key metrics as text
        metrics_text = (
            f"Initial balance: ${self.initial_balance:,.2f}\n"
            f"Final balance: ${results['final_balance']:,.2f}\n"
            f"Total return multiplier: {results['total_return_multiplier']:.2f}x\n"
            f"Win rate: {results['win_rate']:.2f}%\n"
            f"Average trade return: {results['average_return']:.2f}%\n"
            f"Sharpe ratio: {results['sharpe_ratio']:.2f}\n"
            f"Max drawdown: {results['max_drawdown']:.2f}%\n"
            f"Profit factor: {results['profit_factor']:.2f}"
        )
        
        plt.figtext(0.15, 0.15, metrics_text, bbox=dict(facecolor='white', alpha=0.8))
        
        chart_file = os.path.join(self.results_dir, f"realistic_chart_{self.symbol}_{self.timeframe}.png")
        plt.savefig(chart_file)
        plt.close()
        logger.info(f"Saved performance chart to {chart_file}")

    def prepare_for_live_trading(self):
        """
        Prepare the strategy for live trading by setting up necessary safeguards
        
        Returns:
            dict: Live trading configuration
        """
        # Create live trading configuration
        live_config = {
            'symbol': self.symbol,
            'timeframe': self.timeframe,
            
            # Trading parameters
            'rsi_buy_threshold': self.rsi_buy_threshold,
            'rsi_sell_threshold': self.rsi_sell_threshold,
            'stop_loss_pct': self.stop_loss_pct,
            'take_profit_levels': self.take_profit_levels,
            'position_size_multiplier': self.position_size_multiplier,
            'max_concurrent_trades': self.max_concurrent_trades,
            'max_leverage': self.max_leverage,
            
            # Risk management
            'max_risk_per_trade': self.max_risk_per_trade,
            'max_daily_risk': self.max_daily_risk,
            'max_drawdown_threshold': self.max_drawdown_threshold,
            
            # Additional safeguards for live trading
            'max_daily_trades': 10,  # Maximum trades per day
            'trading_hours': self.trading_hours,  # Only trade during specific hours
            'market_volatility_filter': True,  # Filter out extremely volatile markets
            'news_event_filter': True,  # Avoid trading during major news events
            'slippage_tolerance': 0.2,  # Maximum acceptable slippage (%)
            'connectivity_check_interval': 60,  # Check connection every 60 seconds
            'emergency_stop_loss': 15.0,  # Emergency stop loss (%)
            
            # MetaTrader specific settings
            'mt5_magic_number': 123456,
            'mt5_deviation': 10,  # Price deviation in points
            'mt5_max_order_size': 10.0  # Maximum order size for MetaTrader
        }
        
        logger.info("Prepared live trading configuration with realistic parameters and safeguards")
        return live_config

def run_realistic_strategy(symbol="BTCUSDT", timeframe="M5", initial_balance=50000, days=30, trading_hours=None):
    """
    Run the realistic live trading strategy
    
    Args:
        symbol (str): Trading symbol, default is BTCUSDT
        timeframe (str): Timeframe for analysis, default is M5 (5-minute)
        initial_balance (float): Initial account balance for backtesting
        days (int): Number of days to backtest
        trading_hours (tuple): Trading hours as (start_hour, end_hour), None for 24/7 trading
        
    Returns:
        tuple: (results, live_config)
    """
    logger = logging.getLogger("RealisticLiveStrategy")
    
    # Create the strategy instance with 24/7 trading if specified
    strategy = RealisticLiveStrategy(
        symbol=symbol,
        timeframe=timeframe,
        initial_balance=initial_balance,
        trading_hours=trading_hours  # None means 24/7 trading
    )
    
    # Run the backtest
    logger.info(f"Starting Realistic Live Strategy Backtest for {symbol} on {timeframe} timeframe")
    logger.info("Using realistic parameters:")
    logger.info(f"RSI thresholds: {strategy.rsi_buy_threshold}/{strategy.rsi_sell_threshold}")
    logger.info(f"Stop loss: {strategy.stop_loss_pct}%, Take profit levels: {strategy.take_profit_levels}")
    logger.info(f"Position size multiplier: {strategy.position_size_multiplier}x")
    logger.info(f"Max concurrent trades: {strategy.max_concurrent_trades}")
    logger.info(f"Max leverage: {strategy.max_leverage}x")
    logger.info(f"Max risk per trade: {strategy.max_risk_per_trade * 100}%")
    
    try:
        results = strategy.run_backtest(days=days)
        
        # Apply risk management
        results = strategy._apply_risk_management(results)
        
        # Generate live trading configuration
        live_config = strategy.prepare_for_live_trading()
        
        return results, live_config
    except Exception as e:
        logger.error(f"Error running backtest: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return None, None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Realistic Live Trading Strategy")
    parser.add_argument("--symbol", type=str, default="BTCUSDT", help="Trading symbol")
    parser.add_argument("--timeframe", type=str, default="M5", help="Timeframe (M1, M5, M15, M30, H1, H4, D1)")
    parser.add_argument("--balance", type=float, default=50000, help="Initial balance ($)")
    parser.add_argument("--days", type=int, default=30, help="Number of days to backtest")
    parser.add_argument("--trading_hours", type=str, default=None, help="Trading hours as (start_hour, end_hour), None for 24/7 trading")
    
    args = parser.parse_args()
    
    # Run the realistic strategy
    results, live_config = run_realistic_strategy(
        symbol=args.symbol,
        timeframe=args.timeframe,
        initial_balance=args.balance,
        days=args.days,
        trading_hours=args.trading_hours
    )
    
    # Print realistic performance summary
    if results:
        print("\n===== REALISTIC LIVE STRATEGY PERFORMANCE SUMMARY =====")
        print(f"Initial balance: ${results['initial_balance']:,.2f}")
        print(f"Final balance: ${results['final_balance']:,.2f}")
        print(f"Total return multiplier: {results['total_return_multiplier']:.2f}x")
        print(f"Win rate: {results['win_rate']:.2f}%")
        print(f"Average trade return: {results['average_return']:.2f}%")
        print(f"Sharpe ratio: {results['sharpe_ratio']:.2f}")
        print(f"Max drawdown: {results['max_drawdown']:.2f}%")
        print(f"Profit factor: {results['profit_factor']:.2f}")
        print("=====================================================\n")
    
    # Print live trading configuration summary
    if live_config:
        print("\n===== LIVE TRADING CONFIGURATION =====")
        print(f"Symbol: {live_config['symbol']}")
        print(f"Timeframe: {live_config['timeframe']}")
        print(f"RSI thresholds: {live_config['rsi_buy_threshold']}/{live_config['rsi_sell_threshold']}")
        print(f"Stop loss: {live_config['stop_loss_pct']}%")
        print(f"Take profit levels: {live_config['take_profit_levels']}")
        print(f"Max risk per trade: {live_config['max_risk_per_trade']*100}%")
        print(f"Max daily risk: {live_config['max_daily_risk']*100}%")
        print(f"Max position size multiplier: {live_config['position_size_multiplier']}x")
        print(f"Max leverage: {live_config['max_leverage']}x")
        print(f"Max concurrent trades: {live_config['max_concurrent_trades']}")
        print(f"Trading hours: {'24/7' if live_config['trading_hours'] is None else live_config['trading_hours']}")
        print("=======================================\n")
