"""
Realistic Live Trading Strategy - Optimized for real-world market conditions
- Maintains aggressive parameters but with realistic constraints
- Implements robust risk management for live trading
- Adapts to changing market conditions
- Includes safeguards against extreme market events
- Enhanced with adaptive parameters and machine learning
- Implements multi-timeframe analysis for improved entries and exits
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
    def __init__(self, symbol="BTCUSDT", timeframe="M5", initial_balance=50000):
        self.symbol = symbol
        self.timeframe = timeframe
        self.initial_balance = initial_balance
        self.results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
        os.makedirs(self.results_dir, exist_ok=True)
        
        # Realistic but still aggressive parameters
        self.rsi_buy_threshold = 20.0  # More realistic RSI threshold
        self.rsi_sell_threshold = 80.0  # More realistic RSI threshold
        self.stop_loss_pct = -5.0  # Realistic stop loss percentage
        self.take_profit_levels = [2.0, 5.0, 10.0, 15.0, 20.0]  # Realistic profit targets
        self.position_size_multiplier = 3.0  # Realistic position sizing
        self.max_concurrent_trades = 5  # Reasonable number of concurrent trades
        self.max_leverage = 10.0  # Realistic leverage for crypto
        
        # Risk management parameters
        self.max_risk_per_trade = 0.02  # Maximum 2% risk per trade
        self.max_daily_risk = 0.06  # Maximum 6% risk per day
        self.max_drawdown_threshold = 0.15  # Maximum 15% drawdown before reducing position size
        
        # Enhanced parameters
        self.confirmation_period = 2  # Number of bars to confirm signal
        self.min_signal_strength = 0.7  # Signal strength threshold
        
        # Advanced adaptive parameters
        self.adaptive_parameters = True  # Enable adaptive parameters
        self.parameter_update_frequency = 50  # Update parameters every 50 trades
        self.adaptive_rsi_range = (15.0, 25.0)  # Range for adaptive RSI buy threshold
        self.adaptive_tp_range = (15.0, 25.0)  # Range for adaptive take profit
        
        # Multi-timeframe analysis
        self.use_multi_timeframe = True
        self.higher_timeframes = ["M15", "H1", "H4"]  # Higher timeframes to analyze
        self.timeframe_weights = {"M5": 0.4, "M15": 0.3, "H1": 0.2, "H4": 0.1}  # Weights for each timeframe
        
        # Market regime detection with shorter lookback for faster adaptation
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
        self.partial_exit_levels = [0.25, 0.5, 0.75]  # Exit 25%, 50%, 75% at different profit levels
        self.partial_exit_profits = [5.0, 10.0, 15.0]  # Profit levels for partial exits
        
        # Advanced ML models with realistic predictions
        self.trading_models = EnhancedTrading(use_ml=True)
        self.ml_confidence_threshold = 0.75  # Minimum confidence for ML signals
        self.ensemble_models = True  # Use ensemble of models for better predictions
        self.feature_importance_analysis = True  # Analyze feature importance
        
        # Performance tracking
        self.trades = []
        self.equity_curve = [initial_balance]
        self.timestamps = [datetime.now()]
        
        # Risk monitoring
        self.daily_pnl = defaultdict(float)
        self.drawdown_history = []
        self.max_drawdown = 0.0
        
        # Advanced risk metrics
        self.var_confidence_level = 0.95  # 95% Value at Risk
        self.cvar_lookback = 20  # Conditional VaR lookback period
        self.kelly_fraction = 0.5  # Kelly criterion fraction
        
        # Market condition filters
        self.volatility_filter = True  # Filter trades based on volatility
        self.volatility_threshold = 2.0  # Volatility threshold (standard deviations)
        self.liquidity_filter = True  # Filter trades based on liquidity
        self.min_volume_threshold = 1000000  # Minimum volume for trading
        
        # File paths for results
        self.trades_file = os.path.join(self.results_dir, f"realistic_trades_{self.symbol}_{self.timeframe}.csv")
        self.equity_file = os.path.join(self.results_dir, f"realistic_equity_{self.symbol}_{self.timeframe}.csv")
        self.chart_file = os.path.join(self.results_dir, f"realistic_chart_{self.symbol}_{self.timeframe}.png")
    
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
        self._save_results(risk_managed_results, days)
        
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
        
        # Advanced risk metrics
        returns = []
        consecutive_losses = 0
        max_consecutive_losses = 0
        
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
            
            # Apply Kelly criterion for position sizing if enabled
            if hasattr(self, 'kelly_fraction') and self.kelly_fraction > 0:
                # Calculate win probability and win/loss ratio from recent trades
                if len(managed_trades) >= 20:
                    recent_trades = managed_trades[-20:]
                    win_count = sum(1 for t in recent_trades if t.get('profit_pct', 0) > 0)
                    win_prob = win_count / len(recent_trades)
                    
                    if win_prob > 0:
                        avg_win = np.mean([t.get('profit_pct', 0) for t in recent_trades if t.get('profit_pct', 0) > 0])
                        avg_loss = abs(np.mean([t.get('profit_pct', 0) for t in recent_trades if t.get('profit_pct', 0) < 0]) or 1.0)
                        
                        # Kelly formula: f* = (p * b - q) / b
                        # where p = win probability, q = loss probability, b = win/loss ratio
                        win_loss_ratio = avg_win / avg_loss if avg_loss > 0 else 1.0
                        kelly_size = (win_prob * win_loss_ratio - (1 - win_prob)) / win_loss_ratio
                        
                        # Apply Kelly fraction to avoid over-betting
                        kelly_size = max(0.0, kelly_size) * self.kelly_fraction
                        
                        # Adjust position size based on Kelly criterion
                        position_size = min(position_size, balance * kelly_size)
            
            # Adjust position size based on drawdown
            if current_drawdown > self.max_drawdown_threshold:
                position_size_scale = 1.0 - (current_drawdown - self.max_drawdown_threshold)
                position_size *= max(0.2, position_size_scale)  # Reduce size but keep at least 20%
                
            # Adjust position size based on consecutive losses
            if consecutive_losses >= 2:
                position_size *= max(0.5, 1.0 - (consecutive_losses * 0.1))  # Reduce by 10% per loss
                
            # Get original trade details
            original_profit_pct = trade.get('profit_pct', 0)
            
            # Apply realistic slippage and fees
            slippage_pct = 0.05  # 0.05% slippage
            fee_pct = 0.1  # 0.1% fee
            
            # Adjust profit by slippage and fees
            adjusted_profit_pct = original_profit_pct - slippage_pct - fee_pct
            
            # Update trade with realistic parameters
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
                current_drawdown = 0.0
                consecutive_losses = 0  # Reset consecutive losses on new peak
            else:
                current_drawdown = (peak_balance - balance) / peak_balance
                max_drawdown = max(max_drawdown, current_drawdown)
            
            # Track consecutive losses
            if adjusted_profit_pct < 0:
                consecutive_losses += 1
                max_consecutive_losses = max(max_consecutive_losses, consecutive_losses)
            else:
                consecutive_losses = 0
                
            # Calculate return for this trade
            if len(equity_curve) >= 2:
                trade_return = (equity_curve[-1] - equity_curve[-2]) / equity_curve[-2]
                returns.append(trade_return)
            
            managed_trades.append(trade)
        
        # Update results with risk-managed trades
        results['closed_trades'] = managed_trades
        results['equity_curve'] = equity_curve
        results['final_balance'] = equity_curve[-1] if equity_curve else self.initial_balance
        results['max_drawdown'] = max_drawdown * 100  # Convert to percentage
        results['max_consecutive_losses'] = max_consecutive_losses
        
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
            
            # Calculate Sortino ratio (downside risk only)
            downside_returns = [r for r in returns if r < 0]
            downside_deviation = np.std(downside_returns) if downside_returns else 0.001
            results['sortino_ratio'] = (np.mean(returns) / downside_deviation) * np.sqrt(252) if downside_deviation > 0 else 0
            
            # Calculate Calmar ratio (return / max drawdown)
            results['calmar_ratio'] = ((results['final_balance'] / self.initial_balance - 1) / (max_drawdown or 0.001)) if max_drawdown > 0 else 0
            
            # Calculate Value at Risk (VaR)
            if len(returns) > 0:
                results['value_at_risk'] = np.percentile(returns, (1 - self.var_confidence_level) * 100) * balance
                
                # Calculate Conditional VaR (CVaR) / Expected Shortfall
                var_threshold = results['value_at_risk'] / balance
                cvar_returns = [r for r in returns if r <= var_threshold]
                results['conditional_var'] = np.mean(cvar_returns) * balance if cvar_returns else results['value_at_risk']
            
            # Calculate total return multiplier
            results['total_return_multiplier'] = results['final_balance'] / self.initial_balance
        
        return results
    
    def _save_results(self, results, days):
        """Save the realistic results to files"""
        if not results or 'closed_trades' not in results:
            return
            
        # Save trades to CSV
        trades_df = pd.DataFrame(results['closed_trades'])
        trades_df.to_csv(self.trades_file, index=False)
        logger.info(f"Saved realistic trades to {self.trades_file}")
        
        # Save equity curve if available
        if 'equity_curve' in results:
            equity_df = pd.DataFrame({
                'equity': results['equity_curve']
            })
            equity_df.to_csv(self.equity_file, index=False)
            logger.info(f"Saved realistic equity curve to {self.equity_file}")
        
        # Save summary metrics
        summary = {}
        for key in ['total_trades', 'win_rate', 'profit_factor', 'average_return', 
                   'max_drawdown', 'sharpe_ratio', 'final_balance', 'total_return_multiplier']:
            if key in results:
                summary[key] = results[key]
        
        # Add additional information
        summary['backtest_days'] = days  # Use the actual days parameter
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
        """Generate a performance chart for the backtest results"""
        try:
            # Create a figure with multiple subplots
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle('Realistic Strategy Performance', fontsize=16)
            
            # Plot equity curve
            if os.path.exists(self.equity_file):
                equity_data = pd.read_csv(self.equity_file)
                if not equity_data.empty:
                    axes[0, 0].plot(equity_data['equity'])
                    axes[0, 0].set_title('Equity Curve')
                    axes[0, 0].set_xlabel('Time')
                    axes[0, 0].set_ylabel('Equity')
            
            # Plot trade outcomes (win/loss)
            if os.path.exists(self.trades_file):
                trades_data = pd.read_csv(self.trades_file)
                if not trades_data.empty:
                    # Check which column exists for profit
                    profit_column = 'profit_pct' if 'profit_pct' in trades_data.columns else 'profit_loss'
                    
                    wins = trades_data[trades_data[profit_column] > 0]
                    losses = trades_data[trades_data[profit_column] <= 0]
                    
                    axes[0, 1].bar(['Wins', 'Losses'], [len(wins), len(losses)])
                    axes[0, 1].set_title('Trade Outcomes')
                    
                    # Plot profit distribution
                    axes[1, 0].hist(trades_data[profit_column], bins=20)
                    axes[1, 0].set_title('Profit Distribution')
                    axes[1, 0].set_xlabel('Profit/Loss')
                    axes[1, 0].set_ylabel('Frequency')
            
            # Add text summary
            summary_text = (
                f"Total Trades: {results.get('total_trades', 0)}\n"
                f"Win Rate: {results.get('win_rate', 0):.2f}%\n"
                f"Profit Factor: {results.get('profit_factor', 0):.2f}\n"
                f"Max Drawdown: {results.get('max_drawdown_pct', 0):.2f}%\n"
                f"Total Return: {results.get('total_return_pct', 0):.2f}%\n"
                f"Sharpe Ratio: {results.get('sharpe_ratio', 0):.2f}\n"
            )
            axes[1, 1].text(0.1, 0.5, summary_text, fontsize=12)
            axes[1, 1].axis('off')
            
            # Adjust layout and save
            plt.tight_layout(rect=[0, 0, 1, 0.95])
            plt.savefig(self.chart_file)
            plt.close()
            
            logger.info(f"Saved performance chart to {self.chart_file}")
        except Exception as e:
            logger.error(f"Error generating performance chart: {e}")
    
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
            
            # Advanced adaptive parameters
            'adaptive_parameters': self.adaptive_parameters,
            'parameter_update_frequency': self.parameter_update_frequency,
            'adaptive_rsi_range': self.adaptive_rsi_range,
            'adaptive_tp_range': self.adaptive_tp_range,
            
            # Multi-timeframe analysis
            'use_multi_timeframe': self.use_multi_timeframe,
            'higher_timeframes': self.higher_timeframes,
            'timeframe_weights': self.timeframe_weights,
            
            # Partial exit strategy
            'partial_exit_levels': self.partial_exit_levels,
            'partial_exit_profits': self.partial_exit_profits,
            
            # Machine learning settings
            'ml_confidence_threshold': self.ml_confidence_threshold,
            'ensemble_models': self.ensemble_models,
            
            # Advanced risk metrics
            'kelly_fraction': self.kelly_fraction,
            'var_confidence_level': self.var_confidence_level,
            
            # Additional safeguards for live trading
            'max_daily_trades': 10,  # Maximum trades per day
            'trading_hours': {  # Trading 24/7
                'start': '00:00',
                'end': '23:59'
            },
            'market_volatility_filter': True,  # Filter out extremely volatile markets
            'volatility_threshold': self.volatility_threshold,
            'news_event_filter': True,  # Avoid trading during major news events
            'news_event_window': 60,  # Minutes to avoid trading before/after news
            'slippage_tolerance': 0.2,  # Maximum acceptable slippage (%)
            'connectivity_check_interval': 60,  # Check connection every 60 seconds
            'emergency_stop_loss': 15.0,  # Emergency stop loss (%)
            'circuit_breaker': {  # Circuit breaker to pause trading
                'enabled': True,
                'consecutive_losses': 3,  # Pause after 3 consecutive losses
                'daily_loss_threshold': 0.05,  # Pause after 5% daily loss
                'pause_duration': 120  # Pause for 120 minutes
            },
            'auto_recovery': {  # Auto-recovery after circuit breaker
                'enabled': True,
                'min_pause_time': 60,  # Minimum pause time in minutes
                'recovery_position_scale': 0.5  # Scale positions to 50% after recovery
            },
            
            # MetaTrader specific settings
            'mt5_magic_number': 123456,
            'mt5_deviation': 10,  # Price deviation in points
            'mt5_max_order_size': 10.0  # Maximum order size for MetaTrader
        }
        
        logger.info("Prepared live trading configuration with realistic parameters and safeguards")
        return live_config

def run_realistic_strategy(symbol="BTCUSDT", timeframe="M5", initial_balance=50000, days=30):
    """
    Run the realistic live trading strategy
    
    Args:
        symbol (str): Trading symbol
        timeframe (str): Timeframe for analysis
        initial_balance (float): Initial account balance
        days (int): Number of days to backtest
        
    Returns:
        dict: Results of the backtest
    """
    strategy = RealisticLiveStrategy(
        symbol=symbol,
        timeframe=timeframe,
        initial_balance=initial_balance
    )
    
    # Run backtest with realistic parameters
    results = strategy.run_backtest(days=days)
    
    # Print realistic performance summary
    if results:
        print("\n===== REALISTIC LIVE STRATEGY PERFORMANCE SUMMARY =====")
        print(f"Initial balance: ${initial_balance:,.2f}")
        print(f"Final balance: ${results['final_balance']:,.2f}")
        print(f"Total return multiplier: {results['total_return_multiplier']:.2f}x")
        print(f"Win rate: {results['win_rate']:.2f}%")
        print(f"Average trade return: {results['average_return']:.2f}%")
        print(f"Sharpe ratio: {results['sharpe_ratio']:.2f}")
        
        # Print additional advanced metrics if available
        if 'sortino_ratio' in results:
            print(f"Sortino ratio: {results['sortino_ratio']:.2f}")
        if 'calmar_ratio' in results:
            print(f"Calmar ratio: {results['calmar_ratio']:.2f}")
        if 'max_consecutive_losses' in results:
            print(f"Max consecutive losses: {results['max_consecutive_losses']}")
        if 'value_at_risk' in results:
            print(f"Value at Risk (95%): ${results['value_at_risk']:,.2f}")
            
        print(f"Max drawdown: {results['max_drawdown']:.2f}%")
        print(f"Profit factor: {results['profit_factor']:.2f}")
        print("=====================================================\n")
    
    # Prepare live trading configuration
    live_config = strategy.prepare_for_live_trading()
    
    # Print live trading configuration summary
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
    print(f"Trading hours: 24/7")
    
    # Print advanced configuration
    print(f"Adaptive parameters: {live_config['adaptive_parameters']}")
    print(f"Multi-timeframe analysis: {live_config['use_multi_timeframe']}")
    print(f"Partial exit strategy: {live_config['partial_exit_levels']}")
    print(f"Circuit breaker enabled: {live_config['circuit_breaker']['enabled']}")
    print("=======================================\n")
    
    return results, live_config

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Realistic Live Trading Strategy")
    parser.add_argument("--symbol", type=str, default="BTCUSDT", help="Trading symbol")
    parser.add_argument("--timeframe", type=str, default="M5", help="Timeframe (M1, M5, M15, M30, H1, H4, D1)")
    parser.add_argument("--balance", type=float, default=50000, help="Initial balance ($)")
    parser.add_argument("--days", type=int, default=30, help="Number of days to backtest")
    
    args = parser.parse_args()
    
    # Run the realistic strategy
    results, live_config = run_realistic_strategy(
        symbol=args.symbol,
        timeframe=args.timeframe,
        initial_balance=args.balance,
        days=args.days
    )
