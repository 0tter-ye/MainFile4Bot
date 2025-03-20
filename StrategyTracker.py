"""
Ultra-Maximized Strategy Performance Tracker

This script creates and updates a performance tracking spreadsheet for monitoring
the live trading results of the ultra-maximized strategy.
"""

import os
import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import glob
import json
import time
import math

# Configuration
INITIAL_BALANCE = 210781.75
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
TRADES_PATTERN = os.path.join(RESULTS_DIR, "live_trades_*.json")
TRACKER_FILE = os.path.join(RESULTS_DIR, 'strategy_performance_tracker.csv')
CHARTS_FOLDER = os.path.join(RESULTS_DIR, 'performance_charts')

# Constants for backtest comparison
BACKTEST_WIN_RATE = 1.0  # 100% win rate
BACKTEST_AVG_RETURN = 1.5792  # 157.92% average return
BACKTEST_SHARPE = 3.28  # Sharpe ratio from backtest
BACKTEST_FINAL_VALUE = 1.56e15  # 1.56 quadrillion from backtest

class PerformanceTracker:
    def __init__(self, start_value):
        self.start_value = start_value
        self.current_value = start_value
        self.trades = []
        self.win_rate = 0.0
        self.avg_return = 0.0

    def add_trade(self, trade):
        self.trades.append(trade)
        self.current_value += trade.get('pnl', 0)
        self._update_metrics()

    def _update_metrics(self):
        if self.trades:
            self.win_rate = sum(1 for t in self.trades if t.get('pnl', 0) > 0) / len(self.trades)
            self.avg_return = sum(t.get('return', 0) for t in self.trades) / len(self.trades)

    def get_performance(self):
        return {
            'start_value': self.start_value,
            'current_value': self.current_value,
            'win_rate': self.win_rate,
            'avg_return': self.avg_return,
            'trade_count': len(self.trades)
        }

def create_tracker_file():
    """Create a new tracker file with headers"""
    os.makedirs(os.path.dirname(TRACKER_FILE), exist_ok=True)
    
    with open(TRACKER_FILE, 'w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow([
            'Date',
            'Account Balance',
            'P&L',
            'Growth %',
            'Cumulative Growth %',
            'Trade Count',
            'Win Count',
            'Win Rate',
            'Avg Trade Return %',
            'Avg Win %',
            'Avg Loss %',
            'Largest Win %',
            'Largest Loss %',
            'Drawdown %',
            'Max Drawdown %',
            'Sharpe Ratio',
            'Notes'
        ])
    print(f"Created new tracker file: {TRACKER_FILE}")

def load_trades():
    """Load trades from all JSON files in the results directory"""
    trade_files = glob.glob(TRADES_PATTERN)
    
    if not trade_files:
        print(f"Warning: No trade files found matching pattern {TRADES_PATTERN}")
        return []
    
    all_trades = []
    for trade_file in trade_files:
        print(f"Loading trades from {trade_file}")
        try:
            with open(trade_file, 'r') as file:
                trades = json.load(file)
                all_trades.extend(trades)
        except json.JSONDecodeError:
            print(f"Error: Could not parse {trade_file}")
        except Exception as e:
            print(f"Error loading {trade_file}: {str(e)}")
    
    # Sort trades by exit date
    if all_trades:
        all_trades.sort(key=lambda x: x.get('exit_date', ''))
    
    print(f"Loaded {len(all_trades)} trades in total")
    return all_trades

def update_tracker(current_balance, trades=None, pnl=0, growth_pct=0, cumulative_growth_pct=0, notes=''):
    """Update the tracker with the current balance and trade statistics"""
    if not os.path.exists(TRACKER_FILE):
        create_tracker_file()
        return pd.read_csv(TRACKER_FILE, encoding='utf-8')
    
    df = pd.read_csv(TRACKER_FILE, encoding='utf-8')
    initial_balance = df.iloc[0]['Account Balance']
    
    # Trade statistics
    if trades is None:
        trades = []
    
    trade_count = len(trades)
    win_count = sum(1 for t in trades if t.get('pnl', 0) > 0)
    win_rate = (win_count / trade_count) * 100 if trade_count > 0 else 0
    
    # Return statistics
    returns = [t.get('return', 0) * 100 for t in trades]  # Convert to percentage
    avg_return = sum(returns) / len(returns) if returns else 0
    
    win_returns = [r for r in returns if r > 0]
    loss_returns = [r for r in returns if r <= 0]
    
    avg_win = sum(win_returns) / len(win_returns) if win_returns else 0
    avg_loss = sum(loss_returns) / len(loss_returns) if loss_returns else 0
    
    largest_win = max(win_returns) if win_returns else 0
    largest_loss = min(loss_returns) if loss_returns else 0
    
    # Calculate drawdown
    if trade_count > 0:
        balances = [initial_balance]
        for t in trades:
            balances.append(balances[-1] + t.get('pnl', 0))
        
        peak = initial_balance
        drawdowns = []
        
        for balance in balances:
            if balance > peak:
                peak = balance
            drawdown = ((peak - balance) / peak) * 100
            drawdowns.append(drawdown)
        
        current_drawdown = drawdowns[-1] if drawdowns else 0
        max_drawdown = max(drawdowns) if drawdowns else 0
    else:
        current_drawdown = 0
        max_drawdown = 0
    
    # Calculate Sharpe ratio (simplified)
    if returns:
        returns_array = np.array(returns)
        sharpe_ratio = returns_array.mean() / returns_array.std() * np.sqrt(252) if returns_array.std() > 0 else 0
    else:
        sharpe_ratio = 0
    
    # Add new row to tracker
    with open(TRACKER_FILE, 'a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow([
            datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            current_balance,
            pnl,
            growth_pct,
            cumulative_growth_pct,
            trade_count,
            win_count,
            win_rate,
            avg_return,
            avg_win,
            avg_loss,
            largest_win,
            largest_loss,
            current_drawdown,
            max_drawdown,
            sharpe_ratio,
            notes
        ])
    
    print(f"Updated tracker with current balance: {current_balance:.2f}")
    return pd.read_csv(TRACKER_FILE, encoding='utf-8')

def reset_account_statistics(current_balance):
    """Reset all account statistics while keeping the current balance"""
    if not os.path.exists(TRACKER_FILE):
        create_tracker_file()
        return pd.read_csv(TRACKER_FILE, encoding='utf-8')
    
    # Create a new tracker file with just the initial entry
    backup_file = TRACKER_FILE + '.bak'
    if os.path.exists(TRACKER_FILE):
        # Remove existing backup file if it exists
        if os.path.exists(backup_file):
            os.remove(backup_file)
        os.rename(TRACKER_FILE, backup_file)
    
    # Get the initial balance from the user
    initial_balance = None
    while initial_balance is None:
        try:
            initial_balance_input = input(f"Enter the start value (current balance is {current_balance:.2f}): ")
            initial_balance = float(initial_balance_input)
            if initial_balance <= 0:
                print("Start value must be greater than 0")
                initial_balance = None
        except ValueError:
            print("Please enter a valid number for the start value")
    
    # Calculate initial P&L and growth based on the new start value
    pnl = current_balance - initial_balance
    growth = (current_balance / initial_balance - 1) * 100 if initial_balance > 0 else 0
    
    # Create a new tracker file
    with open(TRACKER_FILE, 'w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow([
            'Date',
            'Account Balance',
            'P&L',
            'Growth %',
            'Cumulative Growth %',
            'Trade Count',
            'Win Count',
            'Win Rate',
            'Avg Trade Return %',
            'Avg Win %',
            'Avg Loss %',
            'Largest Win %',
            'Largest Loss %',
            'Drawdown %',
            'Max Drawdown %',
            'Sharpe Ratio',
            'Notes'
        ])
        
        # Add the initial entry with the user-specified start value
        writer.writerow([
            datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            current_balance,  # Current balance
            pnl,              # P&L based on user-specified start value
            growth,           # Growth % based on user-specified start value
            growth,           # Cumulative Growth % starts same as Growth %
            0,                # Trade Count
            0,                # Win Count
            0,                # Win Rate
            0,                # Avg Trade Return %
            0,                # Avg Win %
            0,                # Avg Loss %
            0,                # Largest Win %
            0,                # Largest Loss %
            0,                # Drawdown %
            0,                # Max Drawdown %
            0,                # Sharpe Ratio
            f'Statistics reset - Start value: {initial_balance:.2f}, Current balance: {current_balance:.2f}'
        ])
    
    print(f"Reset account statistics with start value: {initial_balance:.2f} and current balance: {current_balance:.2f}")
    return pd.read_csv(TRACKER_FILE, encoding='utf-8')

def fix_encoding_issues():
    """Fix encoding issues in the tracker file by recreating it with proper encoding"""
    if not os.path.exists(TRACKER_FILE):
        print("Tracker file does not exist. Creating a new one.")
        create_tracker_file()
        return pd.read_csv(TRACKER_FILE, encoding='utf-8')
    
    # Backup the existing file
    backup_file = TRACKER_FILE + '.bak'
    try:
        # Try to read with different encodings
        for encoding in ['utf-8', 'latin1', 'cp1252', 'iso-8859-1']:
            try:
                df = pd.read_csv(TRACKER_FILE, encoding=encoding)
                print(f"Successfully read file with {encoding} encoding")
                
                # Backup the original file
                if os.path.exists(TRACKER_FILE):
                    # Remove existing backup file if it exists
                    if os.path.exists(backup_file):
                        os.remove(backup_file)
                    os.rename(TRACKER_FILE, backup_file)
                
                # Write back with utf-8 encoding
                df.to_csv(TRACKER_FILE, index=False, encoding='utf-8')
                print(f"Fixed encoding issues and saved file with utf-8 encoding")
                return df
            except Exception as e:
                print(f"Failed to read with {encoding} encoding: {e}")
                continue
        
        # If we get here, none of the encodings worked
        print("Could not fix encoding issues automatically.")
        print("Creating a new tracker file...")
        
        # Rename the problematic file
        if os.path.exists(TRACKER_FILE):
            os.rename(TRACKER_FILE, TRACKER_FILE + '.corrupted')
        
        # Create a new file
        create_tracker_file()
        return pd.read_csv(TRACKER_FILE, encoding='utf-8')
    
    except Exception as e:
        print(f"Error fixing encoding issues: {e}")
        # Create a new file as a last resort
        create_tracker_file()
        return pd.read_csv(TRACKER_FILE, encoding='utf-8')

def generate_performance_charts(df):
    """Generate performance charts from the tracker data"""
    # Create charts directory if it doesn't exist
    os.makedirs(CHARTS_FOLDER, exist_ok=True)
    
    if len(df) < 2:
        print("Not enough data points to generate charts.")
        return
    
    # Plot equity curve
    plt.figure(figsize=(12, 6))
    plt.plot(df['Date'], df['Account Balance'], label='Actual Balance')
    
    # Add backtest projection line
    initial_balance = df['Account Balance'].iloc[0]
    dates = pd.to_datetime(df['Date'])
    days_passed = [(date - dates.iloc[0]).days for date in dates]
    
    # Calculate theoretical backtest growth (assuming daily compounding)
    daily_return = (1 + BACKTEST_AVG_RETURN) ** (1/20) - 1  # Assuming 20 trading days per trade
    backtest_balance = [initial_balance * (1 + daily_return) ** day for day in days_passed]
    
    plt.plot(df['Date'], backtest_balance, 'r--', label='Backtest Projection')
    
    plt.title('Equity Curve')
    plt.xlabel('Date')
    plt.ylabel('Balance')
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(CHARTS_FOLDER, 'equity_curve.png'))
    plt.close()
    
    # 2. Cumulative Growth Chart
    plt.figure(figsize=(12, 6))
    plt.plot(df['Date'], df['Cumulative Growth %'], label='Actual Growth')
    
    # Add backtest growth projection
    backtest_growth = [(1 + daily_return) ** day * 100 - 100 for day in days_passed]
    plt.plot(df['Date'], backtest_growth, 'r--', label='Backtest Growth Projection')
    
    plt.title('Cumulative Growth (%)')
    plt.xlabel('Date')
    plt.ylabel('Growth (%)')
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(CHARTS_FOLDER, 'cumulative_growth.png'))
    plt.close()
    
    # 3. Drawdown Chart
    plt.figure(figsize=(12, 6))
    plt.plot(df['Date'], df['Drawdown %'])
    plt.title('Drawdown (%)')
    plt.xlabel('Date')
    plt.ylabel('Drawdown (%)')
    plt.grid(True)
    plt.savefig(os.path.join(CHARTS_FOLDER, 'drawdown.png'))
    plt.close()
    
    # 4. Win Rate Chart
    plt.figure(figsize=(12, 6))
    plt.plot(df['Date'], df['Win Rate'], label='Actual Win Rate')
    plt.axhline(y=BACKTEST_WIN_RATE*100, color='r', linestyle='--', label='Backtest Win Rate (100%)')
    plt.title('Win Rate (%)')
    plt.xlabel('Date')
    plt.ylabel('Win Rate (%)')
    plt.ylim(0, 105)  # Set y-axis limit to 0-105%
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(CHARTS_FOLDER, 'win_rate.png'))
    plt.close()
    
    # 5. Average Trade Return Chart
    plt.figure(figsize=(12, 6))
    plt.plot(df['Date'], df['Avg Trade Return %'], label='Actual Avg Return')
    plt.axhline(y=BACKTEST_AVG_RETURN*100, color='r', linestyle='--', label=f'Backtest Avg Return ({BACKTEST_AVG_RETURN*100:.2f}%)')
    plt.title('Average Trade Return (%)')
    plt.xlabel('Date')
    plt.ylabel('Average Return (%)')
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(CHARTS_FOLDER, 'avg_return.png'))
    plt.close()
    
    # 6. Sharpe Ratio Chart
    plt.figure(figsize=(12, 6))
    plt.plot(df['Date'], df['Sharpe Ratio'], label='Actual Sharpe Ratio')
    plt.axhline(y=BACKTEST_SHARPE, color='r', linestyle='--', label=f'Backtest Sharpe Ratio ({BACKTEST_SHARPE:.2f})')
    plt.title('Sharpe Ratio')
    plt.xlabel('Date')
    plt.ylabel('Sharpe Ratio')
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(CHARTS_FOLDER, 'sharpe_ratio.png'))
    plt.close()
    
    # 7. Performance Comparison (% of backtest performance)
    plt.figure(figsize=(12, 6))
    
    # Calculate percentages of backtest performance
    win_rate_pct = df['Win Rate'] / (BACKTEST_WIN_RATE*100) * 100
    avg_return_pct = df['Avg Trade Return %'] / (BACKTEST_AVG_RETURN*100) * 100
    sharpe_pct = df['Sharpe Ratio'] / BACKTEST_SHARPE * 100
    
    plt.plot(df['Date'], win_rate_pct, label='Win Rate % of Backtest')
    plt.plot(df['Date'], avg_return_pct, label='Avg Return % of Backtest')
    plt.plot(df['Date'], sharpe_pct, label='Sharpe % of Backtest')
    plt.axhline(y=100, color='r', linestyle='--', label='Backtest Performance (100%)')
    
    plt.title('Performance as % of Backtest')
    plt.xlabel('Date')
    plt.ylabel('% of Backtest Performance')
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(CHARTS_FOLDER, 'backtest_comparison.png'))
    plt.close()
    
    # 8. Logarithmic comparison of balance growth
    plt.figure(figsize=(12, 6))
    plt.semilogy(df['Date'], df['Account Balance'], label='Actual Balance')
    plt.semilogy(df['Date'], backtest_balance, 'r--', label='Backtest Projection')
    plt.title('Balance Growth (Log Scale)')
    plt.xlabel('Date')
    plt.ylabel('Balance - Log Scale')
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(CHARTS_FOLDER, 'balance_log_scale.png'))
    plt.close()
    
    print(f"Performance charts generated and saved to {CHARTS_FOLDER}")

def print_performance_summary(df, trades=None):
    """Print a summary of the current performance metrics"""
    if df.empty or len(df) < 1:
        print("No data available for performance summary")
        return
    
    # Get the latest values
    current_balance = df.iloc[-1]['Account Balance']
    
    # Get the initial balance
    initial_balance = df.iloc[0]['Account Balance']
    
    # Calculate total P&L and growth
    total_pnl = current_balance - initial_balance
    total_growth = (current_balance / initial_balance - 1) * 100 if initial_balance > 0 else 0
    
    # Check if this is a reset statistics scenario
    notes = df.iloc[0]['Notes']
    if 'Statistics reset' in str(notes) and 'Start value' in str(notes):
        # Extract the start value from the notes
        start_value_match = re.search(r'Start value: ([\d.]+)', str(notes))
        if start_value_match:
            initial_balance = float(start_value_match.group(1))
            total_pnl = current_balance - initial_balance
            total_growth = (current_balance / initial_balance - 1) * 100
    
    # Get trade count
    trade_count = df.iloc[-1]['Trade Count'] if 'Trade Count' in df.columns and len(df) > 0 else 0
    
    if len(df) <= 1:
        # Very simplified summary when only one entry exists
        print("\n----- ACCOUNT SUMMARY -----")
        print(f"Start Value: {initial_balance:.2f}")
        print(f"Current Value: {current_balance:.2f}")
        print(f"Total P&L: {total_pnl:.2f}")
        print(f"Total Growth: {total_growth:.2f}%")
        print(f"Trade Count: 0")
        print("No trade data available yet.")
        return
    
    # Trade statistics
    if trades is None:
        trades = []
    
    win_count = sum(1 for t in trades if t.get('pnl', 0) > 0)
    win_rate = (win_count / trade_count) * 100 if trade_count > 0 else 0
    
    # Return statistics
    returns = [t.get('return', 0) * 100 for t in trades]  # Convert to percentage
    avg_return = sum(returns) / len(returns) if returns else 0
    
    win_returns = [r for r in returns if r > 0]
    loss_returns = [r for r in returns if r <= 0]
    
    avg_win = sum(win_returns) / len(win_returns) if win_returns else 0
    avg_loss = sum(loss_returns) / len(loss_returns) if loss_returns else 0
    
    largest_win = max(win_returns) if win_returns else 0
    largest_loss = min(loss_returns) if loss_returns else 0
    
    # Calculate drawdown
    if trade_count > 0:
        balances = [initial_balance]
        for t in trades:
            balances.append(balances[-1] + t.get('pnl', 0))
        
        peak = initial_balance
        drawdowns = []
        
        for balance in balances:
            if balance > peak:
                peak = balance
            drawdown = ((peak - balance) / peak) * 100
            drawdowns.append(drawdown)
        
        current_drawdown = drawdowns[-1] if drawdowns else 0
        max_drawdown = max(drawdowns) if drawdowns else 0
    else:
        current_drawdown = 0
        max_drawdown = 0
    
    # Calculate Sharpe ratio (simplified)
    if returns:
        returns_array = np.array(returns)
        sharpe_ratio = returns_array.mean() / returns_array.std() * np.sqrt(252) if returns_array.std() > 0 else 0
    else:
        sharpe_ratio = 0
    
    print("\n===== PERFORMANCE SUMMARY =====")
    
    # Account statistics
    print("\n----- ACCOUNT STATISTICS -----")
    print(f"Start Value: {initial_balance:.2f}")
    print(f"Final Value: {current_balance:.2f}")
    print(f"Total P&L: {total_pnl:.2f}")
    print(f"Total Growth: {total_growth:.2f}%")
    
    # Trade statistics
    print("\n----- TRADE STATISTICS -----")
    print(f"Total Trades: {int(trade_count)}")
    print(f"Winning Trades: {int(win_count)}")
    print(f"Win Rate: {win_rate:.2f}%")
    print(f"Average Trade Return: {avg_return:.2f}%")
    print(f"Average Win: {avg_win:.2f}%")
    print(f"Average Loss: {avg_loss:.2f}%")
    print(f"Largest Win: {largest_win:.2f}%")
    print(f"Largest Loss: {largest_loss:.2f}%")
    
    # Calculate profit factor
    if avg_loss != 0:
        profit_factor = (win_rate/100 * avg_win) / ((1-win_rate/100) * abs(avg_loss)) if avg_loss != 0 else float('inf')
        print(f"Profit Factor: {profit_factor:.2f}")
    
    # Calculate return standard deviation if we have trades
    if trades and len(trades) > 1:
        returns = [trade.get('return', 0) for trade in trades]
        return_std = np.std(returns)
        print(f"Return Std Dev: {return_std:.2f}%")
        
        # Calculate Sharpe ratio (assuming risk-free rate of 0%)
        sharpe = avg_return / return_std if return_std > 0 else 0
        print(f"Sharpe Ratio: {sharpe:.2f}")
    
    # Risk statistics
    current_drawdown = df.iloc[-1]['Drawdown %']
    max_drawdown = df.iloc[-1]['Max Drawdown %']
    sharpe = df.iloc[-1]['Sharpe Ratio']
    
    print("\n----- RISK STATISTICS -----")
    print(f"Current Drawdown: {current_drawdown:.2f}%")
    print(f"Maximum Drawdown: {max_drawdown:.2f}%")
    print(f"Sharpe Ratio: {sharpe:.2f}")
    
    # Comparison to backtest
    print("\n----- BACKTEST COMPARISON -----")
    print(f"Backtest Win Rate: {BACKTEST_WIN_RATE*100:.2f}% | Live Win Rate: {win_rate:.2f}% | Difference: {win_rate - BACKTEST_WIN_RATE*100:.2f}%")
    print(f"Backtest Avg Return: {BACKTEST_AVG_RETURN*100:.2f}% | Live Avg Return: {avg_return:.2f}% | Difference: {avg_return - BACKTEST_AVG_RETURN*100:.2f}%")
    print(f"Backtest Sharpe: {BACKTEST_SHARPE:.2f} | Live Sharpe: {sharpe:.2f} | Difference: {sharpe - BACKTEST_SHARPE:.2f}")
    
    # Add Markov Chain and Decision Tree analysis if we have trades
    if trades and len(trades) >= 5:
        try:
            # Markov Chain Analysis
            markov_probability = analyze_markov_chain(trades)
            print("\n----- MARKOV CHAIN ANALYSIS -----")
            print(f"Predicted Next Trade Win Probability (Markov): {markov_probability:.2f}%")
            
            # Decision Tree Analysis if we have enough trades
            if len(trades) >= 10:
                dt_result = analyze_decision_tree(trades)
                print("\n----- DECISION TREE ANALYSIS -----")
                print(f"Decision Tree Accuracy: {dt_result['accuracy']:.2f}")
                print(f"Decision Tree Precision: {dt_result['precision']:.2f}")
                print(f"Decision Tree Recall: {dt_result['recall']:.2f}")
                print(f"Decision Tree F1 Score: {dt_result['f1']:.2f}")
                print(f"Predicted Next Trade Win Probability (Decision Tree): {dt_result['next_win_probability']:.2f}%")
        except Exception as e:
            print(f"\nWarning: Could not complete advanced analysis: {e}")
            
    # Print a compact summary of key metrics
    print("\n----- SUMMARY -----")
    print(f"Start Value: {initial_balance:.2f}")
    print(f"Final Value: {current_balance:.2f}")
    print(f"Win Rate: {win_rate:.2f}%")
    profit_factor = (win_rate/100 * avg_win) / ((1-win_rate/100) * abs(avg_loss)) if avg_loss != 0 else float('inf')
    print(f"Profit Factor: {profit_factor:.2f}")
    print(f"Average Trade Return: {avg_return:.2f}%")
    return_std = np.std([trade.get('return', 0) for trade in trades]) if trades and len(trades) > 1 else 0
    print(f"Return Std Dev: {return_std:.2f}%")
    print(f"Sharpe Ratio: {sharpe:.2f}")
    
    if trades and len(trades) >= 5:
        try:
            markov_probability = analyze_markov_chain(trades)
            print("\n--- Markov Chain Analysis ---")
            print(f"Predicted Next Trade Win Probability (Markov): {markov_probability:.2f}%")
            
            if len(trades) >= 10:
                dt_result = analyze_decision_tree(trades)
                print("\n--- Decision Tree Analysis ---")
                print(f"Decision Tree Accuracy: {dt_result['accuracy']:.2f}")
                print(f"Decision Tree Precision: {dt_result['precision']:.2f}")
                print(f"Decision Tree Recall: {dt_result['recall']:.2f}")
                print(f"Decision Tree F1 Score: {dt_result['f1']:.2f}")
                print(f"Predicted Next Trade Win Probability (Decision Tree): {dt_result['next_win_probability']:.2f}%")
        except Exception as e:
            print(f"\nWarning: Could not complete advanced analysis summary: {e}")

def parse_trade_logs():
    """Parse the trade log file to extract trade information"""
    log_file = os.path.join(RESULTS_DIR, "trade_log.txt")
    if not os.path.exists(log_file):
        print(f"Warning: Trade log file not found at {log_file}")
        return []
    
    trades = []
    current_trade = None
    
    try:
        with open(log_file, 'r') as file:
            for line in file:
                if "ENTRY" in line:
                    # Parse entry line
                    parts = line.split(" - ")
                    if len(parts) >= 2:
                        entry_time = parts[0]
                        entry_info = parts[1]
                        
                        # Extract price and size
                        price_match = entry_info.split("Price=$")[1].split(",")[0]
                        size_match = entry_info.split("Size=")[1].split(",")[0]
                        stop_match = entry_info.split("Stop=$")[1].split("\n")[0]
                        
                        try:
                            entry_price = float(price_match)
                            position_size = float(size_match)
                            stop_price = float(stop_match)
                            
                            current_trade = {
                                'entry_date': entry_time,
                                'entry_price': entry_price,
                                'position_size': position_size,
                                'stop_price': stop_price
                            }
                        except (ValueError, IndexError):
                            print(f"Error parsing entry line: {line}")
                
                elif "EXIT" in line and current_trade is not None:
                    # Parse exit line
                    parts = line.split(" - ")
                    if len(parts) >= 2:
                        exit_time = parts[0]
                        exit_info = parts[1]
                        
                        # Extract price and return
                        price_match = exit_info.split("Price=$")[1].split(",")[0]
                        return_match = exit_info.split("Return=")[1].split("%")[0]
                        pnl_match = exit_info.split("PnL=$")[1].split("\n")[0]
                        
                        try:
                            exit_price = float(price_match)
                            return_pct = float(return_match) / 100  # Convert to decimal
                            pnl = float(pnl_match)
                            
                            # Complete the trade
                            current_trade['exit_date'] = exit_time
                            current_trade['exit_price'] = exit_price
                            current_trade['return'] = return_pct
                            current_trade['pnl'] = pnl
                            
                            trades.append(current_trade)
                            current_trade = None
                        except (ValueError, IndexError):
                            print(f"Error parsing exit line: {line}")
    
    except Exception as e:
        print(f"Error parsing trade log: {str(e)}")
    
    print(f"Parsed {len(trades)} trades from trade log")
    return trades

def analyze_markov_chain(trades):
    """
    Analyze trades using a Markov Chain to predict next trade probability
    
    Args:
        trades: List of trade dictionaries
        
    Returns:
        float: Probability of next trade being a win (as percentage)
    """
    if len(trades) < 5:
        return 50.0  # Not enough data, return neutral probability
    
    # Extract trade results (1 for win, 0 for loss)
    trade_results = [1 if trade.get('return', 0) > 0 else 0 for trade in trades]
    
    # Count transitions
    win_to_win = 0
    win_to_loss = 0
    loss_to_win = 0
    loss_to_loss = 0
    
    for i in range(len(trade_results) - 1):
        if trade_results[i] == 1 and trade_results[i+1] == 1:
            win_to_win += 1
        elif trade_results[i] == 1 and trade_results[i+1] == 0:
            win_to_loss += 1
        elif trade_results[i] == 0 and trade_results[i+1] == 1:
            loss_to_win += 1
        elif trade_results[i] == 0 and trade_results[i+1] == 0:
            loss_to_loss += 1
    
    # Calculate transition probabilities
    p_win_to_win = win_to_win / (win_to_win + win_to_loss) if (win_to_win + win_to_loss) > 0 else 0.5
    p_loss_to_win = loss_to_win / (loss_to_win + loss_to_loss) if (loss_to_win + loss_to_loss) > 0 else 0.5
    
    # Predict next trade based on last trade
    if len(trade_results) > 0:
        last_trade = trade_results[-1]
        if last_trade == 1:
            next_win_probability = p_win_to_win
        else:
            next_win_probability = p_loss_to_win
    else:
        next_win_probability = 0.5  # No trades, return neutral probability
    
    return next_win_probability * 100  # Return as percentage

def analyze_decision_tree(trades):
    """
    Analyze trades using Decision Tree to predict next trade probability
    
    Args:
        trades: List of trade dictionaries
        
    Returns:
        dict: Dictionary with decision tree metrics and prediction
    """
    if len(trades) < 10:
        return {
            'accuracy': 0,
            'precision': 0,
            'recall': 0,
            'f1': 0,
            'next_win_probability': 50.0
        }
    
    # Create features (X) and target (y)
    X = []
    y = []
    
    try:
        for i in range(3, len(trades)):
            # Features: previous 3 trade results, RSI at entry, trend strength
            features = [
                1 if trades[i-3].get('return', 0) > 0 else 0,
                1 if trades[i-2].get('return', 0) > 0 else 0,
                1 if trades[i-1].get('return', 0) > 0 else 0,
                trades[i-1].get('return', 0),  # Previous trade return
                trades[i-1].get('rsi_at_entry', 50),  # RSI at entry (default 50 if not available)
                trades[i-1].get('trend_strength', 0.01)  # Trend strength (default 0.01 if not available)
            ]
            X.append(features)
            y.append(1 if trades[i].get('return', 0) > 0 else 0)
        
        if len(X) < 5:
            return {
                'accuracy': 0,
                'precision': 0,
                'recall': 0,
                'f1': 0,
                'next_win_probability': 50.0
            }
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        
        # Train decision tree
        clf = DecisionTreeClassifier(random_state=42, max_depth=3)  # Limit depth to prevent overfitting
        clf.fit(X_train, y_train)
        
        # Evaluate model
        y_pred = clf.predict(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        
        # Predict next trade
        last_features = [
            1 if trades[-3].get('return', 0) > 0 else 0,
            1 if trades[-2].get('return', 0) > 0 else 0,
            1 if trades[-1].get('return', 0) > 0 else 0,
            trades[-1].get('return', 0),
            trades[-1].get('rsi_at_entry', 50),
            trades[-1].get('trend_strength', 0.01)
        ]
        
        # Get probability of win for next trade
        try:
            next_win_probability = clf.predict_proba([last_features])[0][1] * 100
        except Exception as e:
            print(f"Warning: Could not predict probability: {e}")
            next_win_probability = 50.0
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'next_win_probability': next_win_probability
        }
    except Exception as e:
        print(f"Error in decision tree analysis: {e}")
        return {
            'accuracy': 0,
            'precision': 0,
            'recall': 0,
            'f1': 0,
            'next_win_probability': 50.0
        }

def print_strategy_parameters():
    """Print the current strategy parameters"""
    print("\n----- STRATEGY PARAMETERS -----")
    print(f"- Long trades: RSI <= 25.0 (regular) or RSI <= 0.1 (ultra-aggressive)")
    print(f"- Short trades: RSI >= 75.0 (regular) or RSI >= 99.9 (ultra-aggressive)")
    print(f"- Stop loss: -500%")
    print(f"- Take profit: 5000%")
    print(f"- Maximum position size: 100.0")
    print(f"- Dynamic leverage: 50.0")

def main():
    """Main function to run the strategy tracker"""
    print("Ultra-Maximized Strategy Performance Tracker")
    
    # Check if tracker file exists
    if os.path.exists(TRACKER_FILE):
        print(f"Tracking file already exists: {TRACKER_FILE}")
    else:
        create_tracker_file()
        # Initialize with initial balance
        with open(TRACKER_FILE, 'a', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow([
                datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                INITIAL_BALANCE,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                'Initial balance'
            ])
        print(f"Initialized tracker with initial balance: {INITIAL_BALANCE}")
    
    while True:
        print("\nOptions:")
        print("1. Update tracker with current balance")
        print("2. Generate performance charts")
        print("3. View performance summary")
        print("4. Parse trade log file")
        print("5. Analyze Markov Chain")
        print("6. Analyze Decision Tree")
        print("7. Reset account statistics")
        print("8. Fix encoding issues")
        print("9. View strategy parameters")
        print("10. Exit")
        
        choice = input("Enter your choice (1-10): ")
        
        if choice == '1':
            try:
                current_balance = float(input("Enter current account balance: "))
                notes = input("Enter any notes about recent performance (optional): ")
                
                # Try to load trades from JSON files first
                trades = load_trades()
                
                # If no trades found in JSON, try to parse from log file
                if not trades:
                    print("No trade files found. Attempting to parse trade log...")
                    trades = parse_trade_logs()
                
                df = update_tracker(current_balance, trades, notes)
                print("Tracker updated successfully!")
                
            except ValueError:
                print("Error: Please enter a valid number for the balance")
        
        elif choice == '2':
            if os.path.exists(TRACKER_FILE):
                for encoding in ['utf-8', 'latin1', 'cp1252', 'iso-8859-1']:
                    try:
                        df = pd.read_csv(TRACKER_FILE, encoding=encoding)
                        generate_performance_charts(df)
                        break
                    except Exception as e:
                        print(f"Failed to read with {encoding} encoding: {e}")
                else:
                    print("Error: Could not read tracker file with any encoding")
            else:
                print("Error: Tracker file not found")
        
        elif choice == '3':
            if os.path.exists(TRACKER_FILE):
                for encoding in ['utf-8', 'latin1', 'cp1252', 'iso-8859-1']:
                    try:
                        df = pd.read_csv(TRACKER_FILE, encoding=encoding)
                        print_performance_summary(df)
                        break
                    except Exception as e:
                        print(f"Failed to read with {encoding} encoding: {e}")
                else:
                    print("Error: Could not read tracker file with any encoding")
            else:
                print("Error: Tracker file not found")
        
        elif choice == '4':
            trades = parse_trade_logs()
            print(f"Parsed {len(trades)} trades from log files")
        
        elif choice == '5':
            trades = parse_trade_logs()
            if len(trades) >= 5:
                probability = analyze_markov_chain(trades)
                print(f"Probability of winning next trade (Markov Chain): {probability:.2f}%")
            else:
                print("Error: Not enough trades for Markov Chain analysis (need at least 5)")
        
        elif choice == '6':
            trades = parse_trade_logs()
            if len(trades) >= 10:
                result = analyze_decision_tree(trades)
                print("\n----- DECISION TREE ANALYSIS -----")
                print(f"Decision Tree Accuracy: {result['accuracy']:.2f}")
                print(f"Decision Tree Precision: {result['precision']:.2f}")
                print(f"Decision Tree Recall: {result['recall']:.2f}")
                print(f"Decision Tree F1 Score: {result['f1']:.2f}")
                print(f"Predicted Next Trade Win Probability: {result['next_win_probability']:.2f}%")
            else:
                print("Error: Not enough trades for Decision Tree analysis (need at least 10)")
        
        elif choice == '7':
            try:
                current_balance = float(input("Enter current account balance: "))
                df = reset_account_statistics(current_balance)
                print("Account statistics reset successfully!")
            except ValueError:
                print("Error: Please enter a valid number for the balance")
        
        elif choice == '8':
            df = fix_encoding_issues()
            print("Encoding issues fixed successfully!")
        
        elif choice == '9':
            print_strategy_parameters()
        
        elif choice == '10':
            print("Exiting...")
            break
        
        else:
            print("Invalid choice. Please enter a number between 1 and 10.")

if __name__ == "__main__":
    main()
