"""
Options Analysis Helper Functions
==================================

This module contains helper functions for options arbitrage analysis,
including P&L calculation, price extraction, and hedge ratio computation.

Functions:
----------
- plot_portfolio_pnl: Generate P&L charts for arbitrage portfolios
- get_effective_price_cross: Extract effective bid/ask prices
- calculate_hedge_ratio: Calculate hedging ratios for cross-asset pairs
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def get_effective_price_cross(row, asset, option_type, position):
    """
    Get effective transaction price considering bid-ask spread for cross_pairs data.

    Parameters:
    -----------
    row : pd.Series
        Row containing option pricing data from cross_pairs
    asset : str
        'upro' or 'spxu' - which underlying asset
    option_type : str
        'call' or 'put'
    position : str
        'buy' (use ask price) or 'sell' (use bid price)

    Returns:
    --------
    float : Effective price for the transaction
    """
    # Determine suffix based on asset
    suffix = '_x' if asset == 'upro' else '_y'

    # Determine column prefix based on option type
    if option_type == 'call':
        ask_col = f'PA_C{suffix}'
        bid_col = f'PB_C{suffix}'
        market_col = f'Option_Price_C{suffix}'
    else:  # put
        ask_col = f'PA_P{suffix}'
        bid_col = f'PB_P{suffix}'
        market_col = f'Option_Price_P{suffix}'

    ask_price = row.get(ask_col)
    bid_price = row.get(bid_col)
    market_price = row.get(market_col)

    if position == 'buy':
        # Buy at ask price if available, otherwise use market price
        if pd.notna(ask_price):
            return ask_price
        elif pd.notna(market_price):
            return market_price
        else:
            # Use mid price if available
            if pd.notna(bid_price):
                return bid_price
            return None
    else:  # sell
        # Sell at bid price if available, otherwise use market price
        if pd.notna(bid_price):
            return bid_price
        elif pd.notna(market_price):
            return market_price
        else:
            # Use mid price if available
            if pd.notna(ask_price):
                return ask_price
            return None


def plot_portfolio_pnl(row, portfolio_id):
    """
    Plot P&L chart for arbitrage portfolio.

    Strategy:
    - Always BUY the undervalued CALL option (1 unit)
    - SELL X units of the corresponding PUT option to hedge
    - Pair 1: Buy UPRO Call (undervalued) + Sell X SPXU Puts
    - Pair 2: Buy SPXU Call (undervalued) + Sell X UPRO Puts

    X-axis: UPRO price at expiry
    Y-axis: Portfolio profit

    P&L Calculation Steps:
    1. Given UPRO price at expiry
    2. Calculate log return: log(UPRO_final / UPRO_current)
    3. SPXU has inverse relationship: SPXU final price = SPXU_current * exp(-log_return)
    4. Calculate payoff for each leg with respective strikes
    5. Combine with hedging ratio to get total portfolio P&L
    """
    # Extract data
    upro_strike = row['Strike_x']
    spxu_strike = row['Strike_y']
    upro_price = row['Under_Price_x']
    spxu_price = row['Under_Price_y']
    iv_gap = row['iv_gap']
    expiry_date = row['EXP']
    pair_type = row['pair_type']

    # Calculate hedging ratio
    hedge_ratio = calculate_hedge_ratio(upro_strike, spxu_strike)

    # Determine strategy based on pair type
    if pair_type == 'UPRO_CALL_SPXU_PUT':
        upro_sign = 1  # Buy UPRO Call
        # Buy UPRO Call (undervalued), Sell X SPXU Puts
        upro_position = 'buy'
        spxu_position = 'sell'
        upro_option_type = 'call'
        spxu_option_type = 'put'
        upro_units = 1  # Positive for buy
        spxu_units = -hedge_ratio  # Negative for sell
        upro_price_eff = get_effective_price_cross(row, 'upro', 'call', 'buy')  # Ask price
        spxu_price_eff = get_effective_price_cross(row, 'spxu', 'put', 'sell')  # Bid price
        strategy_label = f"Buy 1 UPRO Call / Sell {hedge_ratio:.2f} SPXU Put"
    else:  # SPXU_CALL_UPRO_PUT or SPXU_CALL_UPRO_PUT_REVERSE
        # Buy SPXU Call (undervalued), Sell X UPRO Puts
        upro_sign = -1  # Sell UPRO Put
        upro_position = 'sell'
        spxu_position = 'buy'
        upro_option_type = 'put'
        spxu_option_type = 'call'
        upro_units = -hedge_ratio  # Negative for sell
        spxu_units = 1  # Positive for buy
        upro_price_eff = get_effective_price_cross(row, 'upro', 'put', 'sell')  # Bid price
        spxu_price_eff = get_effective_price_cross(row, 'spxu', 'call', 'buy')  # Ask price
        strategy_label = f"Buy 1 SPXU Call / Sell {hedge_ratio:.2f} UPRO Put"

    # Check if prices are valid
    if upro_price_eff is None or spxu_price_eff is None:
        print(f"Warning: Missing price data for portfolio #{portfolio_id}, skipping...")
        return None

    # Step 1: Generate UPRO price range (±50% from current price)
    upro_price_range = np.linspace(upro_price * 0.5, upro_price * 1.5, 200)

    # Step 2: Calculate SPXU prices with inverse relationship
    # Both are 3x leveraged ETFs: UPRO = 3x SPY, SPXU = -3x SPY
    # If SPY has log return r: UPRO has 3r, SPXU has -3r
    # From UPRO: 3r = log(UPRO_final/UPRO_current), so r = (1/3)*log(UPRO_final/UPRO_current)
    # SPXU log return = -3r = -log(UPRO_final/UPRO_current)
    # Therefore: SPXU_final = SPXU_current * exp(-log(UPRO_final/UPRO_current))
    #                       = SPXU_current * (UPRO_current / UPRO_final)
    spxu_price_range = spxu_price * (upro_price / upro_price_range)

    # Step 3 & 4: Calculate P&L for each scenario
    portfolio_pnl = []
    upro_leg_pnl_1unit = []
    spxu_leg_pnl_1unit = []

    for upro_final, spxu_final in zip(upro_price_range, spxu_price_range):
        # UPRO leg P&L (call or put depending on strategy)
        if upro_option_type == 'call':
            upro_intrinsic = np.maximum(upro_final - upro_strike, 0)
        else:  # put
            upro_intrinsic = np.maximum(upro_strike - upro_final, 0)
        upro_leg_payoff = upro_units * (upro_intrinsic - upro_price_eff)
        upro_1unit = upro_sign * (upro_intrinsic - upro_price_eff)

        # SPXU leg P&L (call or put depending on strategy)
        if spxu_option_type == 'call':
            spxu_intrinsic = np.maximum(spxu_final - spxu_strike, 0)
        else:  # put
            spxu_intrinsic = np.maximum(spxu_strike - spxu_final, 0)
        spxu_leg_payoff = spxu_units * (spxu_intrinsic - spxu_price_eff)
        spxu_1unit = -upro_sign * (spxu_intrinsic - spxu_price_eff)

        # Step 5: Total portfolio P&L
        total_pnl = upro_leg_payoff + spxu_leg_payoff
        portfolio_pnl.append(total_pnl)
        upro_leg_pnl_1unit.append(upro_1unit)
        spxu_leg_pnl_1unit.append(spxu_1unit)

    portfolio_pnl = np.array(portfolio_pnl)
    upro_leg_pnl_1unit = np.array(upro_leg_pnl_1unit)
    spxu_leg_pnl_1unit = np.array(spxu_leg_pnl_1unit)

    # Plot with two subplots
    fig, (ax1) = plt.subplots(1, 1, figsize=(14, 12), height_ratios=[2, 1])

    # ===== Top subplot: P&L Chart =====
    # Individual leg P&L (1 unit each)
    upro_color = 'green' if upro_position == 'buy' else 'red'
    spxu_color = 'green' if spxu_position == 'buy' else 'red'

    ax1.plot(upro_price_range, upro_leg_pnl_1unit, linewidth=2, color=upro_color,
            linestyle='--', alpha=0.7,
            label=f'{upro_position.title()} 1 UPRO {upro_option_type.upper()} P&L')
    ax1.plot(upro_price_range, spxu_leg_pnl_1unit, linewidth=2, color=spxu_color,
            linestyle='--', alpha=0.7,
            label=f'{spxu_position.title()} 1 SPXU {spxu_option_type.upper()} P&L')

    # Portfolio P&L
    ax1.plot(upro_price_range, portfolio_pnl, linewidth=3, color='darkblue',
            label=f'Portfolio P&L ({abs(upro_units):.0f} UPRO + {abs(spxu_units):.2f} SPXU)')

    # Zero line
    ax1.axhline(0, color='black', linestyle='-', linewidth=1, alpha=0.3)
    ax1.axvline(upro_price, color='red', linestyle='--', linewidth=1, alpha=0.5,
               label=f'Current UPRO Price (${upro_price:.2f})')

    # Fill profit/loss areas for portfolio
    ax1.fill_between(upro_price_range, 0, portfolio_pnl,
                     where=(portfolio_pnl >= 0), alpha=0.2, color='green', label='Profit Zone')
    ax1.fill_between(upro_price_range, 0, portfolio_pnl,
                     where=(portfolio_pnl < 0), alpha=0.2, color='red', label='Loss Zone')

    # Formatting for P&L chart
    ax1.set_xlabel('UPRO Price at Expiry ($)', fontsize=13, fontweight='bold')
    ax1.set_ylabel('Portfolio Profit ($)', fontsize=13, fontweight='bold')

    # Get bid/ask prices for display (dynamic based on option types)
    if pair_type == 'UPRO_CALL_SPXU_PUT':
        upro_ask = row.get('PA_C_x')
        upro_bid = row.get('PB_C_x')
        spxu_ask = row.get('PA_P_y')
        spxu_bid = row.get('PB_P_y')
    else:  # SPXU_CALL_UPRO_PUT or SPXU_CALL_UPRO_PUT_REVERSE
        upro_ask = row.get('PA_P_x')
        upro_bid = row.get('PB_P_x')
        spxu_ask = row.get('PA_C_y')
        spxu_bid = row.get('PB_C_y')

    # Build detailed title with both legs
    leg1_info = (f"{upro_position.upper()} {abs(upro_units):.2f} UPRO {upro_option_type.upper()} | "
                 f"K=${upro_strike:.0f} | Exp: {expiry_date} | Bid=${upro_bid:.2f}, Ask=${upro_ask:.2f}")
    leg2_info = (f"{spxu_position.upper()} {abs(spxu_units):.2f} SPXU {spxu_option_type.upper()} | "
                 f"K=${spxu_strike:.0f} | Exp: {expiry_date} | Bid=${spxu_bid:.2f}, Ask=${spxu_ask:.2f}")

    ax1.set_title(
        f'Portfolio #{portfolio_id}: {strategy_label}\n'
        f'Leg 1: {leg1_info}\n'
        f'Leg 2: {leg2_info}',
        fontsize=13, fontweight='bold', pad=20
    )
    ax1.legend(fontsize=10, loc='upper right')
    ax1.grid(True, alpha=0.3, linestyle='--')

    # Add annotations (moved to bottom-left to avoid legend overlap)
    max_profit = portfolio_pnl.max()
    max_loss = portfolio_pnl.min()
    current_idx = np.argmin(np.abs(upro_price_range - upro_price))
    current_pnl = portfolio_pnl[current_idx]

    textstr = (f'Max Profit: ${max_profit:.2f}\n'
               f'Max Loss: ${max_loss:.2f}\n'
               f'Current Position: ${current_pnl:.2f}\n'
               f'UPRO: {upro_position.upper()} {abs(upro_units):.0f} @ ${upro_price_eff:.2f}\n'
               f'SPXU: {spxu_position.upper()} {abs(spxu_units):.2f} @ ${spxu_price_eff:.2f}')
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax1.text(0.02, 0.02, textstr, transform=ax1.transAxes, fontsize=10,
            verticalalignment='bottom', bbox=props)

    # # ===== Bottom subplot: UPRO vs SPXU Price Correlation =====
    # ax2.plot(upro_price_range, spxu_price_range, linewidth=2, color='purple', label='SPXU Price')
    # ax2.axvline(upro_price, color='red', linestyle='--', linewidth=1, alpha=0.5,
    #            label=f'Current UPRO: ${upro_price:.2f}')
    # ax2.axhline(spxu_price, color='blue', linestyle='--', linewidth=1, alpha=0.5,
    #            label=f'Current SPXU: ${spxu_price:.2f}')

    # ax2.set_xlabel('UPRO Price at Expiry ($)', fontsize=12, fontweight='bold')
    # ax2.set_ylabel('SPXU Price ($)', fontsize=12, fontweight='bold')
    # ax2.set_title('UPRO vs SPXU Price Correlation (Inverse Relationship)', fontsize=12, fontweight='bold')
    # ax2.legend(fontsize=10, loc='upper right')
    # ax2.grid(True, alpha=0.3, linestyle='--')

    plt.tight_layout()
    plt.show()

    return {
        'iv_gap': iv_gap,
        'strategy': strategy_label,
        'pair_type': pair_type,
        'upro_units': upro_units,
        'spxu_units': spxu_units,
        'upro_option_type': upro_option_type,
        'spxu_option_type': spxu_option_type,
        'max_profit': max_profit,
        'max_loss': max_loss,
        'upro_price': upro_price_eff,
        'spxu_price': spxu_price_eff,
        'upro_position': upro_position,
        'spxu_position': spxu_position
    }
