"""
Helper functions for leveraged ETF pairs analysis.
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional
from datetime import datetime


def standardize_pair_name(undervalued_ticker: str, undervalued_type: str, undervalued_strike: float,
                         hedge_ticker: str, hedge_type: str, hedge_strike: float,
                         expiry_date: str) -> str:
    """
    Standardize pair name format based on which leg is undervalued.
    Format: 'undervalued_ticker_(C/P)_strike_hedge_ticker_(C/P)_strike_expiredate'

    Args:
        undervalued_ticker: Ticker of undervalued option (e.g., 'UPRO')
        undervalued_type: 'C' for call, 'P' for put
        undervalued_strike: Strike price of undervalued option
        hedge_ticker: Ticker of hedge option (e.g., 'SPXU')
        hedge_type: 'C' for call, 'P' for put
        hedge_strike: Strike price of hedge option
        expiry_date: Expiration date (YYYY-MM-DD or timestamp)

    Returns:
        Standardized pair name string
    """
    # Parse expiry date if it's a string
    if isinstance(expiry_date, str):
        try:
            expiry_dt = pd.to_datetime(expiry_date)
        except:
            expiry_dt = expiry_date
    else:
        expiry_dt = expiry_date

    # Format expiry as YYYYMMDD
    if hasattr(expiry_dt, 'strftime'):
        expiry_str = expiry_dt.strftime('%Y%m%d')
    else:
        expiry_str = str(expiry_date).replace('-', '')[:8]

    # Format strikes as integers if whole numbers, otherwise 1 decimal
    undervalued_strike_str = f"{int(undervalued_strike)}" if undervalued_strike == int(undervalued_strike) else f"{undervalued_strike:.1f}"
    hedge_strike_str = f"{int(hedge_strike)}" if hedge_strike == int(hedge_strike) else f"{hedge_strike:.1f}"

    return f"{undervalued_ticker}_{undervalued_type}_{undervalued_strike_str}_{hedge_ticker}_{hedge_type}_{hedge_strike_str}_{expiry_str}"


def calculate_volatility_decay(S_bull: float, S_bear: float, sigma_underlying: float,
                               T: float, leverage: float = 3.0) -> Tuple[float, float]:
    """
    Model the volatility deterioration of leveraged ETFs.

    For leveraged ETFs, the product S_bull * S_bear inevitably shrinks over time due to
    volatility drag. This function models that shrinkage based on the underlying's volatility.

    Theory: For daily rebalanced 3x leveraged ETFs:
    - Bull return: R_bull = 3 * R_underlying - k * σ²
    - Bear return: R_bear = -3 * R_underlying - k * σ²
    - Product decay: S_bull * S_bear ≈ S0_bull * S0_bear * exp(-9 * σ² * T / 2)

    Args:
        S_bull: Current bull ETF price
        S_bear: Current bear ETF price
        sigma_underlying: Annualized volatility of the underlying index (e.g., 0.20 for 20%)
        T: Time horizon in years
        leverage: Leverage factor (default 3.0 for 3x ETFs)

    Returns:
        Tuple of (decay_factor, erosion_rate) where:
        - decay_factor: Multiplier for S_bull * S_bear product (< 1.0)
        - erosion_rate: Annualized rate of erosion
    """
    # Theoretical decay rate for leveraged product
    # Based on: d(S_bull * S_bear) / dt = -leverage² * σ² * (S_bull * S_bear)
    decay_exponent = -leverage**2 * sigma_underlying**2 * T / 2.0
    decay_factor = np.exp(decay_exponent)

    # Annualized erosion rate
    if T > 0:
        erosion_rate = -(np.log(decay_factor) / T)
    else:
        erosion_rate = 0.0

    return decay_factor, erosion_rate


def adjust_hedge_ratio_for_decay(bull_price: float, bear_price: float,
                                 sigma_underlying: float, T: float,
                                 leverage: float = 3.0) -> float:
    """
    Calculate hedge ratio adjusted for volatility decay.

    Args:
        bull_price: Bull option price
        bear_price: Bear option price
        sigma_underlying: Underlying volatility
        T: Time to expiration in years
        leverage: Leverage factor

    Returns:
        Adjusted hedge ratio
    """
    decay_factor, _ = calculate_volatility_decay(bull_price, bear_price,
                                                 sigma_underlying, T, leverage)

    # Base hedge ratio with decay adjustment
    hedge_ratio = (bull_price * decay_factor) / bear_price

    return hedge_ratio


def calculate_pnl_with_decay(strategy_type: str, bull_strike: float, bear_strike: float,
                             bull_price: float, bear_price: float,
                             long_price: float, short_price: float,
                             bull_underlying: float, bear_underlying: float,
                             sigma_underlying: float, T: float,
                             leverage: float = 3.0) -> Dict[str, float]:
    """
    Calculate P&L projections incorporating volatility decay model.

    Args:
        strategy_type: 'bull_call' or 'bear_call'
        bull_strike: Bull option strike
        bear_strike: Bear option strike
        bull_price: Bull option price
        bear_price: Bear option price
        long_price: Price paid for long leg
        short_price: Price received for short leg
        bull_underlying: Current bull ETF underlying price
        bear_underlying: Current bear ETF underlying price
        sigma_underlying: Underlying index volatility
        T: Time to expiration in years
        leverage: Leverage factor

    Returns:
        Dictionary with P&L projections including decay effects
    """
    # Calculate decay factor
    decay_factor, erosion_rate = calculate_volatility_decay(
        bull_underlying, bear_underlying, sigma_underlying, T, leverage
    )

    net_cost = long_price - short_price

    if strategy_type == 'bull_call':
        # Long bull call, short bear put
        # Adjust for decay - the bull/bear relationship deteriorates

        # Best case: Bull up 20%, Bear down 20% with decay
        bull_up = bull_underlying * 1.20 * np.sqrt(decay_factor)
        bear_down = bear_underlying * 0.80 * np.sqrt(decay_factor)
        best_pnl = max(0, bull_up - bull_strike) - max(0, bear_strike - bear_down) - net_cost

        # Expected case: Bull up 5%, Bear down 5% with decay
        bull_exp = bull_underlying * 1.05 * np.sqrt(decay_factor)
        bear_exp = bear_underlying * 0.95 * np.sqrt(decay_factor)
        expected_pnl = max(0, bull_exp - bull_strike) - max(0, bear_strike - bear_exp) - net_cost

        # Worst case: Bull down 20%, Bear up 20% with decay
        bull_down = bull_underlying * 0.80 * np.sqrt(decay_factor)
        bear_up = bear_underlying * 1.20 * np.sqrt(decay_factor)
        worst_pnl = max(0, bull_down - bull_strike) - max(0, bear_strike - bear_up) - net_cost

        # Mid case (unchanged with decay)
        bull_mid = bull_underlying * np.sqrt(decay_factor)
        bear_mid = bear_underlying * np.sqrt(decay_factor)
        mid_pnl = max(0, bull_mid - bull_strike) - max(0, bear_strike - bear_mid) - net_cost

    else:  # bear_call
        # Long bear call, short bull put

        # Best case: Bear up 20%, Bull down 20% with decay
        bear_up = bear_underlying * 1.20 * np.sqrt(decay_factor)
        bull_down = bull_underlying * 0.80 * np.sqrt(decay_factor)
        best_pnl = max(0, bear_up - bear_strike) - max(0, bull_strike - bull_down) - net_cost

        # Expected case
        bear_exp = bear_underlying * 1.05 * np.sqrt(decay_factor)
        bull_exp = bull_underlying * 0.95 * np.sqrt(decay_factor)
        expected_pnl = max(0, bear_exp - bear_strike) - max(0, bull_strike - bull_exp) - net_cost

        # Worst case
        bear_down = bear_underlying * 0.80 * np.sqrt(decay_factor)
        bull_up = bull_underlying * 1.20 * np.sqrt(decay_factor)
        worst_pnl = max(0, bear_down - bear_strike) - max(0, bull_strike - bull_up) - net_cost

        # Mid case
        bear_mid = bear_underlying * np.sqrt(decay_factor)
        bull_mid = bull_underlying * np.sqrt(decay_factor)
        mid_pnl = max(0, bear_mid - bear_strike) - max(0, bull_strike - bull_mid) - net_cost

    return {
        'net_cost': net_cost,
        'pnl_best': best_pnl,
        'pnl_expected': expected_pnl,
        'pnl_mid': mid_pnl,
        'pnl_worst': worst_pnl,
        'decay_factor': decay_factor,
        'erosion_rate': erosion_rate,
    }


def calculate_breakeven_price(strategy_type: str, bull_strike: float, bear_strike: float,
                              net_cost: float, bear_underlying: float = None) -> float:
    """
    Calculate breakeven underlying price for the bull ETF.

    Args:
        strategy_type: 'bull_call' or 'bear_call'
        bull_strike: Bull option strike
        bear_strike: Bear option strike
        net_cost: Net cost of the spread
        bear_underlying: Current bear ETF price (for approximation)

    Returns:
        Breakeven bull underlying price
    """
    if strategy_type == 'bull_call':
        # Long bull call, short bear put
        # Breakeven when: (Bull - bull_strike) - (bear_strike - Bear) = net_cost
        # Assuming Bear moves inversely: Bull * Bear ≈ constant
        # Simplified: breakeven ≈ bull_strike + net_cost
        breakeven = bull_strike + net_cost
    else:
        # Long bear call, short bull put
        # Breakeven when: (Bear - bear_strike) - (bull_strike - Bull) = net_cost
        # Simplified: breakeven ≈ bull_strike - net_cost
        breakeven = bull_strike - net_cost

    return breakeven


def find_delta_neutral_pair(long_option: dict, short_candidates: list,
                           max_total_contracts: int = 12) -> dict:
    """
    Find the best delta-neutral pairing where sum(num_contracts * strike * sign) ≈ 0.
    This creates a more balanced risk profile for the pair.

    Args:
        long_option: Dict with 'Strike', 'IV', etc for the long position
        short_candidates: List of dicts for potential short positions
        max_total_contracts: Maximum sum(abs(num_contracts)) allowed

    Returns:
        Dict with best pairing info including num_long, num_short, delta_neutrality_score
    """
    if len(short_candidates) == 0:
        return None

    best_pair = None
    best_score = float('inf')  # Lower is better

    long_strike = long_option['Strike']

    for short_opt in short_candidates:
        short_strike = short_opt['Strike']

        # Try different contract ratios
        for num_long in range(1, max_total_contracts):
            for num_short in range(1, max_total_contracts - num_long + 1):
                if num_long + num_short > max_total_contracts:
                    continue

                # Calculate delta neutrality: sum(contracts * strikes * sign)
                # Long position contributes +strike, short contributes -strike
                delta_sum = num_long * long_strike - num_short * short_strike

                # Score: how close to zero + penalty for too many contracts
                delta_score = abs(delta_sum)
                contract_penalty = (num_long + num_short) * 0.1  # Prefer fewer contracts
                total_score = delta_score + contract_penalty

                if total_score < best_score:
                    best_score = total_score
                    best_pair = {
                        'short_option': short_opt,
                        'num_long': num_long,
                        'num_short': num_short,
                        'delta_sum': delta_sum,
                        'delta_neutrality_score': delta_score,
                        'total_contracts': num_long + num_short
                    }

    return best_pair
