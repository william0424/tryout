"""
Implied Volatility Calculation Functions
This module contains functions to calculate implied volatility for options using the Black-Scholes model.
"""

import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq, newton


def black_scholes_call(S, K, T, r, sigma):
    """
    Calculate the Black-Scholes price for a European call option.

    Parameters:
    -----------
    S : float
        Current stock price
    K : float
        Strike price
    T : float
        Time to expiration (in years)
    r : float
        Risk-free interest rate (annualized)
    sigma : float
        Volatility (annualized)

    Returns:
    --------
    float : Call option price
    """
    if T <= 0:
        return max(S - K, 0)

    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    call_price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return call_price


def black_scholes_put(S, K, T, r, sigma):
    """
    Calculate the Black-Scholes price for a European put option.

    Parameters:
    -----------
    S : float
        Current stock price
    K : float
        Strike price
    T : float
        Time to expiration (in years)
    r : float
        Risk-free interest rate (annualized)
    sigma : float
        Volatility (annualized)

    Returns:
    --------
    float : Put option price
    """
    if T <= 0:
        return max(K - S, 0)

    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    put_price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    return put_price


def vega(S, K, T, r, sigma):
    """
    Calculate the Vega (sensitivity to volatility) of an option.

    Parameters:
    -----------
    S : float
        Current stock price
    K : float
        Strike price
    T : float
        Time to expiration (in years)
    r : float
        Risk-free interest rate (annualized)
    sigma : float
        Volatility (annualized)

    Returns:
    --------
    float : Vega value
    """
    if T <= 0:
        return 0

    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    vega_value = S * norm.pdf(d1) * np.sqrt(T)
    return vega_value


def implied_volatility_call(option_price, S, K, T, r, method='brentq', initial_guess=0.3, max_iter=100, tol=1e-6):
    """
    Calculate implied volatility for a European call option.

    Parameters:
    -----------
    option_price : float
        Market price of the call option
    S : float
        Current stock price
    K : float
        Strike price
    T : float
        Time to expiration (in years)
    r : float
        Risk-free interest rate (annualized)
    method : str, optional
        Numerical method to use ('brentq' or 'newton'). Default is 'brentq'.
    initial_guess : float, optional
        Initial guess for volatility (used for Newton method). Default is 0.3.
    max_iter : int, optional
        Maximum number of iterations. Default is 100.
    tol : float, optional
        Tolerance for convergence. Default is 1e-6.

    Returns:
    --------
    float : Implied volatility, or None if calculation fails
    """
    # Validate inputs
    if T <= 0 or option_price <= 0 or S <= 0 or K <= 0:
        return None

    # Check for intrinsic value violation
    intrinsic_value = max(S - K, 0)
    if option_price < intrinsic_value * 0.99:  # Allow small tolerance
        return None

    # Check upper bound (call can't be worth more than stock)
    if option_price > S:
        return None

    # Check if option is at or very near intrinsic value
    if option_price <= intrinsic_value * 1.01:
        return 0.01  # Return very low volatility for deep ITM options

    try:
        objective = lambda sigma: black_scholes_call(S, K, T, r, sigma) - option_price

        if method == 'brentq':
            # Check if bounds bracket the solution
            sigma_low = 1e-4
            sigma_high = 5.0

            f_low = objective(sigma_low)
            f_high = objective(sigma_high)

            # If bounds don't bracket, try to find better bounds
            if f_low * f_high > 0:
                # Try wider bounds
                sigma_high = 10.0
                f_high = objective(sigma_high)

                if f_low * f_high > 0:
                    # Bounds still don't work, fall back to Newton method
                    method = 'newton'
                else:
                    iv = brentq(objective, sigma_low, sigma_high, maxiter=max_iter, xtol=tol)
                    return iv
            else:
                iv = brentq(objective, sigma_low, sigma_high, maxiter=max_iter, xtol=tol)
                return iv

        if method == 'newton':
            # Newton-Raphson method (faster but less robust)
            fprime = lambda sigma: vega(S, K, T, r, sigma)
            iv = newton(objective, initial_guess, fprime=fprime, maxiter=max_iter, tol=tol)
            return iv

    except Exception as e:
        # Silent failure - return None for bad data
        return None


def implied_volatility_put(option_price, S, K, T, r, method='brentq', initial_guess=0.3, max_iter=100, tol=1e-6):
    """
    Calculate implied volatility for a European put option.

    Parameters:
    -----------
    option_price : float
        Market price of the put option
    S : float
        Current stock price
    K : float
        Strike price
    T : float
        Time to expiration (in years)
    r : float
        Risk-free interest rate (annualized)
    method : str, optional
        Numerical method to use ('brentq' or 'newton'). Default is 'brentq'.
    initial_guess : float, optional
        Initial guess for volatility (used for Newton method). Default is 0.3.
    max_iter : int, optional
        Maximum number of iterations. Default is 100.
    tol : float, optional
        Tolerance for convergence. Default is 1e-6.

    Returns:
    --------
    float : Implied volatility, or None if calculation fails
    """
    # Validate inputs
    if T <= 0 or option_price <= 0 or S <= 0 or K <= 0:
        return None

    # Check for intrinsic value violation
    intrinsic_value = max(K - S, 0)
    if option_price < intrinsic_value * 0.99:  # Allow small tolerance
        return None

    # Check upper bound (put can't be worth more than strike)
    discounted_strike = K * np.exp(-r * T)
    if option_price > discounted_strike:
        return None

    # Check if option is at or very near intrinsic value
    if option_price <= intrinsic_value * 1.01:
        return 0.01  # Return very low volatility for deep ITM options

    try:
        objective = lambda sigma: black_scholes_put(S, K, T, r, sigma) - option_price

        if method == 'brentq':
            # Check if bounds bracket the solution
            sigma_low = 1e-4
            sigma_high = 5.0

            f_low = objective(sigma_low)
            f_high = objective(sigma_high)

            # If bounds don't bracket, try to find better bounds
            if f_low * f_high > 0:
                # Try wider bounds
                sigma_high = 10.0
                f_high = objective(sigma_high)

                if f_low * f_high > 0:
                    # Bounds still don't work, fall back to Newton method
                    method = 'newton'
                else:
                    iv = brentq(objective, sigma_low, sigma_high, maxiter=max_iter, xtol=tol)
                    return iv
            else:
                iv = brentq(objective, sigma_low, sigma_high, maxiter=max_iter, xtol=tol)
                return iv

        if method == 'newton':
            # Newton-Raphson method (faster but less robust)
            fprime = lambda sigma: vega(S, K, T, r, sigma)
            iv = newton(objective, initial_guess, fprime=fprime, maxiter=max_iter, tol=tol)
            return iv

    except Exception as e:
        # Silent failure - return None for bad data
        return None


def implied_risk_free_rate_call(option_price, S, K, T, sigma, method='brentq', initial_guess=0.05, max_iter=100, tol=1e-6):
    """
    Calculate implied risk-free rate for a European call option.

    Parameters:
    -----------
    option_price : float
        Market price of the call option
    S : float
        Current stock price
    K : float
        Strike price
    T : float
        Time to expiration (in years)
    sigma : float
        Volatility (annualized)
    method : str, optional
        Numerical method to use ('brentq' or 'newton'). Default is 'brentq'.
    initial_guess : float, optional
        Initial guess for risk-free rate (used for Newton method). Default is 0.05.
    max_iter : int, optional
        Maximum number of iterations. Default is 100.
    tol : float, optional
        Tolerance for convergence. Default is 1e-6.

    Returns:
    --------
    float : Implied risk-free rate, or None if calculation fails
    """
    # Validate inputs
    if T <= 0 or option_price <= 0 or S <= 0 or K <= 0 or sigma <= 0:
        return None

    # Check for intrinsic value violation
    intrinsic_value = max(S - K, 0)
    if option_price < intrinsic_value * 0.99:
        return None

    try:
        objective = lambda r: black_scholes_call(S, K, T, r, sigma) - option_price

        if method == 'brentq':
            # Use reasonable bounds for risk-free rate (-0.1 to 1.0, i.e., -10% to 100%)
            r_low = -0.1
            r_high = 1.0

            f_low = objective(r_low)
            f_high = objective(r_high)

            # If bounds don't bracket, try to find better bounds
            if f_low * f_high > 0:
                # Try wider bounds
                r_high = 2.0
                f_high = objective(r_high)

                if f_low * f_high > 0:
                    # Bounds still don't work, fall back to Newton method
                    method = 'newton'
                else:
                    r = brentq(objective, r_low, r_high, maxiter=max_iter, xtol=tol)
                    return r
            else:
                r = brentq(objective, r_low, r_high, maxiter=max_iter, xtol=tol)
                return r

        if method == 'newton':
            # Newton-Raphson method
            # Derivative of call price with respect to r
            def fprime(r):
                return -K * T * np.exp(-r * T) * norm.cdf((np.log(S / K) + (r - 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T)))

            r = newton(objective, initial_guess, fprime=fprime, maxiter=max_iter, tol=tol)
            return r

    except Exception as e:
        # Silent failure - return None for bad data
        return None


def implied_risk_free_rate_put(option_price, S, K, T, sigma, method='brentq', initial_guess=0.05, max_iter=100, tol=1e-6):
    """
    Calculate implied risk-free rate for a European put option.

    Parameters:
    -----------
    option_price : float
        Market price of the put option
    S : float
        Current stock price
    K : float
        Strike price
    T : float
        Time to expiration (in years)
    sigma : float
        Volatility (annualized)
    method : str, optional
        Numerical method to use ('brentq' or 'newton'). Default is 'brentq'.
    initial_guess : float, optional
        Initial guess for risk-free rate (used for Newton method). Default is 0.05.
    max_iter : int, optional
        Maximum number of iterations. Default is 100.
    tol : float, optional
        Tolerance for convergence. Default is 1e-6.

    Returns:
    --------
    float : Implied risk-free rate, or None if calculation fails
    """
    # Validate inputs
    if T <= 0 or option_price <= 0 or S <= 0 or K <= 0 or sigma <= 0:
        return None

    # Check for intrinsic value violation
    intrinsic_value = max(K - S, 0)
    if option_price < intrinsic_value * 0.99:
        return None

    try:
        objective = lambda r: black_scholes_put(S, K, T, r, sigma) - option_price

        if method == 'brentq':
            # Use reasonable bounds for risk-free rate (-0.1 to 1.0, i.e., -10% to 100%)
            r_low = -0.1
            r_high = 1.0

            f_low = objective(r_low)
            f_high = objective(r_high)

            # If bounds don't bracket, try to find better bounds
            if f_low * f_high > 0:
                # Try wider bounds
                r_high = 2.0
                f_high = objective(r_high)

                if f_low * f_high > 0:
                    # Bounds still don't work, fall back to Newton method
                    method = 'newton'
                else:
                    r = brentq(objective, r_low, r_high, maxiter=max_iter, xtol=tol)
                    return r
            else:
                r = brentq(objective, r_low, r_high, maxiter=max_iter, xtol=tol)
                return r

        if method == 'newton':
            # Newton-Raphson method
            # Derivative of put price with respect to r
            def fprime(r):
                return K * T * np.exp(-r * T) * norm.cdf(-(np.log(S / K) + (r - 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T)))

            r = newton(objective, initial_guess, fprime=fprime, maxiter=max_iter, tol=tol)
            return r

    except Exception as e:
        # Silent failure - return None for bad data
        return None


def implied_risk_free_rate(option_price, S, K, T, sigma, option_type='call', method='brentq', initial_guess=0.05, max_iter=100, tol=1e-6):
    """
    Unified wrapper function to calculate implied risk-free rate for both call and put options.

    Parameters:
    -----------
    option_price : float
        Market price of the option
    S : float
        Current stock price
    K : float
        Strike price
    T : float
        Time to expiration (in years)
    sigma : float
        Volatility (annualized)
    option_type : str, optional
        Type of option ('call' or 'put'). Default is 'call'.
    method : str, optional
        Numerical method to use ('brentq' or 'newton'). Default is 'brentq'.
    initial_guess : float, optional
        Initial guess for risk-free rate (used for Newton method). Default is 0.05.
    max_iter : int, optional
        Maximum number of iterations. Default is 100.
    tol : float, optional
        Tolerance for convergence. Default is 1e-6.

    Returns:
    --------
    float : Implied risk-free rate, or None if calculation fails
    """
    if option_type.upper() == 'CALL':
        return implied_risk_free_rate_call(option_price, S, K, T, sigma, method, initial_guess, max_iter, tol)
    elif option_type.upper() == 'PUT':
        return implied_risk_free_rate_put(option_price, S, K, T, sigma, method, initial_guess, max_iter, tol)
    else:
        raise ValueError("option_type must be 'call' or 'put'")


def black_scholes_price(S, K, T, r, sigma, option_type='call'):
    """
    Unified wrapper function to calculate Black-Scholes option price for both calls and puts.

    Parameters:
    -----------
    S : float
        Current stock price
    K : float
        Strike price
    T : float
        Time to expiration (in years)
    r : float
        Risk-free interest rate (annualized)
    sigma : float
        Volatility (annualized)
    option_type : str, optional
        Type of option ('call' or 'put'). Default is 'call'.

    Returns:
    --------
    float : Option price
    """
    if option_type.upper() == 'CALL':
        return black_scholes_call(S, K, T, r, sigma)
    elif option_type.upper() == 'PUT':
        return black_scholes_put(S, K, T, r, sigma)
    else:
        raise ValueError("option_type must be 'call' or 'put'")


def implied_volatility(option_price, S, K, T, r, option_type='call', method='brentq', initial_guess=0.3, max_iter=100, tol=1e-6):
    """
    Unified wrapper function to calculate implied volatility for both call and put options.

    Parameters:
    -----------
    option_price : float
        Market price of the option
    S : float
        Current stock price
    K : float
        Strike price
    T : float
        Time to expiration (in years)
    r : float
        Risk-free interest rate (annualized)
    option_type : str, optional
        Type of option ('call' or 'put'). Default is 'call'.
    method : str, optional
        Numerical method to use ('brentq' or 'newton'). Default is 'brentq'.
    initial_guess : float, optional
        Initial guess for volatility (used for Newton method). Default is 0.3.
    max_iter : int, optional
        Maximum number of iterations. Default is 100.
    tol : float, optional
        Tolerance for convergence. Default is 1e-6.

    Returns:
    --------
    float : Implied volatility, or None if calculation fails
    """
    if option_type.upper() == 'CALL':
        return implied_volatility_call(option_price, S, K, T, r, method, initial_guess, max_iter, tol)
    elif option_type.upper() == 'PUT':
        return implied_volatility_put(option_price, S, K, T, r, method, initial_guess, max_iter, tol)
    else:
        raise ValueError("option_type must be 'call' or 'put'")


# Example usage
if __name__ == "__main__":
    # Example parameters
    S = 100  # Current stock price
    K = 105  # Strike price
    T = 0.25  # Time to expiration (3 months)
    r = 0.05  # Risk-free rate (5%)
    sigma = 0.2  # Volatility (20%)

    # Calculate option price using wrapper functions
    call_price = black_scholes_price(S, K, T, r, sigma, option_type='call')
    put_price = black_scholes_price(S, K, T, r, sigma, option_type='put')

    print(f"Call option price: ${call_price:.2f}")
    print(f"Put option price: ${put_price:.2f}")

    # Calculate implied volatility using wrapper function
    iv_call = implied_volatility(call_price, S, K, T, r, option_type='call')
    iv_put = implied_volatility(put_price, S, K, T, r, option_type='put')

    print(f"\nImplied volatility (call): {iv_call:.4f}")
    print(f"Implied volatility (put): {iv_put:.4f}")
    print(f"Original volatility: {sigma:.4f}")

    # Calculate implied risk-free rate using wrapper function
    ir_call = implied_risk_free_rate(call_price, S, K, T, sigma, option_type='call')
    ir_put = implied_risk_free_rate(put_price, S, K, T, sigma, option_type='put')

    print(f"\nImplied risk-free rate (call): {ir_call:.4f}")
    print(f"Implied risk-free rate (put): {ir_put:.4f}")
    print(f"Original risk-free rate: {r:.4f}")
