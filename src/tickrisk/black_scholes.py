"""Closed-form Black-Scholes-Merton pricing, Greeks and implied volatility.

Pure functions with no dependencies beyond the standard library, so they are
easy to test against textbook values and to reuse outside the portfolio layer.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

OptionKind = Literal["call", "put"]

_SQRT_2PI = math.sqrt(2.0 * math.pi)


def norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def norm_pdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / _SQRT_2PI


@dataclass(frozen=True)
class BSResult:
    """Price and Greeks for one option on one unit of underlying.

    vega is per 1.00 of volatility (divide by 100 for per vol point).
    theta is per year (divide by 365 for per calendar day).
    rho is per 1.00 of rate (divide by 10,000 for per basis point).
    """

    price: float
    delta: float
    gamma: float
    vega: float
    theta: float
    rho: float


def _check_kind(kind: str) -> None:
    if kind not in ("call", "put"):
        raise ValueError(f"kind must be 'call' or 'put', got {kind!r}")


def price(kind: OptionKind, s: float, k: float, t: float, vol: float, r: float, q: float = 0.0) -> float:
    return greeks(kind, s, k, t, vol, r, q).price


def greeks(kind: OptionKind, s: float, k: float, t: float, vol: float, r: float, q: float = 0.0) -> BSResult:
    """Black-Scholes-Merton with continuous dividend yield q."""
    _check_kind(kind)
    if s <= 0 or k <= 0:
        raise ValueError("spot and strike must be positive")

    if t <= 0.0 or vol <= 0.0:
        # Expired or zero vol: value is discounted intrinsic on the forward.
        t = max(t, 0.0)
        fwd = s * math.exp((r - q) * t)
        df = math.exp(-r * t)
        intrinsic = max(fwd - k, 0.0) if kind == "call" else max(k - fwd, 0.0)
        itm = (fwd > k) if kind == "call" else (fwd < k)
        delta = (math.exp(-q * t) if itm else 0.0) * (1 if kind == "call" else -1)
        return BSResult(price=df * intrinsic, delta=delta, gamma=0.0, vega=0.0, theta=0.0, rho=0.0)

    sqrt_t = math.sqrt(t)
    d1 = (math.log(s / k) + (r - q + 0.5 * vol * vol) * t) / (vol * sqrt_t)
    d2 = d1 - vol * sqrt_t
    dq = math.exp(-q * t)
    dr = math.exp(-r * t)
    pdf_d1 = norm_pdf(d1)

    gamma = dq * pdf_d1 / (s * vol * sqrt_t)
    vega = s * dq * pdf_d1 * sqrt_t
    decay = -s * dq * pdf_d1 * vol / (2.0 * sqrt_t)

    if kind == "call":
        nd1, nd2 = norm_cdf(d1), norm_cdf(d2)
        px = s * dq * nd1 - k * dr * nd2
        delta = dq * nd1
        theta = decay - r * k * dr * nd2 + q * s * dq * nd1
        rho = k * t * dr * nd2
    else:
        nmd1, nmd2 = norm_cdf(-d1), norm_cdf(-d2)
        px = k * dr * nmd2 - s * dq * nmd1
        delta = -dq * nmd1
        theta = decay + r * k * dr * nmd2 - q * s * dq * nmd1
        rho = -k * t * dr * nmd2

    return BSResult(price=px, delta=delta, gamma=gamma, vega=vega, theta=theta, rho=rho)


def implied_vol(
    kind: OptionKind,
    target_price: float,
    s: float,
    k: float,
    t: float,
    r: float,
    q: float = 0.0,
    tol: float = 1e-10,
    max_iter: int = 200,
) -> float:
    """Implied volatility by safeguarded Newton-Raphson with bisection fallback.

    Raises ValueError if the price is outside no-arbitrage bounds.
    """
    _check_kind(kind)
    if t <= 0:
        raise ValueError("cannot imply volatility for an expired option")

    dq, dr = math.exp(-q * t), math.exp(-r * t)
    lower = max(s * dq - k * dr, 0.0) if kind == "call" else max(k * dr - s * dq, 0.0)
    upper = s * dq if kind == "call" else k * dr
    if not (lower - tol <= target_price <= upper + tol):
        raise ValueError(f"price {target_price} outside arbitrage bounds [{lower:.6f}, {upper:.6f}]")

    lo, hi = 1e-6, 5.0
    vol = 0.2
    for _ in range(max_iter):
        res = greeks(kind, s, k, t, vol, r, q)
        diff = res.price - target_price
        if abs(diff) < tol:
            return vol
        if diff > 0:
            hi = vol
        else:
            lo = vol
        step_ok = res.vega > 1e-12
        newton = vol - diff / res.vega if step_ok else None
        vol = newton if (newton is not None and lo < newton < hi) else 0.5 * (lo + hi)
    return vol
