"""A small illustrative multi-asset book and synthetic market history.

Prices and levels are made up for demonstration. Do not use them for anything real.
"""

from __future__ import annotations

from datetime import date

import numpy as np

from .instruments import Equity, EuropeanOption, FixedRateBond, FXForward
from .market import MarketSnapshot
from .portfolio import Portfolio, Position

AS_OF = date(2026, 10, 14)


def sample_market(as_of: date = AS_OF) -> MarketSnapshot:
    return MarketSnapshot(
        as_of=as_of,
        spots={"AAPL": 230.0, "MSFT": 420.0, "SPY": 580.0},
        fx={"GBPUSD": 1.27},
        vols={"AAPL": 0.27, "MSFT": 0.24, "SPY": 0.17},
        rates={"USD": 0.042, "GBP": 0.041},
        dividends={"AAPL": 0.005, "MSFT": 0.008, "SPY": 0.013},
    )


def sample_portfolio() -> Portfolio:
    """A GBP-reported book: US equities with an overwrite and a tail hedge,
    a US and a UK government-style bond, and a forward hedging part of the USD."""
    return Portfolio.of(
        "GBP",
        [
            Position("AAPL-stock", Equity("AAPL", "USD"), 2_000),
            Position("MSFT-stock", Equity("MSFT", "USD"), 1_500),
            Position(
                "AAPL-Dec26-240C",
                EuropeanOption("AAPL", 240.0, date(2026, 12, 18), "call", "USD", multiplier=100),
                -20,
            ),
            Position(
                "SPY-Mar27-540P",
                EuropeanOption("SPY", 540.0, date(2027, 3, 19), "put", "USD", multiplier=100),
                10,
            ),
            Position("UST-4.25-2031", FixedRateBond(2_000_000, 0.0425, date(2031, 1, 31), "USD"), 1),
            Position("UKT-4.00-2034", FixedRateBond(1_500_000, 0.04, date(2034, 3, 7), "GBP"), 1),
            Position("GBPUSD-fwd-Jan27", FXForward("GBPUSD", 1_500_000, 1.265, date(2027, 1, 14)), 1),
        ],
    )


def synthetic_history(market: MarketSnapshot, days: int = 500, seed: int = 7) -> dict[str, list[float]]:
    """Correlated GBM price paths ending at today's spot, one per spot and FX key."""
    keys = [*market.spots, *market.fx]
    rng = np.random.default_rng(seed)
    daily_vol = np.array([0.018, 0.016, 0.011, 0.006][: len(keys)] + [0.012] * max(0, len(keys) - 4))
    n = len(keys)
    corr = np.full((n, n), 0.55)
    np.fill_diagonal(corr, 1.0)
    for i, k in enumerate(keys):
        if k in market.fx:  # FX loosely tied to equities
            corr[i, :] = corr[:, i] = 0.15
            corr[i, i] = 1.0
    cov = np.outer(daily_vol, daily_vol) * corr
    # Fat tails: Student-t shocks scaled to unit variance.
    z = rng.standard_t(df=4, size=(days, n)) / np.sqrt(2.0)
    rets = z @ np.linalg.cholesky(cov).T
    paths = np.exp(np.vstack([np.zeros(n), np.cumsum(rets, axis=0)]))
    paths = paths / paths[-1] * np.array([market.spot(k) for k in keys])
    return {k: paths[:, i].tolist() for i, k in enumerate(keys)}
