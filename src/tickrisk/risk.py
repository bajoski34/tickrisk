"""Portfolio risk: sensitivities, P&L attribution, stress tests and Value at Risk.

All measures use full revaluation of the portfolio against bumped or shocked
MarketSnapshots, so they work for any instrument that implements `pv`.
Results are reported in the portfolio's base currency.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from statistics import NormalDist

import numpy as np

from .market import MarketSnapshot
from .portfolio import Portfolio

SPOT_BUMP_REL = 1e-3
VOL_BUMP = 0.01  # one vol point
RATE_BUMP = 1e-4  # one basis point
_EPS = 1e-9


# ---------------------------------------------------------------- sensitivities


@dataclass(frozen=True)
class Sensitivities:
    """Bump-and-revalue Greeks of the whole book, in base currency.

    delta:      dPV/dS per unit move in each spot or FX rate.
    delta_1pct: PV change for a +1% move in each spot or FX rate (delta * S / 100).
    gamma:      d2PV/dS2 per unit move.
    vega_1pt:   PV change for a +1 vol point move (e.g. 20% to 21%).
    dv01:       PV change for a +1bp move in each currency's rate curve.
    theta_1d:   PV change from moving the valuation date forward one calendar day.
    """

    base_ccy: str
    pv: float
    delta: Mapping[str, float]
    delta_1pct: Mapping[str, float]
    gamma: Mapping[str, float]
    vega_1pt: Mapping[str, float]
    dv01: Mapping[str, float]
    theta_1d: float

    def to_dict(self) -> dict:
        return {
            "base_ccy": self.base_ccy,
            "pv": self.pv,
            "delta": dict(self.delta),
            "delta_1pct": dict(self.delta_1pct),
            "gamma": dict(self.gamma),
            "vega_1pt": dict(self.vega_1pt),
            "dv01": dict(self.dv01),
            "theta_1d": self.theta_1d,
        }


def sensitivities(portfolio: Portfolio, market: MarketSnapshot) -> Sensitivities:
    pv0 = portfolio.pv(market)
    delta, delta_1pct, gamma, vega, dv01 = {}, {}, {}, {}, {}

    for key in [*market.spots, *market.fx]:
        s = market.spot(key)
        h = s * SPOT_BUMP_REL
        up = portfolio.pv(market.with_spot(key, s + h))
        dn = portfolio.pv(market.with_spot(key, s - h))
        d = (up - dn) / (2 * h)
        g = (up - 2 * pv0 + dn) / (h * h)
        if abs(up - pv0) > _EPS or abs(dn - pv0) > _EPS:
            delta[key] = d
            delta_1pct[key] = d * s * 0.01
            gamma[key] = g

    for key in market.vols:
        up = portfolio.pv(market.bump_vol(key, VOL_BUMP))
        dn = portfolio.pv(market.bump_vol(key, -VOL_BUMP))
        v = (up - dn) / 2
        if abs(v) > _EPS:
            vega[key] = v

    for ccy in market.rates:
        up = portfolio.pv(market.bump_rate(ccy, RATE_BUMP))
        dn = portfolio.pv(market.bump_rate(ccy, -RATE_BUMP))
        r = (up - dn) / 2
        if abs(r) > _EPS:
            dv01[ccy] = r

    theta = portfolio.pv(market.roll(1)) - pv0
    return Sensitivities(portfolio.base_ccy, pv0, delta, delta_1pct, gamma, vega, dv01, theta)


# ---------------------------------------------------------------- P&L attribution


@dataclass(frozen=True)
class PnLExplain:
    """Splits actual P&L between two snapshots into risk-factor buckets.

    actual = theta + sum(delta) + sum(gamma) + sum(vega) + sum(rates) + unexplained

    A large `unexplained` means the move was too big for a second-order view,
    or there are cross effects (spot-vol, cross-gamma) worth modelling.
    """

    base_ccy: str
    actual: float
    theta: float
    delta: Mapping[str, float]
    gamma: Mapping[str, float]
    vega: Mapping[str, float]
    rates: Mapping[str, float]
    unexplained: float

    @property
    def explained(self) -> float:
        return self.actual - self.unexplained

    def to_dict(self) -> dict:
        return {
            "base_ccy": self.base_ccy,
            "actual": self.actual,
            "theta": self.theta,
            "delta": dict(self.delta),
            "gamma": dict(self.gamma),
            "vega": dict(self.vega),
            "rates": dict(self.rates),
            "unexplained": self.unexplained,
        }


def explain_pnl(portfolio: Portfolio, before: MarketSnapshot, after: MarketSnapshot) -> PnLExplain:
    if after.as_of < before.as_of:
        raise ValueError("'after' snapshot must not be dated before 'before'")
    s = sensitivities(portfolio, before)
    actual = portfolio.pv(after) - s.pv

    days = (after.as_of - before.as_of).days
    theta = portfolio.pv(before.roll(days)) - s.pv if days else 0.0

    d_pnl, g_pnl = {}, {}
    for key, d in s.delta.items():
        ds = after.spot(key) - before.spot(key)
        d_pnl[key] = d * ds
        g_pnl[key] = 0.5 * s.gamma[key] * ds * ds
    v_pnl = {k: v * (after.vol(k) - before.vol(k)) / VOL_BUMP for k, v in s.vega_1pt.items()}
    r_pnl = {c: v * (after.rate(c) - before.rate(c)) / RATE_BUMP for c, v in s.dv01.items()}

    explained = theta + sum(d_pnl.values()) + sum(g_pnl.values()) + sum(v_pnl.values()) + sum(r_pnl.values())
    return PnLExplain(s.base_ccy, actual, theta, d_pnl, g_pnl, v_pnl, r_pnl, actual - explained)


# ---------------------------------------------------------------- stress testing


@dataclass(frozen=True)
class Scenario:
    """A deterministic market shock. Use "*" as a key to hit every factor of that kind.

    equity: relative shocks to equity spots (-0.2 = down 20%).
    fx:     relative shocks to FX pairs (-0.1 = base currency down 10%).
    vol:    absolute shocks to implied vol (0.1 = +10 vol points).
    rates:  absolute shocks to rates (0.01 = +100bp).
    """

    name: str
    equity: Mapping[str, float] = field(default_factory=dict)
    fx: Mapping[str, float] = field(default_factory=dict)
    vol: Mapping[str, float] = field(default_factory=dict)
    rates: Mapping[str, float] = field(default_factory=dict)
    description: str = ""

    @staticmethod
    def _pick(shocks: Mapping[str, float], key: str) -> float:
        return shocks.get(key, shocks.get("*", 0.0))

    def apply(self, market: MarketSnapshot) -> MarketSnapshot:
        m = market
        for k in market.spots:
            if x := self._pick(self.equity, k):
                m = m.bump_spot(k, x)
        for k in market.fx:
            if x := self._pick(self.fx, k):
                m = m.bump_spot(k, x)
        for k in market.vols:
            if x := self._pick(self.vol, k):
                m = m.bump_vol(k, x)
        for c in market.rates:
            if x := self._pick(self.rates, c):
                m = m.bump_rate(c, x)
        return m

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "equity": dict(self.equity),
            "fx": dict(self.fx),
            "vol": dict(self.vol),
            "rates": dict(self.rates),
            "description": self.description,
        }

    @classmethod
    def from_dict(cls, d: Mapping) -> Scenario:
        return cls(**d)


# Hypothetical shocks loosely shaped like well-known episodes. Calibrate your own
# from data before relying on them for decisions.
STANDARD_SCENARIOS: tuple[Scenario, ...] = (
    Scenario(
        "Equity crash",
        equity={"*": -0.30},
        vol={"*": 0.25},
        rates={"*": -0.01},
        description="Broad equity sell-off with a volatility spike and rate cuts.",
    ),
    Scenario(
        "Pandemic shock",
        equity={"*": -0.25},
        vol={"*": 0.30},
        rates={"*": -0.015},
        fx={"*": -0.08},
        description="Fast drawdown, vol explosion, flight to the US dollar.",
    ),
    Scenario(
        "Rates up 200bp",
        rates={"*": 0.02},
        equity={"*": -0.15},
        vol={"*": 0.05},
        description="Inflation-driven tightening hitting bonds and equities together.",
    ),
    Scenario(
        "Sterling crisis",
        fx={"GBPUSD": -0.08},
        rates={"GBP": 0.015},
        description="GBP sells off while gilt yields jump.",
    ),
    Scenario(
        "Vol crush", vol={"*": -0.08}, equity={"*": 0.05}, description="Relief rally with implied vol collapsing."
    ),
)


@dataclass(frozen=True)
class StressResult:
    scenario: str
    pnl: float
    by_position: Mapping[str, float]

    def to_dict(self) -> dict:
        return {"scenario": self.scenario, "pnl": self.pnl, "by_position": dict(self.by_position)}


def stress_test(
    portfolio: Portfolio, market: MarketSnapshot, scenarios: Sequence[Scenario] = STANDARD_SCENARIOS
) -> list[StressResult]:
    base = portfolio.value(market).by_position
    results = []
    for sc in scenarios:
        shocked = portfolio.value(sc.apply(market)).by_position
        by_pos = {pid: shocked[pid] - base[pid] for pid in base}
        results.append(StressResult(sc.name, sum(by_pos.values()), by_pos))
    return results


# ---------------------------------------------------------------- value at risk


@dataclass(frozen=True)
class VaRResult:
    """VaR and Expected Shortfall as positive loss amounts in base currency."""

    method: str
    confidence: float
    horizon_days: int
    var: float
    expected_shortfall: float
    n_scenarios: int
    pnl_vector: tuple[float, ...] = ()

    def to_dict(self) -> dict:
        return {
            "method": self.method,
            "confidence": self.confidence,
            "horizon_days": self.horizon_days,
            "var": self.var,
            "expected_shortfall": self.expected_shortfall,
            "n_scenarios": self.n_scenarios,
        }


def _returns(prices: Sequence[float]) -> np.ndarray:
    p = np.asarray(prices, dtype=float)
    if p.ndim != 1 or len(p) < 2 or np.any(p <= 0):
        raise ValueError("price history needs at least two positive prices")
    return p[1:] / p[:-1] - 1.0


def historical_var(
    portfolio: Portfolio,
    market: MarketSnapshot,
    price_history: Mapping[str, Sequence[float]],
    confidence: float = 0.99,
    horizon_days: int = 1,
    vol_history: Mapping[str, Sequence[float]] | None = None,
    rate_history: Mapping[str, Sequence[float]] | None = None,
) -> VaRResult:
    """Full-revaluation historical simulation.

    price_history: aligned daily closes per spot or FX key (same length for all).
    vol_history / rate_history: optional aligned daily levels; their daily
    changes are applied as absolute shocks.
    Multi-day horizons scale one-day shocks by sqrt(horizon), a standard but
    simplifying assumption.
    """
    if not 0.5 < confidence < 1.0:
        raise ValueError("confidence must be between 0.5 and 1")
    rets = {k: _returns(v) for k, v in price_history.items()}
    vol_ch = {k: np.diff(np.asarray(v, float)) for k, v in (vol_history or {}).items()}
    rate_ch = {k: np.diff(np.asarray(v, float)) for k, v in (rate_history or {}).items()}
    lengths = {len(x) for x in (*rets.values(), *vol_ch.values(), *rate_ch.values())}
    if len(lengths) != 1:
        raise ValueError("all histories must have the same length")
    n = lengths.pop()
    scale = math.sqrt(horizon_days)

    pv0 = portfolio.pv(market)
    pnls = np.empty(n)
    for i in range(n):
        m = market
        for k, r in rets.items():
            m = m.bump_spot(k, r[i] * scale)
        for k, d in vol_ch.items():
            m = m.bump_vol(k, d[i] * scale)
        for c, d in rate_ch.items():
            m = m.bump_rate(c, d[i] * scale)
        pnls[i] = portfolio.pv(m) - pv0

    var = -float(np.quantile(pnls, 1 - confidence, method="lower"))
    tail = pnls[pnls <= -var]
    es = -float(tail.mean()) if tail.size else var
    return VaRResult("historical", confidence, horizon_days, var, es, n, tuple(float(x) for x in pnls))


def parametric_var(
    portfolio: Portfolio,
    market: MarketSnapshot,
    price_history: Mapping[str, Sequence[float]],
    confidence: float = 0.99,
    horizon_days: int = 1,
) -> VaRResult:
    """Delta-normal VaR from cash deltas and the covariance of daily returns.

    Fast and transparent, but ignores option convexity. Compare it with
    historical_var to see how much gamma matters for the book.
    """
    sens = sensitivities(portfolio, market)
    keys = [k for k in price_history if k in sens.delta]
    if not keys:
        return VaRResult("parametric", confidence, horizon_days, 0.0, 0.0, 0)
    exposure = np.array([sens.delta[k] * market.spot(k) for k in keys])
    rets = np.vstack([_returns(price_history[k]) for k in keys])
    cov = np.atleast_2d(np.cov(rets))
    sigma = math.sqrt(max(float(exposure @ cov @ exposure), 0.0)) * math.sqrt(horizon_days)
    z = NormalDist().inv_cdf(confidence)
    var = z * sigma
    es = sigma * NormalDist().pdf(z) / (1 - confidence)
    return VaRResult("parametric", confidence, horizon_days, var, es, rets.shape[1])
