"""Market data snapshot: the single input every pricing and risk call reads from.

A MarketSnapshot is immutable. Every bump or scenario returns a new snapshot,
which is what makes runs reproducible and replayable from the event log.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from datetime import date, timedelta
from types import MappingProxyType


def _frozen(d: Mapping[str, float] | None) -> Mapping[str, float]:
    return MappingProxyType({k: float(v) for k, v in (d or {}).items()})


@dataclass(frozen=True)
class MarketSnapshot:
    """Point-in-time market state.

    Attributes:
        as_of: valuation date.
        spots: equity and index prices keyed by ticker, e.g. {"AAPL": 230.0}.
        fx: FX rates keyed by 6-letter pair, base then quote, e.g. {"GBPUSD": 1.27}
            meaning 1 GBP = 1.27 USD.
        vols: annualised implied volatilities keyed by underlying, e.g. {"AAPL": 0.25}.
        rates: continuously compounded flat zero rates keyed by currency, e.g. {"USD": 0.045}.
        dividends: continuous dividend yields keyed by underlying.
    """

    as_of: date
    spots: Mapping[str, float] = field(default_factory=dict)
    fx: Mapping[str, float] = field(default_factory=dict)
    vols: Mapping[str, float] = field(default_factory=dict)
    rates: Mapping[str, float] = field(default_factory=dict)
    dividends: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in ("spots", "fx", "vols", "rates", "dividends"):
            object.__setattr__(self, name, _frozen(getattr(self, name)))
        for pair in self.fx:
            if len(pair) != 6:
                raise ValueError(f"FX pair must be 6 letters like 'GBPUSD', got {pair!r}")

    # ---------- lookups ----------

    def spot(self, key: str) -> float:
        if key in self.spots:
            return self.spots[key]
        if key in self.fx:
            return self.fx[key]
        raise KeyError(f"No spot or FX rate for {key!r}")

    def vol(self, key: str) -> float:
        try:
            return self.vols[key]
        except KeyError:
            raise KeyError(f"No volatility for {key!r}") from None

    def rate(self, ccy: str) -> float:
        try:
            return self.rates[ccy]
        except KeyError:
            raise KeyError(f"No rate for currency {ccy!r}") from None

    def dividend(self, key: str) -> float:
        return self.dividends.get(key, 0.0)

    def fx_rate(self, from_ccy: str, to_ccy: str) -> float:
        """Units of to_ccy per 1 unit of from_ccy. Supports direct and inverse pairs."""
        if from_ccy == to_ccy:
            return 1.0
        direct = from_ccy + to_ccy
        if direct in self.fx:
            return self.fx[direct]
        inverse = to_ccy + from_ccy
        if inverse in self.fx:
            return 1.0 / self.fx[inverse]
        raise KeyError(f"No FX rate between {from_ccy} and {to_ccy}")

    def year_fraction(self, to: date) -> float:
        """ACT/365 year fraction from as_of to a date. Negative if in the past."""
        return (to - self.as_of).days / 365.0

    # ---------- bumps (each returns a new snapshot) ----------

    def with_spot(self, key: str, value: float) -> MarketSnapshot:
        if key in self.spots:
            return replace(self, spots={**self.spots, key: value})
        if key in self.fx:
            return replace(self, fx={**self.fx, key: value})
        raise KeyError(f"No spot or FX rate for {key!r}")

    def bump_spot(self, key: str, rel: float) -> MarketSnapshot:
        return self.with_spot(key, self.spot(key) * (1.0 + rel))

    def with_vol(self, key: str, value: float) -> MarketSnapshot:
        if key not in self.vols:
            raise KeyError(f"No volatility for {key!r}")
        return replace(self, vols={**self.vols, key: max(value, 1e-8)})

    def bump_vol(self, key: str, abs_bump: float) -> MarketSnapshot:
        return self.with_vol(key, self.vol(key) + abs_bump)

    def with_rate(self, ccy: str, value: float) -> MarketSnapshot:
        if ccy not in self.rates:
            raise KeyError(f"No rate for currency {ccy!r}")
        return replace(self, rates={**self.rates, ccy: value})

    def bump_rate(self, ccy: str, abs_bump: float) -> MarketSnapshot:
        return self.with_rate(ccy, self.rate(ccy) + abs_bump)

    def roll(self, days: int = 1) -> MarketSnapshot:
        """Same market levels, valuation date moved forward. Used for theta."""
        return replace(self, as_of=self.as_of + timedelta(days=days))

    # ---------- serialisation ----------

    def to_dict(self) -> dict:
        return {
            "as_of": self.as_of.isoformat(),
            "spots": dict(self.spots),
            "fx": dict(self.fx),
            "vols": dict(self.vols),
            "rates": dict(self.rates),
            "dividends": dict(self.dividends),
        }

    @classmethod
    def from_dict(cls, d: Mapping) -> MarketSnapshot:
        return cls(
            as_of=date.fromisoformat(d["as_of"]),
            spots=d.get("spots", {}),
            fx=d.get("fx", {}),
            vols=d.get("vols", {}),
            rates=d.get("rates", {}),
            dividends=d.get("dividends", {}),
        )
