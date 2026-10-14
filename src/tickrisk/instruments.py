"""Instruments. Each one knows how to price a single unit off a MarketSnapshot.

Every instrument:
  * has a currency (`ccy`) that its present value is expressed in,
  * implements `pv(market)` for one unit,
  * round-trips through `to_dict()` / `instrument_from_dict()` so runs can be
    stored in the event log and replayed byte-for-byte.

Greeks for the whole book are computed by bump-and-revalue in `risk.py`, so a
new instrument only has to implement `pv`. Options also expose analytic Greeks.
"""

from __future__ import annotations

import calendar
import math
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from datetime import date
from typing import Any, ClassVar

from . import black_scholes as bs
from .market import MarketSnapshot

_REGISTRY: dict[str, Callable[[Mapping[str, Any]], Instrument]] = {}


def _add_months(d: date, months: int) -> date:
    """Shift a date by whole months, clamping to month end (31 Jan - 1m = 31 Dec, 31 Mar - 1m = 28/29 Feb)."""
    idx = d.year * 12 + (d.month - 1) + months
    year, month = divmod(idx, 12)
    month += 1
    return date(year, month, min(d.day, calendar.monthrange(year, month)[1]))


def _register(cls):
    _REGISTRY[cls.TYPE] = cls._from_dict
    return cls


class Instrument:
    TYPE: ClassVar[str] = "instrument"
    ccy: str

    def pv(self, market: MarketSnapshot) -> float:  # pragma: no cover - interface
        raise NotImplementedError

    def to_dict(self) -> dict:
        d = asdict(self)  # type: ignore[call-overload]
        for k, v in list(d.items()):
            if isinstance(v, date):
                d[k] = v.isoformat()
        return {"type": self.TYPE, **d}

    @classmethod
    def _from_dict(cls, d: Mapping[str, Any]) -> Instrument:
        kwargs = {k: v for k, v in d.items() if k != "type"}
        for k, v in kwargs.items():
            if k in ("expiry", "maturity") and isinstance(v, str):
                kwargs[k] = date.fromisoformat(v)
        return cls(**kwargs)  # type: ignore[call-arg]


def instrument_from_dict(d: Mapping[str, Any]) -> Instrument:
    try:
        return _REGISTRY[d["type"]](d)
    except KeyError:
        raise ValueError(f"Unknown instrument type {d.get('type')!r}") from None


@_register
@dataclass(frozen=True)
class Equity(Instrument):
    """A share or index unit. PV is the spot price."""

    TYPE: ClassVar[str] = "equity"
    underlying: str
    ccy: str

    def pv(self, market: MarketSnapshot) -> float:
        return market.spot(self.underlying)


@_register
@dataclass(frozen=True)
class EuropeanOption(Instrument):
    """European call or put priced with Black-Scholes-Merton.

    `multiplier` is the contract size (100 for US listed equity options).
    """

    TYPE: ClassVar[str] = "european_option"
    underlying: str
    strike: float
    expiry: date
    kind: str
    ccy: str
    multiplier: float = 1.0

    def __post_init__(self) -> None:
        if self.kind not in ("call", "put"):
            raise ValueError("kind must be 'call' or 'put'")
        if self.strike <= 0:
            raise ValueError("strike must be positive")

    def _inputs(self, m: MarketSnapshot) -> tuple:
        return (
            self.kind,
            m.spot(self.underlying),
            self.strike,
            m.year_fraction(self.expiry),
            m.vol(self.underlying),
            m.rate(self.ccy),
            m.dividend(self.underlying),
        )

    def pv(self, market: MarketSnapshot) -> float:
        return bs.price(*self._inputs(market)) * self.multiplier

    def analytic_greeks(self, market: MarketSnapshot) -> bs.BSResult:
        g = bs.greeks(*self._inputs(market))
        m = self.multiplier
        return bs.BSResult(g.price * m, g.delta * m, g.gamma * m, g.vega * m, g.theta * m, g.rho * m)


@_register
@dataclass(frozen=True)
class FixedRateBond(Instrument):
    """Bullet fixed-coupon bond priced off the flat currency curve plus a spread.

    PV is the dirty price for the full notional, using continuous discounting.
    Coupons are paid `frequency` times a year, scheduled backwards from maturity.
    """

    TYPE: ClassVar[str] = "fixed_rate_bond"
    notional: float
    coupon: float
    maturity: date
    ccy: str
    frequency: int = 2
    spread: float = 0.0

    def payment_dates(self, after: date) -> list[date]:
        """Coupon dates strictly after `after`, rolled back from maturity in calendar months."""
        if 12 % self.frequency:
            raise ValueError("frequency must divide 12 (1, 2, 3, 4, 6 or 12)")
        months = 12 // self.frequency
        dates, n = [], 0
        while (d := _add_months(self.maturity, -months * n)) > after:
            dates.append(d)
            n += 1
        return dates[::-1]

    def cashflows(self, market: MarketSnapshot) -> list[tuple[float, float]]:
        """(time in years, amount) for every remaining payment."""
        dates = self.payment_dates(market.as_of)
        if not dates:
            return []
        cpn = self.notional * self.coupon / self.frequency
        flows = [(market.year_fraction(d), cpn) for d in dates]
        flows[-1] = (flows[-1][0], cpn + self.notional)
        return flows

    def yield_(self, market: MarketSnapshot) -> float:
        return market.rate(self.ccy) + self.spread

    def pv(self, market: MarketSnapshot) -> float:
        y = self.yield_(market)
        return sum(cf * math.exp(-y * t) for t, cf in self.cashflows(market))

    def duration(self, market: MarketSnapshot) -> float:
        """Macaulay duration in years (equals modified duration under continuous compounding)."""
        y = self.yield_(market)
        flows = self.cashflows(market)
        p = sum(cf * math.exp(-y * t) for t, cf in flows)
        return sum(t * cf * math.exp(-y * t) for t, cf in flows) / p if p else 0.0

    def convexity(self, market: MarketSnapshot) -> float:
        y = self.yield_(market)
        flows = self.cashflows(market)
        p = sum(cf * math.exp(-y * t) for t, cf in flows)
        return sum(t * t * cf * math.exp(-y * t) for t, cf in flows) / p if p else 0.0


@_register
@dataclass(frozen=True)
class FXForward(Instrument):
    """Outright FX forward: buy `notional` of the base currency at `strike`.

    Pair "GBPUSD" with notional 1,000,000 and strike 1.25 means: at maturity
    receive 1,000,000 GBP and pay 1,250,000 USD. Use a negative notional to sell.
    PV is in the quote currency (USD in this example).
    """

    TYPE: ClassVar[str] = "fx_forward"
    pair: str
    notional: float
    strike: float
    maturity: date

    @property
    def base(self) -> str:
        return self.pair[:3]

    @property
    def quote(self) -> str:
        return self.pair[3:]

    @property
    def ccy(self) -> str:  # type: ignore[override]
        return self.quote

    def forward_rate(self, market: MarketSnapshot) -> float:
        t = max(market.year_fraction(self.maturity), 0.0)
        r_base, r_quote = market.rate(self.base), market.rate(self.quote)
        return market.spot(self.pair) * math.exp((r_quote - r_base) * t)

    def pv(self, market: MarketSnapshot) -> float:
        t = market.year_fraction(self.maturity)
        if t < 0:
            return 0.0
        df_quote = math.exp(-market.rate(self.quote) * t)
        return self.notional * (self.forward_rate(market) - self.strike) * df_quote
