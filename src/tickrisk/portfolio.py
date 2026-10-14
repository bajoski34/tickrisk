"""Positions and portfolio valuation in a single reporting currency."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field

from .instruments import Instrument, instrument_from_dict
from .market import MarketSnapshot


@dataclass(frozen=True)
class Position:
    id: str
    instrument: Instrument
    quantity: float

    def pv(self, market: MarketSnapshot, base_ccy: str) -> float:
        local = self.instrument.pv(market) * self.quantity
        return local * market.fx_rate(self.instrument.ccy, base_ccy)

    def to_dict(self) -> dict:
        return {"id": self.id, "quantity": self.quantity, "instrument": self.instrument.to_dict()}

    @classmethod
    def from_dict(cls, d: Mapping) -> Position:
        return cls(id=d["id"], instrument=instrument_from_dict(d["instrument"]), quantity=float(d["quantity"]))


@dataclass(frozen=True)
class Valuation:
    base_ccy: str
    total: float
    by_position: Mapping[str, float]

    def to_dict(self) -> dict:
        return {"base_ccy": self.base_ccy, "total": self.total, "by_position": dict(self.by_position)}


@dataclass(frozen=True)
class Portfolio:
    """A book of positions reported in `base_ccy`. Immutable; `with_position` returns a copy."""

    base_ccy: str
    positions: tuple[Position, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        object.__setattr__(self, "positions", tuple(self.positions))
        ids = [p.id for p in self.positions]
        if len(ids) != len(set(ids)):
            raise ValueError("position ids must be unique")

    @classmethod
    def of(cls, base_ccy: str, positions: Iterable[Position]) -> Portfolio:
        return cls(base_ccy=base_ccy, positions=tuple(positions))

    def with_position(self, position: Position) -> Portfolio:
        return Portfolio(self.base_ccy, (*self.positions, position))

    def pv(self, market: MarketSnapshot) -> float:
        return sum(p.pv(market, self.base_ccy) for p in self.positions)

    def value(self, market: MarketSnapshot) -> Valuation:
        by_pos = {p.id: p.pv(market, self.base_ccy) for p in self.positions}
        return Valuation(self.base_ccy, sum(by_pos.values()), by_pos)

    def to_dict(self) -> dict:
        return {"base_ccy": self.base_ccy, "positions": [p.to_dict() for p in self.positions]}

    @classmethod
    def from_dict(cls, d: Mapping) -> Portfolio:
        return cls(base_ccy=d["base_ccy"], positions=tuple(Position.from_dict(p) for p in d["positions"]))
