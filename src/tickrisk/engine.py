"""RiskEngine: the stateful front door.

* Runs valuation, Greeks, stress tests, VaR and P&L explain, recording each
  run (inputs and outputs) in an optional hash-chained EventLog.
* Streams: feed it market ticks and it reprices the book, recomputes Greeks,
  checks limits and notifies subscribers.
* `replay()` rebuilds any logged run from its stored inputs and proves the
  numbers are reproducible.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

from .events import EventLog, RunRecord
from .market import MarketSnapshot
from .portfolio import Portfolio, Valuation
from .risk import (
    STANDARD_SCENARIOS,
    PnLExplain,
    Scenario,
    Sensitivities,
    StressResult,
    VaRResult,
    explain_pnl,
    historical_var,
    parametric_var,
    sensitivities,
    stress_test,
)

Metric = Literal["delta_1pct", "vega_1pt", "dv01", "gamma", "loss"]


# ---------------------------------------------------------------- limits


@dataclass(frozen=True)
class Limit:
    """Absolute limit on one risk number.

    metric "loss" checks P&L since the engine started (key is ignored).
    For the others, key is the factor (ticker, FX pair, vol key or currency),
    or "*" to apply the limit to every factor.
    """

    name: str
    metric: Metric
    max_abs: float
    key: str = "*"


@dataclass(frozen=True)
class Breach:
    limit: str
    metric: str
    key: str
    value: float
    max_abs: float

    def __str__(self) -> str:
        return f"{self.limit}: {self.metric}[{self.key}] = {self.value:,.2f} exceeds {self.max_abs:,.2f}"


def check_limits(limits: Sequence[Limit], sens: Sensitivities, pnl: float) -> list[Breach]:
    breaches = []
    for lim in limits:
        if lim.metric == "loss":
            if -pnl > lim.max_abs:
                breaches.append(Breach(lim.name, "loss", "*", -pnl, lim.max_abs))
            continue
        values: Mapping[str, float] = getattr(sens, lim.metric)
        keys = values.keys() if lim.key == "*" else [lim.key]
        for k in keys:
            v = values.get(k, 0.0)
            if abs(v) > lim.max_abs:
                breaches.append(Breach(lim.name, lim.metric, k, v, lim.max_abs))
    return breaches


# ---------------------------------------------------------------- streaming


@dataclass(frozen=True)
class TickUpdate:
    seq: int
    factor: str
    value: float
    pv: float
    pnl_tick: float
    pnl_total: float
    sensitivities: Sensitivities
    breaches: tuple[Breach, ...] = ()
    run_id: str | None = None


# ---------------------------------------------------------------- compute registry
#
# Each entry rebuilds typed inputs from plain JSON and returns a JSON-able dict.
# Live runs and replays both go through these, so they cannot drift apart.


def _pm(inp: Mapping[str, Any]) -> tuple[Portfolio, MarketSnapshot]:
    return Portfolio.from_dict(inp["portfolio"]), MarketSnapshot.from_dict(inp["market"])


def _c_valuation(inp):
    p, m = _pm(inp)
    return p.value(m)


def _c_sensitivities(inp):
    p, m = _pm(inp)
    return sensitivities(p, m)


def _c_stress(inp):
    p, m = _pm(inp)
    return stress_test(p, m, [Scenario.from_dict(s) for s in inp["params"]["scenarios"]])


def _c_hvar(inp):
    p, m = _pm(inp)
    return historical_var(p, m, **inp["params"])


def _c_pvar(inp):
    p, m = _pm(inp)
    return parametric_var(p, m, **inp["params"])


def _c_explain(inp):
    p = Portfolio.from_dict(inp["portfolio"])
    return explain_pnl(p, MarketSnapshot.from_dict(inp["market"]), MarketSnapshot.from_dict(inp["params"]["after"]))


COMPUTE: dict[str, Callable[[Mapping[str, Any]], Any]] = {
    "valuation": _c_valuation,
    "sensitivities": _c_sensitivities,
    "stress": _c_stress,
    "historical_var": _c_hvar,
    "parametric_var": _c_pvar,
    "pnl_explain": _c_explain,
}


def _outputs(result: Any) -> dict:
    if isinstance(result, list):
        return {"results": [r.to_dict() for r in result]}
    return result.to_dict()


# ---------------------------------------------------------------- engine


class RiskEngine:
    def __init__(
        self,
        portfolio: Portfolio,
        market: MarketSnapshot,
        log: EventLog | None = None,
        limits: Sequence[Limit] = (),
        log_ticks: bool = False,
    ) -> None:
        self.portfolio = portfolio
        self.market = market
        self.log = log
        self.limits = tuple(limits)
        self.log_ticks = log_ticks
        self._start_pv = portfolio.pv(market)
        self._last_pv = self._start_pv
        self._seq = 0
        self._subscribers: list[Callable[[TickUpdate], None]] = []
        self.last_run: RunRecord | None = None

    # ----- recorded runs

    def _run(self, kind: str, params: Mapping[str, Any] | None = None, market: MarketSnapshot | None = None):
        inputs = {
            "portfolio": self.portfolio.to_dict(),
            "market": (market or self.market).to_dict(),
            "params": dict(params or {}),
        }
        result = COMPUTE[kind](inputs)
        if self.log is not None:
            self.last_run = self.log.record(kind, inputs, _outputs(result))
        return result

    def value(self) -> Valuation:
        return self._run("valuation")

    def sensitivities(self) -> Sensitivities:
        return self._run("sensitivities")

    def stress(self, scenarios: Sequence[Scenario] = STANDARD_SCENARIOS) -> list[StressResult]:
        return self._run("stress", {"scenarios": [s.to_dict() for s in scenarios]})

    def historical_var(
        self,
        price_history: Mapping[str, Sequence[float]],
        confidence: float = 0.99,
        horizon_days: int = 1,
        vol_history: Mapping[str, Sequence[float]] | None = None,
        rate_history: Mapping[str, Sequence[float]] | None = None,
    ) -> VaRResult:
        params: dict[str, Any] = {
            "price_history": {k: list(map(float, v)) for k, v in price_history.items()},
            "confidence": confidence,
            "horizon_days": horizon_days,
        }
        if vol_history:
            params["vol_history"] = {k: list(map(float, v)) for k, v in vol_history.items()}
        if rate_history:
            params["rate_history"] = {k: list(map(float, v)) for k, v in rate_history.items()}
        return self._run("historical_var", params)

    def parametric_var(
        self, price_history: Mapping[str, Sequence[float]], confidence: float = 0.99, horizon_days: int = 1
    ) -> VaRResult:
        params = {
            "price_history": {k: list(map(float, v)) for k, v in price_history.items()},
            "confidence": confidence,
            "horizon_days": horizon_days,
        }
        return self._run("parametric_var", params)

    def explain(self, after: MarketSnapshot) -> PnLExplain:
        return self._run("pnl_explain", {"after": after.to_dict()})

    # ----- streaming

    def subscribe(self, callback: Callable[[TickUpdate], None]) -> None:
        self._subscribers.append(callback)

    def tick(self, factor: str, value: float) -> TickUpdate:
        """Apply one market update and reprice.

        factor is "spot:AAPL", "fx:GBPUSD", "vol:AAPL" or "rate:USD".
        """
        kind, _, key = factor.partition(":")
        if not key:
            raise ValueError(f"factor must look like 'spot:AAPL', got {factor!r}")
        if not math.isfinite(value):
            raise ValueError("tick value must be finite")
        if kind in ("spot", "fx"):
            self.market = self.market.with_spot(key, value)
        elif kind == "vol":
            self.market = self.market.with_vol(key, value)
        elif kind == "rate":
            self.market = self.market.with_rate(key, value)
        else:
            raise ValueError(f"unknown factor kind {kind!r}")

        if self.log_ticks and self.log is not None:
            sens = self._run("sensitivities")
            run_id = self.last_run.run_id if self.last_run else None
        else:
            sens = sensitivities(self.portfolio, self.market)
            run_id = None

        pnl_total = sens.pv - self._start_pv
        update = TickUpdate(
            seq=self._seq,
            factor=factor,
            value=value,
            pv=sens.pv,
            pnl_tick=sens.pv - self._last_pv,
            pnl_total=pnl_total,
            sensitivities=sens,
            breaches=tuple(check_limits(self.limits, sens, pnl_total)),
            run_id=run_id,
        )
        self._seq += 1
        self._last_pv = sens.pv
        for cb in self._subscribers:
            cb(update)
        return update


# ---------------------------------------------------------------- replay


@dataclass(frozen=True)
class ReplayResult:
    run_id: str
    kind: str
    matches: bool
    differences: tuple[str, ...] = field(default_factory=tuple)


def _diff(a: Any, b: Any, path: str, out: list[str], rel: float, abs_: float) -> None:
    if isinstance(a, Mapping) and isinstance(b, Mapping):
        for k in sorted(set(a) | set(b)):
            if k not in a or k not in b:
                out.append(f"{path}.{k}: present in only one side")
            else:
                _diff(a[k], b[k], f"{path}.{k}", out, rel, abs_)
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            out.append(f"{path}: length {len(a)} != {len(b)}")
        for i, (x, y) in enumerate(zip(a, b, strict=False)):
            _diff(x, y, f"{path}[{i}]", out, rel, abs_)
    elif isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool):
        if not math.isclose(a, b, rel_tol=rel, abs_tol=abs_):
            out.append(f"{path}: stored {a!r} vs replayed {b!r}")
    elif a != b:
        out.append(f"{path}: stored {a!r} vs replayed {b!r}")


def replay(log: EventLog, run_id: str, rel_tol: float = 1e-9, abs_tol: float = 1e-9) -> ReplayResult:
    """Recompute a logged run from its stored inputs and compare to stored outputs."""
    rec = log.get(run_id)
    fresh = _outputs(COMPUTE[rec.kind](rec.inputs))
    diffs: list[str] = []
    _diff(rec.outputs, fresh, "outputs", diffs, rel_tol, abs_tol)
    return ReplayResult(rec.run_id, rec.kind, not diffs, tuple(diffs))
