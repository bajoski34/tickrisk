"""tickrisk: open-source, auditable pricing and risk for multi-asset books."""

from importlib.metadata import PackageNotFoundError, version

from .black_scholes import BSResult, implied_vol
from .engine import Breach, Limit, ReplayResult, RiskEngine, TickUpdate, replay
from .events import EventLog, IntegrityError, RunRecord
from .instruments import Equity, EuropeanOption, FixedRateBond, FXForward, Instrument
from .market import MarketSnapshot
from .portfolio import Portfolio, Position, Valuation
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

try:
    __version__ = version("tickrisk")
except PackageNotFoundError:  # running from a source checkout without installing
    __version__ = "0.0.0+unknown"

__all__ = [
    "STANDARD_SCENARIOS",
    "BSResult",
    "Breach",
    "Equity",
    "EuropeanOption",
    "EventLog",
    "FXForward",
    "FixedRateBond",
    "Instrument",
    "IntegrityError",
    "Limit",
    "MarketSnapshot",
    "PnLExplain",
    "Portfolio",
    "Position",
    "ReplayResult",
    "RiskEngine",
    "RunRecord",
    "Scenario",
    "Sensitivities",
    "StressResult",
    "TickUpdate",
    "VaRResult",
    "Valuation",
    "explain_pnl",
    "historical_var",
    "implied_vol",
    "parametric_var",
    "replay",
    "sensitivities",
    "stress_test",
]
