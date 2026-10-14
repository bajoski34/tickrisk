"""Runnable version of the README quick start."""

from datetime import date

from tickrisk import (
    Equity,
    EuropeanOption,
    EventLog,
    FixedRateBond,
    FXForward,
    Limit,
    MarketSnapshot,
    Portfolio,
    Position,
    RiskEngine,
    replay,
)

market = MarketSnapshot(
    as_of=date(2026, 10, 14),
    spots={"AAPL": 230.0},
    fx={"GBPUSD": 1.27},
    vols={"AAPL": 0.27},
    rates={"USD": 0.042, "GBP": 0.041},
)

book = Portfolio.of(
    "GBP",
    [
        Position("aapl", Equity("AAPL", "USD"), 2_000),
        Position("aapl-call", EuropeanOption("AAPL", 240, date(2026, 12, 18), "call", "USD", multiplier=100), -20),
        Position("gilt", FixedRateBond(1_500_000, 0.04, date(2034, 3, 7), "GBP"), 1),
        Position("hedge", FXForward("GBPUSD", 500_000, 1.265, date(2027, 1, 14)), 1),
    ],
)

engine = RiskEngine(
    book,
    market,
    log=EventLog(),
    limits=[Limit("AAPL delta", "delta_1pct", 2_000, "AAPL"), Limit("Daily stop", "loss", 25_000)],
)

print(f"PV: {engine.value().total:,.0f} GBP")
g = engine.sensitivities()
print("delta +1%:", {k: round(v) for k, v in g.delta_1pct.items()})
print("vega +1pt:", {k: round(v) for k, v in g.vega_1pt.items()})
print("DV01:", {k: round(v) for k, v in g.dv01.items()}, "theta 1d:", round(g.theta_1d))
for r in engine.stress():
    print(f"  {r.scenario:<18}{r.pnl:>12,.0f}")

engine.subscribe(lambda u: u.breaches and print(f"tick {u.seq}:", [str(b) for b in u.breaches]))
for factor, value in [("spot:AAPL", 226.4), ("vol:AAPL", 0.31), ("fx:GBPUSD", 1.262), ("rate:USD", 0.0435)]:
    u = engine.tick(factor, value)
    print(f"{factor:<11}{value:>9}  pnl {u.pnl_total:>+10,.0f}")

assert all(replay(engine.log, r.run_id).matches for r in engine.log)
print(f"{len(engine.log)} runs logged and replayed")
