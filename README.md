# tickrisk

**Open-source, auditable pricing and risk for multi-asset books.**

Price a portfolio of equities, options, bonds and FX forwards, then get Greeks, stress tests, Value at Risk and a P&L explain in your reporting currency. Every run is written to a tamper-evident log and can be replayed later to prove the numbers have not changed.

Built for emerging managers, family offices, fintech teams and quants who need institutional-style risk numbers without a multi-year vendor implementation.

```bash
pip install tickrisk
tickrisk demo
```

## Why tickrisk

| Problem | What tickrisk does |
| --- | --- |
| Risk is recomputed overnight in batch | Stream ticks into `RiskEngine` and reprice the book, Greeks and limits on every update |
| "Why did the number move?" | `explain_pnl` splits P&L into theta, delta, gamma, vega and rates, and shows what is left unexplained |
| Auditors and investors ask for proof | Every run stores its full inputs and outputs in a SHA-256 hash-chained log. Edit or delete a record and verification fails |
| Results drift between systems | `replay()` rebuilds any run from its stored inputs and checks the outputs match |
| Duplicate runs clutter reports | Run ids are derived from inputs, so recording the same run twice is a no-op (idempotent) |
| Vendor systems are hard to extend | Instruments only implement `pv()`. Greeks, VaR and stress come for free via full revaluation |

## Quick start

```python
from datetime import date
from tickrisk import (
    MarketSnapshot, Portfolio, Position, Equity, EuropeanOption,
    FixedRateBond, FXForward, RiskEngine, EventLog, Limit, replay,
)

market = MarketSnapshot(
    as_of=date(2026, 10, 14),
    spots={"AAPL": 230.0},
    fx={"GBPUSD": 1.27},             # 1 GBP = 1.27 USD
    vols={"AAPL": 0.27},
    rates={"USD": 0.042, "GBP": 0.041},
)

book = Portfolio.of("GBP", [
    Position("aapl", Equity("AAPL", "USD"), 2_000),
    Position("aapl-call", EuropeanOption("AAPL", 240, date(2026, 12, 18), "call", "USD", multiplier=100), -20),
    Position("gilt", FixedRateBond(1_500_000, 0.04, date(2034, 3, 7), "GBP"), 1),
    Position("hedge", FXForward("GBPUSD", 500_000, 1.265, date(2027, 1, 14)), 1),
])

engine = RiskEngine(book, market, log=EventLog("runs.jsonl"))

print(engine.value().total)              # PV in GBP
greeks = engine.sensitivities()
print(greeks.delta_1pct, greeks.vega_1pt, greeks.dv01, greeks.theta_1d)

for result in engine.stress():           # built-in scenario library
    print(result.scenario, round(result.pnl))

# Every call above was logged. Prove it is reproducible:
for run in engine.log:
    assert replay(engine.log, run.run_id).matches
```

### Streaming with limits

```python
engine = RiskEngine(book, market, limits=[
    Limit("AAPL delta", "delta_1pct", 2_000, "AAPL"),
    Limit("Vega per name", "vega_1pt", 5_000),      # "*" applies to every factor
    Limit("Daily stop", "loss", 25_000),
])
engine.subscribe(lambda u: u.breaches and print(u.seq, [str(b) for b in u.breaches]))

engine.tick("spot:AAPL", 226.4)
engine.tick("vol:AAPL", 0.31)
engine.tick("fx:GBPUSD", 1.262)
engine.tick("rate:USD", 0.0435)
```

Plug `tick()` into any market data feed (websocket, Kafka consumer, broker API). Each update returns PV, tick P&L, total P&L, the full sensitivity set and any limit breaches.

### VaR and P&L explain

```python
var = engine.historical_var(price_history, confidence=0.99)   # full revaluation
dn  = engine.parametric_var(price_history, confidence=0.99)   # delta-normal
print(var.var, var.expected_shortfall, dn.var)

after = market.roll(1).bump_spot("AAPL", -0.03).bump_vol("AAPL", 0.02)
pnl = engine.explain(after)
print(pnl.actual, pnl.delta, pnl.gamma, pnl.vega, pnl.theta, pnl.unexplained)
```

`price_history` is a dict of aligned daily closes per ticker or FX pair. The gap between historical and delta-normal VaR is a quick read on how much option convexity matters for the book.

## Demo output

`tickrisk demo` runs a sample GBP book (US equities, a covered call, a SPY put hedge, US and UK government-style bonds, and a GBPUSD forward) on synthetic data:

```
Sensitivities (GBP)
-------------------
  factor           delta +1%       vega +1pt
  AAPL                 2,182            -589
  MSFT                 4,961               0
  SPY                   -949             853
  GBPUSD              -9,636               0
  DV01 USD              -596
  DV01 GBP            -1,000
  theta 1d               445

1-day 99% VaR (GBP, 500 synthetic days)
---------------------------------------
  historical VaR                24,394    ES 28,710
  delta-normal VaR              23,525    ES 26,951

P&L explain for a 1-day move (GBP)
----------------------------------
  actual                        -8,528
  theta                            445
  delta                         -4,648
  gamma                           -317
  vega                          -1,178
  rates                         -2,979
  unexplained                      149

Audit
-----
  6 runs recorded, hash chain verified, head d170550c4304b8c3...
  replayed 6/6 runs with identical results
```

`tickrisk stream` simulates live ticks against the same book with limit checks:

```
#0  spot:SPY      584.5585  pv  3,939,987  tick   -710  total   -710  BREACH: AAPL delta: delta_1pct[AAPL] = 2,182.04 exceeds 2,150.00
#9  spot:MSFT     418.8027  pv  3,937,412  tick -3,884  total -3,286  BREACH: Daily stop: loss[*] = 3,285.93 exceeds 3,000.00
```

## Audit trail

```bash
tickrisk demo --log runs.jsonl
tickrisk verify runs.jsonl        # checks every hash and link in the chain
tickrisk replay runs.jsonl        # recomputes every run and compares outputs
```

Each line of the log is one run:

```json
{"seq":0,"run_id":"…","kind":"sensitivities","inputs":{"portfolio":{…},"market":{…},"params":{…}},
 "outputs":{…},"prev_hash":"000…","hash":"9f2…","recorded_at":"2026-10-14T20:51:03.123456+00:00"}
```

* **Tamper evident:** each record's hash covers its contents and the previous record's hash. Changing a number or deleting a line breaks `verify()`.
* **Idempotent:** `run_id` is a hash of the run kind and inputs. Recording an identical run returns the existing record.
* **Reproducible:** inputs are stored as canonical JSON, so `replay()` rebuilds exactly the same portfolio and market.

The log proves integrity of the chain you hold. To protect against someone rewriting the whole file, publish the head hash somewhere external (a daily email, a ticket, a notary service).

## What is in the box

| Area | Coverage |
| --- | --- |
| Instruments | Equity, European option (Black-Scholes-Merton with dividends), fixed-rate bullet bond, FX forward |
| Pricing analytics | Closed-form Greeks, implied volatility (Newton with bisection fallback), bond duration and convexity, FX forward rates |
| Book risk | Delta, delta per 1%, gamma, vega per vol point, DV01 per currency, 1-day theta, all in base currency including FX translation |
| Scenarios | Composable `Scenario` shocks with wildcards, plus a starter library (equity crash, pandemic shock, rates up 200bp, sterling crisis, vol crush) |
| VaR | Full-revaluation historical VaR and ES with optional vol and rate histories; delta-normal VaR and ES |
| P&L explain | Theta, delta, gamma, vega and rates attribution with an unexplained residual |
| Streaming | `RiskEngine.tick()` with subscribers and limit checks |
| Audit | Hash-chained JSON Lines log, `verify`, `replay`, idempotent run ids |

Only dependency: NumPy.

## Modelling notes

* Rate curves are flat, continuously compounded zero rates per currency. Bonds use a flat yield (curve plus spread) and ACT/365 year fractions, with coupon dates rolled back from maturity by calendar month.
* Options are European. American exercise, local or stochastic vol and volatility smiles are on the roadmap.
* Multi-day VaR scales one-day shocks by the square root of time.
* Scenario shocks in the starter library are hypothetical. Calibrate your own before relying on them.

This is risk analytics software, not investment advice. Validate it against your own models before using it for decisions.

## Roadmap

- [ ] Yield curves with key-rate DV01s
- [ ] Volatility surfaces and American options
- [ ] Interest rate swaps and FX options
- [ ] Monte Carlo VaR
- [ ] Position loaders for broker and fund admin files (CSV, Interactive Brokers)
- [ ] REST and WebSocket service with a live dashboard
- [ ] Investor-ready PDF risk reports

Ideas and pull requests are welcome. See [CONTRIBUTING.md](CONTRIBUTING.md).

## Development

```bash
git clone https://github.com/bajoski34/tickrisk
cd tickrisk
pip install -e ".[dev]"
pytest
ruff check src tests
```

The test suite checks Black-Scholes against textbook values, put-call parity, analytic Greeks against finite differences, bond duration and convexity against bumped prices, book Greeks against analytic Greeks, VaR against hand-computed quantiles, and that tampering with the log is caught.

## License

Apache 2.0. See [LICENSE](LICENSE).
