"""Command line interface.

tickrisk demo   [--log runs.jsonl]   full risk report on the sample book
tickrisk stream [--ticks 15] [--seed 1]   simulated live ticks with limit checks
tickrisk verify runs.jsonl           check the hash chain of a run log
tickrisk replay runs.jsonl [RUN_ID]  recompute logged runs and compare
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

from .engine import Limit, RiskEngine, TickUpdate, replay
from .events import EventLog, IntegrityError
from .sample import sample_market, sample_portfolio, synthetic_history


def _money(x: float) -> str:
    return f"{x:>16,.0f}"


def _rule(title: str) -> None:
    print(f"\n{title}\n{'-' * len(title)}")


def cmd_demo(args: argparse.Namespace) -> int:
    log = EventLog(args.log) if args.log else EventLog()
    market, book = sample_market(), sample_portfolio()
    eng = RiskEngine(book, market, log=log)
    ccy = book.base_ccy

    val = eng.value()
    _rule(f"Valuation ({ccy}) as of {market.as_of}")
    for pid, v in val.by_position.items():
        print(f"  {pid:<20}{_money(v)}")
    print(f"  {'TOTAL':<20}{_money(val.total)}")

    s = eng.sensitivities()
    _rule(f"Sensitivities ({ccy})")
    print(f"  {'factor':<10}{'delta +1%':>16}{'vega +1pt':>16}")
    for k in s.delta_1pct:
        print(f"  {k:<10}{_money(s.delta_1pct[k])}{_money(s.vega_1pt.get(k, 0.0))}")
    for c, v in s.dv01.items():
        print(f"  DV01 {c:<5}{_money(v)}")
    print(f"  theta 1d  {_money(s.theta_1d)}")

    _rule(f"Stress tests ({ccy})")
    for r in eng.stress():
        print(f"  {r.scenario:<20}{_money(r.pnl)}")

    hist = synthetic_history(market)
    h = eng.historical_var(hist, confidence=0.99)
    p = eng.parametric_var(hist, confidence=0.99)
    _rule(f"1-day 99% VaR ({ccy}, {h.n_scenarios} synthetic days)")
    print(f"  {'historical VaR':<20}{_money(h.var)}    ES {h.expected_shortfall:,.0f}")
    print(f"  {'delta-normal VaR':<20}{_money(p.var)}    ES {p.expected_shortfall:,.0f}")

    after = market.roll(1)
    after = after.bump_spot("AAPL", -0.03).bump_spot("SPY", -0.02).bump_vol("AAPL", 0.02).bump_rate("USD", 0.0005)
    e = eng.explain(after)
    _rule(f"P&L explain for a 1-day move ({ccy})")
    print(f"  {'actual':<20}{_money(e.actual)}")
    print(f"  {'theta':<20}{_money(e.theta)}")
    print(f"  {'delta':<20}{_money(sum(e.delta.values()))}")
    print(f"  {'gamma':<20}{_money(sum(e.gamma.values()))}")
    print(f"  {'vega':<20}{_money(sum(e.vega.values()))}")
    print(f"  {'rates':<20}{_money(sum(e.rates.values()))}")
    print(f"  {'unexplained':<20}{_money(e.unexplained)}")

    log.verify()
    checks = [replay(log, r.run_id) for r in log]
    _rule("Audit")
    print(f"  {len(log)} runs recorded, hash chain verified, head {log.head[:16]}...")
    print(f"  replayed {sum(c.matches for c in checks)}/{len(checks)} runs with identical results")
    if args.log:
        print(f"  log written to {args.log}")
    return 0


def cmd_stream(args: argparse.Namespace) -> int:
    market, book = sample_market(), sample_portfolio()
    limits = [
        Limit("AAPL delta", "delta_1pct", 2_150, "AAPL"),
        Limit("Vega per name", "vega_1pt", 6_000),
        Limit("Daily stop", "loss", 3_000),
    ]
    eng = RiskEngine(book, market, limits=limits)
    rng = np.random.default_rng(args.seed)

    def show(u: TickUpdate) -> None:
        flag = "  BREACH: " + "; ".join(map(str, u.breaches)) if u.breaches else ""
        d = u.sensitivities.delta_1pct
        print(
            f"#{u.seq:<3}{u.factor:<12}{u.value:>10.4f}  pv {u.pv:>14,.0f}  "
            f"tick {u.pnl_tick:>+10,.0f}  total {u.pnl_total:>+10,.0f}  "
            f"AAPL d1% {d.get('AAPL', 0):>8,.0f}{flag}"
        )

    eng.subscribe(show)
    factors = ["spot:AAPL", "spot:MSFT", "spot:SPY", "fx:GBPUSD", "vol:AAPL", "rate:USD"]
    print(f"Streaming {args.ticks} simulated ticks (book in {book.base_ccy})")
    for _ in range(args.ticks):
        f = factors[rng.integers(len(factors))]
        kind, _, key = f.partition(":")
        m = eng.market
        if kind in ("spot", "fx"):
            new = m.spot(key) * (1 + rng.normal(-0.002, 0.012 if kind == "spot" else 0.004))
        elif kind == "vol":
            new = max(m.vol(key) + rng.normal(0.004, 0.01), 0.01)
        else:
            new = m.rate(key) + rng.normal(0, 0.0004)
        eng.tick(f, float(new))
    return 0


def cmd_verify(args: argparse.Namespace) -> int:
    try:
        log = EventLog(args.path)
    except IntegrityError as e:
        print(f"FAILED: {e}")
        return 1
    print(f"OK: {len(log)} runs, head {log.head}")
    return 0


def cmd_replay(args: argparse.Namespace) -> int:
    log = EventLog(args.path)
    ids = [args.run_id] if args.run_id else [r.run_id for r in log]
    bad = 0
    for rid in ids:
        res = replay(log, rid)
        print(f"{'MATCH' if res.matches else 'DIFF '}  {res.kind:<16}{rid}")
        for d in res.differences[:5]:
            print(f"        {d}")
        bad += not res.matches
    return 1 if bad else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="tickrisk", description="Auditable pricing and risk.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("demo", help="full risk report on the sample book")
    d.add_argument("--log", help="write runs to this JSON Lines file")
    d.set_defaults(func=cmd_demo)

    s = sub.add_parser("stream", help="simulated live ticks with limit checks")
    s.add_argument("--ticks", type=int, default=15)
    s.add_argument("--seed", type=int, default=1)
    s.set_defaults(func=cmd_stream)

    v = sub.add_parser("verify", help="verify a run log's hash chain")
    v.add_argument("path")
    v.set_defaults(func=cmd_verify)

    r = sub.add_parser("replay", help="recompute logged runs and compare")
    r.add_argument("path")
    r.add_argument("run_id", nargs="?")
    r.set_defaults(func=cmd_replay)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
