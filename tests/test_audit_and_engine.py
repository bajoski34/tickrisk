import json

import pytest

from tickrisk import EventLog, IntegrityError, Limit, MarketSnapshot, Portfolio, RiskEngine, replay
from tickrisk.cli import main
from tickrisk.sample import sample_market, sample_portfolio, synthetic_history


@pytest.fixture
def engine():
    return RiskEngine(sample_portfolio(), sample_market(), log=EventLog())


# ---------------- serialisation


def test_portfolio_and_market_round_trip():
    book, m = sample_portfolio(), sample_market()
    book2 = Portfolio.from_dict(json.loads(json.dumps(book.to_dict())))
    m2 = MarketSnapshot.from_dict(json.loads(json.dumps(m.to_dict())))
    assert book2 == book and m2 == m
    assert book2.pv(m2) == book.pv(m)


# ---------------- event log


def test_identical_runs_are_idempotent(engine):
    engine.value()
    first = engine.last_run
    engine.value()
    assert engine.last_run == first
    assert len(engine.log) == 1


def test_different_inputs_make_new_runs(engine):
    engine.value()
    engine.tick("spot:AAPL", 231.0)
    engine.value()
    assert len(engine.log) == 2
    a, b = list(engine.log)
    assert b.prev_hash == a.hash


def test_every_run_kind_replays_exactly(engine):
    hist = synthetic_history(engine.market, days=120)
    engine.value()
    engine.sensitivities()
    engine.stress()
    engine.historical_var(hist)
    engine.parametric_var(hist)
    engine.explain(engine.market.roll(1).bump_spot("MSFT", 0.02))
    kinds = {r.kind for r in engine.log}
    assert kinds == {"valuation", "sensitivities", "stress", "historical_var", "parametric_var", "pnl_explain"}
    for rec in engine.log:
        res = replay(engine.log, rec.run_id)
        assert res.matches, res.differences


def test_file_log_survives_reopen(tmp_path):
    path = tmp_path / "runs.jsonl"
    eng = RiskEngine(sample_portfolio(), sample_market(), log=EventLog(path))
    eng.value()
    eng.sensitivities()
    reopened = EventLog(path)
    assert len(reopened) == 2 and reopened.head == eng.log.head
    assert all(replay(reopened, r.run_id).matches for r in reopened)


def test_tampering_is_detected(tmp_path):
    path = tmp_path / "runs.jsonl"
    eng = RiskEngine(sample_portfolio(), sample_market(), log=EventLog(path))
    eng.value()
    eng.stress()
    lines = path.read_text().splitlines()
    rec = json.loads(lines[0])
    rec["outputs"]["total"] += 1_000_000  # someone "improves" the numbers
    lines[0] = json.dumps(rec)
    path.write_text("\n".join(lines) + "\n")
    with pytest.raises(IntegrityError, match="modified"):
        EventLog(path)


def test_deleting_a_record_is_detected(tmp_path):
    path = tmp_path / "runs.jsonl"
    eng = RiskEngine(sample_portfolio(), sample_market(), log=EventLog(path))
    eng.value()
    eng.sensitivities()
    eng.stress()
    lines = path.read_text().splitlines()
    path.write_text("\n".join([lines[0], lines[2]]) + "\n")
    with pytest.raises(IntegrityError):
        EventLog(path)


# ---------------- streaming


def test_tick_updates_pv_and_notifies(engine):
    seen = []
    engine.subscribe(seen.append)
    start = engine.portfolio.pv(engine.market)
    u = engine.tick("spot:AAPL", 240.0)
    assert seen == [u]
    assert u.pv == pytest.approx(engine.portfolio.pv(engine.market))
    assert u.pnl_total == pytest.approx(u.pv - start)
    u2 = engine.tick("fx:GBPUSD", 1.28)
    assert u2.pnl_tick == pytest.approx(u2.pv - u.pv)
    assert u2.seq == 1


def test_limits_breach(engine):
    engine.limits = (Limit("tight AAPL", "delta_1pct", 10.0, "AAPL"), Limit("stop", "loss", 1.0))
    u = engine.tick("spot:AAPL", 200.0)
    names = {b.limit for b in u.breaches}
    assert names == {"tight AAPL", "stop"}


def test_logged_ticks_are_replayable(engine):
    engine.log_ticks = True
    u = engine.tick("vol:AAPL", 0.30)
    assert u.run_id and replay(engine.log, u.run_id).matches


@pytest.mark.parametrize("factor", ["AAPL", "price:AAPL", "spot:NOPE"])
def test_bad_ticks_rejected(engine, factor):
    with pytest.raises((ValueError, KeyError)):
        engine.tick(factor, 1.0)


# ---------------- CLI


def test_cli_demo_and_audit_commands(tmp_path, capsys):
    log = tmp_path / "runs.jsonl"
    assert main(["demo", "--log", str(log)]) == 0
    out = capsys.readouterr().out
    assert "replayed 6/6 runs" in out
    assert main(["verify", str(log)]) == 0
    assert main(["replay", str(log)]) == 0
    assert main(["stream", "--ticks", "5"]) == 0
