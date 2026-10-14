import math
from datetime import date
from statistics import NormalDist

import numpy as np
import pytest

from tickrisk import (
    Equity,
    EuropeanOption,
    FixedRateBond,
    MarketSnapshot,
    Portfolio,
    Position,
    Scenario,
    explain_pnl,
    historical_var,
    parametric_var,
    sensitivities,
    stress_test,
)
from tickrisk.sample import sample_market, sample_portfolio, synthetic_history


@pytest.fixture
def mkt():
    return MarketSnapshot(
        as_of=date(2026, 1, 2),
        spots={"ABC": 100.0},
        fx={"GBPUSD": 1.25},
        vols={"ABC": 0.2},
        rates={"USD": 0.04, "GBP": 0.05},
    )


def test_equity_delta_includes_fx_translation(mkt):
    book = Portfolio.of("GBP", [Position("abc", Equity("ABC", "USD"), 1000)])
    s = sensitivities(book, mkt)
    assert s.pv == pytest.approx(1000 * 100 / 1.25)
    assert s.delta["ABC"] == pytest.approx(1000 / 1.25, rel=1e-9)
    assert s.delta_1pct["ABC"] == pytest.approx(800.0, rel=1e-9)
    assert s.gamma["ABC"] == pytest.approx(0.0, abs=1e-4)
    # A stronger pound makes USD assets worth less in GBP.
    assert s.delta_1pct["GBPUSD"] == pytest.approx(-800.0, rel=1e-3)
    assert "ABC" not in s.vega_1pt


def test_option_greeks_match_analytic(mkt):
    opt = EuropeanOption("ABC", 105, date(2026, 7, 2), "call", "USD", multiplier=100)
    book = Portfolio.of("USD", [Position("c", opt, -5)])
    s = sensitivities(book, mkt)
    g = opt.analytic_greeks(mkt)
    assert s.delta["ABC"] == pytest.approx(-5 * g.delta, rel=1e-5)
    assert s.gamma["ABC"] == pytest.approx(-5 * g.gamma, rel=1e-4)
    assert s.vega_1pt["ABC"] == pytest.approx(-5 * g.vega / 100, rel=1e-3)
    assert s.dv01["USD"] == pytest.approx(-5 * g.rho / 1e4, rel=1e-3)
    assert s.theta_1d == pytest.approx(-5 * g.theta / 365, rel=2e-2)


def test_bond_dv01(mkt):
    bond = FixedRateBond(1_000_000, 0.05, date(2031, 1, 2), "GBP")
    book = Portfolio.of("GBP", [Position("b", bond, 1)])
    s = sensitivities(book, mkt)
    assert s.dv01["GBP"] == pytest.approx(-bond.duration(mkt) * bond.pv(mkt) * 1e-4, rel=1e-6)
    assert s.delta == {}


def test_pnl_explain_small_move_is_well_explained():
    m0, book = sample_market(), sample_portfolio()
    m1 = m0.roll(1).bump_spot("AAPL", -0.01).bump_vol("SPY", 0.005).bump_rate("GBP", 0.0002)
    e = explain_pnl(book, m0, m1)
    assert e.actual == pytest.approx(book.pv(m1) - book.pv(m0))
    assert abs(e.unexplained) < 0.02 * abs(e.actual)
    assert e.explained + e.unexplained == pytest.approx(e.actual)


def test_pnl_explain_no_move():
    m0, book = sample_market(), sample_portfolio()
    e = explain_pnl(book, m0, m0)
    assert e.actual == 0.0 and e.theta == 0.0 and e.unexplained == pytest.approx(0.0, abs=1e-6)


def test_pnl_explain_rejects_backwards_dates():
    m0, book = sample_market(), sample_portfolio()
    with pytest.raises(ValueError):
        explain_pnl(book, m0.roll(1), m0)


def test_scenario_wildcards_and_specific_keys(mkt):
    sc = Scenario("t", equity={"*": -0.1}, fx={"GBPUSD": 0.02}, vol={"*": 0.05}, rates={"USD": 0.01})
    m = sc.apply(mkt)
    assert m.spot("ABC") == pytest.approx(90.0)
    assert m.spot("GBPUSD") == pytest.approx(1.275)
    assert m.vol("ABC") == pytest.approx(0.25)
    assert m.rate("USD") == pytest.approx(0.05) and m.rate("GBP") == 0.05


def test_stress_by_position_sums_to_total():
    m, book = sample_market(), sample_portfolio()
    for r in stress_test(book, m):
        assert sum(r.by_position.values()) == pytest.approx(r.pnl)
    assert stress_test(book, m, [Scenario("nothing")])[0].pnl == 0.0


def test_historical_var_linear_book_matches_empirical_quantile(mkt):
    rng = np.random.default_rng(0)
    prices = 100 * np.cumprod(1 + rng.normal(0, 0.01, 1001))
    m = mkt.with_spot("ABC", float(prices[-1]))
    book = Portfolio.of("USD", [Position("abc", Equity("ABC", "USD"), 1000)])
    res = historical_var(book, m, {"ABC": prices}, confidence=0.99)
    pnl = 1000 * prices[-1] * (prices[1:] / prices[:-1] - 1)
    assert res.var == pytest.approx(-np.quantile(pnl, 0.01, method="lower"))
    assert res.expected_shortfall >= res.var > 0
    assert res.n_scenarios == 1000


def test_parametric_var_single_asset(mkt):
    rng = np.random.default_rng(1)
    prices = 100 * np.cumprod(1 + rng.normal(0, 0.02, 501))
    m = mkt.with_spot("ABC", float(prices[-1]))
    book = Portfolio.of("USD", [Position("abc", Equity("ABC", "USD"), 500)])
    res = parametric_var(book, m, {"ABC": prices}, confidence=0.95, horizon_days=10)
    sd = np.std(prices[1:] / prices[:-1] - 1, ddof=1)
    expected = NormalDist().inv_cdf(0.95) * 500 * prices[-1] * sd * math.sqrt(10)
    assert res.var == pytest.approx(expected, rel=1e-6)


def test_option_book_historical_vs_parametric_differ():
    """Convexity shows up as a gap between full revaluation and delta-normal."""
    m, book = sample_market(), sample_portfolio()
    hist = synthetic_history(m)
    h = historical_var(book, m, hist)
    p = parametric_var(book, m, hist)
    assert h.var > 0 and p.var > 0
    assert h.var != pytest.approx(p.var, rel=1e-6)


def test_var_input_validation(mkt):
    book = Portfolio.of("USD", [Position("abc", Equity("ABC", "USD"), 1)])
    with pytest.raises(ValueError):
        historical_var(book, mkt, {"ABC": [100.0]})
    with pytest.raises(ValueError):
        historical_var(book, mkt, {"ABC": [100.0, 101.0]}, confidence=0.3)
    with pytest.raises(ValueError):
        historical_var(book, mkt, {"ABC": [1, 2, 3], "GBPUSD": [1, 2]})


def test_duplicate_position_ids_rejected():
    with pytest.raises(ValueError):
        Portfolio.of("USD", [Position("x", Equity("A", "USD"), 1), Position("x", Equity("B", "USD"), 1)])
