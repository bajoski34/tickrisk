import math
from datetime import date

import pytest

from tickrisk import Equity, EuropeanOption, FixedRateBond, FXForward, MarketSnapshot
from tickrisk import black_scholes as bs

# ---------------- Black-Scholes


def test_textbook_values():
    # Hull, S=100 K=100 T=1 r=5% vol=20%
    assert bs.price("call", 100, 100, 1.0, 0.2, 0.05) == pytest.approx(10.450583572185565, abs=1e-12)
    assert bs.price("put", 100, 100, 1.0, 0.2, 0.05) == pytest.approx(5.573526022256971, abs=1e-12)


@pytest.mark.parametrize("s,k,t,vol,r,q", [(100, 90, 0.5, 0.3, 0.03, 0.01), (50, 70, 2.0, 0.45, 0.06, 0.0)])
def test_put_call_parity(s, k, t, vol, r, q):
    c = bs.price("call", s, k, t, vol, r, q)
    p = bs.price("put", s, k, t, vol, r, q)
    assert c - p == pytest.approx(s * math.exp(-q * t) - k * math.exp(-r * t), abs=1e-10)


@pytest.mark.parametrize("kind", ["call", "put"])
def test_analytic_greeks_match_finite_differences(kind):
    s, k, t, vol, r, q = 105.0, 100.0, 0.75, 0.25, 0.04, 0.015
    g = bs.greeks(kind, s, k, t, vol, r, q)
    h = 1e-4
    f = lambda **kw: bs.price(kind, **{"s": s, "k": k, "t": t, "vol": vol, "r": r, "q": q, **kw})
    assert g.delta == pytest.approx((f(s=s + h) - f(s=s - h)) / (2 * h), rel=1e-7)
    assert g.gamma == pytest.approx((f(s=s + h) - 2 * f() + f(s=s - h)) / h**2, rel=1e-4)
    assert g.vega == pytest.approx((f(vol=vol + h) - f(vol=vol - h)) / (2 * h), rel=1e-7)
    assert g.rho == pytest.approx((f(r=r + h) - f(r=r - h)) / (2 * h), rel=1e-7)
    assert g.theta == pytest.approx(-(f(t=t + h) - f(t=t - h)) / (2 * h), rel=1e-6)


@pytest.mark.parametrize("kind,vol", [("call", 0.05), ("call", 0.8), ("put", 0.2), ("put", 1.5)])
def test_implied_vol_round_trip(kind, vol):
    px = bs.price(kind, 100, 110, 0.5, vol, 0.03, 0.01)
    assert bs.implied_vol(kind, px, 100, 110, 0.5, 0.03, 0.01) == pytest.approx(vol, abs=1e-8)


def test_implied_vol_rejects_arbitrage_price():
    with pytest.raises(ValueError):
        bs.implied_vol("call", 150.0, 100, 100, 1.0, 0.05)


def test_expired_option_is_intrinsic():
    assert bs.price("call", 120, 100, 0.0, 0.3, 0.05) == pytest.approx(20.0)
    assert bs.price("put", 120, 100, -0.1, 0.3, 0.05) == 0.0


def test_bad_inputs():
    with pytest.raises(ValueError):
        bs.price("straddle", 100, 100, 1, 0.2, 0.0)
    with pytest.raises(ValueError):
        bs.price("call", -1, 100, 1, 0.2, 0.0)


# ---------------- instruments


@pytest.fixture
def mkt():
    return MarketSnapshot(
        as_of=date(2026, 1, 2),
        spots={"ABC": 100.0},
        fx={"GBPUSD": 1.25},
        vols={"ABC": 0.2},
        rates={"USD": 0.04, "GBP": 0.05},
    )


def test_equity_pv(mkt):
    assert Equity("ABC", "USD").pv(mkt) == 100.0


def test_option_multiplier(mkt):
    one = EuropeanOption("ABC", 100, date(2027, 1, 2), "call", "USD")
    hundred = EuropeanOption("ABC", 100, date(2027, 1, 2), "call", "USD", multiplier=100)
    assert hundred.pv(mkt) == pytest.approx(100 * one.pv(mkt))
    assert hundred.analytic_greeks(mkt).delta == pytest.approx(100 * one.analytic_greeks(mkt).delta)


def test_bond_pv_and_duration(mkt):
    bond = FixedRateBond(1_000_000, 0.05, date(2031, 1, 2), "GBP", frequency=1)
    flows = bond.cashflows(mkt)
    assert len(flows) == 5
    assert flows[-1][1] == pytest.approx(1_050_000)
    manual = sum(cf * math.exp(-0.05 * t) for t, cf in flows)
    assert bond.pv(mkt) == pytest.approx(manual)
    # dP/dy = -D * P under continuous compounding
    h = 1e-6
    dp = (bond.pv(mkt.bump_rate("GBP", h)) - bond.pv(mkt.bump_rate("GBP", -h))) / (2 * h)
    assert dp == pytest.approx(-bond.duration(mkt) * bond.pv(mkt), rel=1e-6)
    d2p = (bond.pv(mkt.bump_rate("GBP", 1e-4)) - 2 * bond.pv(mkt) + bond.pv(mkt.bump_rate("GBP", -1e-4))) / 1e-8
    assert d2p == pytest.approx(bond.convexity(mkt) * bond.pv(mkt), rel=1e-4)


def test_matured_bond_is_worthless(mkt):
    assert FixedRateBond(100, 0.05, date(2025, 1, 1), "USD").pv(mkt) == 0.0


def test_fx_forward_at_market_is_zero(mkt):
    mat = date(2026, 7, 2)
    probe = FXForward("GBPUSD", 1_000_000, 1.0, mat)
    fwd = probe.forward_rate(mkt)
    assert fwd < 1.25  # USD rate below GBP rate means GBP trades at a forward discount
    assert FXForward("GBPUSD", 1_000_000, fwd, mat).pv(mkt) == pytest.approx(0.0, abs=1e-6)
    assert probe.ccy == "USD"


def test_fx_conversion(mkt):
    assert mkt.fx_rate("GBP", "USD") == 1.25
    assert mkt.fx_rate("USD", "GBP") == pytest.approx(0.8)
    assert mkt.fx_rate("USD", "USD") == 1.0
    with pytest.raises(KeyError):
        mkt.fx_rate("EUR", "USD")


def test_snapshot_is_immutable(mkt):
    bumped = mkt.bump_spot("ABC", 0.1)
    assert mkt.spot("ABC") == 100.0 and bumped.spot("ABC") == pytest.approx(110.0)
    with pytest.raises(TypeError):
        mkt.spots["ABC"] = 1.0  # type: ignore[index]


def test_bond_schedule_rolls_by_calendar_month(mkt):
    bond = FixedRateBond(100, 0.04, date(2031, 1, 31), "USD", frequency=4)
    dates = bond.payment_dates(mkt.as_of)
    assert dates[0] == date(2026, 1, 31) and dates[-1] == date(2031, 1, 31)
    assert date(2026, 4, 30) in dates and date(2026, 7, 31) in dates
    assert len(dates) == 21
    with pytest.raises(ValueError):
        FixedRateBond(100, 0.04, date(2031, 1, 31), "USD", frequency=5).payment_dates(mkt.as_of)
