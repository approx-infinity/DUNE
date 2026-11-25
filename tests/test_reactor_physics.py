"""Unit tests for the nuclear reactor physics engine."""

import numpy as np
import pytest

from dune import reactorPhysics as rp


def _fresh_state(
    n=1e8,
    T_fuel=rp.Tin,
    T_coolant=rp.Tin,
    rod=50.0,
    I135=0.0,
    Xe135=0.0,
    Nd149=0.0,
    Pm149=0.0,
    Sm149=0.0,
    N235=rp.N235_0,
    N238=rp.N238_0,
    N239Pu=rp.N239Pu_0,
    NFP=rp.NFP_0,
    burnup=0.0,
):
    precursors = list(rp.beta_i * n / (rp.lambda_i * rp.Lamb))
    return np.array(
        [n]
        + precursors
        + [T_fuel, T_coolant, rod, I135, Xe135, Nd149, Pm149, Sm149, N235, N238, N239Pu, NFP, burnup],
        dtype=float,
    )


def test_delayed_neutron_constants():
    assert len(rp.beta_i) == rp.NUM_GROUPS == 6
    assert len(rp.lambda_i) == rp.NUM_GROUPS == 6
    assert rp.beta == pytest.approx(np.sum(rp.beta_i))
    assert rp.beta > 0.0
    assert rp.Lamb > 0.0
    assert rp.S_source > 0.0


def test_qfuel_scales_linearly_with_neutron_density():
    assert rp.qFuel(0.0) == 0.0
    q1 = rp.qFuel(1.0e8)
    q2 = rp.qFuel(2.0e8)
    assert q1 > 0.0
    assert q2 == pytest.approx(2.0 * q1)


def test_rod_worth_properties():
    assert rp.intRodWorth(50.0, 50.0) == pytest.approx(0.0)
    total_worth = rp.intRodWorth(0.0, 100.0)
    assert total_worth > 0.0
    assert rp.intRodWorth(0.0, 50.0) == pytest.approx(rp.intRodWorth(50.0, 100.0))
    assert rp.diffRodWorth(50.0) > rp.diffRodWorth(10.0)
    assert rp.diffRodWorth(25.0) == pytest.approx(rp.diffRodWorth(75.0))


def test_reactivity_components_and_signs():
    s_inserted = _fresh_state(rod=0.0)
    s_mid = _fresh_state(rod=50.0)
    s_withdrawn = _fresh_state(rod=100.0)

    assert rp.rho(s_inserted, 0.0, 0.0, 0.0) < 0.0
    assert rp.rho(s_withdrawn, 0.0, 0.0, 0.0) > rp.rho(s_mid, 0.0, 0.0, 0.0)
    assert rp.rho(s_mid, 0.0, 0.0, 0.0) > rp.rho(s_inserted, 0.0, 0.0, 0.0)

    # Temperature feedback must be negative
    s_hot = _fresh_state(rod=50.0, T_fuel=rp.Tin + 200.0)
    assert rp.rho(s_hot, 0.0, 0.0, 0.0) < rp.rho(s_mid, 0.0, 0.0, 0.0)

    # Fission product poisons must insert negative reactivity
    s_xe = _fresh_state(rod=50.0, Xe135=1e15)
    s_sm = _fresh_state(rod=50.0, Sm149=1e15)
    assert rp.rho(s_xe, 0.0, 0.0, 0.0) < rp.rho(s_mid, 0.0, 0.0, 0.0)
    assert rp.rho(s_sm, 0.0, 0.0, 0.0) < rp.rho(s_mid, 0.0, 0.0, 0.0)

    # Burnup: U-235 depletion & FP buildup reduce reactivity; Pu-239 adds reactivity
    s_depleted = _fresh_state(rod=50.0, N235=0.9 * rp.N235_0, NFP=1e20)
    s_pu = _fresh_state(rod=50.0, N239Pu=1e19)
    assert rp.rho(s_depleted, 0.0, 0.0, 0.0) < rp.rho(s_mid, 0.0, 0.0, 0.0)
    assert rp.rho(s_pu, 0.0, 0.0, 0.0) > rp.rho(s_mid, 0.0, 0.0, 0.0)


def test_kinetics_and_depletion_derivatives():
    s = _fresh_state(n=1e9)
    dS = rp.reactorSystem(s, 0.0, hrate=0.5, deltaT=0.01, mdotC=500.0e3)
    assert len(dS) == 20
    assert np.all(np.isfinite(dS))

    # Rod movement derivative equals hrate
    assert dS[9] == pytest.approx(0.5)
    # I-135, Xe-135, Nd-149 grow from zero initial concentration
    assert dS[10] > 0.0
    assert dS[11] > 0.0
    assert dS[12] > 0.0
    # U-235 and U-238 deplete while Pu-239, FP, and burnup increase
    assert dS[15] < 0.0
    assert dS[16] < 0.0
    assert dS[17] > 0.0
    assert dS[18] > 0.0
    assert dS[19] > 0.0


def test_thermal_equilibrium_at_zero_power():
    s = _fresh_state(n=0.0, T_fuel=rp.Tin, T_coolant=rp.Tin)
    assert rp.dTfdt(s, 0.0, mdotC=200.0e3) == pytest.approx(0.0)
    assert rp.dTcdt(s, 0.0, mdotC=200.0e3) == pytest.approx(0.0)


def test_neutron_and_precursor_non_negative_clamping():
    s = _fresh_state(n=0.0)
    s[1:7] = -1.0e9
    assert rp.dndt(s, 0.0, reactivity=-1.0) == 0.0
    for group in range(rp.NUM_GROUPS):
        assert rp.dCdt(s, 0.0, group) == pytest.approx(rp.lambda_i[group] * 1.0e9)
