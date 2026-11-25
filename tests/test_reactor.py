"""Integration and unit tests for DUNEReactor control and simulation."""

import numpy as np
import pytest

import dune
from dune.reactor import DUNEReactor
from dune.reactorPhysics import qFuel


def test_package_metadata():
    assert dune.__version__ == "0.3.0"
    assert dune.DUNEReactor is DUNEReactor


def test_initial_state_dimensions_and_defaults():
    reactor = DUNEReactor(tstep=0.01)
    assert len(reactor.S) == 20
    assert reactor.storVals.shape[0] == 13
    assert reactor.storVals.shape[1] == len(reactor.time)
    assert reactor.S[9] == 0.0  # Rods start fully inserted
    assert reactor.reactivity < 0.0  # Subcritical at shutdown
    assert np.all(np.isfinite(reactor.S))


def test_time_step_advances_time_and_records_history():
    reactor = DUNEReactor(tstep=0.01)
    t_before = reactor.t[-1]
    reactor.timeStep()
    assert reactor.t[-1] == pytest.approx(t_before + 0.01)
    assert reactor.time[-1] == pytest.approx(reactor.t[-1])
    assert np.all(np.isfinite(reactor.S))


def test_rod_withdrawal_increases_reactivity_and_power():
    reactor = DUNEReactor(tstep=0.01)
    initial_power = qFuel(reactor.S[0])
    reactor.setRodPosition(80.0)
    for _ in range(100):
        reactor.timeStep()
    assert reactor.S[9] > 0.0
    assert qFuel(reactor.S[0]) > initial_power


def test_manual_scram_inserts_rods():
    reactor = DUNEReactor(tstep=0.01)
    reactor.S[9] = 60.0
    reactor.setRodPosition(60.0)
    reactor.SCRAM(True)
    reactor.timeStep()
    assert reactor.scramToggle is True
    assert reactor.S[9] == 0.0
    assert reactor.hrate == 0.0


def test_automatic_scram_on_fuel_and_coolant_overtemp():
    reactor_fuel = DUNEReactor(tstep=0.01)
    reactor_fuel.S[9] = 50.0
    reactor_fuel.S[7] = 1750.0  # Above 1700 K fuel SCRAM limit
    reactor_fuel.timeStep()
    assert reactor_fuel.scramToggle is True
    assert reactor_fuel.S[9] == 0.0

    reactor_coolant = DUNEReactor(tstep=0.01)
    reactor_coolant.S[9] = 50.0
    reactor_coolant.S[8] = 750.0  # Above 700 K coolant SCRAM limit
    reactor_coolant.timeStep()
    assert reactor_coolant.scramToggle is True
    assert reactor_coolant.S[9] == 0.0


def test_pid_power_control_bounds_rod_rate():
    reactor = DUNEReactor(tstep=0.01)
    reactor.togglePwrCtrl(200.0, True)
    assert reactor.pwrCtrl is True
    assert reactor.pwrSet == 200.0
    reactor.timeStep()
    assert abs(reactor.hrate) <= 0.60 + 1e-12


def test_manual_coolant_control_sets_flow_rate():
    reactor = DUNEReactor(tstep=0.01)
    reactor.toggleCoolantCtrl(750.0, True)
    assert reactor.coolantCtrl is True
    assert reactor.coolantSetPoint == pytest.approx(750.0e3)
    assert reactor.mdotC == pytest.approx(750.0e3)


def test_prompt_jump_mode_withdraws_rod_and_clamps_at_100():
    reactor = DUNEReactor(tstep=0.01)
    reactor.S[9] = 50.0
    reactor.togglePromptJumpMode(True)
    assert reactor.S[9] == pytest.approx(56.0)
    assert reactor.hrate == 0.0

    reactor.S[9] = 98.0
    reactor.togglePromptJumpMode(True)
    assert reactor.S[9] == pytest.approx(100.0)
