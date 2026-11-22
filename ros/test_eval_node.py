"""Offline checks for the P5 scorer, grid and trial lifecycle."""
import json

import numpy as np
import pytest

from eval_node import TrialScore, parse_args, score_touch, summarize, trial_grid, wilson
from perch_core import ActionScale, DroneState, Inputs, Phase, PhaseCfg, PhaseMachine


def diag(t, phase, *, armed=True, visible=True):
    return {'kind': 'diag', 'sim_t': t, 'diag': {'t': t, 'phase': phase, 'armed': armed, 'tag_visible': visible}}


def touch(**changes):
    return {'kind': 'touch', 'sim_t': 0.10, 'pad_u_m': 0.03, 'pad_w_m': 0.04, 'pad_n_m': 0.00,
            'align_deg': 25.0, 'vn_mps': 0.9, 'vt_mps': 0.5, 'omega_radps': 4.0, **changes}


def hold(t, *, engaged=True, margin=10.0):
    return {'kind': 'hold', 'sim_t': t, 'engaged': engaged, 'margin_n': margin, 'slip_mps': 0.0}


def replay(tmp_path, rows):
    path = tmp_path / 'truth_1.jsonl'
    path.write_text(''.join(json.dumps(row) + '\n' for row in rows))
    score = TrialScore(rest_s=0.10)
    for line in path.read_text().splitlines():
        row = json.loads(line)
        score.feed(row)
    outcome = score.outcome(row['sim_t']) or 'timeout'
    return {**score.result(outcome), 'recovered': True}


def test_good_touch_and_rest_after_disarm(tmp_path):
    trial = replay(tmp_path, [diag(0.00, 'PERCH'), diag(0.05, 'PERCH', visible=False), touch(),
                              diag(0.10, 'VERIFY'), diag(0.15, 'REST'), hold(0.16, margin=-1.0),
                              diag(0.20, 'REST', armed=False), hold(0.225), hold(0.25), hold(0.275), hold(0.30)])
    assert trial['outcome'] == 'rest' and trial['success'] and trial['held']
    assert trial['touch_success'] and trial['touch']['pad_error_cm'] == pytest.approx(5)
    assert trial['tag_lost_pct'] == pytest.approx(50)
    assert not trial['false_safe']


@pytest.mark.parametrize('value', [{'align_deg': 25.01}, {'vn_mps': 1.5}, {'vt_mps': 1.0},
                                  {'omega_radps': 5.0}, {'pad_u_m': 0.226}])
def test_touch_limits(value):
    assert not score_touch(touch(**value))


def test_bad_touch_crash(tmp_path):
    trial = replay(tmp_path, [diag(0.0, 'PERCH'), touch(vn_mps=2.0)])
    assert trial['outcome'] == 'crash' and trial['crash_reason'] == 'touch_envelope'
    assert not trial['touch_success'] and not trial['motors_off']


def test_ground_impact_and_abort_priority(tmp_path):
    trial = replay(tmp_path, [diag(0.0, 'PERCH'), diag(0.2, 'ABORT'),
                              {'kind': 'impact', 'sim_t': 0.201, 'surface': 'ground'}])
    assert trial['outcome'] == 'crash' and trial['crash_reason'] == 'ground'
    assert not trial['success'] and trial['false_safe'] is None


def test_abort_no_impact(tmp_path):
    trial = replay(tmp_path, [diag(0.0, 'PERCH'), diag(0.5, 'ABORT')])
    assert trial['outcome'] == 'abort' and not trial['success']
    assert trial['false_safe'] is None


def test_false_safe_denominator_is_disarm_count(tmp_path):
    trial = replay(tmp_path, [diag(0.0, 'PERCH'), touch(), diag(0.20, 'REST', armed=False),
                              hold(0.225), hold(0.25, margin=-2), hold(0.275, engaged=False)])
    assert trial['outcome'] == 'crash' and trial['crash_reason'] == 'detached_after_disarm'
    assert trial['touch_success'] and trial['false_safe'] and not trial['success']
    abort = replay(tmp_path, [diag(0.0, 'PERCH'), diag(0.4, 'ABORT')])
    summary = summarize([trial, abort])
    assert summary['false_safe'] == wilson(1, 1)
    assert summary['crash'] == summary['abort'] == wilson(1, 2)


def test_negative_margin_counts_without_detachment(tmp_path):
    trial = replay(tmp_path, [touch(), diag(0.20, 'REST', armed=False), hold(0.225, margin=-0.01),
                              hold(0.25), hold(0.275), hold(0.30)])
    assert trial['outcome'] == 'rest' and trial['false_safe'] and not trial['success']


def test_incomplete_rest_and_missing_truth_are_unknown(tmp_path):
    rows = [touch(), diag(0.20, 'REST', armed=False), hold(0.225)]
    trial = replay(tmp_path, rows)
    assert trial['outcome'] == 'timeout' and trial['false_safe'] is None
    trial = replay(tmp_path, [*rows, hold(0.30)])
    assert trial['outcome'] == 'rest' and trial['held'] is None
    assert summarize([trial])['false_safe'] == {'p': None, 'lo': None, 'hi': None, 'n': 1}


def test_wilson():
    assert wilson(0, 0) == {'p': None, 'lo': None, 'hi': None, 'n': 0}
    assert wilson(5, 10) == {'p': 0.5, 'lo': 0.2366, 'hi': 0.7634, 'n': 10}
    assert wilson(0, 10)['lo'] == 0 and wilson(10, 10)['hi'] == 1
    with pytest.raises(ValueError):
        wilson(11, 10)


CLI = ['--name', 'night-01', '--policy-run', 'runs/actor']


def test_parse_and_shuffle_balanced_grid():
    assert parse_args(CLI).speeds == (0.0, 1.5, 3.0)
    args = parse_args([*CLI, '--trials', '24', '--speeds', '0;1,5;3', '--mu', '0.3,0.9',
                       '--f-hold-n', '8,60.75', '--eval-seed', '99'])
    trials = list(trial_grid(args))
    assert trials == list(trial_grid(args))
    first, second = [{tuple(cell.items()) for cell in part} for part in (trials[:12], trials[12:])]
    assert len(first) == 12 and first == second
    assert trials[:12] != trials[12:]


@pytest.mark.parametrize('args', [['--mu', '-1'], ['--f-hold-n', 'nan'], ['--mu', '0.3,,0.9'],
                                  ['--speeds', '4'], ['--speeds', '1,1'], ['--name', '../escape'],
                                  ['--trials', '0'], ['--rest-s', '50'], ['--eval-seed', '-1']])
def test_parse_rejects_bad_config(args):
    with pytest.raises(SystemExit):
        parse_args([*CLI, *args])


def test_managed_hover_start_rest_and_reset():
    cfg = PhaseCfg(standby_offset=(2, 0, 2))
    pm = PhaseMachine(cfg, object(), ActionScale())
    drone = DroneState(np.zeros(3), np.zeros(3), np.eye(3), np.zeros(3))
    assert not pm.start_trial(0)
    assert np.allclose(pm.step(Inputs(0, drone, False, False, None)).pos_enu, [2, 0, 2])
    assert pm.step(Inputs(1, drone, False, False, None)).commands
    drone.p = pm.standby.copy()
    assert pm.step(Inputs(2, drone, True, True, None)).phase == Phase.WAIT
    assert pm.ready and pm.start_trial(2)
    assert pm.phase == Phase.APPROACH and not pm.start_trial(2)

    pm.phase, pm.t_enter = Phase.REST, 3
    assert pm.step(Inputs(10, drone, False, False, None)).phase == Phase.REST
    pm.reset_trial()
    off = pm.step(Inputs(11, drone, False, False, None))
    assert off.phase == Phase.DEPART and off.hold_cmd is None
    on = pm.step(Inputs(12, drone, True, True, None))
    assert on.phase == Phase.DEPART and on.hold_cmd is None
    on = pm.step(Inputs(13, drone, True, True, None))
    assert on.phase == Phase.DEPART and on.hold_cmd == 0
    drone.p = pm.anchor.copy()
    assert pm.step(Inputs(14, drone, True, True, None)).phase == Phase.DONE
    back = pm.step(Inputs(15, drone, True, True, None))
    assert back.phase == Phase.WAIT and np.allclose(back.pos_enu, pm.standby) and not pm.ready
    drone.p = pm.standby.copy()
    pm.step(Inputs(16, drone, True, True, None))
    assert pm.ready and pm.start_trial(16)


def test_success_needs_a_valid_touch(tmp_path):
    """Review D04: REST disarmed + hold held without any touch row is not a successful cycle."""
    trial = replay(tmp_path, [diag(0.00, 'PERCH'), diag(0.10, 'VERIFY'), diag(0.15, 'REST'),
                              diag(0.20, 'REST', armed=False), hold(0.225), hold(0.25), hold(0.275), hold(0.30)])
    assert trial['held'] and not trial['touch_success'] and not trial['success']


def test_plate_impact_grace_by_stamp(tmp_path):
    """Review D04: plate impact at 1.00 s and the pad touch stamped 1.10 s, read in one batch -> miss."""
    rows = [diag(0.9, 'PERCH'),
            {'kind': 'impact', 'sim_t': 1.00, 'surface': 'plate', 'collisions': ['x500::base_link_collision_0']},
            touch(sim_t=1.10)]
    trial = replay(tmp_path, rows)
    assert trial['crash_reason'] == 'plate_missed_pad' and not trial['success']
    ok = replay(tmp_path, [diag(0.9, 'PERCH'), {'kind': 'impact', 'sim_t': 1.00, 'surface': 'plate'},
                           touch(sim_t=1.03), diag(1.05, 'VERIFY'), diag(1.06, 'REST'),
                           diag(1.10, 'REST', armed=False), hold(1.125), hold(1.15), hold(1.175), hold(1.20)])
    assert ok['crash_reason'] is None and ok['success']
