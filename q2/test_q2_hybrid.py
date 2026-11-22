"""CPU tests of the hybrid actions and exact joint threshold search."""
import itertools
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from q2.decide_v4 import dthr_b4e, dthr_slip_abort, select, select_slip  # noqa: E402


def test_hybrid_actions():
    alive = np.array([[1, 1, 1]] * 5 + [[1, 0, 0]], dtype=bool)
    ep = {
        "alive": alive, "held": alive, "strict": alive, "fell": np.zeros_like(alive),
        "p": np.array([[[0.9, 0.9, 0.9]] * 2 + [[0.1, 0.9, 0.9]] + [[0.9, 0.9, 0.9]] * 3]),
        "ps": np.zeros((1, 6, 3)),
        "slip_mm": np.array([[3, 1, 1], [2, 1, 1], [3, 4, 1], [3, 4, 5], [1, 1, 1], [3, 1, 1]]),
    }
    ep["ps"][0, 4, 0] = 0.8
    prm = {"tau": 0.5, "s": 0.8}
    gate = dthr_b4e(ep, {**prm, "slip_off_mm": [2, 3, 4]})
    assert gate["d"].tolist() == [1, 1, 2, -1, -1, -1]
    assert gate["abort"].tolist() == [False, False, False, True, True, False]
    assert gate["self_abort"].tolist() == [False, False, False, False, False, True]

    abort = dthr_slip_abort(ep, {**prm, "slip_abort_mm": 2})
    assert abort["d"].tolist() == [-1, 0, -1, -1, -1, -1]
    assert abort["early_stop"].tolist() == [True, False, True, True, True, True]
    assert not abort["self_abort"].any()

    # Outcome labels change scoring only.
    ep.update(held=~alive, strict=~alive, fell=alive)
    for fn, extra, original in ((dthr_b4e, {"slip_off_mm": [2, 3, 4]}, gate),
                                (dthr_slip_abort, {"slip_abort_mm": 2}, abort)):
        changed = fn(ep, {**prm, **extra})
        for key in ("d", "off", "abort", "self_abort"):
            np.testing.assert_array_equal(changed[key], original[key])


def test_joint_search_matches_v4_selector():
    rng = np.random.default_rng(2028)
    n = 2_500
    shape = (n, 3)
    alive = np.logical_and.accumulate(rng.random(shape) > 0.05, axis=1)
    slip = rng.integers(0, 5, size=shape).astype(np.float32)
    fell = (slip > 1) & (rng.random(shape) < 0.022)
    ep = {"alive": alive, "held": ~fell, "strict": ~fell, "fell": fell, "slip_mm": slip,
          "p": rng.random((1, *shape)), "ps": rng.random((1, *shape))}
    grid = [{"tau": tau, "s": s} for tau, s in itertools.product((0.1, 0.5), (0.8, 1.01))]
    gates = [[0.0, 2.0, 5.0], [1.0, 3.0, 5.0], [0.0, 1.0, 3.0, 5.0]]
    aborts = [0.0, 1.0, 3.0, 5.0]
    for fn, axis, key, options in (
        (dthr_b4e, gates, "slip_off_mm", list(map(list, itertools.product(*gates)))),
        (dthr_slip_abort, aborts, "slip_abort_mm", aborts),
    ):
        brute = select(fn, ep, [{**params, key: value} for params, value in itertools.product(grid, options)])
        assert all(brute.values())
        assert select_slip(ep, grid, axis, abort=fn is dthr_slip_abort) == brute
