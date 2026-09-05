import mujoco
import numpy as np
import pytest

from mjlab_microduck.tasks.step_terrain import (
    STEP_HEIGHT_MAX,
    STEP_HEIGHT_MIN,
    SingleStepTerrainCfg,
    step_height_by_difficulty,
)


def test_step_height_endpoints():
    assert step_height_by_difficulty(0.0) == pytest.approx(STEP_HEIGHT_MIN)
    assert step_height_by_difficulty(1.0) == pytest.approx(STEP_HEIGHT_MAX)


def test_step_height_midpoint():
    mid = (STEP_HEIGHT_MIN + STEP_HEIGHT_MAX) / 2.0
    assert step_height_by_difficulty(0.5) == pytest.approx(mid)


def test_step_height_clamps_out_of_range():
    assert step_height_by_difficulty(-1.0) == pytest.approx(STEP_HEIGHT_MIN)
    assert step_height_by_difficulty(2.0) == pytest.approx(STEP_HEIGHT_MAX)


def test_step_height_is_monotonic():
    heights = [step_height_by_difficulty(d) for d in np.linspace(0.0, 1.0, 11)]
    assert all(b > a for a, b in zip(heights, heights[1:]))


def _empty_terrain_spec():
    spec = mujoco.MjSpec()
    spec.worldbody.add_body(name="terrain")
    return spec


def _build(difficulty, **kwargs):
    cfg = SingleStepTerrainCfg(**kwargs)
    cfg.size = (6.0, 4.0)  # normally set by the generator
    spec = _empty_terrain_spec()
    out = cfg.function(difficulty=difficulty, spec=spec, rng=np.random.default_rng(0))
    return cfg, out


def test_builds_two_geoms():
    _, out = _build(0.5)
    assert len(out.geometries) == 2


def test_origin_is_on_the_approach_flat_facing_the_riser():
    cfg, out = _build(0.5)
    # On the flat, a spawn_distance short of the riser, and at ground level.
    assert out.origin[0] == pytest.approx(cfg.approach_length - cfg.spawn_distance)
    assert out.origin[1] == pytest.approx(0.0)
    assert out.origin[2] == pytest.approx(0.0)
    # The riser is ahead of the robot (+x), not behind it.
    assert out.origin[0] < cfg.approach_length


def test_origin_height_does_not_depend_on_difficulty():
    # Regression guard for the BoxInvertedPyramidStairs bug: an origin z that
    # tracks the terrain feature spawns the robot inside/below the geometry.
    for difficulty in (0.0, 0.5, 1.0):
        _, out = _build(difficulty)
        assert out.origin[2] == pytest.approx(0.0)


def test_riser_join_is_flush():
    cfg, out = _build(0.5)
    approach, tread = (g.geom for g in out.geometries)
    approach_end = approach.pos[0] + approach.size[0]
    tread_start = tread.pos[0] - tread.size[0]
    assert approach_end == pytest.approx(tread_start)
    assert approach_end == pytest.approx(cfg.approach_length)


def test_tread_top_is_at_step_height_and_approach_top_at_zero():
    cfg, out = _build(0.5)
    approach, tread = (g.geom for g in out.geometries)
    expected = step_height_by_difficulty(
        0.5, cfg.step_height_range[0], cfg.step_height_range[1]
    )
    assert approach.pos[2] + approach.size[2] == pytest.approx(0.0)
    assert tread.pos[2] + tread.size[2] == pytest.approx(expected)


def test_harder_difficulty_raises_the_tread():
    _, easy = _build(0.0)
    _, hard = _build(1.0)
    easy_top = easy.geometries[1].geom.pos[2] + easy.geometries[1].geom.size[2]
    hard_top = hard.geometries[1].geom.pos[2] + hard.geometries[1].geom.size[2]
    assert hard_top > easy_top


def test_tread_still_reaches_below_ground_at_max_height():
    # The raised platform must not float: its underside stays below z=0 so there
    # is no gap under the riser.
    _, out = _build(1.0)
    tread = out.geometries[1].geom
    assert tread.pos[2] - tread.size[2] < 0.0


def test_rejects_geometry_that_does_not_fit_the_tile():
    with pytest.raises(AssertionError):
        _build(0.5, approach_length=4.0, tread_length=4.0)


def test_rejects_spawn_beyond_the_riser():
    with pytest.raises(AssertionError):
        _build(0.5, approach_length=2.0, spawn_distance=2.5)
