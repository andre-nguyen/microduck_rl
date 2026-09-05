import torch

from mjlab_microduck.tasks.mdp import step_move_masks

RISER = 1.0  # spawn_distance 0.6 + tread margin 0.4
HEIGHT = torch.tensor([0.03, 0.03])  # step height of the row the env sits on


def test_move_up_requires_both_distance_and_height():
    # Past the riser AND lifted the base by more than half the step height.
    dist = torch.tensor([1.5, 2.0])
    gain = torch.tensor([0.03, 0.05])
    up, down = step_move_masks(dist, gain, RISER, HEIGHT)
    assert bool(up[0]) and bool(up[1])
    assert not bool(down[0]) and not bool(down[1])


def test_distance_without_height_does_not_promote():
    # Belly-flopped onto the tread: travelled far, never rose.
    dist = torch.tensor([2.0])
    gain = torch.tensor([0.0])
    up, _ = step_move_masks(dist, gain, RISER, HEIGHT[:1])
    assert not bool(up[0])


def test_height_without_distance_does_not_promote():
    # Hopping on the approach: gained height, never crossed the riser.
    dist = torch.tensor([0.2])
    gain = torch.tensor([0.05])
    up, down = step_move_masks(dist, gain, RISER, HEIGHT[:1])
    assert not bool(up[0])
    assert bool(down[0])  # and it stalled, so it gets demoted


def test_move_down_when_stuck_before_the_riser():
    dist = torch.tensor([0.1, 0.4])
    gain = torch.tensor([0.0, 0.0])
    up, down = step_move_masks(dist, gain, RISER, HEIGHT)
    assert not bool(up[0]) and not bool(up[1])
    assert bool(down[0]) and bool(down[1])


def test_middle_band_neither_promotes_nor_demotes():
    # Reached the riser but did not get onto the tread.
    dist = torch.tensor([0.8])
    gain = torch.tensor([0.0])
    up, down = step_move_masks(dist, gain, RISER, HEIGHT[:1])
    assert not bool(up[0]) and not bool(down[0])


def test_climb_fraction_boundary():
    dist = torch.tensor([1.5, 1.5])
    # Exactly half the step height does not count; just above does.
    gain = torch.tensor([0.015, 0.016])
    up, _ = step_move_masks(dist, gain, RISER, HEIGHT)
    assert not bool(up[0])
    assert bool(up[1])


def test_taller_rows_demand_more_height_gain():
    dist = torch.tensor([1.5, 1.5])
    gain = torch.tensor([0.02, 0.02])
    heights = torch.tensor([0.03, 0.05])  # easy row vs hard row
    up, _ = step_move_masks(dist, gain, RISER, heights)
    assert bool(up[0])  # 0.02 > 0.015
    assert not bool(up[1])  # 0.02 < 0.025


def test_promotion_and_demotion_are_mutually_exclusive():
    dist = torch.tensor([0.1, 0.8, 1.5])
    gain = torch.tensor([0.05, 0.05, 0.05])
    heights = torch.tensor([0.03, 0.03, 0.03])
    up, down = step_move_masks(dist, gain, RISER, heights)
    assert not bool((up & down).any())
