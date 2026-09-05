"""Custom "flat approach + single step" terrain for the step-climbing task.

The robot spawns on a flat approach facing +x, walks forward, and must climb a
single riser onto a raised platform. The step height is interpolated by the
difficulty (terrain-level curriculum) over
[STEP_HEIGHT_MIN, STEP_HEIGHT_MAX] metres.

mjlab's built-in stair terrains are all concentric pyramids whose spawn origin
sits on TOP of the pyramid (the robot walks down, not up), so a dedicated
SubTerrainCfg is required here — same reasoning as ``slope_terrain.py``.
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from mjlab.terrains.terrain_generator import (
    SubTerrainCfg,
    TerrainGeometry,
    TerrainOutput,
)

# Kinematic max foot lift measured by FK sweep on robot_walk (soft limits,
# trunk pinned at HOME) is ~19 cm, so these bounds are control-limited, not
# reach-limited. The trained walking gait clears ~2 cm (foot_clearance target).
STEP_HEIGHT_MIN = 0.005
STEP_HEIGHT_MAX = 0.05


def step_height_by_difficulty(
    difficulty: float,
    h_min: float = STEP_HEIGHT_MIN,
    h_max: float = STEP_HEIGHT_MAX,
) -> float:
    """Step height (m) linearly interpolated by difficulty over [0, 1]."""
    d = float(np.clip(difficulty, 0.0, 1.0))
    return h_min + d * (h_max - h_min)


@dataclass(kw_only=True)
class SingleStepTerrainCfg(SubTerrainCfg):
    """Flat approach → one riser → raised platform.

    Two boxes aligned along +x:
      1. approach platform, top surface at z=0, where the robot spawns;
      2. raised platform, top surface at z=step_height, starting exactly where
         the approach ends so the riser face is flush (no gap, no overhang).

    The riser is the -x face of box 2. There is a single step: once the robot
    is on the raised platform the task is done.
    """

    approach_length: float = 2.0  # flat run-up before the riser (m)
    tread_length: float = 2.0  # raised platform after the riser (m)
    step_height_range: tuple = (STEP_HEIGHT_MIN, STEP_HEIGHT_MAX)
    spawn_distance: float = 0.6  # spawn this far BEFORE the riser (m)
    thickness: float = 0.5  # box thickness (m)

    def function(self, difficulty: float, spec: mujoco.MjSpec, rng) -> TerrainOutput:
        total = self.approach_length + self.tread_length
        assert total <= self.size[0], (
            f"approach+tread ({total}) must fit in size[0] ({self.size[0]})"
        )
        assert self.spawn_distance < self.approach_length, (
            f"spawn_distance ({self.spawn_distance}) must be less than "
            f"approach_length ({self.approach_length})"
        )
        del rng  # geometry is fully determined by difficulty

        body = spec.body("terrain")
        height = step_height_by_difficulty(
            difficulty, self.step_height_range[0], self.step_height_range[1]
        )
        width = self.size[1]
        t = self.thickness
        assert t / 2.0 > height, (
            f"thickness/2 ({t / 2.0}) must exceed max step height ({height}) so the "
            "raised platform still reaches below z=0"
        )

        # 1) Approach: top surface at z=0, x in [0, approach_length].
        approach = body.add_geom(
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=(self.approach_length / 2.0, width / 2.0, t / 2.0),
            pos=(self.approach_length / 2.0, 0.0, -t / 2.0),
        )

        # 2) Raised platform: top surface at z=height, x in
        # [approach_length, approach_length + tread_length]. Its -x face is the
        # riser; sharing the x boundary with the approach keeps the join flush.
        tread = body.add_geom(
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=(self.tread_length / 2.0, width / 2.0, t / 2.0),
            pos=(
                self.approach_length + self.tread_length / 2.0,
                0.0,
                height - t / 2.0,
            ),
        )

        # Spawn on the approach flat, facing the riser.
        origin = np.array(
            [self.approach_length - self.spawn_distance, 0.0, 0.0]
        )
        return TerrainOutput(
            origin=origin,
            geometries=[
                TerrainGeometry(geom=approach, color=(0.5, 0.5, 0.5, 1.0)),
                TerrainGeometry(geom=tread, color=(0.45, 0.55, 0.75, 1.0)),
            ],
        )
