"""View the single-step terrain (step-climbing task) in the MuJoCo viewer.

Builds ONLY the terrain (SingleStepTerrainCfg) across several rows of
increasing difficulty (step height 0.5 cm -> 5 cm) and opens the native MuJoCo
viewer. No trained policy needed — this is for eyeballing the geometry (flush
riser join, climb direction) before wiring up the env.

Red spheres mark each tile's spawn origin, i.e. where the robot's base will be
placed on reset (before reset_base's pose_range offset is applied).

Usage:
    uv run python scripts/view_step_terrain.py
    uv run python scripts/view_step_terrain.py --rows 6 --height-max 0.08
    uv run python scripts/view_step_terrain.py --build-only   # no GUI

In the viewer: scroll to zoom, left-drag to orbit, right-drag to pan. Each row
is a taller step (difficulty 0 -> 1). "Forward" (+x) must climb UP.

Note: launch_passive is native-window only and will not work over SSH.
"""

import argparse

import mujoco
import mujoco.viewer

from mjlab.terrains.terrain_generator import TerrainGenerator, TerrainGeneratorCfg
from mjlab_microduck.tasks.step_terrain import (
    STEP_HEIGHT_MAX,
    STEP_HEIGHT_MIN,
    SingleStepTerrainCfg,
    step_height_by_difficulty,
)

MARKER_RADIUS = 0.03


def build_model(rows, size, approach, tread, height_range, spawn_distance, markers=True):
    """Build the terrain-only MuJoCo model (rows steps of increasing height)."""
    cfg = TerrainGeneratorCfg(
        seed=0,
        size=size,
        num_rows=rows,
        num_cols=1,
        curriculum=True,  # difficulty increases along rows
        difficulty_range=(0.0, 1.0),
        add_lights=True,
        sub_terrains={
            "single_step": SingleStepTerrainCfg(
                approach_length=approach,
                tread_length=tread,
                step_height_range=height_range,
                spawn_distance=spawn_distance,
            )
        },
    )
    generator = TerrainGenerator(cfg)
    spec = mujoco.MjSpec()
    generator.compile(spec)

    if markers:
        for row in range(generator.terrain_origins.shape[0]):
            for col in range(generator.terrain_origins.shape[1]):
                x, y, z = generator.terrain_origins[row, col]
                spec.worldbody.add_geom(
                    type=mujoco.mjtGeom.mjGEOM_SPHERE,
                    size=(MARKER_RADIUS, 0, 0),
                    pos=(x, y, z + MARKER_RADIUS),
                    rgba=(0.9, 0.1, 0.1, 1.0),
                    contype=0,
                    conaffinity=0,
                )

    return generator, spec.compile()


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--rows", type=int, default=5, help="Number of rows = step heights shown")
    p.add_argument("--size", type=float, nargs=2, default=(6.0, 4.0), help="Tile size (x y) in m")
    p.add_argument("--approach", type=float, default=2.0, help="Flat run-up length (m)")
    p.add_argument("--tread", type=float, default=2.0, help="Raised platform length (m)")
    p.add_argument("--height-min", type=float, default=STEP_HEIGHT_MIN, help="Step height at difficulty 0 (m)")
    p.add_argument("--height-max", type=float, default=STEP_HEIGHT_MAX, help="Step height at difficulty 1 (m)")
    p.add_argument("--spawn-distance", type=float, default=0.6, help="Spawn this far before the riser (m)")
    p.add_argument("--no-markers", action="store_true", help="Hide the spawn-origin markers")
    p.add_argument("--build-only", action="store_true", help="Build the model and exit (no GUI)")
    args = p.parse_args()

    generator, model = build_model(
        rows=args.rows,
        size=tuple(args.size),
        approach=args.approach,
        tread=args.tread,
        height_range=(args.height_min, args.height_max),
        spawn_distance=args.spawn_distance,
        markers=not args.no_markers,
    )
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    print(
        f"Terrain built: {args.rows} steps, height "
        f"{args.height_min * 100:.1f}cm -> {args.height_max * 100:.1f}cm, "
        f"approach {args.approach}m + tread {args.tread}m, {model.ngeom} geoms."
    )
    for row in range(args.rows):
        difficulty = (row + 0.5) / args.rows
        h = step_height_by_difficulty(difficulty, args.height_min, args.height_max)
        origin = generator.terrain_origins[row, 0]
        print(
            f"  row {row}: difficulty~{difficulty:.2f}  step {h * 100:5.2f} cm  "
            f"spawn origin ({origin[0]:.2f}, {origin[1]:.2f}, {origin[2]:.2f})"
        )
    if args.build_only:
        print("--build-only: OK, no GUI.")
        return

    print("Opening the MuJoCo viewer (Ctrl+C to quit)…")
    with mujoco.viewer.launch_passive(
        model, data, show_left_ui=False, show_right_ui=False
    ) as viewer:
        while viewer.is_running():
            mujoco.mj_forward(model, data)
            viewer.sync()


if __name__ == "__main__":
    main()
