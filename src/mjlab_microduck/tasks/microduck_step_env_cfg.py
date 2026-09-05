"""Microduck step climbing — walk forward and climb a single step.

Built on the velocity walking recipe (DR / obs / noise / delays inherited
untouched), with three task-specific changes:

  1. Robot swapped to the GROUNDCONTACT model. robot_walk.xml gives collision
     geoms to the feet only — the legs are ``self_collision_only`` and pass
     straight through terrain, so the walk model cannot feel a riser at all.
  2. Terrain swapped to SingleStepTerrainCfg (flat approach → one riser →
     raised platform), 10 difficulty rows from 0.5 cm to 5 cm.
  3. Terrain-level curriculum (terrain_levels_step): each env is promoted to a
     taller step once it actually climbs, demoted if it stalls before the riser.
     Per-env and performance-gated, not a global iteration schedule.

Obs stays 61D → hot-swappable at runtime. The policy is BLIND: height_scan is
deleted upstream to keep the contract, so the step must be felt, not seen.

Kinematic headroom (FK sweep, soft limits, trunk at HOME): max foot lift is
~19 cm, so 5 cm is control-limited, not reach-limited.
"""

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as base_mdp
from mjlab.managers import CurriculumTermCfg, RewardTermCfg, TerminationTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.rl import RslRlOnPolicyRunnerCfg, RslRlModelCfg
from mjlab.terrains import TerrainEntityCfg
from mjlab.terrains.terrain_generator import TerrainGeneratorCfg

from mjlab_microduck.robot.microduck_constants import MICRODUCK_STANDUP_ROBOT_CFG
from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks.microduck_velocity_env_cfg import (
    _soften_terrain_contacts,
    make_microduck_velocity_env_cfg,
)
from mjlab_microduck.tasks.step_terrain import (
    STEP_HEIGHT_MAX,
    STEP_HEIGHT_MIN,
    SingleStepTerrainCfg,
)
from mjlab_microduck.tasks.symmetry import PpoWithSymmetryCfg

# Terrain geometry. Tile is 6 m so approach + tread (4 m) leaves 2 m of slack.
APPROACH_LENGTH = 2.0
TREAD_LENGTH = 2.0
SPAWN_DISTANCE = 0.6  # spawn this far before the riser
TREAD_MARGIN = 0.4  # how far onto the tread counts as "climbed"
TILE_SIZE = (6.0, 4.0)
NUM_LEVELS = 10

# Forward walking command: always driving at the step, no strafing target.
# y/yaw keep tiny non-zero ranges so their obs slots never go dead.
STEP_LIN_VEL_X = (0.15, 0.35)
STEP_LIN_VEL_Y = (-0.02, 0.02)
STEP_ANG_VEL_Z = (-0.05, 0.05)

# Nominal base height while standing, used to measure a real climb.
NOMINAL_BASE_Z = 0.115

# Below the approach surface by a clear margin: only fires if the robot leaves
# the solid (walks off the far end of the tread), never during a normal climb.
VOID_FLOOR = -0.5


def make_microduck_step_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    cfg = make_microduck_velocity_env_cfg(play=play, rough=False)

    # === ROBOT: groundcontact model (legs collide with terrain) ===
    cfg.scene.entities = {"robot": MICRODUCK_STANDUP_ROBOT_CFG}

    # === TERRAIN: flat approach + single riser + raised platform ===
    cfg.scene.terrain = TerrainEntityCfg(
        terrain_type="generator",
        terrain_generator=TerrainGeneratorCfg(
            size=TILE_SIZE,
            curriculum=True,
            num_rows=NUM_LEVELS,
            num_cols=1,
            difficulty_range=(0.0, 1.0),
            sub_terrains={
                "single_step": SingleStepTerrainCfg(
                    approach_length=APPROACH_LENGTH,
                    tread_length=TREAD_LENGTH,
                    step_height_range=(STEP_HEIGHT_MIN, STEP_HEIGHT_MAX),
                    spawn_distance=SPAWN_DISTANCE,
                )
            },
        ),
        max_init_terrain_level=0,  # start everyone on the lowest step
    )
    if play:
        cfg.scene.terrain.max_init_terrain_level = None  # spread across all heights

    # Box terrain numerics, ported from the velocity rough mode. Each of these
    # has a documented NaN failure mode from box-edge contacts.
    cfg.scene.spec_fn = _soften_terrain_contacts
    cfg.sim.nconmax = 200
    cfg.sim.mujoco.iterations = 30
    cfg.sim.mujoco.ls_iterations = 50

    # === COMMAND: drive forward at the riser ===
    command = cfg.commands["twist"]
    command.rel_standing_envs = 0.0  # standing still never climbs anything
    command.rel_heading_envs = 0.0
    command.ranges.lin_vel_x = STEP_LIN_VEL_X
    command.ranges.lin_vel_y = STEP_LIN_VEL_Y
    if getattr(command.ranges, "ang_vel_z", None) is not None:
        command.ranges.ang_vel_z = STEP_ANG_VEL_Z

    # === RESET: always facing the riser (+x) ===
    cfg.events["reset_base"].params["pose_range"]["yaw"] = (0.0, 0.0)

    # === TERMINATIONS ===
    cfg.terminations["fell_over"] = TerminationTermCfg(
        func=base_mdp.bad_orientation,
        params={
            "limit_angle": 1.0,
            "asset_cfg": SceneEntityCfg("robot", body_names=("trunk_base",)),
        },
    )
    # The tile is only 4 m of solid; the inherited bounds check ends episodes
    # before a slow climber can be promoted, so drop it and catch the real
    # failure (leaving the solid) with a floor check instead.
    if "out_of_terrain_bounds" in cfg.terminations:
        del cfg.terminations["out_of_terrain_bounds"]
    cfg.terminations["fell_into_void"] = TerminationTermCfg(
        func=microduck_mdp.root_height_below,
        params={
            "min_height": VOID_FLOOR,
            "asset_cfg": SceneEntityCfg("robot", body_names=("trunk_base",)),
        },
    )
    cfg.terminations["nan_state"] = TerminationTermCfg(
        func=microduck_mdp.robot_state_is_nan, time_out=False,
    )

    # === OBS: sanitize rare contact divergences on box terrain ===
    for grp in ("actor", "critic"):
        cfg.observations[grp].nan_policy = "sanitize"

    # === CURRICULUM: step height easy → hard, per env ===
    cfg.curriculum["terrain_levels"] = CurriculumTermCfg(
        func=microduck_mdp.terrain_levels_step,
        params={
            "riser_distance": SPAWN_DISTANCE + TREAD_MARGIN,
            "nominal_base_z": NOMINAL_BASE_Z,
            "step_height_min": STEP_HEIGHT_MIN,
            "step_height_max": STEP_HEIGHT_MAX,
        },
    )

    return cfg


MicroduckStepRlCfg = RslRlOnPolicyRunnerCfg(
    actor=RslRlModelCfg(
        hidden_dims=(512, 256, 128),
        activation="elu",
        obs_normalization=True,
        distribution_cfg={
            "class_name": "GaussianDistribution",
            "init_std": 1.0,
            "std_type": "scalar",
        },
    ),
    critic=RslRlModelCfg(
        hidden_dims=(512, 256, 128), activation="elu", obs_normalization=True
    ),
    algorithm=PpoWithSymmetryCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        symmetry_cfg=None,
    ),
    wandb_project="mjlab_microduck",
    experiment_name="step_up",
    run_name="step_up",
    save_interval=250,
    num_steps_per_env=24,
    max_iterations=6_000,
)
