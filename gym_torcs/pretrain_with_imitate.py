"""
pretrain_bc.py  —  Behavioural Cloning pre-train → PPO fine-tune

SETUP:
  1. Start TORCS manually and configure/start the race as normal
  2. Run this script - it will connect automatically
  Monitor: tensorboard --logdir ./logs/
"""

import numpy as np
from pathlib import Path

from imitation.algorithms import bc
from imitation.data.types import Transitions

from stable_baselines3 import PPO
from stable_baselines3.common.running_mean_std import RunningMeanStd
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback

from gym_torcs2 import TorcsEnv

class TorcsPauseCallback(BaseCallback):
    def __init__(self, torcs_exe="wtorcs.exe"):
        super().__init__()
        self.torcs_exe = torcs_exe

    def _find_torcs_procs(self):
        return [p for p in psutil.process_iter(['name'])
                if p.info['name'] == self.torcs_exe]

    def _on_step(self):
        return True  # required by SB3, just return True to continue training

    def on_rollout_end(self):
        for p in self._find_torcs_procs():
            p.suspend()

    def on_rollout_start(self):
        for p in self._find_torcs_procs():
            p.resume()

# ─────────────────────────────────────────────────────────────────────────────
BC_EPOCHS        = 50
RL_TIMESTEPS     = 1_000_000
CHECKPOINT_EVERY = 10_000
# ─────────────────────────────────────────────────────────────────────────────


# ── Load demos ────────────────────────────────────────────────────────────────

data    = np.load("demos.npz")
obs_raw = data["obs"].astype(np.float32)      # (T, OBS_DIM)
acts    = data["actions"].astype(np.float32)  # (T, ACT_DIM)
T       = len(obs_raw)
print(f"Loaded {T} demo transitions  obs{obs_raw.shape}  acts{acts.shape}")


# ── Env ───────────────────────────────────────────────────────────────────────

def make_env(port: int, episode_offset: int):
    def _init():
        return TorcsEnv(
            vision=False, throttle=True, gear_change=False,
            port=port, episode_offset=episode_offset,
        )
    return _init

env = DummyVecEnv([lambda: TorcsEnv(vision=False, throttle=True, gear_change=False, port=3001)])
env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)


# ── Bootstrap VecNormalize stats from demo data ───────────────────────────────
#
#  BC trains the policy in-place.  The policy will be used under VecNormalize
#  during RL, so BC must also see normalised observations.
#  Instead of throwing away the demo obs structure we bootstrap the running
#  mean/variance directly from the demo buffer, then apply the same normalisation
#  to produce the BC training set.  This way VecNormalize is already "warm"
#  when RL starts and no distribution shift occurs between the two phases.

obs_rms = RunningMeanStd(shape=env.observation_space.shape)
obs_rms.update(obs_raw)
env.obs_rms = obs_rms   # hand stats to the wrapper

def _normalise(x: np.ndarray) -> np.ndarray:
    return np.clip(
        (x - obs_rms.mean) / np.sqrt(obs_rms.var + 1e-8),
        -env.clip_obs, env.clip_obs,
    ).astype(np.float32)

obs_norm = _normalise(obs_raw)

transitions = Transitions(
    obs      = obs_norm[:-1],
    acts     = acts[:-1],
    next_obs = obs_norm[1:],
    dones    = np.zeros(T - 1, dtype=bool),
    infos    = np.array([{}] * (T - 1)),
)


# ── PPO model (policy will be pre-trained in-place by BC) ─────────────────────

model = PPO(
    "MlpPolicy", env,
    learning_rate   = 3e-4,
    n_steps         = 2048,
    batch_size      = 128,
    n_epochs        = 10,
    gamma           = 0.999,
    gae_lambda      = 0.95,
    clip_range      = 0.2,
    ent_coef        = 0.01,
    policy_kwargs   = {"net_arch": [256, 256]},
    verbose         = 1,
    tensorboard_log = "./logs/",
)


# ── BC pre-training ───────────────────────────────────────────────────────────
#
#  env.training = False during BC so VecNormalize doesn't update its running
#  stats from the (non-interactive) BC pass — the stats were already set above.

env.training = False

bc_trainer = bc.BC(
    observation_space = env.observation_space,
    action_space      = env.action_space,
    demonstrations    = transitions,
    policy            = model.policy,   # modifies the PPO policy in-place
    rng               = np.random.default_rng(42),
    batch_size        = 256,
)

print(f"Running BC pre-training ({BC_EPOCHS} epochs)…")
bc_trainer.train(n_epochs=BC_EPOCHS)
print("BC done — starting RL fine-tune.")

# Save the BC-only checkpoint so you can diff against the RL result later
model.save("torcs_ppo_bc_only")
env.save("torcs_vecnorm_bc_only.pkl")


# ── RL fine-tune from the pre-trained policy ──────────────────────────────────

env.training = True   # re-enable running-stat updates for RL

checkpoint_cb = CheckpointCallback(
    save_freq   = CHECKPOINT_EVERY,
    save_path   = "./checkpoints/",
    name_prefix = "torcs_ppo_v0",
    save_vecnormalize = True,
    verbose     = 1,
)

model.learn(
    total_timesteps    = RL_TIMESTEPS,
    reset_num_timesteps= False,   # keep global step counter from BC warm-up
    callback           = checkpoint_cb,
    tb_log_name        = "torcs_ppo_v0",
)

model.save("torcs_ppo_v0")
env.save("torcs_vecnorm_v0.pkl")
print("Done.  Saved torcs_ppo_v0 + torcs_vecnorm_v0.pkl")