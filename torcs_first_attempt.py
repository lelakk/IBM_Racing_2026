from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback

from gym_torcs2 import TorcsEnv

# Workflow:
#   1. Start TORCS manually and configure/start the race as normal
#   2. Run this script - it will connect automatically
#   3. Monitor training:  tensorboard --logdir ./logs/
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

CHECKPOINT_EVERY = 10_000
TOTAL_TIMESTEPS  = 1_000_000

def make_env():
    return TorcsEnv(vision=False, throttle=True, gear_change=False, port=3001)

env = DummyVecEnv([make_env])
env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)

model = PPO(
    "MlpPolicy",
    env,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=128,
    n_epochs=10,
    gamma=0.999,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.01,
    policy_kwargs={"net_arch": [256, 256]},
    verbose=1,
    tensorboard_log="./logs/",
)

checkpoint_cb = CheckpointCallback(
    save_freq=CHECKPOINT_EVERY,
    save_path="./checkpoints/",
    name_prefix="torcs_ppo_v0",
    save_vecnormalize=True,
    verbose=1,
)

model.learn(
    total_timesteps=TOTAL_TIMESTEPS,
    callback=checkpoint_cb,
    tb_log_name="torcs_ppo_v0",
)

model.save("torcs_ppo_v0")
env.save("torcs_vecnorm_v0.pkl")
print("Training complete. Model saved.")