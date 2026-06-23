import os
import re
import threading
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
from gym_torcs2 import TorcsEnv
import numpy as np
import torch
import sys
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')


WM_KEYDOWN = 0x0100
WM_KEYUP = 0x0101

def expand_model_obs(model, env, old_dim: int, new_dim: int):
    for net in [model.policy.mlp_extractor.policy_net,
                model.policy.mlp_extractor.value_net]:
        first_layer = net[0]
        old_w = first_layer.weight.data
        new_w = torch.zeros(old_w.shape[0], new_dim, device=old_w.device)
        new_w[:, :old_dim] = old_w
        new_w[:, old_dim:] = torch.randn(old_w.shape[0], new_dim - old_dim, device=old_w.device) * 0.01
        first_layer.weight.data = new_w

    model.observation_space = env.observation_space

    # resetuj optimizer — stare momenty mają zły rozmiar
    model.policy.optimizer = torch.optim.Adam(
        model.policy.parameters(),
        lr=model.learning_rate(1.0) if callable(model.learning_rate) else model.learning_rate,
        eps=1e-5
    )

    print(f"Rozszerzono wagi sieci: {old_dim} → {new_dim}")

class SaveEpisodeOffsetCallback(BaseCallback):
    def __init__(self, save_freq: int):
        super().__init__()
        self.save_freq = save_freq

    def _on_step(self):
        if self.num_timesteps % self.save_freq == 0 and self.num_timesteps > 0:
            try:
                episode = self.training_env.get_attr("episode")[0]
                np.save("episode_offset.npy", np.array([episode]))
            except Exception as e:
                print(f"[WARN] Episode save failed: {e}")
        return True

class PauseDuringUpdateCallback(BaseCallback):
    def __init__(self, torcs_env: TorcsEnv, n_steps: int):
        super().__init__()
        self.torcs_env = torcs_env
        self.n_steps = int(n_steps)
        self._update_started = False

    def _on_step(self) -> bool:
        if self.n_steps > 0 and self.num_timesteps % self.n_steps == 0:
            if not self._update_started:
                self.torcs_env.begin_update()
                self._update_started = True
        return True

    def on_rollout_end(self) -> None:
        # Startuje PRZED update'em PPO
        if not self._update_started:
            self.torcs_env.begin_update()
            self._update_started = True

    def on_rollout_start(self) -> None:
        # Zatrzymuje PO update'cie PPO, przed nową kolekcją
        self.torcs_env.end_update()
        self._update_started = False


class RewardLoggerCallback(BaseCallback):
    def __init__(self):
        super().__init__()
        self._episode_rewards = []
        self._current_reward = 0.0

    def _on_step(self) -> bool:
        self._current_reward += self.locals["rewards"][0]

        if self.locals["dones"][0]:
            self._episode_rewards.append(self._current_reward)
            self.logger.record("episode/reward", self._current_reward)
            self.logger.record("episode/reward_mean_10",
                               np.mean(self._episode_rewards[-10:]))
            self._current_reward = 0.0

        return True

#  Monitor training:  tensorboard --logdir ./logs/
# ─────────────────────────────────────────────────────────────────────────────

CHECKPOINT_EVERY = 98_304
BASE_TIMESTEPS = 19_660_800

# ── Find latest completed version ─────────────────────────────────────────────

versions = [
    int(re.search(r'torcs_vecnorm_v(\d+)\.pkl$', f).group(1))
    for f in os.listdir()
    if re.search(r'torcs_vecnorm_v(\d+)\.pkl$', f)
]
i = max(versions) if versions else 0
TOTAL_TIMESTEPS = (i + 1) * BASE_TIMESTEPS
print(f"Base version: torcs_ppo_v{i} → will save as v{i + 1}")

# ── Env ───────────────────────────────────────────────────────────────────────

custom_objects = {
    "n_steps": 16384,
    "batch_size": 512,
    "learning_rate": 3e-4,
    "ent_coef": 0.02,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_range": 0.3
}

episode_offset = int(np.load("episode_offset.npy")[0]) if Path("episode_offset.npy").exists() else 0
torcs_env = TorcsEnv(vision=False, throttle=True, gear_change=True, port=3001,
                     episode_offset=episode_offset, n_steps_per_update=custom_objects["n_steps"])

env = DummyVecEnv([lambda: torcs_env])

# ── Load model: checkpoint → final save ───────────────────────────────────────

checkpoint_dir = Path("./checkpoints")
step_ckpts = list(checkpoint_dir.glob(f"torcs_ppo_v{i}_*_steps.zip"))

if step_ckpts:
    latest_ckpt_path = max(
        step_ckpts,
        key=lambda p: int(re.search(r'(\d+)_steps', p.name).group(1)),
    )
    step_match = re.search(r'(\d+)_steps', latest_ckpt_path.name)
    resumed_steps = int(step_match.group(1))

    vecnorm_name = f"torcs_ppo_v{i}_vecnormalize_{resumed_steps}_steps.pkl"
    vecnorm_ckpt = latest_ckpt_path.with_name(vecnorm_name)

    print(f"Resuming from checkpoint : {latest_ckpt_path.name}")
    print(f"Loading VecNormalize     : {vecnorm_name}")

    if not vecnorm_ckpt.exists():
        print("--- DEBUG ---")
        print(f"Looking for: {vecnorm_ckpt}")
        print(f"In folder  : {os.listdir(str(checkpoint_dir))}")
        raise FileNotFoundError(f"VecNormalize file not found: {vecnorm_ckpt}")

    # ── If expanding turn off code: ───────────────────────────────────────

    env = VecNormalize.load(str(vecnorm_ckpt), env)
    env.training = True

    model = PPO.load(
        str(latest_ckpt_path),
        env=env,
        tensorboard_log="./logs/",
        custom_objects=custom_objects,
        device="cuda",
        verbose=2
    )

    # ── Model expansion code ───────────────────────────────────────

    # with open(str(vecnorm_ckpt), 'rb') as f:
    #     old_vecnorm_data = pickle.load(f)
    #
    # env = VecNormalize(env, norm_obs=True, norm_reward=True, gamma=0.99)
    #
    # old_mean = old_vecnorm_data.obs_rms.mean
    # old_var = old_vecnorm_data.obs_rms.var
    # old_count = old_vecnorm_data.obs_rms.count
    #
    # new_dim = 42 #tu edytować dimension ze starego na nowe
    # new_mean = np.zeros(new_dim, dtype=np.float64)
    # new_var = np.ones(new_dim, dtype=np.float64)
    # new_mean[:len(old_mean)] = old_mean
    # new_var[:len(old_var)] = old_var
    #
    # env.obs_rms.mean = new_mean
    # env.obs_rms.var = new_var
    # env.obs_rms.count = old_count
    # env.ret_rms = old_vecnorm_data.ret_rms
    # env.training = True
    #
    # model = PPO.load(
    #     str(latest_ckpt_path),
    #     env=None,
    #     tensorboard_log="./logs/",
    #     custom_objects=custom_objects,
    #     device="cuda",
    #     verbose=2
    # )
    #
    # expand_model_obs(model, env, old_dim=41, new_dim=new_dim) #tu edytować stare dimension
    #
    # model.set_env(env)
    #
    # from stable_baselines3.common.buffers import RolloutBuffer
    #
    # model.rollout_buffer = RolloutBuffer(
    #     model.n_steps,
    #     model.observation_space,
    #     model.action_space,
    #     device=model.device,
    #     gamma=model.gamma,
    #     gae_lambda=model.gae_lambda,
    #     n_envs=model.n_envs,
    # )

else:
    final_model = Path(f"torcs_ppo_v{i}.zip")
    final_vec = Path(f"torcs_vecnorm_v{i}.pkl")

    if final_model.exists() and final_vec.exists():
        print(f"Resuming from final model: {final_model.name}")
        print(f"Loading VecNormalize     : {final_vec.name}")

        env = VecNormalize.load(str(final_vec), env)
        env.training = True

        model = PPO.load(
            str(final_model),
            env=env,
            tensorboard_log="./logs/",
            custom_objects=custom_objects,
            device="cuda",
            verbose=2
        )
    else:
        print("No checkpoint and no base model found — starting from scratch")
        env = VecNormalize(env, norm_obs=True, norm_reward=True, gamma=0.99)
        env.training = True
        model = PPO(
            "MlpPolicy",
            env=env,
            n_steps=custom_objects["n_steps"],
            batch_size=custom_objects["batch_size"],
            learning_rate=custom_objects["learning_rate"],
            ent_coef=custom_objects["ent_coef"],
            gamma=custom_objects["gamma"],
            gae_lambda=custom_objects["gae_lambda"],
            clip_range=custom_objects["clip_range"],
            tensorboard_log="./logs/",
            device="cuda",
            verbose=2,
        )

# ── Callbacks ─────────────────────────────────────────────────────────────────

checkpoint_cb = CheckpointCallback(
    save_freq=CHECKPOINT_EVERY,
    save_path=str(checkpoint_dir),
    name_prefix=f"torcs_ppo_v{i}",
    save_vecnormalize=True,
    verbose=1,
)

pause_cb = PauseDuringUpdateCallback(torcs_env, custom_objects["n_steps"])

reward_cb = RewardLoggerCallback()

# ── Train ─────────────────────────────────────────────────────────────────────

try:
    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        reset_num_timesteps=False,
        callback=[checkpoint_cb, SaveEpisodeOffsetCallback(CHECKPOINT_EVERY), pause_cb, reward_cb],
        tb_log_name=f"torcs_ppo_v{i}",
    )
    learning_completed = True
except KeyboardInterrupt:
    print("\n!!! Training interrupted (Ctrl+C) !!!")
    learning_completed = False
except Exception as e:
    print(f"✗ Learning ending failed: {e}")
    learning_completed = False

# ── Save with explicit flush ───────────────────────────────────────────────

if learning_completed:
    print("Saving model...")
    import sys

    sys.stdout.flush()

    try:
        model.save(f"torcs_ppo_v{i + 1}")
        print(f"✓ Model saved")
        sys.stdout.flush()

        env.save(f"torcs_vecnorm_v{i + 1}.pkl")
        print(f"✓ VecNormalize saved")
        sys.stdout.flush()

        np.save("episode_offset.npy", np.array([env.get_attr("episode")[0]]))
        print(f"✓ Episode offset saved")
        sys.stdout.flush()

        print(f"Fine-tuning run {i + 1} complete! ✓")
    except Exception as e:
        print(f"✗ Save failed: {e}")
        import traceback

        traceback.print_exc()
else:
    print("Training incomplete - checkpoint remains in ./checkpoints/")