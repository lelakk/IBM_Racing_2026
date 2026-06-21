# record_demos.py
import numpy as np
import keyboard  # pip install keyboard
from gym_torcs2 import TorcsEnv
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

SAVE_PATH = "demos.npz"

env = TorcsEnv(vision=False, throttle=True, gear_change=False)
obs = env.reset()

observations, actions = [], []

print("Drive! WASD controls. Press Q to quit and save.")

try:
    while True:
        # Map keypresses to action
        steer = 0.0
        accel = 0.0
        if keyboard.is_pressed("d"):
            steer = -1.0
        elif keyboard.is_pressed("a"):
            steer = 1.0
        if keyboard.is_pressed("w"):
            accel = 1.0
        elif keyboard.is_pressed("s"):
            accel = -1.0   # brake
        if keyboard.is_pressed("q"):
            break

        action = np.array([steer, accel], dtype=np.float32)
        observations.append(obs)
        actions.append(action)

        obs, reward, done, _ = env.step(action)
        if done:
            obs = env.reset()

finally:
    np.savez(SAVE_PATH, obs=np.array(observations), actions=np.array(actions))
    print(f"Saved {len(observations)} transitions to {SAVE_PATH}")
    env.close()