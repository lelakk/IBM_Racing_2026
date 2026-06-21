import gym
from gym import spaces
import numpy as np
import snakeoil3_gym as snakeoil3
import copy
import collections as col
import os
import time
import subprocess

TORCS_DIR = r"C:\Users\janik\Desktop\torcs"
TORCS_EXE = "wtorcs.exe"

# Observation vector:
# focus(5) + speedX(1) + speedY(1) + speedZ(1) + opponents(36)
# + rpm(1) + track(19) + wheelSpinVel(4) = 68
OBS_DIM = 68


class TorcsEnv(gym.Env):
    terminal_judge_start       = 500
    termination_limit_progress = 5
    default_speed              = 50

    initial_reset = True

    def __init__(self, vision=False, throttle=False, gear_change=False):
        super().__init__()
        self.vision      = vision
        self.throttle    = throttle
        self.gear_change = gear_change
        self.initial_run = True

        # NOTE: Do NOT kill or launch TORCS here.
        # Start TORCS manually, configure the race, then run the training script.

        # Action space: [steer] or [steer, accel]
        n_actions = 2 if throttle else 1
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(n_actions,), dtype=np.float32
        )

        # Observation space - must match get_obs() exactly (68 values)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(OBS_DIM,), dtype=np.float32
        )

    def auto_gear(self, obs):
        speed = float(obs["speedX"])  # km/h
        gear  = int(float(obs["gear"]))
       # print(f"gear={obs['gear']} type={type(obs['gear'])}  rpm={obs['rpm']} type={type(obs['rpm'])}")

        # Speed thresholds for each gear
        # Adjust these if the car feels wrong
        up_thresholds   = [0,  40,  70, 100, 130, 160]  # shift up above this speed
        down_thresholds = [0,   0,  30,  60,  90, 120]  # shift down below this speed

        gear = max(1, min(gear, 6))  # clamp to valid range first

        if gear < 6 and speed > up_thresholds[gear]:
            gear += 1
        elif gear > 1 and speed < down_thresholds[gear]:
            gear -= 1

        return gear

    def step(self, u):
        client = self.client
        this_action  = self.agent_to_torcs(u)
        action_torcs = client.R.d

        # ── Steering (always from agent) ────────────────────────────────────
        action_torcs["steer"] = this_action["steer"]

        # ── Throttle / Brake ────────────────────────────────────────────────
        if not self.throttle:
            obs   = client.S.d
            sx    = float(obs["speedX"])
            track = np.array(obs["track"])   # 19 rangefinder distances

            # Forward-facing sensors (centre 5, roughly +-7 degrees)
            # Low value = wall close ahead = corner coming
            forward_sensors = track[7:12]
            min_forward     = float(forward_sensors.min())

            # Asymmetry between left and right sensors detects how sharp the curve is
            left_sensors    = track[0:5].mean()
            right_sensors   = track[14:19].mean()
            track_asymmetry = abs(float(left_sensors) - float(right_sensors))

            # Dynamic target speed:
            #   long straight (min_forward near 200) -> up to 120 km/h
            #   tight corner  (min_forward small, high asymmetry) -> as low as 15 km/h
            target_speed = float(np.clip(
                0.6 * min_forward - 10.0 * track_asymmetry,
                15.0,    # floor:   never target below 15 km/h
                120.0    # ceiling: never target above 120 km/h
            ))

            speed_error = target_speed - sx

            if speed_error > 0:
                # Need to accelerate - proportional to how far below target we are
                client.R.d["accel"] = float(np.clip(0.02 * speed_error, 0.0, 1.0))
                client.R.d["brake"] = 0.0
            else:
                # Need to slow down - use actual brake, not just lifting throttle
                client.R.d["accel"] = 0.0
                client.R.d["brake"] = float(np.clip(-0.05 * speed_error, 0.0, 1.0))

            # Traction control: rear wheels spinning faster than front = wheelspin
            wsv = obs["wheelSpinVel"]
            if (wsv[2] + wsv[3]) - (wsv[0] + wsv[1]) > 5:
                client.R.d["accel"] -= 0.3

            # Standstill kick: strong push if nearly stopped
            if sx < 5:
                client.R.d["accel"] += 1.0 / (sx + 0.1)
                client.R.d["brake"] = 0.0

        else:
            accel_action = float(this_action["accel"])
            if accel_action >= 0:
                action_torcs["accel"] = accel_action
                client.R.d["brake"]   = 0.0
            else:
                action_torcs["accel"] = 0.0
                client.R.d["brake"]   = -accel_action  # negative accel becomes brake

        # ── Gear ────────────────────────────────────────────────────────────
        action_torcs["gear"] = self.auto_gear(client.S.d)

        # ── Send controls, receive new state ────────────────────────────────
        obs_pre = copy.deepcopy(client.S.d)
        client.respond_to_server()
        client.get_servers_input()

        obs = client.S.d
        self.observation = self.make_observaton(obs)

        # ── Reward ──────────────────────────────────────────────────────────
        track    = np.array(obs["track"])
        sp       = float(obs["speedX"])
        speed_bonus   = 0.004 * sp
        #angle_penalty   = 0.5  * abs(np.angle)        # penalise being angled to track

        progress = sp * np.cos(obs["angle"]) 
        reward   = progress + speed_bonus #- angle_penalty

        if obs["damage"] - obs_pre["damage"] > 0:
            reward -= 10.0

        # ── Termination ─────────────────────────────────────────────────────
        done = False

        if track.min() < 0:                          # any sensor negative = off track
            reward = -100.0
            done   = True
            client.R.d["meta"] = True

        if self.terminal_judge_start < self.time_step:
            if progress < self.termination_limit_progress:  # stuck or barely moving
                done = True
                client.R.d["meta"] = True

        if np.cos(obs["angle"]) < 0:                 # facing backwards
            done = True
            reward = -100.0
            client.R.d["meta"] = True

        if client.R.d["meta"]:
            self.initial_run = False
            client.respond_to_server()

        self.time_step += 1
        return self.get_obs(), reward, done, {}

    def reset(self, relaunch=False):
        self.time_step = 0

        if not self.initial_reset:
            self.client.R.d["meta"] = True
            self.client.respond_to_server()

            if relaunch:
                self.reset_torcs()

        # Port 3001 = SCR server default
        self.client = snakeoil3.Client(p=3001, vision=self.vision)
        self.client.MAX_STEPS = float("inf")

        self.client.get_servers_input()
        obs = self.client.S.d
        self.observation = self.make_observaton(obs)

        self.last_u        = None
        self.initial_reset = False
        return self.get_obs()

    def reset_torcs(self):
        """Kill and relaunch TORCS. Only called when reset(relaunch=True)."""
        subprocess.call(
            ["taskkill", "/f", "/im", TORCS_EXE],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
        time.sleep(0.5)
        exe_path = os.path.join(TORCS_DIR, TORCS_EXE)
        flags = ["-nofuel", "-nodamage", "-nolaptime"]
        if self.vision:
            flags.append("-vision")
        subprocess.Popen([exe_path] + flags, cwd=TORCS_DIR)
        time.sleep(3.0)

    def end(self):
        subprocess.call(
            ["taskkill", "/f", "/im", TORCS_EXE],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )

    def get_obs(self):
        o = self.observation
        return np.hstack([
            o.focus,         # 5
            o.speedX,        # 1
            o.speedY,        # 1
            o.speedZ,        # 1
            o.opponents,     # 36
            o.rpm,           # 1
            o.track,         # 19
            o.wheelSpinVel   # 4
        ]).astype(np.float32)  # 68 total

    def agent_to_torcs(self, u):
        action = {"steer": float(u[0])}
        if self.throttle:
            action["accel"] = float(u[1])
        action["gear"] = int(u[2]) if self.gear_change else 1
        return action

    def make_observaton(self, raw_obs):
        names = ["focus", "speedX", "speedY", "speedZ",
                 "opponents", "rpm", "track", "wheelSpinVel"]
        Obs = col.namedtuple("Observation", names)
        sp  = self.default_speed
        return Obs(
            focus        = np.array(raw_obs["focus"],        dtype=np.float32) / 200.0,
            speedX       = np.array(raw_obs["speedX"],       dtype=np.float32) / sp,
            speedY       = np.array(raw_obs["speedY"],       dtype=np.float32) / sp,
            speedZ       = np.array(raw_obs["speedZ"],       dtype=np.float32) / sp,
            opponents    = np.array(raw_obs["opponents"],    dtype=np.float32) / 200.0,
            rpm          = np.array(raw_obs["rpm"],          dtype=np.float32),
            track        = np.array(raw_obs["track"],        dtype=np.float32) / 200.0,
            wheelSpinVel = np.array(raw_obs["wheelSpinVel"], dtype=np.float32),
        )

    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]