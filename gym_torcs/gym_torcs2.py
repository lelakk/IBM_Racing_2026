from math import floor

from pathlib import Path
import gym
import threading
from gym import spaces
import numpy as np
# add near imports in gym_torcs2.py
import ctypes
import logging
try:
    import psutil
except Exception:
    psutil = None

Path("logs").mkdir(exist_ok=True)

logging.basicConfig(
    filename="logs/training.log",
    level=logging.INFO,
    format="%(asctime)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    encoding="utf-8",
)

def tlog(msg: str):
    print(msg)
    logging.info(msg)

SPEEDUP_PRESSES = 7  # 7 - max
WM_KEYDOWN = 0x0100
WM_KEYUP = 0x0101
VK_ADD = 0x6B        # numpad +
VK_SUBTRACT = 0x6D   # numpad -
VK_PERIOD = 0xBE  # klawisz .
import snakeoil3_gym as snakeoil3
import os
import csv
import time
import subprocess
from pathlib import Path

TORCS_DIR = r"C:\torcs"
TORCS_EXE = "wtorcs.exe"

OBS_DIM = 49

TRACK_LEN = 3608.45
CP_AMOUNT = 160
_CP_SPACING = TRACK_LEN / CP_AMOUNT
CHECKPOINTS = [_CP_SPACING * i for i in range(1, CP_AMOUNT+1)]
CP_REWARD = 2.0

CORNER_APPROACH_M = 80


CORNER_ZONES = [
    (380, 550, 120, -1),
    (710, 810, 130, 1),
    (990,1060, 150, 1),
    (1430, 1610, 160, -1),
    (1880,2000, 160, -1),
    (2400,2535, 45, -1),
    (2630,2780, 140, -1),
    (2930,3020, 150, 1),
    (3240,3300, 65, -1)
]

SLOW_ZONES = [
    (300,379, 150),
    (2250,2399, 80),
    (2536,2629, 180),
    (2830,2929, 180),
    (3030,3239, 80)
]


# ── Crash telemetry log ────────────────────────────────────────────────────────
CRASH_LOG = Path("./telemetry/crashes.csv")
CRASH_LOG.parent.mkdir(exist_ok=True)
if not CRASH_LOG.exists():
    with open(CRASH_LOG, "w", newline="") as f:
        csv.writer(f).writerow(["episode", "step", "dist_raced", "track_pos",
                                 "speed_x", "angle", "reason"])



def _append_crash(episode: int, step: int, obs: dict, reason: str):
    with open(CRASH_LOG, "a", newline="") as f:
        csv.writer(f).writerow([
            episode, step,
            float(obs.get("distRaced", 0.0)),
            float(obs.get("trackPos", 0.0)),
            float(obs.get("speedX", 0.0)),
            float(obs.get("angle", 0.0)),
            reason,
        ])


class TorcsEnv(gym.Env):
    terminal_judge_start       = 500
    default_speed              = 50
    termination_limit_progress = -1

    def _speedup_torcs(self, presses=SPEEDUP_PRESSES):
        user32 = ctypes.windll.user32

        EnumWindowsProc = ctypes.WINFUNCTYPE(
            ctypes.c_bool,
            ctypes.c_void_p,
            ctypes.c_void_p
        )

        def callback(hwnd, _):
            if not user32.IsWindowVisible(hwnd):
                return True

            length = user32.GetWindowTextLengthW(hwnd)
            if length == 0:
                return True

            buff = ctypes.create_unicode_buffer(length + 1)
            user32.GetWindowTextW(hwnd, buff, length + 1)

            title = buff.value.lower()

            # only TORCS windows
            if "torcs.exe" not in title:
                return True

            user32.SetForegroundWindow(hwnd)

            user32.PostMessageW(hwnd, WM_KEYDOWN, VK_PERIOD, 0)
            user32.PostMessageW(hwnd, WM_KEYUP, VK_PERIOD, 0)
            for _ in range(presses):
                user32.PostMessageW(hwnd, WM_KEYDOWN, VK_ADD, 0)
                user32.PostMessageW(hwnd, WM_KEYUP, VK_ADD, 0)

            return True

        user32.EnumWindows(EnumWindowsProc(callback), 0)

    def _pause_torcs(self):
        user32 = ctypes.windll.user32
        VK_P = 0x50

        EnumWindowsProc = ctypes.WINFUNCTYPE(
            ctypes.c_bool,
            ctypes.c_void_p,
            ctypes.c_void_p
        )

        def callback(hwnd, _):
            if not user32.IsWindowVisible(hwnd):
                return True

            length = user32.GetWindowTextLengthW(hwnd)
            if length == 0:
                return True

            buff = ctypes.create_unicode_buffer(length + 1)
            user32.GetWindowTextW(hwnd, buff, length + 1)

            title = buff.value.lower()

            if "torcs.exe" not in title:
                return True

            user32.SetForegroundWindow(hwnd)
            user32.SendMessageW(hwnd, WM_KEYDOWN, VK_P, 0)
            user32.SendMessageW(hwnd, WM_KEYUP, VK_P, 0)

            return True

        user32.EnumWindows(EnumWindowsProc(callback), 0)

    def _slowdown_torcs(self, presses=SPEEDUP_PRESSES):
        user32 = ctypes.windll.user32

        EnumWindowsProc = ctypes.WINFUNCTYPE(
            ctypes.c_bool,
            ctypes.c_void_p,
            ctypes.c_void_p
        )

        def callback(hwnd, _):
            if not user32.IsWindowVisible(hwnd):
                return True

            length = user32.GetWindowTextLengthW(hwnd)
            if length == 0:
                return True

            buff = ctypes.create_unicode_buffer(length + 1)
            user32.GetWindowTextW(hwnd, buff, length + 1)

            title = buff.value.lower()

            if "torcs.exe" not in title:
                return True

            user32.SetForegroundWindow(hwnd)

            user32.PostMessageW(hwnd, WM_KEYDOWN, VK_PERIOD, 0)
            user32.PostMessageW(hwnd, WM_KEYUP, VK_PERIOD, 0)
            for _ in range(presses):
                user32.PostMessageW(hwnd, WM_KEYDOWN, VK_SUBTRACT, 0)
                user32.PostMessageW(hwnd, WM_KEYUP, VK_SUBTRACT, 0)

            return True

        user32.EnumWindows(EnumWindowsProc(callback), 0)

    def pause(self):
        if not self.paused:
            self._pause_torcs()
            self.paused = True
            time.sleep(0.1)  # daj czas na przetworzenie inputu
            self._slowdown_torcs()  # zwolnij do 1x

            tlog(f"[PAUSE] episode {self.episode}")

    def unpause(self):
        tlog(f"[UNPAUSE START] paused={self.paused}")
        if self.paused:
            tlog(f"[UNPAUSE] stopping keepalive...")
            self._keepalive_running = False
            if self._keepalive_thread is not None:
                tlog(f"[UNPAUSE] joining thread...")
                self._keepalive_thread.join(timeout=2.0)  # ← timeout żeby nie zablokować
                self._keepalive_thread = None
            tlog(f"[UNPAUSE] pressing p...")
            self._pause_torcs()
            self.paused = False
            time.sleep(0.1)
            tlog(f"[UNPAUSE] speedup...")
            self._speedup_torcs()
            tlog(f"[UNPAUSE END]")

    def __init__(self, vision=False, throttle=False, gear_change=False,
                 port=3001, episode_offset=0, n_steps_per_update = 16384):
        super().__init__()
        self.vision                   = vision
        self.throttle                 = throttle
        self.gear_change              = gear_change
        self.port                     = port
        self.initial_reset            = True
        self._meta_sent               = False
        self.last_u                   = np.zeros(2, dtype=np.float32)
        self.episode                  = episode_offset
        self.prev_steer               = 0.0
        self.stopped_steps            = 0
        self.prev_laps                = 0
        self._cp_reached: set[int]    = set()
        self.steps_from_gearchange    = 0
        self._last_steer_delta        = 0
        self.zone_penality_mult       = 1.0
        self.in_corner                = False
        self.target_speed_corner      = 0
        self.in_slow_zone             = False
        self.target_speed_slow_zone   = 0
        self.paused                   = False
        self.steps_from_update        = 0
        self.n_steps_per_update       = n_steps_per_update
        self._keepalive_running       = False
        self._keepalive_thread        = None
        self._socket_lock             = threading.Lock()
        self._in_update               = False
        self._torcs_procs             = []
        self._cp_reached_corners: set = set()
        self.corner_direction         = 0

        # Bufor na obserwację — alokowany RAZ, wypełniany in-place ──────────
        self._obs_buf = np.zeros(OBS_DIM, dtype=np.float32)

        if throttle and gear_change:
            self.action_space = spaces.Box(
                low=np.array([-1.0, -1.0, -1.0], dtype=np.float32),
                high=np.array([1.0, 1.0, 1.0], dtype=np.float32),
            )
            # u[0]=steer, u[1]=accel/brake, u[2]=zmiana biegu (-1/0/+1)
        elif throttle:
            self.action_space = spaces.Box(
                low=np.array([-1.0, -1.0], dtype=np.float32),
                high=np.array([1.0, 1.0], dtype=np.float32),
            )
        else:
            self.action_space = spaces.Box(
                low=-1.0, high=1.0, shape=(1,), dtype=np.float32
            )

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(OBS_DIM,), dtype=np.float32
        )

    # ── Gear ──────────────────────────────────────────────────────────────────

    def auto_gear(self, obs):
        gear = int(obs["gear"])

        rpm = float(obs["rpm"])

        if gear <= 0:
            self.steps_from_gearchange = 0
            return 1

        if self.steps_from_gearchange >= 50:

            if rpm > 11500 and gear < 6:
                self.steps_from_gearchange = 0
                return gear + 1

            if rpm < 6300 and gear > 1:
                self.steps_from_gearchange = 0
                return gear - 1

        self.steps_from_gearchange += 1
        return gear

    # ── Track Position ──────────────────────────────────────────────────────────────────

    def _is_corner(self, dist: float) -> tuple[bool, float | None, int]:
        for start, end, target_sp, direction in CORNER_ZONES:
            if start <= dist % TRACK_LEN <= end:
                return True, target_sp, direction
        return False, None, 0

    def _is_slow_zone(self, dist: float) -> tuple[bool, float | None]:
        for start, end, target_sp in SLOW_ZONES:
            if start <= dist % TRACK_LEN <= end:
                return True, target_sp
        return False, None

    # ── Step ──────────────────────────────────────────────────────────────────

    def _safe_sync(self):
        # Jedna sekcja krytyczna dla respond/get aby nie mieszać z keepalive.
        with self._socket_lock:
            self.client.respond_to_server()
            self.client.get_servers_input()

    def _safe_respond(self):
        with self._socket_lock:
            self.client.respond_to_server()

    def _safe_get_input(self):
        with self._socket_lock:
            self.client.get_servers_input()

    def _get_torcs_procs(self):
        if psutil is None:
            return []
        procs = []
        for p in psutil.process_iter(attrs=["name", "pid"]):
            name = (p.info.get("name") or "").lower()
            if name in {"wtorcs.exe", "torcs.exe"}:
                procs.append(p)
        return procs

    def _suspend_torcs(self):
        if psutil is None:
            return False
        self._torcs_procs = self._get_torcs_procs()
        if not self._torcs_procs:
            return False
        for p in self._torcs_procs:
            try:
                p.suspend()
            except Exception:
                pass
        tlog("[TORCS] suspended for PPO update")
        return True

    def _resume_torcs(self):
        if psutil is None or not self._torcs_procs:
            return
        for p in self._torcs_procs:
            try:
                p.resume()
            except Exception:
                pass
        self._torcs_procs = []
        tlog("[TORCS] resumed after PPO update")

    def begin_update(self):
        if self._keepalive_running:
            return
        self._in_update = True
        if not self._suspend_torcs():
            self._keepalive_running = True
            self._keepalive_thread = threading.Thread(
                target=self._keepalive_loop, daemon=True
            )
            self._keepalive_thread.start()

    def end_update(self):
        if self._keepalive_running:
            self._keepalive_running = False
            if self._keepalive_thread is not None:
                self._keepalive_thread.join(timeout=2.0)
                self._keepalive_thread = None
        self._resume_torcs()
        self._in_update = False

    def _keepalive_loop(self):
        tlog("[KEEPALIVE] start")
        while self._keepalive_running:
            try:
                self._safe_respond()
            except Exception as e:
                tlog(f"[KEEPALIVE ERR] {e}")
                break
            time.sleep(0.02)
        tlog("[KEEPALIVE] stop")

    def step(self, u):
        client = self.client
        action_torcs = client.R.d

        steer_val = float(u[0])
        raw_accel = float(u[1])

        action_torcs["steer"] = steer_val
        action_torcs["accel"] = raw_accel if raw_accel > 0.0 else 0.0
        action_torcs["brake"] = -raw_accel if raw_accel < 0.0 else 0.0
        current_gear = int(client.S.d.get("gear", 1))

        if self.gear_change and len(u) >= 3 and self.episode>5000:
            gear_action = float(u[2])
            if gear_action > 0.5 and self.steps_from_gearchange >= 30:
                new_gear = min(current_gear + 1, 6)
                self.steps_from_gearchange = 0
            elif gear_action < -0.5 and self.steps_from_gearchange >= 30:
                new_gear = max(current_gear - 1, 1)
                self.steps_from_gearchange = 0
            else:
                new_gear = current_gear
                self.steps_from_gearchange += 1
        else:
            new_gear = self.auto_gear(client.S.d)

        action_torcs["gear"] = new_gear

        obs_pre_damage = float(client.S.d.get("damage", 0.0))

        self._safe_sync()
        obs = client.S.d
        self._last_steer_delta = steer_val - self.last_u[0]

        # Aktualizuj last_u in-place
        self.last_u[0] = steer_val
        self.last_u[1] = action_torcs["accel"] - action_torcs["brake"]

        angle = float(obs["angle"])
        sp = float(obs["speedX"])
        track_pos = float(obs["trackPos"])
        track = obs["track"]  # lista z serwera — indeksujemy bezpośrednio

        min_forward = min(track[7], track[8], track[9], track[10], track[11])
        min_left = min(track[0], track[1], track[2], track[3])
        min_right = min(track[15], track[16], track[17], track[18])

        cos_angle = np.cos(angle)
        progress = sp * cos_angle

        if sp <= 1:
            self.stopped_steps += 1
        else:
            self.stopped_steps = 0

        if self.episode <= 1000:
            self.zone_penality_mult = 0.0
        elif self.episode < 3000:
            self.zone_penality_mult = (self.episode - 1000) / 2000.0
        else:
            self.zone_penality_mult = 1.0


        distance_raced = float(obs.get("distRaced", 0.0))
        distance_from_start = float(obs.get("distFromStart", 0.0))

        self.in_corner, self.target_speed_corner, self.corner_direction = self._is_corner(distance_from_start)
        self.in_slow_zone, self.target_speed_slow_zone  = self._is_slow_zone(distance_from_start)

        steer_delta = abs(steer_val - self.prev_steer)

        laps = int(floor(distance_raced / TRACK_LEN))

        cp_idx = int(distance_from_start / _CP_SPACING)
        cp_idx = min(cp_idx, CP_AMOUNT - 1)

        self._fill_obs(obs)

        # ── Reward ────────────────────────────────────────────────────────────

        reward = np.tanh(progress / 150.0) * 1.5  # prędkość rzutowana na kierunek toru, ograniczona do (-1, 1)
        reward += (sp / 300.0) * 0.8 # silniejszy bonus za prędkość
        reward += (sp / 300.0) ** 2 * 0.5  # dodatkowy bonus rosnący z prędkością
        reward += 0.001  # mały bonus za przeżycie kroku
        reward += cos_angle * 0.1  # lekka nagroda za równoległość do toru

        if abs(track_pos) > 0.9:
            reward -= ((abs(track_pos) + 0.1) ** 1.2) * 0.4  # łagodniejsza kara za zjazd z toru
        reward -= abs(angle) * 0.1  # łagodniejsza kara za kąt do osi toru
        reward -= (0.01 * self.stopped_steps) ** 1.5  # rosnąca kara za stanie w miejscu

        rpm = float(obs["rpm"])
        if (rpm > 12000 and current_gear < 6) or (rpm < 5500 and current_gear > 1):
            overshoot = max(0, int(rpm - 12000)) / 3000.0
            undershoot = max(0, int(5500 - rpm)) / 4500.0
            reward -= (overshoot + undershoot) * 0.1  # łagodniejsza kara za złe obroty

        if cp_idx not in self._cp_reached and distance_from_start > 0:
            self._cp_reached.add(cp_idx)  # zalicz nowy checkpoint
            reward += CP_REWARD #nagroda za checkpoint

        if laps > self.prev_laps:
            reward += 40  # bonus za ukończenie okrążenia
            self._cp_reached = set()  # reset checkpointów na nowe okrążenie
            self._cp_reached_corners = set()
            self.prev_laps += 1

        # Jednorazowa nagroda za pozycję przy wejściu w zakręt
        for start, end, target_sp, direction in CORNER_ZONES:
            approach_start = (start - CORNER_APPROACH_M) % TRACK_LEN
            corner_key = start  # identyfikator zakrętu

            in_approach = approach_start <= distance_from_start % TRACK_LEN < start

            if in_approach and corner_key not in self._cp_reached_corners:
                # track_pos w kierunku idealnym: direction=+1 chcemy trackPos>0, direction=-1 chcemy trackPos<0
                aligned = track_pos * direction  # >0 jeśli po właściwej stronie
                approach_reward = np.clip((aligned + 1.0) / 2.0, 0.0, 1.0)  # 0–1 w zależności od tego czy agent ustawia się po dobrej stronie toru
                reward += approach_reward
                self._cp_reached_corners.add(corner_key)
                tlog(
                    f"[APPROACH] corner@{start} | trackPos={track_pos:.2f} | dir={direction} | r={approach_reward:.3f}")
                break

        if self.in_corner:
            if sp < self.target_speed_corner:
                reward -= (self.target_speed_corner - sp) * 0.05 * self.zone_penality_mult  # kara za wolną jazdę w zakręcie
            if sp > (self.target_speed_corner + 20):
                reward -= (sp - (self.target_speed_corner + 20)) * 0.05 * self.zone_penality_mult  # kara za wolną jazdę w zakręcie
        elif self.in_slow_zone:
            if steer_delta <= 0.1 and abs(steer_val) <= 0.1:
                reward += 0.02  # bonus za płynną jazdę prosto
            if sp < self.target_speed_slow_zone:
                reward -= (self.target_speed_slow_zone - sp) * 0.01 * self.zone_penality_mult
            if sp > (self.target_speed_slow_zone + 20):
                reward -= (sp - (self.target_speed_slow_zone + 20)) * 0.06 * self.zone_penality_mult# kara za szybką jazdę w slow zone
            reward -= abs(steer_val)*0.2* self.zone_penality_mult  # kara za wychylenie kierownicy
            reward -= steer_delta * 0.5* self.zone_penality_mult  # kara za gwałtowne skręty
        else:
            reward -= steer_delta * 0.7 * self.zone_penality_mult
            reward -= (steer_delta ** 2) * 0.4 * self.zone_penality_mult
            reward -= abs(steer_val) * 0.2 * self.zone_penality_mult
            reward += (sp / 300.0) * 0.1
            reward += action_torcs["accel"] * 0.08 * self.zone_penality_mult
            reward -= action_torcs["brake"] * 0.05 * self.zone_penality_mult

        self.prev_steer = steer_val

        damage_diff = float(obs["damage"]) - obs_pre_damage
        if damage_diff > 0:
            reward -= 10.0  # łagodniejsza kara za kolizję


        # ── Termination ───────────────────────────────────────────────────────
        done   = False
        reason = None

        if abs(track_pos) > 1.2:
            reward -= 100.0
            done    = True
            reason  = "off_track"
        elif self.terminal_judge_start < self.time_step and progress < self.termination_limit_progress:
            reward -= 100.0
            done    = True
            reason  = "backing"
        elif cos_angle < -0.3:
            reward -= 100.0
            done    = True
            reason  = "wrong_way"
        elif self.stopped_steps >= 1000:
            reward -= 120.0
            done = True
            reason = "no_progress"
        elif self.time_step >= 21000:
            done = True
            reason = "timeout"


        if done:
            _append_crash(self.episode, self.time_step, obs, reason)
            tlog(f"[port {self.port}] episode {self.episode} | step {self.time_step:>5} | dist {float(obs.get('distFromStart', 0.0)):.1f}m | "
                  f"angle={angle:.3f} | trackPos={track_pos:.3f} | speedX={sp:.1f} | "
                  f"reward {reward:.3f} | Episode {self.episode} ended: {reason}")
            client.R.d["meta"] = True
            self._safe_respond()
            self._meta_sent = True

        self.time_step += 1
        if self.time_step % 250 == 0:
            tlog(
                f"[port {self.port}] ep {self.episode} step {self.time_step} | "
                f"reward {reward:.3f} | sp {sp:.1f} | steer {steer_val:.3f} | steer_δ {steer_delta:.3f} | "
                f"accel {action_torcs['accel']:.2f} | brake {action_torcs['brake']:.2f} | gear {new_gear} | rpm {rpm:.0f} | "
                f"track_pos {track_pos:.3f} | angle {angle:.3f} | "
                f"min_fwd {min_forward:.1f}m | min_r {min_right:.1f}m | min_l {min_left:.1f}m | "
                f"dist {distance_from_start:.1f}m | dist_raced {distance_raced:.1f}m | lap {laps} | "
                f"corner {self.in_corner} | slow {self.in_slow_zone} | zone_mult {self.zone_penality_mult:.2f} | "
                f"stopped {self.stopped_steps} | damage_d {damage_diff:.1f}"
            )

        return self._obs_buf.copy(), reward, done, {}

    # ── Reset ─────────────────────────────────────────────────────────────────

    def reset(self, relaunch=False):
        self.time_step                = 0
        self.prev_steer               = 0.0
        self.stopped_steps            = 0
        self.prev_laps                = 0
        self.episode                 += 1
        self._cp_reached              = set()
        self.steps_from_gearchange    = 0
        self._last_steer_delta        = 0.0
        self.zone_penality_mult       = 1.0
        self.in_corner                = False
        self.target_speed_corner      = 0
        self.in_slow_zone             = False
        self.target_speed_slow_zone   = 0
        self._keepalive_running       = False
        self.paused                   = False
        self._cp_reached_corners: set = set()
        self.corner_direction         = 0
        if self._keepalive_thread is not None:
            self._keepalive_thread.join()
            self._keepalive_thread = None

        if not self.initial_reset:
            if not self._meta_sent:
                try:
                    self.client.R.d["meta"] = True
                    self.client.respond_to_server()
                except Exception:
                    pass
            try:
                self.client.shutdown()
            except Exception:
                pass

        for attempt in range(5):
            try:
                self._meta_sent = False
                self.client = snakeoil3.Client(p=self.port, vision=self.vision)
                if getattr(self.client, "so", None) is not None:
                    self.client.so.settimeout(1.0)
                self.client.MAX_STEPS = float("inf")
                self.client.R.d["focus"] = [-60, -30, 0, 30, 60]
                self._safe_respond()
                self._safe_get_input()
                break
            except Exception as e:
                tlog(f"[port {self.port}] Connection attempt {attempt+1} failed: {e}")
                time.sleep(2.0)

        self._speedup_torcs()
        time.sleep(0.01)
        self.last_u[:] = 0.0
        self._fill_obs(self.client.S.d)
        self.initial_reset = False
        return self._obs_buf.copy()

    # ── Obs helper — wypełnia bufor IN-PLACE, bez alokacji ───────────────────

    def _fill_obs(self, raw_obs):
        buf = self._obs_buf

        buf[0]  = np.clip(raw_obs["speedX"] / 320.0, -1.0, 1.0)
        buf[1]  = np.clip(raw_obs["speedY"] / 50.0, -1.0, 1.0)
        buf[2]  = np.clip(raw_obs["speedZ"] / 5.0, -1.0, 1.0)
        buf[3]  = np.clip(raw_obs["rpm"]    / 18000.0, 0.0, 1.0)

        track = raw_obs["track"]
        for i in range(19):
            buf[4 + i] = np.clip(track[i] / 200.0, 0.0, 1.0)          # indeksy 4–22

        wsv = raw_obs["wheelSpinVel"]
        for i in range(4):
            buf[23 + i] = np.clip(wsv[i] / 100.0, 0.0, 1.0)            # indeksy 23–26

        buf[27] = np.clip(raw_obs["angle"] / np.pi, -1.0, 1.0)
        buf[28] = np.clip(raw_obs["trackPos"]/1.2, -1.0, 1.0)

        front_wsv = wsv[0]
        if front_wsv != 0:
            buf[29] = np.clip(0.5555555555 * raw_obs["speedX"] / front_wsv - 0.66124, -1.0, 1.0)
        else:
            buf[29] = 0.0

        buf[30] = np.clip((wsv[2] + wsv[3] - wsv[0] - wsv[1]) / 100.0, -1.0, 1.0)  # slip

        buf[31] = self.last_u[0]
        buf[32] = self.last_u[1]

        focus = raw_obs.get("focus", [200.0] * 5)
        for i in range(5):
            buf[33 + i] = np.clip(focus[i] / 200.0, 0.0, 1.0)          # indeksy 33–37

        buf[38] = np.clip(raw_obs.get("distFromStart", 0.0) / TRACK_LEN, 0.0, 1.0)
        buf[39] = np.clip(self.stopped_steps / 1000.0, 0.0, 1.0)
        buf[40] = np.clip(int(floor(raw_obs.get("distRaced", 0.0) / TRACK_LEN)) / 20.0, 0.0, 1.0)
        buf[41] = np.clip(raw_obs.get("curLapTime", 0.0) / 150.0, 0.0, 1.0)
        buf[42] = np.clip(float(raw_obs.get("gear", 1)) / 6.0, 0.0, 1.0)
        buf[43] = np.clip(self._last_steer_delta / 2.0, -1.0, 1.0)
        buf[44] = self.zone_penality_mult
        buf[45] = 1.0 if self.in_corner else 0.0
        buf[46] = 1.0 if self.in_slow_zone else 0.0
        buf[47] = np.clip((self.target_speed_corner or self.target_speed_slow_zone or 320.0) / 320.0, 0.0, 1.0)
        buf[48] = float(self.corner_direction)

    # ── Misc ──────────────────────────────────────────────────────────────────

    def reset_torcs(self):
        subprocess.call(
            ["taskkill", "/f", "/im", TORCS_EXE],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        time.sleep(0.5)
        exe_path = os.path.join(TORCS_DIR, TORCS_EXE)
        flags = ["-nofuel", "-nodamage", "-nolaptime", "-port", str(self.port)]
        if self.vision:
            flags.append("-vision")

        proc = subprocess.Popen(
            [exe_path] + flags,
            cwd=TORCS_DIR,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )

        def _pipe_torcs_tlog(p):
            for line in p.stdout:
                line = line.rstrip()
                if line:
                    tlog(f"[TORCS] {line}")

        threading.Thread(target=_pipe_torcs_tlog, args=(proc,), daemon=True).start()
        time.sleep(3.0)

    def end(self):
        subprocess.call(
            ["taskkill", "/f", "/im", TORCS_EXE],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )

    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]

    def close(self):
        try:
            self.client.socket.close()
        except Exception:
            pass
        self.end()
