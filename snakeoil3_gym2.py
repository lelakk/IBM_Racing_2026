#!/usr/bin/python
# snakeoil3_gym.py — fixed for multi-agent parallel training
# Zmiany względem oryginału:
#   1. Usunięto parse_the_command_line() z __init__ (bezpieczne dla multiprocessing)
#   2. Dodano time.sleep(0.002) w busy-wait get_servers_input → brak 100% CPU
#   3. Socket timeout ustawiony explicite na 1s (nie 10s) — szybszy restart
#   4. Bezpieczny shutdown (idempotentny)

import socket
import sys
import os
import time

PI = 3.14159265359
data_size = 2 ** 17


def clip(v, lo, hi):
    if v < lo:
        return lo
    elif v > hi:
        return hi
    else:
        return v


class Client:
    def __init__(self, H=None, p=None, i=None, e=None, t=None, s=None,
                 d=None, vision=False):
        self.vision = vision
        self.host = H or 'localhost'
        self.port = p or 3001
        self.sid = i or 'SCR'
        self.maxEpisodes = e or 1
        self.trackname = t or 'unknown'
        self.stage = s if s is not None else 3
        self.debug = d or False
        self.maxSteps = 100000  # 50 steps/sec → ~33 min max

        self.S = ServerState()
        self.R = DriverAction()
        self.so = None
        self.setup_connection()

    # ── Connection ────────────────────────────────────────────────────────────

    def setup_connection(self):
        try:
            self.so = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        except socket.error:
            print(f'[port {self.port}] Error: Could not create socket.')
            sys.exit(-1)

        # Timeout na tyle krótki żeby nie blokować workera zbyt długo,
        # ale wystarczający żeby serwer zdążył odpowiedzieć.
        self.so.settimeout(1.0)

        a = "-45 -19 -12 -7 -4 -2.5 -1.7 -1 -.5 0 .5 1 1.7 2.5 4 7 12 19 45"
        initmsg = f'{self.sid}(init {a})'

        n_fail = 10
        while True:
            try:
                self.so.sendto(initmsg.encode(), (self.host, self.port))
            except socket.error:
                sys.exit(-1)

            sockdata = ''
            try:
                raw, _ = self.so.recvfrom(data_size)
                sockdata = raw.decode('utf-8')
            except socket.timeout:
                print(f"[port {self.port}] Waiting for server... ({n_fail} retries left)")
                n_fail -= 1
                if n_fail < 0:
                    raise RuntimeError(
                        f"[port {self.port}] TORCS server not responding. "
                        "Start TORCS first!"
                    )
                time.sleep(1.0)
                continue

            if '***identified***' in sockdata:
                print(f"[port {self.port}] Client connected.")
                break

    # ── Main receive loop ─────────────────────────────────────────────────────

    def get_servers_input(self):
        """Odbiera stan gry z serwera. Blokuje do 1s, NIE spina CPU."""
        if not self.so:
            return

        while True:
            sockdata = ''
            try:
                raw, _ = self.so.recvfrom(data_size)
                sockdata = raw.decode('utf-8')
            except socket.timeout:
                # Serwer nie odpowiedział w ciągu 1s — czekamy dalej
                # ale oddajemy CPU zamiast spinować
                time.sleep(0.002)
                continue
            except socket.error as emsg:
                print(f"[port {self.port}] Socket error: {emsg}")
                time.sleep(0.002)
                continue

            if '***identified***' in sockdata:
                print(f"[port {self.port}] Re-identified.")
                continue
            elif '***shutdown***' in sockdata:
                print(f"[port {self.port}] Server shutdown received.")
                self.shutdown()
                return
            elif '***restart***' in sockdata:
                print(f"[port {self.port}] Server restart received.")
                self.shutdown()
                return
            elif not sockdata:
                time.sleep(0.002)
                continue
            else:
                self.S.parse_server_str(sockdata)
                if self.debug:
                    sys.stderr.write("\x1b[2J\x1b[H")
                    print(self.S)
                break

    # ── Send action ───────────────────────────────────────────────────────────

    def respond_to_server(self):
        if not self.so:
            return
        try:
            message = repr(self.R)
            self.so.sendto(message.encode(), (self.host, self.port))
        except socket.error as emsg:
            print(f"[port {self.port}] Error sending: {emsg}")
            sys.exit(-1)
        if self.debug:
            print(self.R.fancyout())

    # ── Shutdown ──────────────────────────────────────────────────────────────

    def shutdown(self):
        if not self.so:
            return
        print(f"[port {self.port}] Shutting down client.")
        try:
            self.so.close()
        except Exception:
            pass
        self.so = None


# ── Server state ───────────────────────────────────────────────────────────────

class ServerState:
    def __init__(self):
        self.servstr = str()
        self.d = dict()

    def parse_server_str(self, server_string):
        self.servstr = server_string.strip()[:-1]
        sslisted = self.servstr.strip().lstrip('(').rstrip(')').split(')(')
        for i in sslisted:
            w = i.split(' ')
            self.d[w[0]] = destringify(w[1:])

    def __repr__(self):
        out = str()
        for k in sorted(self.d):
            v = self.d[k]
            strout = ', '.join(str(x) for x in v) if type(v) is list else str(v)
            out += f"{k}: {strout}\n"
        return out


# ── Driver action ──────────────────────────────────────────────────────────────

class DriverAction:
    """Akcja wysyłana do serwera TORCS."""

    def __init__(self):
        self.d = {
            'accel':  0.2,
            'brake':  0,
            'clutch': 0,
            'gear':   1,
            'steer':  0,
            'focus':  [-60, -30, 0, 30, 60],
            'meta':   0,
        }

    def clip_to_limits(self):
        self.d['steer']  = clip(self.d['steer'],  -1, 1)
        self.d['brake']  = clip(self.d['brake'],   0, 1)
        self.d['accel']  = clip(self.d['accel'],   0, 1)
        self.d['clutch'] = clip(self.d['clutch'],  0, 1)
        if self.d['gear'] not in [-1, 0, 1, 2, 3, 4, 5, 6]:
            self.d['gear'] = 0
        if self.d['meta'] not in [0, 1]:
            self.d['meta'] = 0
        if (type(self.d['focus']) is not list
                or min(self.d['focus']) < -180
                or max(self.d['focus']) > 180):
            self.d['focus'] = 0

    def __repr__(self):
        self.clip_to_limits()
        out = str()
        for k, v in self.d.items():
            out += f'({k} '
            out += ' '.join(str(x) for x in v) if type(v) is list else '%.3f' % v
            out += ')'
        return out

    def fancyout(self):
        out = str()
        od = {k: v for k, v in self.d.items()
              if k not in ('gear', 'meta', 'focus')}
        for k in sorted(od):
            if k in ('clutch', 'brake', 'accel', 'steer'):
                out += f"{k}: {od[k]:.3f}\n"
            else:
                out += f"{k}: {od[k]}\n"
        return out


# ── Utility ────────────────────────────────────────────────────────────────────

def destringify(s):
    if not s:
        return s
    if type(s) is str:
        try:
            return float(s)
        except ValueError:
            return s
    elif type(s) is list:
        if len(s) < 2:
            return destringify(s[0])
        else:
            return [destringify(i) for i in s]