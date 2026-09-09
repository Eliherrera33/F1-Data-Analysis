# -*- coding: utf-8 -*-
"""
Quasi-steady lap-time model. Reference implementation for lapsim.js.

Given a track as curvature against arc length, and a car as mass, power,
tyre grip and the aero coefficients from the wind tunnel model, produce the
speed profile and lap time. The method is the standard one:

  1. grip-limited speed at every point, solving v^2 k = mu (g + DF(v)/m)
     with DF = 0.5 rho v^2 sCz  ->  v^2 = mu g / (k - mu rho sCz / 2m)
  2. a forward pass accelerating from each point, limited by the friction
     circle (what is left after cornering) and by power against drag
  3. a backward pass braking into each point, friction circle plus drag
  4. v = the minimum of the three; lap time = sum(ds / v)

Two full forward passes make the flying lap independent of where it starts.

The JavaScript port must keep these constants and this order of operations
exactly, so the number the visitor sees on the page is the number this file
was calibrated to.
"""

from __future__ import annotations

import numpy as np

RHO = 1.225
G0 = 9.81
MASS = 798.0

# Calibrated in export_web_data.py so the best-found setup lands on the real
# pole time at Monaco, Spa and Monza. See that file for the residuals.
MU = 1.40            # peak tyre friction with aero load
POWER_W = 600000.0   # W at the wheels, qualifying mode
DRS_DF = 0.35        # rear wing load retained with DRS open
DRS_DRAG = 0.30      # rear wing drag retained with DRS open
DRS_RW_SHARE_CZ = 0.2619   # share of sCz the rear wing carries (PERRINN)
DRS_RW_SHARE_CX = 0.325    # share of sCx the rear wing carries
V_MIN = 8.0          # m/s floor so the hairpin never divides by zero


def simulate(kappa, ds, drs_open, s_cz, s_cx, mu=MU, power=POWER_W):
    """
    kappa    : curvature per point (1/m), closed lap
    ds       : uniform spacing (m)
    drs_open : bool per point, taken from the real pole lap's DRS channel
    s_cz/s_cx: area-inclusive coefficients (m^2) for the setup being tested
    returns  : dict(v=speed m/s per point, lap_time s, sector times)
    """
    n = len(kappa)
    kappa = np.abs(np.asarray(kappa, dtype=float))

    # DRS changes the rear wing only.
    cz = np.where(drs_open, s_cz * (1 - DRS_RW_SHARE_CZ * (1 - DRS_DF)), s_cz)
    cx = np.where(drs_open, s_cx * (1 - DRS_RW_SHARE_CX * (1 - DRS_DRAG)), s_cx)

    # 1. grip-limited speed
    denom = kappa - mu * RHO * cz / (2 * MASS)
    v_lim = np.where(denom > 1e-9, np.sqrt(mu * G0 / np.maximum(denom, 1e-9)), np.inf)
    # top speed where power balances drag, as a ceiling everywhere
    v_top = np.cbrt(power / (0.5 * RHO * cx))
    v_lim = np.minimum(v_lim, v_top)
    v_lim = np.maximum(v_lim, V_MIN)

    def a_lat_max(v, i):
        return mu * (G0 + 0.5 * RHO * v * v * cz[i] / MASS)

    def spare(v, i):
        """Fraction of the friction circle left over after cornering."""
        need = v * v * kappa[i]
        have = a_lat_max(v, i)
        r = min(need / max(have, 1e-9), 1.0)
        return np.sqrt(max(0.0, 1.0 - r * r))

    def drag_acc(v, i):
        return 0.5 * RHO * v * v * cx[i] / MASS

    # 2. forward pass, twice, so the start condition washes out
    v_f = v_lim.copy()
    v = v_lim[0]
    for _ in range(2):
        for i in range(n):
            j = (i + 1) % n
            a_trac = a_lat_max(v, i) * spare(v, i)
            a_pow = power / (MASS * max(v, V_MIN))
            a = min(a_trac, a_pow) - drag_acc(v, i)
            v_next = np.sqrt(max(v * v + 2 * a * ds, V_MIN * V_MIN))
            v = min(v_next, v_lim[j])
            v_f[j] = v

    # 3. backward pass, twice
    v_b = v_f.copy()
    v = v_f[-1]
    for _ in range(2):
        for i in range(n - 1, -1, -1):
            j = (i - 1) % n
            a_brk = a_lat_max(v, i) * spare(v, i) + drag_acc(v, i)
            v_prev = np.sqrt(v * v + 2 * a_brk * ds)
            v = min(v_prev, v_f[j])
            v_b[j] = v

    v_out = np.minimum(v_f, v_b)
    v_out = np.maximum(v_out, V_MIN)
    dt = ds / v_out
    lap_time = float(dt.sum())

    third = n // 3
    sectors = [float(dt[:third].sum()), float(dt[third:2 * third].sum()),
               float(dt[2 * third:].sum())]
    return dict(v=v_out, lap_time=lap_time, sectors=sectors, v_lim=v_lim)
