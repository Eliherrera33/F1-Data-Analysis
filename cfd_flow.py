# -*- coding: utf-8 -*-
"""
Potential-flow CFD animations - the offline twin of the site's wind tunnel.

The previous cfd_animations.py did not solve anything. It set U to the
freestream everywhere, added a few hand-placed Gaussian bumps near the wing and
floor, and masked a car silhouette on top. It looks like flow but carries no
physics, and anyone who works in aero would see that immediately.

This module runs the same model as windtunnel.js on the website:

  * doublets give the bodywork its thickness
  * Rankine vortices carry the circulation that makes downforce
  * every element is mirrored about y = 0, which makes the ground a streamline
  * pressure is Bernoulli, Cp = 1 - (V/Vinf)^2

Circulation is split between front wing, floor, diffuser and rear wing using
the same PERRINN-derived force model, so the picture and the numbers agree -
and the renders here and the interactive tool on the page validate each other.

    python cfd_flow.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.collections import LineCollection
from matplotlib.path import Path as MplPath
from matplotlib.patches import PathPatch

import f1_style as S

OUT = Path(__file__).parent / 'cfd_visualizations'
OUT.mkdir(exist_ok=True)

# ---- physical constants, identical to windtunnel.js ----------------------
RHO = 1.225
G0 = 9.81
CAR_MASS = 798.0
FRONTAL_AREA = 1.5
BASE_SCZ = 3.25          # downforce coefficient x area, m^2 (PERRINN)
BASE_SCX = 1.16          # drag coefficient x area, m^2
DF_SPLIT = dict(floor=0.4048, rear_wing=0.2619, front_wing=0.2143, diffuser=0.1190)
DR_SPLIT = dict(rear_wing=0.325, wheels=0.350, body=0.125, front_wing=0.120, floor=0.080)
REF = dict(rear_wing=20.0, front_wing=10.0, front_rh=40.0, rear_rh=50.0)

VIS_GAIN = 2.0           # legibility scaling on circulation
SPAN_EFF = 1.8           # m, turns force into 2D circulation
RIDE_BASE = 0.05         # m, drawn clearance floor
RIDE_VIS = 5.0           # ride height is drawn exaggerated; numbers stay true
WHEEL_R = 0.36

X0, X1, Y0, Y1 = -0.62, 5.72, 0.0, 1.95


def ground_effect(h_mm: float) -> float:
    """Floor multiplier vs front ride height. Peaks near 24 mm, then falls off
    a cliff as the underfloor stalls - the 2022 porpoising story."""
    def raw(x):
        return 1.0 / (1.0 + ((x - 24.0) / 38.0) ** 1.6) if x >= 24 else (x / 24.0) ** 1.5
    return raw(h_mm) / raw(REF['front_rh'])


def solve(speed_kmh=250.0, rear_wing=20.0, front_wing=10.0,
          front_rh=40.0, rear_rh=50.0, drs=False) -> dict:
    """Forces from the PERRINN model. sCz/sCx already include the reference
    area, so force is 0.5*rho*v^2*sC - never multiply by frontal area again."""
    v = speed_kmh / 3.6
    q = 0.5 * RHO * v * v
    rake = rear_rh - front_rh

    f_rw = np.clip(1 + (rear_wing - REF['rear_wing']) * 0.030, 0.45, 1.55)
    f_fw = np.clip(1 + (front_wing - REF['front_wing']) * 0.020, 0.70, 1.35)
    f_ge = ground_effect(front_rh)
    f_rake = np.clip(1 + (rake - 10) * 0.006, 0.90, 1.16)
    drs_df, drs_dr = (0.35, 0.30) if drs else (1.0, 1.0)

    cz = dict(
        floor=BASE_SCZ * DF_SPLIT['floor'] * f_ge * f_rake,
        diffuser=BASE_SCZ * DF_SPLIT['diffuser'] * f_ge * f_rake * f_rake,
        front_wing=BASE_SCZ * DF_SPLIT['front_wing'] * f_fw * (1 + (REF['front_rh'] - front_rh) * 0.004),
        rear_wing=BASE_SCZ * DF_SPLIT['rear_wing'] * f_rw * drs_df,
    )
    s_cz = sum(cz.values())
    s_cx = (BASE_SCX * DR_SPLIT['rear_wing'] * f_rw ** 1.6 * drs_dr
            + BASE_SCX * DR_SPLIT['front_wing'] * f_fw ** 1.5
            + BASE_SCX * DR_SPLIT['floor'] * f_ge
            + BASE_SCX * DR_SPLIT['wheels']
            + BASE_SCX * DR_SPLIT['body'])

    downforce = q * s_cz
    drag = q * s_cx
    return dict(v=v, q=q, cz=cz, s_cz=s_cz, s_cx=s_cx,
                downforce=downforce, drag=drag,
                downforce_kg=downforce / G0,
                efficiency=s_cz / s_cx,
                lat_g=1.8 * (G0 + downforce / CAR_MASS) / G0,
                power_kw=drag * v / 1000.0,
                front_rh=front_rh, rear_rh=rear_rh, speed_kmh=speed_kmh)


def floor_y(x, front_rh, rear_rh):
    t = np.clip(np.asarray(x, dtype=float) / 5.0, 0, 1)
    mm = front_rh + (rear_rh - front_rh) * t
    return RIDE_BASE + mm * RIDE_VIS / 1000.0


class Field:
    """Superposition of doublets and Rankine vortices, each with a ground image."""

    def __init__(self, res: dict):
        self.U = res['v']
        fr, rr = res['front_rh'], res['rear_rh']
        U = self.U
        self.doublets = []   # (x, y, mu, rc)
        self.vortices = []   # (x, y, gamma, rc)

        def dbl(x, y, a):
            self.doublets.append((x, y + float(floor_y(x, fr, rr)),
                                  2 * np.pi * U * a * a, a * 0.85))
        dbl(0.55, 0.20, 0.16)
        dbl(2.40, 0.38, 0.23)
        dbl(3.25, 0.42, 0.21)
        # wheels sit on the ground, not on the floor plane
        self.doublets.append((1.05, WHEEL_R, 2 * np.pi * U * 0.30 ** 2, 0.30))
        self.doublets.append((4.05, WHEEL_R, 2 * np.pi * U * 0.30 ** 2, 0.30))

        k = VIS_GAIN / (RHO * U * SPAN_EFF) * res['q']
        for x, y, key, rc in ((0.32, 0.10, 'front_wing', 0.16),
                              (2.55, 0.13, 'floor', 0.42),
                              (3.95, 0.12, 'diffuser', 0.30),
                              (4.85, 0.80, 'rear_wing', 0.18)):
            self.vortices.append((x, y + float(floor_y(x, fr, rr)),
                                  res['cz'][key] * k, rc))

    def velocity(self, x, y):
        """Vectorised over any array shape."""
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        u = np.full(x.shape, self.U)
        v = np.zeros(x.shape)

        for ex, ey, mu, rc in self.doublets:
            for sy in (ey, -ey):
                dx, dy = x - ex, y - sy
                r2 = np.maximum(dx * dx + dy * dy, rc * rc)
                r4 = r2 * r2
                c = mu / (2 * np.pi)
                u -= c * (dx * dx - dy * dy) / r4
                v -= c * (2 * dx * dy) / r4

        for ex, ey, g, rc in self.vortices:
            for sy, gs in ((ey, g), (-ey, -g)):
                dx, dy = x - ex, y - sy
                r2 = dx * dx + dy * dy
                f = np.where(r2 < rc * rc,
                             gs / (2 * np.pi * rc * rc),
                             gs / (2 * np.pi * np.maximum(r2, 1e-9)))
                u -= f * dy
                v += f * dx
        return u, v


# ---- car geometry ---------------------------------------------------------

BODY = [(0.00, 0.055), (0.38, 0.085), (0.72, 0.22), (1.12, 0.27), (1.48, 0.34),
        (1.78, 0.44), (1.95, 0.56), (2.20, 0.50), (2.55, 0.52), (3.05, 0.54),
        (3.55, 0.50), (4.05, 0.44), (4.45, 0.36), (4.72, 0.30), (4.72, 0.10),
        (3.20, 0.055), (1.60, 0.045), (0.55, 0.045)]


def car_polys(front_rh, rear_rh, rear_wing_deg, drs=False):
    """Every filled shape that makes up the car, in world coordinates."""
    def lift(pts):
        return [(x, y + float(floor_y(x, front_rh, rear_rh))) for x, y in pts]

    polys = [lift(BODY),
             lift([(-0.22, 0.015), (0.62, 0.060), (0.62, 0.175), (-0.22, 0.115)]),
             lift([(-0.26, 0.010), (-0.14, 0.010), (-0.14, 0.30), (-0.26, 0.26)]),
             lift([(4.58, 0.24), (4.72, 0.24), (4.86, 0.70), (4.72, 0.70)]),
             lift([(4.68, 0.64), (5.16, 0.66), (5.16, 1.06), (4.68, 0.98)])]

    ang = np.deg2rad(3 if drs else rear_wing_deg)
    cx, cy, c, th = 4.86, 0.86, 0.50, 0.035
    plane = []
    for dx, dy in ((-c / 2, -th), (c / 2, -th), (c / 2, th), (-c / 2, th)):
        x = cx + dx * np.cos(ang) - dy * np.sin(ang)
        y = cy + dx * np.sin(ang) + dy * np.cos(ang)
        plane.append((x, y + float(floor_y(x, front_rh, rear_rh))))
    return polys, plane


def solid_mask(X, Y, front_rh, rear_rh):
    """True where a sample point is inside the car."""
    fy = floor_y(X, front_rh, rear_rh)
    yy = Y - fy
    m = np.zeros(X.shape, dtype=bool)
    m |= (X >= -0.28) & (X <= 0.62) & (yy >= 0.01) & (yy <= 0.30)
    m |= (X >= 4.56) & (X <= 5.20) & (yy >= 0.62) & (yy <= 1.10)
    m |= ((X - 1.05) ** 2 + (Y - WHEEL_R) ** 2) < WHEEL_R ** 2
    m |= ((X - 4.05) ** 2 + (Y - WHEEL_R) ** 2) < WHEEL_R ** 2

    bx = np.array([p[0] for p in BODY[:14]])
    by = np.array([p[1] for p in BODY[:14]])
    top = np.interp(X, bx, by, left=0, right=0)
    m |= (X >= 0) & (X <= 4.72) & (yy >= 0.035) & (yy < top)
    return m


def draw_car(ax, front_rh, rear_rh, rear_wing_deg, drs=False):
    polys, plane = car_polys(front_rh, rear_rh, rear_wing_deg, drs)
    artists = []
    for poly in polys:
        p = PathPatch(MplPath(poly + [poly[0]], closed=True),
                      facecolor='#20202b', edgecolor='white', lw=1.4,
                      zorder=6, joinstyle='round')
        ax.add_patch(p)
        artists.append(p)
    p = PathPatch(MplPath(plane + [plane[0]], closed=True),
                  facecolor=(S.TEAL if drs else S.RED), alpha=0.85,
                  edgecolor='white', lw=1.2, zorder=7)
    ax.add_patch(p)
    artists.append(p)
    for wx in (1.05, 4.05):
        c = plt.Circle((wx, WHEEL_R), WHEEL_R, facecolor='#131319',
                       edgecolor='white', lw=1.4, zorder=8)
        ax.add_patch(c)
        artists.append(c)
        artists.append(ax.add_patch(plt.Circle(
            (wx, WHEEL_R), WHEEL_R * 0.5, facecolor='none',
            edgecolor='white', lw=0.8, alpha=0.25, zorder=9)))
    return artists


def cp_rgba(field, nx=520, ny=170, gain=1.2):
    """
    Pressure field as RGBA.

    A plain diverging colormap is wrong here: with vmin/vmax spanning -2.2..1,
    Cp = 0 lands two thirds up the ramp, so undisturbed air renders as bright
    "high pressure" and floods the frame. Instead the field is drawn with alpha
    proportional to |Cp|, so freestream air is transparent and only worked flow
    takes colour - suction teal, stagnation red.

    No solid mask: the car is drawn opaque on top, and masking rectangles that
    do not match the bodywork punched black boxes through the field.
    """
    xs = np.linspace(X0, X1, nx)
    ys = np.linspace(Y0, Y1, ny)
    X, Y = np.meshgrid(xs, ys)
    u, v = field.velocity(X, Y)
    cp = 1.0 - (np.hypot(u, v) / field.U) ** 2

    suction = np.clip(-cp / 2.4, 0, 1)     # Cp < 0
    stag = np.clip(cp / 1.0, 0, 1)         # Cp > 0
    rgba = np.zeros(cp.shape + (4,))
    # teal for suction, red for stagnation
    rgba[..., 0] = 0.00 * suction + 0.88 * stag
    rgba[..., 1] = 0.82 * suction + 0.02 * stag
    rgba[..., 2] = 0.75 * suction + 0.00 * stag
    rgba[..., 3] = np.clip(np.abs(cp) / gain, 0, 1) * 0.80
    return rgba


# ==========================================================================
# animation 1 - streamlines over the pressure field
# ==========================================================================

def render_streamlines(speed_kmh=250.0, front_rh=40.0, rear_rh=50.0,
                       rear_wing=20.0, n_particles=900, trail=14,
                       seconds=12, fps=50, crf=24):
    """Particles advected through the real field, over the Cp map."""
    res = solve(speed_kmh, rear_wing, 10.0, front_rh, rear_rh)
    field = Field(res)

    S.apply()
    fig = plt.figure(figsize=(19.2, 7.2), dpi=100)
    ax = fig.add_axes([0.028, 0.085, 0.945, 0.70])
    S.strip(ax)
    ax.set_xlim(X0, X1)
    ax.set_ylim(Y0, Y1)
    ax.set_aspect('equal')
    ax.set_facecolor('#08080c')

    ax.imshow(cp_rgba(field), extent=(X0, X1, Y0, Y1), origin='lower',
              aspect='equal', zorder=1, interpolation='bilinear')

    draw_car(ax, front_rh, rear_rh, rear_wing)
    ax.axhline(0, color='white', lw=1.2, alpha=0.35, zorder=10)

    streaks = LineCollection([], linewidths=1.25, zorder=5, capstyle='round')
    ax.add_collection(streaks)

    rng = np.random.default_rng(7)

    def spawn(n, spread=False):
        # `spread` fills the whole domain (used once, at startup); respawns come
        # in at the inlet. Without the initial spread the frame starts empty and
        # takes hundreds of steps to populate.
        x = rng.uniform(X0, X1, n) if spread else rng.uniform(X0, X0 + 0.5, n)
        # bias low - the interesting flow is under and around the floor
        y = Y0 + rng.random(n) ** 1.8 * (Y1 - Y0)
        gap = float(floor_y(0.0, front_rh, rear_rh))
        under = rng.random(n) < 0.30
        y[under] = rng.uniform(0.012, max(gap - 0.02, 0.02), under.sum())
        return x, y

    px, py = spawn(n_particles, spread=True)
    hist = np.zeros((n_particles, trail, 2))
    hist[:, :, 0] = px[:, None]
    hist[:, :, 1] = py[:, None]

    S.title_block(fig, 'PRESSURE FIELD AND STREAMLINES',
                  f'Potential flow · {speed_kmh:.0f} km/h · front ride height '
                  f'{front_rh:.0f} mm · rear wing {rear_wing:.0f}°',
                  x=0.028, y=0.975, size=26)
    S.accent_rule(fig, x=0.028, y=0.845, w=0.048)
    fig.text(0.973, 0.975,
             f"{res['downforce']:,.0f} N   ·   L/D {res['efficiency']:.2f}",
             color=S.INK, fontsize=17, ha='right', va='top', fontweight='bold',
             family='monospace')
    fig.text(0.973, 0.938, 'downforce at this setup', color=S.INK_3,
             fontsize=10, ha='right', va='top')
    S.footer(fig, 'Doublets + Rankine vortices with ground-plane images · '
                  'Cp = 1 − (V/V∞)² · circulation split from the PERRINN '
                  'CFD dataset · ride height drawn exaggerated')

    def step(_):
        nonlocal px, py, hist
        u, v = field.velocity(px, py)
        # A streak should be a short tick of flow, not a stripe across the whole
        # domain: at 69 m/s a 0.01 s step is 0.7 m, and 14 of those is longer
        # than the car. Size the step so the full trail spans about half a metre.
        dt = 0.55 / (trail * field.U)
        nx, ny = px + u * dt, py + v * dt

        dead = ((nx > X1) | (ny > Y1) | (ny < 0.004)
                | solid_mask(nx, ny, front_rh, rear_rh)
                | (np.hypot(u, v) < field.U * 0.06))
        if dead.any():
            sx, sy = spawn(int(dead.sum()))
            nx[dead], ny[dead] = sx, sy
            hist[dead] = np.stack([np.repeat(sx[:, None], trail, 1),
                                   np.repeat(sy[:, None], trail, 1)], axis=-1)

        px, py = nx, ny
        hist = np.roll(hist, -1, axis=1)
        hist[:, -1, 0] = px
        hist[:, -1, 1] = py

        streaks.set_segments(list(hist))
        uu, vv = field.velocity(px, py)
        ratio = np.hypot(uu, vv) / field.U
        # Colour by speed, but let brightness track how far the flow departs
        # from freestream, so undisturbed air recedes and the worked flow reads.
        rgba = np.zeros((len(px), 4))
        slow = np.clip((1.0 - ratio) / 0.55, 0, 1)
        fast = np.clip((ratio - 1.0) / 0.85, 0, 1)
        rgba[:, 0] = 0.80 * fast + 0.55 * (1 - slow - fast).clip(0, 1)
        rgba[:, 1] = 0.90 * (1 - fast) * (1 - slow) + 0.78 * slow + 0.10 * fast
        rgba[:, 2] = 0.88 * slow + 0.75 * (1 - slow - fast).clip(0, 1) + 0.04 * fast
        rgba[:, 3] = 0.14 + 0.62 * np.clip(np.abs(ratio - 1.0) / 0.55, 0, 1)
        streaks.set_color(rgba)
        return (streaks,)

    # Fill the trails properly: a particle needs roughly domain_width /
    # (U * dt) steps to cross, and the trail history has to be real flow
    # rather than the spawn point repeated.
    for _ in range(trail * 3):
        step(0)

    out = OUT / 'streamlines_pressure.mp4'
    # A dense particle field is close to noise, which H.264 hates: at CRF 19
    # this clip came out at 8.3 MB. CRF 24 is visually indistinguishable here
    # and roughly a third of the bytes.
    anim = FuncAnimation(fig, step, frames=seconds * fps, interval=1000 / fps, blit=False)
    anim.save(str(out), writer=S.writer(fps=fps, crf=crf), dpi=100)
    fig.savefig(OUT / 'streamlines_pressure.jpg', dpi=100, facecolor=S.BG,
                pil_kwargs={'quality': 88})
    plt.close(fig)
    print(f'  {out.name}  {out.stat().st_size / 1e6:.2f} MB')
    return out


# ==========================================================================
# animation 2 - ride height sweep: ground effect, and the cliff
# ==========================================================================

def render_ride_height_sweep(speed_kmh=250.0, rear_wing=20.0, rake=10.0,
                             lo=15.0, hi=60.0, seconds=16, fps=50):
    """
    Lower the floor and downforce climbs, because the underfloor works harder
    the closer it gets to the ground. Keep going and it collapses: the venturi
    stalls, the car loses load and starts bouncing. That cliff is why 2022 cars
    porpoised, and it is the single most instructive thing this model can show.
    """
    n = seconds * fps
    # down then back up, eased, so the cliff gets crossed slowly in both directions
    phase = np.linspace(0, 2 * np.pi, n, endpoint=False)
    heights = lo + (hi - lo) * (0.5 + 0.5 * np.cos(phase))

    sweep_h = np.linspace(lo, hi, 400)
    sweep_df = np.array([solve(speed_kmh, rear_wing, 10.0, h, h + rake)['downforce']
                         for h in sweep_h])
    peak_i = int(np.argmax(sweep_df))

    S.apply()
    fig = plt.figure(figsize=(19.2, 10.0), dpi=100)
    ax = fig.add_axes([0.028, 0.335, 0.945, 0.505])
    ax_curve = fig.add_axes([0.028, 0.085, 0.60, 0.20])
    ax_bars = fig.add_axes([0.685, 0.085, 0.288, 0.20])

    S.strip(ax)
    ax.set_xlim(X0, X1)
    ax.set_ylim(Y0, Y1)
    ax.set_aspect('equal')
    ax.set_facecolor('#08080c')

    S.title_block(fig, 'GROUND EFFECT AND THE STALL CLIFF',
                  f'Front ride height swept {lo:.0f}–{hi:.0f} mm at '
                  f'{speed_kmh:.0f} km/h, {rake:.0f} mm rake',
                  x=0.028, y=0.978, size=26)
    S.accent_rule(fig, x=0.028, y=0.878, w=0.048)

    # headline readouts
    rh_txt = fig.text(0.973, 0.978, '', color=S.INK, fontsize=30, ha='right',
                      va='top', fontweight='bold', family='monospace')
    fig.text(0.973, 0.930, 'front ride height', color=S.INK_3, fontsize=10,
             ha='right', va='top')
    df_txt = fig.text(0.760, 0.978, '', color=S.TEAL, fontsize=30, ha='right',
                      va='top', fontweight='bold', family='monospace')
    fig.text(0.760, 0.930, 'downforce', color=S.INK_3, fontsize=10,
             ha='right', va='top')

    warn = fig.text(0.5, 0.305, '', color=S.YELLOW, fontsize=15, ha='center',
                    va='bottom', fontweight='bold')

    # ---- the curve ---------------------------------------------------------
    S.panel(ax_curve, 'DOWNFORCE vs FRONT RIDE HEIGHT  (N)')
    ax_curve.plot(sweep_h, sweep_df, color=S.INK_2, lw=2.2)
    ax_curve.axvline(sweep_h[peak_i], color=S.TEAL, lw=1.4, ls='--', alpha=0.8)
    ax_curve.text(sweep_h[peak_i] + 0.6, sweep_df.min(),
                  f'peak {sweep_h[peak_i]:.0f} mm', color=S.TEAL, fontsize=10,
                  va='bottom')
    ax_curve.axvspan(lo, 22, color=S.RED, alpha=0.12, lw=0)
    ax_curve.text(lo + 0.5, sweep_df.max(), 'stall', color=S.RED, fontsize=10,
                  va='top', fontweight='bold')
    ax_curve.set_xlim(lo, hi)
    ax_curve.set_xlabel('Front ride height (mm)')
    marker, = ax_curve.plot([], [], 'o', ms=11, color=S.INK, mec=S.RED, mew=2.5,
                            zorder=5)

    # ---- component bars ----------------------------------------------------
    S.panel(ax_bars, 'DOWNFORCE BY COMPONENT  (N)')
    keys = ['floor', 'diffuser', 'front_wing', 'rear_wing']
    names = ['Underfloor', 'Diffuser', 'Front wing', 'Rear wing']
    bars = ax_bars.barh(names[::-1], [0] * 4,
                        color=[S.RED, S.AMBER, S.TEAL, S.TEAL][::-1], height=0.62)
    ax_bars.set_xlim(0, max(sweep_df) * 0.55)
    ax_bars.grid(axis='y', visible=False)
    bar_labels = [ax_bars.text(0, i, '', color=S.INK, fontsize=10, va='center',
                               ha='left', family='monospace')
                  for i in range(4)]

    S.footer(fig, 'Same potential-flow model as the interactive wind tunnel · '
                  'ground-effect curve from the PERRINN ride-height study · '
                  'ride height drawn exaggerated, numbers are true')

    car_artists, field_img = [], []

    def step(i):
        nonlocal car_artists, field_img
        h = float(heights[i])
        res = solve(speed_kmh, rear_wing, 10.0, h, h + rake)
        field = Field(res)

        for a in car_artists:
            a.remove()
        for im in field_img:
            im.remove()

        field_img = [ax.imshow(cp_rgba(field, nx=420, ny=140),
                               extent=(X0, X1, Y0, Y1), origin='lower',
                               aspect='equal', zorder=1, interpolation='bilinear')]
        car_artists = draw_car(ax, h, h + rake, rear_wing)

        rh_txt.set_text(f'{h:4.0f} mm')
        df_txt.set_text(f"{res['downforce']:>6,.0f} N")
        marker.set_data([h], [res['downforce']])

        for bar, lab, key in zip(bars[::-1], bar_labels[::-1], keys):
            val = res['q'] * res['cz'][key]
            bar.set_width(val)
            lab.set_x(val + max(sweep_df) * 0.012)
            lab.set_text(f'{val:,.0f}')

        if h <= 22:
            warn.set_text('UNDERFLOOR STALLED  ·  suction lost, car starts porpoising')
            warn.set_color(S.RED)
        elif h <= 27:
            warn.set_text('peak ground effect  ·  the edge teams live on')
            warn.set_color(S.YELLOW)
        else:
            warn.set_text('')
        return ()

    out = OUT / 'ride_height_sweep.mp4'
    anim = FuncAnimation(fig, step, frames=n, interval=1000 / fps, blit=False)
    anim.save(str(out), writer=S.writer(fps=fps), dpi=100)
    step(int(n * 0.72))
    fig.savefig(OUT / 'ride_height_sweep.jpg', dpi=100, facecolor=S.BG,
                pil_kwargs={'quality': 88})
    plt.close(fig)
    print(f'  {out.name}  {out.stat().st_size / 1e6:.2f} MB')
    return out


def main():
    print('Rendering CFD animations')
    render_streamlines()
    render_ride_height_sweep()


if __name__ == '__main__':
    main()
