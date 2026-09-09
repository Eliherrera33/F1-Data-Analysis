# -*- coding: utf-8 -*-
"""
Every static figure on the site, rebuilt on the shared style.

The originals were each written in isolation on matplotlib's `dark_background`,
so they had different palettes, different type, primary red/blue/yellow fills,
and in the CFD case a car drawn as an unrecognisable blob. This module
regenerates them all through f1_style, and the aero figures reuse the real car
geometry and potential-flow solver from cfd_flow rather than a silhouette.

    python figures.py            # everything
    python figures.py cfd        # just one group
"""

from __future__ import annotations

import io
import sys
import warnings
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

import f1_style as S
import cfd_flow as C

warnings.filterwarnings('ignore')

ROOT = Path(__file__).parent
CFD_OUT = ROOT / 'cfd_visualizations'
TEL_OUT = ROOT / 'telemetry_visualizations'
TYRE_OUT = ROOT / 'tire_pitstop_visualizations'
SIM_OUT = ROOT / 'race_simulations'
for d in (CFD_OUT, TEL_OUT, TYRE_OUT, SIM_OUT):
    d.mkdir(exist_ok=True)

DPI = 150

# The flow domain is much wider than it is tall. Matplotlib's aspect='equal'
# honours that by shrinking the image inside whatever box it is given, which
# left the old figures with a thin strip of data floating in dead space. These
# helpers size the axes box to the data instead.
DOMAIN_ASPECT = (C.X1 - C.X0) / (C.Y1 - C.Y0)


def field_axes(fig, left, bottom, width):
    """Axes whose box matches the flow domain's aspect ratio exactly."""
    fig_w, fig_h = fig.get_size_inches()
    height = (width * fig_w / DOMAIN_ASPECT) / fig_h
    ax = fig.add_axes([left, bottom, width, height])
    S.strip(ax)
    ax.set_aspect('equal')
    ax.set_xlim(C.X0, C.X1)
    ax.set_ylim(C.Y0, C.Y1)
    return ax


from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

# Diverging ramps that sit on a dark page: the neutral value is dark, not white,
# so undisturbed flow recedes instead of glaring.
# Stops are positioned explicitly. Spacing them evenly puts the neutral colour
# at 0.333 rather than 0.5, which is what flooded the velocity panel orange:
# TwoSlopeNorm sends the centre value to 0.5, so that is where the dark has to be.
CP_CMAP = LinearSegmentedColormap.from_list('cp', [
    (0.00, '#00f0d0'), (0.22, '#00a894'), (0.40, '#0d3a3a'), (0.50, '#141420'),
    (0.62, '#3d1010'), (0.82, '#a80500'), (1.00, '#ff2d1a')])
VEL_CMAP = LinearSegmentedColormap.from_list('vel', [
    (0.00, '#1f6dff'), (0.28, '#1b3a8f'), (0.44, '#10101a'), (0.50, '#141420'),
    (0.58, '#4a2a00'), (0.76, '#ff8700'), (0.90, '#ffd97a'), (1.00, '#ffffff')])


def save(fig, path, source):
    S.footer(fig, source)
    fig.savefig(path, dpi=DPI, facecolor=S.BG)
    plt.close(fig)
    print(f'  {path.name}  {path.stat().st_size / 1e3:.0f} KB')


# ==========================================================================
# aerodynamics - real geometry, real field
# ==========================================================================

def fig_pressure_velocity():
    """Cp and velocity ratio through the centreline, side by side."""
    res = C.solve(300.0, 20.0, 10.0, 40.0, 50.0)
    field = C.Field(res)

    xs = np.linspace(C.X0, C.X1, 640)
    ys = np.linspace(C.Y0, C.Y1, 210)
    X, Y = np.meshgrid(xs, ys)
    u, v = field.velocity(X, Y)
    ratio = np.hypot(u, v) / field.U
    cp = 1.0 - ratio ** 2
    solid = C.solid_mask(X, Y, 40.0, 50.0)
    cp = np.ma.array(cp, mask=solid)
    ratio = np.ma.array(ratio, mask=solid)

    S.apply()
    fig = plt.figure(figsize=(17, 6.6), dpi=DPI)
    S.title_block(fig, 'PRESSURE AND VELOCITY FIELDS',
                  'Potential-flow solution through the centreline plane at 300 km/h',
                  x=0.035, y=0.955, size=23)
    S.accent_rule(fig, x=0.035, y=0.792, w=0.05)

    panels = [
        (cp, 'PRESSURE COEFFICIENT  Cp', CP_CMAP,
         TwoSlopeNorm(vcenter=0.0, vmin=-2.2, vmax=1.0),
         'suction   <   freestream   >   stagnation'),
        (ratio, 'VELOCITY  V / V∞', VEL_CMAP,
         TwoSlopeNorm(vcenter=1.0, vmin=0.0, vmax=2.0),
         'slowed   <   freestream   >   accelerated'),
    ]
    for k, (data, label, cmap, norm, cblabel) in enumerate(panels):
        left = 0.045 + k * 0.487
        ax = field_axes(fig, left, 0.30, 0.425)
        im = ax.imshow(data, extent=(C.X0, C.X1, C.Y0, C.Y1), origin='lower',
                       aspect='equal', cmap=cmap, norm=norm,
                       interpolation='bilinear', zorder=1)
        C.draw_car(ax, 40.0, 50.0, 20.0)
        ax.axhline(0, color='white', lw=1.1, alpha=0.4, zorder=10)
        ax.set_title(label, color=S.INK_2, fontsize=11.5, loc='left', pad=10)

        cax = fig.add_axes([left, 0.175, 0.425, 0.020])
        cb = fig.colorbar(im, cax=cax, orientation='horizontal')
        cb.outline.set_edgecolor(S.LINE)
        cb.ax.tick_params(colors=S.INK_3, labelsize=8.5)
        cb.set_label(cblabel, color=S.INK_3, fontsize=9)

    fig.text(0.035, 0.745,
             'Undisturbed air is dark in both panels. The teal sheet under the floor is the '
             'suction that makes the downforce; the red patches ahead of each tyre are '
             'stagnation, where the flow is brought to rest.',
             color=S.INK_3, fontsize=10.5)
    save(fig, CFD_OUT / 'pressure_velocity_fields.png',
         'Doublets + Rankine vortices with ground-plane images · Cp = 1 − (V/V∞)² · '
         'PERRINN 2017 coefficients · ride height drawn exaggerated')


def fig_ground_effect():
    """Downforce vs ride height, with the stall region called out."""
    heights = np.linspace(12, 90, 400)
    rake = 10.0
    df = np.array([C.solve(250, 20, 10, h, h + rake)['downforce'] for h in heights])
    ld = np.array([C.solve(250, 20, 10, h, h + rake)['efficiency'] for h in heights])
    peak = int(np.argmax(df))

    S.apply()
    fig = plt.figure(figsize=(15, 8.5), dpi=DPI)
    S.title_block(fig, 'GROUND EFFECT vs RIDE HEIGHT',
                  'Total downforce at 250 km/h as the floor is lowered, 10 mm rake',
                  x=0.045, y=0.965, size=23)
    S.accent_rule(fig, x=0.045, y=0.868, w=0.05)

    ax = fig.add_axes([0.065, 0.135, 0.60, 0.70])
    S.panel(ax)
    ax.plot(heights, df, color=S.TEAL, lw=2.8, zorder=4)
    ax.fill_between(heights, df, df.min(), color=S.TEAL, alpha=0.10, lw=0)
    ax.axvspan(12, 22, color=S.RED, alpha=0.14, lw=0, zorder=1)
    ax.axvline(heights[peak], color=S.INK, lw=1.2, ls='--', alpha=0.7, zorder=5)

    ax.annotate(f'peak {heights[peak]:.0f} mm\n{df[peak]:,.0f} N',
                xy=(heights[peak], df[peak]), xytext=(heights[peak] + 12, df[peak]),
                color=S.INK, fontsize=11, va='center',
                arrowprops=dict(arrowstyle='-', color=S.INK_3, lw=1))
    ax.text(13.5, df.max() * 0.985, 'STALL', color=S.RED, fontsize=12,
            fontweight='bold', va='top')
    ax.text(13.5, df.max() * 0.94,
            'underfloor loses suction;\nthe car porpoises', color=S.INK_2,
            fontsize=9.5, va='top')
    ax.set_xlabel('Front ride height (mm)')
    ax.set_ylabel('Downforce (N)')
    ax.set_xlim(12, 90)

    axr = fig.add_axes([0.725, 0.135, 0.24, 0.70])
    S.panel(axr, 'AERODYNAMIC EFFICIENCY  L/D')
    axr.plot(ld, heights, color=S.AMBER, lw=2.4)
    axr.axhspan(12, 22, color=S.RED, alpha=0.14, lw=0)
    axr.set_ylim(12, 90)
    axr.set_ylabel('Front ride height (mm)')
    axr.set_xlabel('L/D')

    save(fig, CFD_OUT / 'ground_effect.png',
         'PERRINN ride-height study · downforce = ½ρv²·sCz with the ground-effect '
         'multiplier normalised to 1.0 at the 40 mm reference')


def fig_force_vectors():
    """Where the load acts, drawn on the real car."""
    res = C.solve(300.0, 20.0, 10.0, 40.0, 50.0)
    q = res['q']

    S.apply()
    fig = plt.figure(figsize=(16, 8), dpi=DPI)
    S.title_block(fig, 'AERODYNAMIC FORCE BREAKDOWN',
                  f"Where the {res['downforce']:,.0f} N of downforce acts at 300 km/h",
                  x=0.035, y=0.965, size=23)
    S.accent_rule(fig, x=0.035, y=0.862, w=0.05)

    ax = fig.add_axes([0.035, 0.10, 0.63, 0.66])
    S.strip(ax)
    ax.set_aspect('equal')
    ax.set_xlim(C.X0 - 0.2, C.X1 + 0.2)
    ax.set_ylim(-0.35, C.Y1)
    C.draw_car(ax, 40.0, 50.0, 20.0)
    ax.axhline(0, color='white', lw=1.2, alpha=0.4)

    scale = 1.10 / max(q * v for v in res['cz'].values())
    points = [('front_wing', 0.30, 'Front wing'), ('floor', 2.45, 'Underfloor'),
              ('diffuser', 3.95, 'Diffuser'), ('rear_wing', 4.86, 'Rear wing')]
    for key, x, label in points:
        f = q * res['cz'][key]
        top = float(C.floor_y(x, 40.0, 50.0)) + 1.42
        ax.annotate('', xy=(x, top - f * scale), xytext=(x, top),
                    arrowprops=dict(arrowstyle='-|>', color=S.TEAL, lw=3.2,
                                    mutation_scale=22))
        ax.text(x, top + 0.06, f'{label}\n{f:,.0f} N', color=S.TEAL, fontsize=10,
                ha='center', va='bottom', fontweight='bold')

    # drag, acting rearward
    ax.annotate('', xy=(C.X1 - 0.1, 1.05), xytext=(C.X1 - 0.1 - res['drag'] * scale * 0.9, 1.05),
                arrowprops=dict(arrowstyle='-|>', color=S.RED, lw=3.2, mutation_scale=22))
    ax.text(C.X1 - 0.1, 1.14, f"Drag\n{res['drag']:,.0f} N", color=S.RED,
            fontsize=10, ha='right', va='bottom', fontweight='bold')

    for i in range(4):
        ax.annotate('', xy=(C.X0 + 0.35, 1.45 - i * 0.22),
                    xytext=(C.X0 - 0.15, 1.45 - i * 0.22),
                    arrowprops=dict(arrowstyle='-|>', color=S.INK_3, lw=1.4,
                                    mutation_scale=12))
    ax.text(C.X0 - 0.15, 1.60, '300 km/h', color=S.INK_3, fontsize=10)

    axb = fig.add_axes([0.715, 0.10, 0.25, 0.70])
    S.panel(axb, 'SHARE OF TOTAL DOWNFORCE')
    labels = ['Rear wing', 'Diffuser', 'Front wing', 'Underfloor']
    keys = ['rear_wing', 'diffuser', 'front_wing', 'floor']
    vals = [q * res['cz'][k] for k in keys]
    colors = [S.TEAL, S.AMBER, '#00a0c0', S.RED]
    axb.barh(labels, vals, color=colors, height=0.6)
    axb.grid(axis='y', visible=False)
    for i, v in enumerate(vals):
        axb.text(v + max(vals) * 0.03, i, f'{v:,.0f} N\n{v / res["downforce"] * 100:.0f}%',
                 color=S.INK, fontsize=9.5, va='center')
    axb.set_xlim(0, max(vals) * 1.42)
    axb.set_xlabel('Newtons')

    fig.text(0.035, 0.845,
             f"Total {res['downforce']:,.0f} N = {res['downforce_kg']:,.0f} kg, "
             f"{res['downforce'] / (C.CAR_MASS * C.G0) * 100:.0f}% of car weight  ·  "
             f"L/D {res['efficiency']:.2f}",
             color=S.INK_2, fontsize=11)

    save(fig, CFD_OUT / 'force_vectors.png',
         'Component split from the PERRINN CFD dataset · forces at 300 km/h · '
         'arrow lengths proportional to load')


def fig_component_breakdown():
    """Downforce and drag contributions, side by side."""
    res = C.solve(300.0, 20.0, 10.0, 40.0, 50.0)
    q = res['q']

    S.apply()
    fig = plt.figure(figsize=(15, 7.5), dpi=DPI)
    S.title_block(fig, 'WHERE DOWNFORCE AND DRAG COME FROM',
                  'Component contributions at 300 km/h, PERRINN 2017 reference car',
                  x=0.045, y=0.962, size=23)
    S.accent_rule(fig, x=0.045, y=0.852, w=0.05)

    ax1 = fig.add_axes([0.055, 0.135, 0.40, 0.665])
    S.panel(ax1, 'DOWNFORCE')
    keys = ['floor', 'rear_wing', 'front_wing', 'diffuser']
    names = ['Underfloor', 'Rear wing', 'Front wing', 'Diffuser']
    vals = [q * res['cz'][k] for k in keys]
    order = np.argsort(vals)
    ax1.barh([names[i] for i in order], [vals[i] for i in order],
             color=[S.RED, S.TEAL, '#00a0c0', S.AMBER], height=0.6)
    ax1.grid(axis='y', visible=False)
    for i, idx in enumerate(order):
        ax1.text(vals[idx] + max(vals) * 0.02, i,
                 f'{vals[idx]:,.0f} N  ({vals[idx] / sum(vals) * 100:.0f}%)',
                 color=S.INK, fontsize=10, va='center')
    ax1.set_xlim(0, max(vals) * 1.38)
    ax1.set_xlabel('Newtons')

    ax2 = fig.add_axes([0.555, 0.135, 0.40, 0.665])
    S.panel(ax2, 'DRAG')
    dnames = ['Wheels', 'Rear wing', 'Body', 'Front wing', 'Floor']
    dshare = [C.DR_SPLIT['wheels'], C.DR_SPLIT['rear_wing'], C.DR_SPLIT['body'],
              C.DR_SPLIT['front_wing'], C.DR_SPLIT['floor']]
    dvals = [q * C.BASE_SCX * s for s in dshare]
    dorder = np.argsort(dvals)
    ax2.barh([dnames[i] for i in dorder], [dvals[i] for i in dorder],
             color=[S.INK_3, S.RED, '#8f0400', S.AMBER, S.TEAL], height=0.6)
    ax2.grid(axis='y', visible=False)
    for i, idx in enumerate(dorder):
        ax2.text(dvals[idx] + max(dvals) * 0.02, i,
                 f'{dvals[idx]:,.0f} N  ({dshare[idx] * 100:.0f}%)',
                 color=S.INK, fontsize=10, va='center')
    ax2.set_xlim(0, max(dvals) * 1.38)
    ax2.set_xlabel('Newtons')

    fig.text(0.045, 0.828,
             'Open wheels are the single biggest drag source on the car — 35% of it, '
             'and almost nothing can be done about them under the regulations.',
             color=S.INK_3, fontsize=10)

    save(fig, CFD_OUT / 'component_breakdown.png',
         'Component splits from windtunnel_data/perrinn_cfd_data.csv')


def fig_speed_comparison():
    """Everything scales with v-squared. Show it."""
    speeds = np.linspace(60, 360, 260)
    rows = [C.solve(s, 20, 10, 40, 50) for s in speeds]
    df = np.array([r['downforce'] for r in rows])
    dg = np.array([r['drag'] for r in rows])
    pw = np.array([r['power_kw'] for r in rows])
    lg = np.array([r['lat_g'] for r in rows])

    S.apply()
    fig = plt.figure(figsize=(15, 8.5), dpi=DPI)
    S.title_block(fig, 'AERODYNAMIC LOADS ACROSS THE SPEED RANGE',
                  'Reference setup: 40 mm front ride height, 20° rear wing',
                  x=0.045, y=0.965, size=23)
    S.accent_rule(fig, x=0.045, y=0.868, w=0.05)

    ax = fig.add_axes([0.06, 0.42, 0.42, 0.40])
    S.panel(ax, 'DOWNFORCE AND DRAG  (N)')
    ax.plot(speeds, df, color=S.TEAL, lw=2.6, label='Downforce')
    ax.plot(speeds, dg, color=S.RED, lw=2.6, label='Drag')
    weight = C.CAR_MASS * C.G0
    ax.axhline(weight, color=S.INK_3, lw=1.2, ls='--')
    cross = speeds[np.argmin(np.abs(df - weight))]
    ax.text(speeds[-1], weight, ' car weight', color=S.INK_3, fontsize=9,
            va='center', ha='left')
    ax.axvline(cross, color=S.YELLOW, lw=1.2, ls=':')
    ax.text(cross + 4, df.max() * 0.5,
            f'{cross:.0f} km/h\ndownforce exceeds\nthe car\'s weight',
            color=S.YELLOW, fontsize=9.5, va='center')
    ax.set_xlabel('Speed (km/h)')
    ax.legend(loc='upper left')

    ax2 = fig.add_axes([0.56, 0.42, 0.40, 0.40])
    S.panel(ax2, 'POWER CONSUMED BY DRAG  (kW)')
    ax2.plot(speeds, pw, color=S.AMBER, lw=2.6)
    ax2.fill_between(speeds, pw, color=S.AMBER, alpha=0.10, lw=0)
    ax2.set_xlabel('Speed (km/h)')

    ax3 = fig.add_axes([0.06, 0.115, 0.42, 0.215])
    S.panel(ax3, 'PEAK LATERAL LOAD  (G)')
    ax3.plot(speeds, lg, color=S.TEAL, lw=2.4)
    ax3.set_xlabel('Speed (km/h)')

    ax4 = fig.add_axes([0.56, 0.115, 0.40, 0.215])
    S.panel(ax4, 'DOWNFORCE AS A MULTIPLE OF CAR WEIGHT')
    ax4.plot(speeds, df / weight, color='#00a0c0', lw=2.4)
    ax4.axhline(1, color=S.INK_3, lw=1, ls='--')
    ax4.set_xlabel('Speed (km/h)')

    save(fig, CFD_OUT / 'speed_comparison.png',
         'Forces scale with the square of speed; drag power with the cube · '
         'grip assumes a 1.8 peak friction coefficient')


def fig_streamlines_still():
    """A clean single-frame streamline picture."""
    res = C.solve(250.0, 20.0, 10.0, 40.0, 50.0)
    field = C.Field(res)

    S.apply()
    fig = plt.figure(figsize=(16, 7), dpi=DPI)
    S.title_block(fig, 'STREAMLINE STRUCTURE',
                  'Integrated streamlines at 250 km/h over the pressure field',
                  x=0.035, y=0.962, size=23)
    S.accent_rule(fig, x=0.035, y=0.842, w=0.05)

    ax = field_axes(fig, 0.035, 0.14, 0.93)
    ax.imshow(C.cp_rgba(field, nx=620, ny=200), extent=(C.X0, C.X1, C.Y0, C.Y1),
              origin='lower', aspect='equal', interpolation='bilinear', zorder=1)

    segs, colors = [], []
    for y0 in np.concatenate([np.linspace(0.012, 0.30, 16),
                              np.linspace(0.33, C.Y1 - 0.05, 30)]):
        x, y = C.X0, y0
        pts = [(x, y)]
        for _ in range(1800):
            u, v = field.velocity(np.array([x]), np.array([y]))
            spd = float(np.hypot(u, v))
            if spd < field.U * 0.05:
                break
            dt = 0.012 / (spd / field.U) / field.U * 4
            x += float(u) * dt
            y += float(v) * dt
            if x > C.X1 or y > C.Y1 or y < 0.004 or C.solid_mask(
                    np.array([x]), np.array([y]), 40.0, 50.0)[0]:
                break
            pts.append((x, y))
        if len(pts) < 3:
            continue
        pts = np.array(pts)
        for i in range(len(pts) - 1):
            segs.append(pts[i:i + 2])
            u, v = field.velocity(np.array([pts[i, 0]]), np.array([pts[i, 1]]))
            colors.append(float(np.hypot(u, v)) / field.U)

    colors = np.array(colors)
    lc = LineCollection(segs, linewidths=1.35, zorder=4,
                        colors=plt.get_cmap('RdYlBu_r')(np.clip((colors - 0.45) / 1.3, 0, 1)),
                        alpha=0.85)
    ax.add_collection(lc)
    C.draw_car(ax, 40.0, 50.0, 20.0)
    ax.axhline(0, color='white', lw=1.2, alpha=0.4, zorder=10)

    save(fig, CFD_OUT / 'streamlines_visualization.png',
         'Streamlines integrated through the potential-flow field · colour is local '
         'velocity relative to freestream')


def build_cfd():
    print('CFD figures')
    fig_pressure_velocity()
    fig_ground_effect()
    fig_force_vectors()
    fig_component_breakdown()
    fig_speed_comparison()
    fig_streamlines_still()


GROUPS = {'cfd': build_cfd}


# ==========================================================================
# telemetry
# ==========================================================================

import fastf1
fastf1.Cache.enable_cache(str(ROOT / 'cache'))

TEL_YEAR, TEL_GP, TEL_TAG = 2024, 'Monaco', '2024_Monaco_Grand_Prix'


def _session(year, gp, kind='Q'):
    ses = fastf1.get_session(year, gp, kind)
    ses.load()
    return ses


def _fastest(ses, abbr):
    laps = ses.laps.pick_drivers(abbr) if hasattr(ses.laps, 'pick_drivers') \
        else ses.laps.pick_driver(abbr)
    lap = laps.pick_fastest()
    return lap, lap.get_telemetry()


def _rotate(x, y, deg):
    r = np.deg2rad(deg)
    return x * np.cos(r) - y * np.sin(r), x * np.sin(r) + y * np.cos(r)


def _circuit(ses):
    try:
        ci = ses.get_circuit_info()
        return float(ci.rotation), ci.corners
    except Exception:
        return 0.0, None


def fig_track_map():
    """The lap drawn as a speed-coloured trace - the standard F1 track map."""
    ses = _session(TEL_YEAR, TEL_GP)
    abbr = ses.results.iloc[0]['Abbreviation']
    lap, tel = _fastest(ses, abbr)
    rot, corners = _circuit(ses)
    x, y = _rotate(tel['X'].to_numpy(float), tel['Y'].to_numpy(float), rot)
    spd = tel['Speed'].to_numpy(float)

    # Size the canvas to the circuit. A fixed portrait figure left Monaco - which
    # is far wider than it is tall - floating in dead space.
    aspect = (np.ptp(y) * 1.14) / (np.ptp(x) * 1.14)
    width = 14.0
    height = max(7.0, min(15.0, width * 0.90 * aspect / 0.775 + 2.2))

    S.apply()
    fig = plt.figure(figsize=(width, height), dpi=DPI)
    S.title_block(fig, f'{ses.event["EventName"].upper()} {TEL_YEAR}',
                  f'Fastest qualifying lap · {abbr} · '
                  f'{S.fmt_laptime(lap["LapTime"].total_seconds())}',
                  x=0.045, y=0.968, size=25)
    S.accent_rule(fig, x=0.045, y=0.888, w=0.055)

    ax = fig.add_axes([0.045, 0.085, 0.90, 0.775])
    S.strip(ax)
    ax.set_aspect('equal')

    pts = np.column_stack([x, y]).reshape(-1, 1, 2)
    segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
    ax.add_collection(LineCollection(segs, colors='#000000', linewidths=16,
                                     capstyle='round', zorder=1, alpha=0.6))
    lc = LineCollection(segs, cmap=S.SPEED_CMAP, linewidths=9.5,
                        capstyle='round', zorder=2)
    lc.set_array(spd[:-1])
    ax.add_collection(lc)

    pad = 0.07 * max(np.ptp(x), np.ptp(y))
    ax.set_xlim(x.min() - pad, x.max() + pad)
    ax.set_ylim(y.min() - pad, y.max() + pad)

    if corners is not None and len(corners):
        cx, cy = _rotate(corners['X'].to_numpy(float),
                         corners['Y'].to_numpy(float), rot)
        for i in range(len(corners)):
            ax.text(cx[i], cy[i], str(corners['Number'].iloc[i]), color=S.INK,
                    fontsize=8.5, ha='center', va='center', zorder=6,
                    bbox=dict(boxstyle='circle,pad=0.22', fc=S.BG, ec=S.LINE, lw=0.8))

    cax = fig.add_axes([0.30, 0.052, 0.40, 0.016])
    cb = fig.colorbar(lc, cax=cax, orientation='horizontal')
    cb.set_label('Speed (km/h)', color=S.INK_3, fontsize=9.5)
    cb.outline.set_edgecolor(S.LINE)
    cb.ax.tick_params(colors=S.INK_3, labelsize=8.5)

    fig.text(0.955, 0.888, f'{spd.max():.0f} km/h', color=S.INK, fontsize=22,
             ha='right', va='top', fontweight='bold', family='monospace')
    fig.text(0.955, 0.855, 'top speed', color=S.INK_3, fontsize=10,
             ha='right', va='top')

    save(fig, TEL_OUT / f'{TEL_TAG}_track_map.png',
         f'Source: FastF1 position + car telemetry · '
         f'{ses.event["EventName"]} {TEL_YEAR} qualifying')


def fig_speed_trace():
    """Speed against distance for the top three, with the corners marked."""
    ses = _session(TEL_YEAR, TEL_GP)
    abbrs = [r['Abbreviation'] for _, r in ses.results.head(3).iterrows()]
    rot, corners = _circuit(ses)

    S.apply()
    fig = plt.figure(figsize=(16, 8), dpi=DPI)
    S.title_block(fig, 'SPEED TRACE',
                  f'{ses.event["EventName"]} {TEL_YEAR} qualifying · '
                  f'top three, fastest laps',
                  x=0.04, y=0.965, size=24)
    S.accent_rule(fig, x=0.04, y=0.862, w=0.05)

    ax = fig.add_axes([0.055, 0.30, 0.90, 0.535])
    S.panel(ax, 'SPEED  (km/h)')
    axg = fig.add_axes([0.055, 0.105, 0.90, 0.155])
    S.panel(axg, 'GEAR')

    for abbr in abbrs:
        try:
            lap, tel = _fastest(ses, abbr)
        except Exception:
            continue
        d = tel['Distance'].to_numpy(float)
        col = S.team_color(lap['Team'])
        ax.plot(d, tel['Speed'].to_numpy(float), color=col, lw=2.0,
                label=f'{abbr}  {S.fmt_laptime(lap["LapTime"].total_seconds())}')
        axg.step(d, tel['nGear'].to_numpy(float), where='post', color=col, lw=1.5)

    if corners is not None and len(corners):
        top = ax.get_ylim()[1]
        for _, c in corners.iterrows():
            ax.axvline(c['Distance'], color=S.LINE, lw=0.8, zorder=0)
            ax.text(c['Distance'], top, str(int(c['Number'])),
                    color=S.INK_3, fontsize=7.5, ha='center', va='bottom')

    ax.set_xlim(0, None)
    ax.set_xticklabels([])
    ax.legend(loc='lower right', ncol=3)
    axg.set_ylim(0.5, 8.5)
    axg.set_yticks([2, 4, 6, 8])
    axg.set_xlim(ax.get_xlim())
    axg.set_xlabel('Lap distance (m)')

    save(fig, TEL_OUT / f'{TEL_TAG}_speed_trace.png',
         'Source: FastF1 car telemetry · numbers along the top are corner numbers')


def fig_throttle_brake():
    """Driver inputs: throttle and brake for the pole lap."""
    ses = _session(TEL_YEAR, TEL_GP)
    abbr = ses.results.iloc[0]['Abbreviation']
    lap, tel = _fastest(ses, abbr)
    d = tel['Distance'].to_numpy(float)
    thr = tel['Throttle'].to_numpy(float)
    brk = tel['Brake'].to_numpy(float).astype(bool)

    S.apply()
    fig = plt.figure(figsize=(16, 8), dpi=DPI)
    S.title_block(fig, 'DRIVER INPUTS',
                  f'{abbr} · pole lap · {ses.event["EventName"]} {TEL_YEAR}',
                  x=0.04, y=0.965, size=24)
    S.accent_rule(fig, x=0.04, y=0.862, w=0.05)

    ax = fig.add_axes([0.055, 0.46, 0.90, 0.375])
    S.panel(ax, 'THROTTLE  (%)')
    ax.fill_between(d, thr, color=S.TEAL, alpha=0.22, lw=0)
    ax.plot(d, thr, color=S.TEAL, lw=1.8)
    ax.set_ylim(-3, 105)
    ax.set_xticklabels([])

    axb = fig.add_axes([0.055, 0.23, 0.90, 0.185])
    S.panel(axb, 'BRAKE')
    axb.fill_between(d, brk.astype(float) * 100, color=S.RED, alpha=0.55, lw=0,
                     step='post')
    axb.set_ylim(0, 105)
    axb.set_yticks([])
    axb.set_xticklabels([])

    axs = fig.add_axes([0.055, 0.105, 0.90, 0.09])
    S.panel(axs, 'SPEED  (km/h)')
    axs.plot(d, tel['Speed'].to_numpy(float), color=S.INK_2, lw=1.4)
    axs.set_xlabel('Lap distance (m)')

    for a in (ax, axb, axs):
        a.set_xlim(0, d.max())

    full = float((thr > 98).sum()) / len(thr) * 100
    braking = float(brk.sum()) / len(brk) * 100
    fig.text(0.955, 0.862,
             f'{full:.0f}% full throttle   ·   {braking:.0f}% braking',
             color=S.INK_2, fontsize=12, ha='right', va='top')

    save(fig, TEL_OUT / f'{TEL_TAG}_throttle_brake.png',
         'Source: FastF1 car telemetry · brake is a boolean channel in the F1 feed')


def fig_driver_comparison():
    """Two drivers, every channel, plus the delta."""
    import qualifying_duel as Q
    ses, grid, drv = Q.load_duel(TEL_YEAR, TEL_GP, 'Q')
    a, b = drv
    delta = b['t'] - a['t']

    S.apply()
    fig = plt.figure(figsize=(16, 10), dpi=DPI)
    S.title_block(fig, f'{a["abbr"]}  vs  {b["abbr"]}',
                  f'{ses.event["EventName"]} {TEL_YEAR} qualifying · '
                  f'fastest laps compared',
                  x=0.04, y=0.972, size=25)
    S.accent_rule(fig, x=0.04, y=0.888, w=0.05)
    fig.text(0.955, 0.972,
             f'{S.fmt_laptime(a["laptime"])}   vs   {S.fmt_laptime(b["laptime"])}',
             color=S.INK, fontsize=15, ha='right', va='top', family='monospace')
    fig.text(0.955, 0.938, f'{S.fmt_delta(b["laptime"] - a["laptime"])} s',
             color=S.INK_2, fontsize=12, ha='right', va='top', family='monospace')

    rows = [
        ('SPEED  (km/h)', 'speed', 0.615, 0.245),
        ('THROTTLE  (%)', 'throttle', 0.455, 0.135),
        ('GEAR', 'gear', 0.325, 0.105),
    ]
    axes = []
    for label, key, bottom, height in rows:
        ax = fig.add_axes([0.055, bottom, 0.90, height])
        S.panel(ax, label)
        for d in drv:
            if key == 'gear':
                ax.step(grid, d[key], where='post', color=d['color'], lw=1.4)
            else:
                ax.plot(grid, d[key], color=d['color'], lw=1.8, label=d['abbr'])
        axes.append(ax)
    axes[0].legend(loc='lower right', ncol=2)
    axes[2].set_ylim(0.5, 8.5)
    axes[2].set_yticks([2, 4, 6, 8])

    axd = fig.add_axes([0.055, 0.105, 0.90, 0.175])
    S.panel(axd, f'CUMULATIVE DELTA  ·  {b["abbr"]} minus {a["abbr"]}  (s)')
    axd.axhline(0, color=S.INK_3, lw=1)
    axd.fill_between(grid, delta, 0, where=delta >= 0, color=a['color'],
                     alpha=0.30, interpolate=True, lw=0)
    axd.fill_between(grid, delta, 0, where=delta < 0, color=b['color'],
                     alpha=0.30, interpolate=True, lw=0)
    axd.plot(grid, delta, color=S.INK, lw=1.8)
    axd.set_xlabel('Lap distance (m)')
    axes.append(axd)

    for ax in axes[:-1]:
        ax.set_xticklabels([])
    for ax in axes:
        ax.set_xlim(0, grid[-1])

    save(fig, TEL_OUT / f'{TEL_TAG}_detailed_LEC_vs_PIA.png',
         'Source: FastF1 telemetry · laps aligned on lap fraction so the delta '
         'closes on the official gap')


def build_telemetry():
    print('Telemetry figures')
    fig_track_map()
    fig_speed_trace()
    fig_throttle_brake()
    fig_driver_comparison()


GROUPS['telemetry'] = build_telemetry


# ==========================================================================
# race strategy
# ==========================================================================

def fig_tyre_strategy():
    """Stint chart: every driver's race, coloured by compound."""
    ses = _session(TEL_YEAR, TEL_GP, 'R')
    laps = ses.laps
    order = [r['Abbreviation'] for _, r in ses.results.iterrows()]

    S.apply()
    fig = plt.figure(figsize=(15, 10), dpi=DPI)
    S.title_block(fig, 'TYRE STRATEGY',
                  f'{ses.event["EventName"]} {TEL_YEAR} · stints by compound, '
                  f'finishing order',
                  x=0.045, y=0.968, size=24)
    S.accent_rule(fig, x=0.045, y=0.878, w=0.05)

    ax = fig.add_axes([0.085, 0.105, 0.885, 0.755])
    S.panel(ax)
    ax.grid(axis='y', visible=False)

    # Monaco 2024 had a lap-one red flag, so most of the grid changed tyres
    # while stopped and then ran to the end. Without a retirement marker the
    # one-lap bars from the first-lap crashes look like broken data.
    status = {r['Abbreviation']: str(r.get('Status', '')) for _, r in ses.results.iterrows()}
    total_laps = float(laps['LapNumber'].max())

    seen = {}
    for row, abbr in enumerate(order):
        dl = laps.pick_drivers(abbr) if hasattr(laps, 'pick_drivers') \
            else laps.pick_driver(abbr)
        if dl is None or not len(dl):
            continue
        dl = dl.sort_values('LapNumber')
        stint_no = dl['Stint'].to_numpy()
        comp = dl['Compound'].to_numpy()
        lapn = dl['LapNumber'].to_numpy(float)

        start = 0
        for i in range(1, len(dl) + 1):
            if i == len(dl) or stint_no[i] != stint_no[start]:
                c = str(comp[start])
                ax.barh(row, lapn[i - 1] - lapn[start] + 1, left=lapn[start] - 0.5,
                        color=S.tyre_color(c), height=0.72,
                        edgecolor=S.BG, linewidth=1.1)
                seen[c] = S.tyre_color(c)
                start = i

        done = float(lapn.max())
        st = status.get(abbr, '')
        if done < total_laps - 1 and not st.startswith('+'):
            ax.text(done + 1.2, row, f'RETIRED  lap {done:.0f}', color=S.RED,
                    fontsize=8.5, va='center', fontweight='bold')

    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(order, fontsize=9.5)
    ax.invert_yaxis()
    ax.set_xlabel('Lap')
    ax.set_xlim(0.5, laps['LapNumber'].max() + 0.5)

    handles = [plt.Rectangle((0, 0), 1, 1, color=v) for v in seen.values()]
    ax.legend(handles, list(seen), loc='lower right', ncol=len(seen),
              frameon=True, facecolor=S.PANEL_HI, edgecolor=S.LINE)

    save(fig, TYRE_OUT / f'{TEL_TAG}_tire_strategy.png',
         'Source: FastF1 lap data · stint boundaries from the Stint and Compound '
         'channels')


def fig_pit_stops():
    """Pit lane times by team - and an honest label on what is being measured."""
    import csv
    path = ROOT / 'data' / 'f1db_csv' / f'{TEL_YEAR}_pit_stops.csv'
    rows = list(csv.DictReader(io.open(path, encoding='utf-8')))

    by_team = {}
    for r in rows:
        try:
            t = float(r['time'])
        except (TypeError, ValueError):
            continue
        if t <= 0 or t > 120:
            continue
        by_team.setdefault(r['constructorId'], []).append(t)

    pretty = {
        'red-bull': 'Red Bull', 'ferrari': 'Ferrari', 'mclaren': 'McLaren',
        'mercedes': 'Mercedes', 'aston-martin': 'Aston Martin', 'alpine': 'Alpine',
        'williams': 'Williams', 'rb': 'RB', 'kick-sauber': 'Kick Sauber',
        'haas': 'Haas F1 Team',
    }
    teams = sorted(by_team, key=lambda k: np.median(by_team[k]))
    med = [float(np.median(by_team[t])) for t in teams]
    best = [float(np.min(by_team[t])) for t in teams]
    names = [pretty.get(t, t.replace('-', ' ').title()) for t in teams]
    colors = [S.team_color(n, S.INK_3) for n in names]

    S.apply()
    fig = plt.figure(figsize=(15, 8.5), dpi=DPI)
    S.title_block(fig, 'PIT LANE PERFORMANCE',
                  f'{TEL_YEAR} season · median and best pit lane time per team',
                  x=0.045, y=0.965, size=24)
    S.accent_rule(fig, x=0.045, y=0.862, w=0.05)

    ax = fig.add_axes([0.16, 0.135, 0.80, 0.695])
    S.panel(ax)
    ax.grid(axis='y', visible=False)
    y = np.arange(len(teams))
    ax.barh(y, med, color=colors, height=0.62, label='Median')
    ax.plot(best, y, 'o', color=S.INK, ms=7, mec=S.BG, mew=1.5, label='Season best',
            zorder=5)
    for i, (m, b) in enumerate(zip(med, best)):
        ax.text(m + 0.35, i, f'{m:.2f}s', color=S.INK, fontsize=9.5, va='center')
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel('Seconds')
    ax.set_xlim(0, max(med) * 1.18)
    ax.legend(loc='lower right')

    fig.text(0.045, 0.838,
             'These are full pit lane times — entry line to exit line, including the '
             'speed-limited run in and out. They are not the two-second stationary '
             'stops shown on television, which are a subset of this.',
             color=S.INK_3, fontsize=10)

    save(fig, TYRE_OUT / 'fastest_pit_stops_2024.png',
         f'Source: data/f1db_csv/{TEL_YEAR}_pit_stops.csv · '
         f'{sum(len(v) for v in by_team.values()):,} stops across the season')


# ==========================================================================
# circuit comparison
# ==========================================================================

CIRCUITS = [(2024, 'Monaco', 'Monaco'), (2024, 'Belgium', 'Spa'),
            (2024, 'Italy', 'Monza')]


def fig_track_comparison():
    """Three circuits with completely different demands, measured the same way."""
    data = []
    for year, gp, label in CIRCUITS:
        try:
            ses = _session(year, gp)
            abbr = ses.results.iloc[0]['Abbreviation']
            lap, tel = _fastest(ses, abbr)
            rot, corners = _circuit(ses)
            x, y = _rotate(tel['X'].to_numpy(float), tel['Y'].to_numpy(float), rot)
            spd = tel['Speed'].to_numpy(float)
            thr = tel['Throttle'].to_numpy(float)
            data.append(dict(
                label=label, abbr=abbr, x=x, y=y, speed=spd,
                laptime=lap['LapTime'].total_seconds(),
                length=float(tel['Distance'].max()),
                corners=0 if corners is None else len(corners),
                vmax=float(spd.max()), vmean=float(spd.mean()),
                full=float((thr > 98).sum()) / len(thr) * 100,
            ))
        except Exception as exc:
            print(f'    skipped {gp}: {exc}')

    if not data:
        return

    S.apply()
    fig = plt.figure(figsize=(16, 10), dpi=DPI)
    S.title_block(fig, 'CIRCUIT COMPARISON',
                  'Three tracks that ask for opposite cars — measured from the '
                  'same qualifying telemetry',
                  x=0.04, y=0.972, size=25)
    S.accent_rule(fig, x=0.04, y=0.906, w=0.05)

    # track outlines, drawn at a shared metres-per-pixel scale
    spans = [max(np.ptp(d['x']), np.ptp(d['y'])) for d in data]
    span = max(spans) * 1.08
    for i, d in enumerate(data):
        ax = fig.add_axes([0.045 + i * 0.315, 0.515, 0.28, 0.335])
        S.strip(ax)
        ax.set_aspect('equal')
        pts = np.column_stack([d['x'], d['y']]).reshape(-1, 1, 2)
        segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
        lc = LineCollection(segs, cmap=S.SPEED_CMAP, linewidths=3.4,
                            capstyle='round')
        lc.set_array(d['speed'][:-1])
        lc.set_clim(50, 360)
        ax.add_collection(lc)
        cx, cy = d['x'].mean(), d['y'].mean()
        ax.set_xlim(cx - span / 2, cx + span / 2)
        ax.set_ylim(cy - span / 2, cy + span / 2)
        left = 0.045 + i * 0.315
        fig.text(left, 0.870, d['label'], color=S.INK, fontsize=15,
                 fontweight='bold', va='baseline')
        fig.text(left + 0.075, 0.870, S.fmt_laptime(d['laptime']), color=S.INK_2,
                 fontsize=13, va='baseline', family='monospace')
        fig.text(left, 0.487,
                 f'{d["length"] / 1000:.2f} km   ·   {d["corners"]} corners   ·   '
                 f'{d["vmax"]:.0f} km/h peak',
                 color=S.INK_3, fontsize=10, va='baseline')

    fig.text(0.04, 0.452,
             'All three maps are drawn at one shared scale, so the size difference '
             'between them is real.', color=S.INK_3, fontsize=10, va='baseline')

    # comparative bars
    metrics = [
        ('Lap length (km)', [d['length'] / 1000 for d in data], '{:.2f}'),
        ('Top speed (km/h)', [d['vmax'] for d in data], '{:.0f}'),
        ('Average speed (km/h)', [d['vmean'] for d in data], '{:.0f}'),
        ('Full throttle (%)', [d['full'] for d in data], '{:.0f}'),
    ]
    palette = [S.RED, S.TEAL, S.AMBER]
    for i, (label, vals, fmt) in enumerate(metrics):
        ax = fig.add_axes([0.045 + (i % 4) * 0.238, 0.095, 0.195, 0.315])
        S.panel(ax, label.upper())
        ax.grid(axis='x', visible=False)
        bars = ax.bar([d['label'] for d in data], vals, color=palette[:len(data)],
                      width=0.62)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, v, fmt.format(v),
                    ha='center', va='bottom', color=S.INK, fontsize=10.5,
                    fontweight='bold')
        ax.set_ylim(0, max(vals) * 1.22)
        ax.tick_params(axis='x', labelsize=9.5)

    save(fig, SIM_OUT / 'track_comparison.png',
         'Source: FastF1 qualifying telemetry, pole lap at each circuit · '
         'corner counts from the FastF1 circuit info')


def build_strategy():
    print('Strategy figures')
    fig_tyre_strategy()
    fig_pit_stops()


def build_circuits():
    print('Circuit figures')
    fig_track_comparison()


GROUPS['strategy'] = build_strategy
GROUPS['circuits'] = build_circuits




def main():
    which = sys.argv[1:] or list(GROUPS)
    for name in which:
        if name in GROUPS:
            GROUPS[name]()
        else:
            print(f'unknown group: {name}')


if __name__ == '__main__':
    main()
