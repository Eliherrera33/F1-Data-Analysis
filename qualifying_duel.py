# -*- coding: utf-8 -*-
"""
Qualifying head-to-head: the two fastest laps of a session, side by side.

This replaces the old race_simulation / race_simulation_advanced animations.
Those drew cars going round a track at an arbitrary playback rate, which looks
busy but says nothing - you could not tell who was quicker or where.

What an engineer actually wants from two laps is: where did the time go. So
this shows, all locked to one clock:

  * the track coloured by minisector dominance - who owned which stretch
  * both cars, positioned by TIME, so the quicker driver visibly pulls away
  * the cumulative delta trace, the single graphic every pit wall watches
  * speed, throttle/brake and gear for both drivers under a shared playhead

Everything comes from real FastF1 telemetry in ./cache. Output is H.264 MP4.

    python qualifying_duel.py
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.collections import LineCollection

import fastf1

import f1_style as S

warnings.filterwarnings('ignore')

ROOT = Path(__file__).parent
CACHE = ROOT / 'cache'
OUT = ROOT / 'race_simulations'
OUT.mkdir(exist_ok=True)
CACHE.mkdir(exist_ok=True)
fastf1.Cache.enable_cache(str(CACHE))

GRID = 3000        # samples along the lap for the common distance grid
MINISECTORS = 26   # dominance segments
CLIP_SECONDS = 18  # target playback length
FPS = 50


# --------------------------------------------------------------------------
# data
# --------------------------------------------------------------------------

def _pick_driver(laps, abbr):
    """fastf1 renamed pick_driver -> pick_drivers; support both."""
    if hasattr(laps, 'pick_drivers'):
        return laps.pick_drivers(abbr)
    return laps.pick_driver(abbr)


def load_duel(year: int, gp: str, session_type: str = 'Q'):
    """Session plus the two quickest drivers' fastest laps, resampled onto a
    shared distance grid so every channel is directly comparable."""
    ses = fastf1.get_session(year, gp, session_type)
    ses.load()

    order = [r['Abbreviation'] for _, r in ses.results.iterrows()]
    picked = []
    for abbr in order:
        try:
            lap = _pick_driver(ses.laps, abbr).pick_fastest()
            if lap is None or lap.isnull().all():
                continue
            tel = lap.get_telemetry()
            if tel is None or len(tel) < 50 or 'X' not in tel:
                continue
            picked.append((abbr, lap, tel))
        except Exception:
            continue
        if len(picked) == 2:
            break

    if len(picked) < 2:
        raise RuntimeError(f'could not get two clean laps for {year} {gp}')

    # Align on LAP FRACTION, not raw distance. FastF1 integrates distance from
    # speed, so two drivers round the same lap record different totals - at
    # Monaco 2024 it is 3264.9 m for LEC against 3295.8 m for PIA, purely from
    # racing line. Cutting both to the shorter distance compares a full lap
    # against 99% of one, which put the delta 0.4 s out. Normalising each lap to
    # 0-1 puts both drivers at the same point on track, and makes the delta at
    # u = 1 exactly the official lap time difference.
    drivers = []
    grid_u = np.linspace(0.0, 1.0, GRID)
    dist_ref = None

    for abbr, lap, tel in picked:
        d = tel['Distance'].to_numpy(dtype=float)
        t = (tel['Time'] - tel['Time'].iloc[0]).dt.total_seconds().to_numpy()
        # Distance must increase strictly for np.interp to behave.
        keep = np.concatenate(([True], np.diff(d) > 0))
        d, t = d[keep], t[keep]
        d = d - d[0]
        u = d / d[-1]
        if dist_ref is None:
            dist_ref = d[-1]

        def on_grid(col, default=0.0):
            if col not in tel:
                return np.full_like(grid_u, default)
            v = tel[col].to_numpy(dtype=float)[keep]
            return np.interp(grid_u, u, v)

        team = lap['Team'] if 'Team' in lap else ''
        drivers.append({
            'abbr': abbr,
            'team': team,
            'color': S.team_color(team, S.TEAL if not drivers else S.RED),
            'laptime': lap['LapTime'].total_seconds() if lap['LapTime'] is not None else float('nan'),
            'compound': str(lap['Compound']) if 'Compound' in lap else 'UNKNOWN',
            't': np.interp(grid_u, u, t),
            'distance_m': d[-1],
            'x': on_grid('X'), 'y': on_grid('Y'),
            'speed': on_grid('Speed'),
            'throttle': on_grid('Throttle'),
            'brake': on_grid('Brake'),
            'gear': on_grid('nGear', 1.0),
        })

    # Distinct colours even for team mates
    if drivers[0]['color'] == drivers[1]['color']:
        drivers[0]['color'], drivers[1]['color'] = S.TEAL, S.RED

    # The endpoint delta must reproduce the official gap; if it does not, the
    # alignment is wrong and every number downstream is wrong with it.
    gap_official = drivers[1]['laptime'] - drivers[0]['laptime']
    gap_telemetry = drivers[1]['t'][-1] - drivers[0]['t'][-1]
    if abs(gap_official - gap_telemetry) > 0.005:
        print(f'    WARNING: delta endpoint {gap_telemetry:+.3f}s does not match '
              f'official gap {gap_official:+.3f}s')

    # Report the x axis in metres using the reference driver's lap length.
    grid = grid_u * dist_ref
    return ses, grid, drivers


def minisector_winners(grid, drivers, n=MINISECTORS):
    """Time through each minisector; index of the quicker driver per segment.

    `grid` is uniform in lap fraction, so equal bins are equal stretches of
    track for both drivers even though their measured distances differ."""
    edges = np.linspace(0, grid[-1], n + 1)
    idx = np.searchsorted(grid, edges)
    idx[-1] = len(grid) - 1
    winners, margins = [], []
    for i in range(n):
        a, b = idx[i], idx[i + 1]
        ta = drivers[0]['t'][b] - drivers[0]['t'][a]
        tb = drivers[1]['t'][b] - drivers[1]['t'][a]
        winners.append(0 if ta <= tb else 1)
        margins.append(abs(ta - tb))
    return edges, np.array(winners), np.array(margins)


def rotate(x, y, deg):
    r = np.deg2rad(deg)
    return x * np.cos(r) - y * np.sin(r), x * np.sin(r) + y * np.cos(r)


# --------------------------------------------------------------------------
# figure
# --------------------------------------------------------------------------

def build(year, gp, label, session_type='Q', still=False):
    ses, grid, drv = load_duel(year, gp, session_type)
    a, b = drv
    delta = b['t'] - a['t']          # >0 means driver b is behind
    edges, winners, margins = minisector_winners(grid, drv)

    # orient the circuit the way the FIA map does, when fastf1 knows the angle
    rot = 0.0
    corners = None
    try:
        ci = ses.get_circuit_info()
        rot = float(ci.rotation)
        corners = ci.corners
    except Exception:
        pass
    ax_, ay_ = rotate(a['x'], a['y'], rot)
    bx_, by_ = rotate(b['x'], b['y'], rot)

    S.apply()
    fig = plt.figure(figsize=(19.2, 10.8), dpi=100)

    ax_map = fig.add_axes([0.028, 0.115, 0.465, 0.655])
    ax_delta = fig.add_axes([0.545, 0.635, 0.430, 0.250])
    ax_speed = fig.add_axes([0.545, 0.375, 0.430, 0.215])
    ax_pedal = fig.add_axes([0.545, 0.225, 0.430, 0.115])
    ax_gear = fig.add_axes([0.545, 0.125, 0.430, 0.070])

    S.strip(ax_map)
    ax_map.set_aspect('equal')

    # ---- header -----------------------------------------------------------
    ev = ses.event
    name = f"{ev['EventName']} {year}" if 'EventName' in ev else f'{gp} {year}'
    fig.text(0.028, 0.975, name.upper(), color=S.INK, fontsize=30,
             fontweight='bold', va='top', ha='left')
    fig.text(0.028, 0.929, 'QUALIFYING · FASTEST LAP HEAD-TO-HEAD',
             color=S.INK_2, fontsize=11.5, va='top', ha='left',
             fontweight='600')
    S.accent_rule(fig, x=0.028, y=0.908, w=0.048)

    # ---- track: minisector dominance --------------------------------------
    pts = np.column_stack([ax_, ay_]).reshape(-1, 1, 2)
    segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
    seg_owner = np.zeros(len(segs), dtype=int)
    for i in range(len(winners)):
        m = (grid[:-1] >= edges[i]) & (grid[:-1] < edges[i + 1])
        seg_owner[m] = winners[i]

    ax_map.add_collection(LineCollection(
        segs, colors=[drv[o]['color'] for o in seg_owner], linewidths=7.5,
        capstyle='round', zorder=2))
    ax_map.add_collection(LineCollection(
        segs, colors=['#000000'] * len(segs), linewidths=12,
        capstyle='round', zorder=1, alpha=0.55))

    pad = 0.06 * max(np.ptp(ax_), np.ptp(ay_))
    ax_map.set_xlim(ax_.min() - pad, ax_.max() + pad)
    ax_map.set_ylim(ay_.min() - pad, ay_.max() + pad)

    if corners is not None and len(corners):
        cx, cy = rotate(corners['X'].to_numpy(float), corners['Y'].to_numpy(float), rot)
        for i in range(len(corners)):
            ax_map.text(cx[i], cy[i], str(corners['Number'].iloc[i]),
                        color=S.INK_3, fontsize=7.5, ha='center', va='center',
                        zorder=5)

    # Two rings each: a dark halo to lift the marker off the coloured track,
    # then the team colour with a white edge.
    for _c in (a, b):
        ax_map.plot([], [], 'o', ms=22, color=S.BG, alpha=0.9, zorder=7)
    halo_a, = ax_map.plot([], [], 'o', ms=23, color=S.BG, zorder=7)
    halo_b, = ax_map.plot([], [], 'o', ms=23, color=S.BG, zorder=7)
    car_a, = ax_map.plot([], [], 'o', ms=16, color=a['color'], mec='white',
                         mew=2.0, zorder=9)
    car_b, = ax_map.plot([], [], 'o', ms=16, color=b['color'], mec='white',
                         mew=2.0, zorder=9)

    ax_map.text(0.5, -0.055,
                f'Track coloured by minisector — {MINISECTORS} segments, quicker driver takes the segment',
                transform=ax_map.transAxes, color=S.INK_3, fontsize=9.5, ha='center')

    # ---- delta ------------------------------------------------------------
    S.panel(ax_delta, f"CUMULATIVE DELTA  ·  {b['abbr']} minus {a['abbr']}  (s)")
    ax_delta.axhline(0, color=S.INK_3, lw=1)
    ax_delta.fill_between(grid, delta, 0, where=delta >= 0, color=a['color'],
                          alpha=0.30, interpolate=True, lw=0)
    ax_delta.fill_between(grid, delta, 0, where=delta < 0, color=b['color'],
                          alpha=0.30, interpolate=True, lw=0)
    ax_delta.plot(grid, delta, color=S.INK, lw=1.9)
    span = max(0.12, float(np.abs(delta).max()) * 1.25)
    ax_delta.set_ylim(-span, span)
    ax_delta.set_xlim(0, grid[-1])
    ax_delta.set_xticklabels([])
    ax_delta.text(0.012, 0.90, f"{a['abbr']} ahead", transform=ax_delta.transAxes,
                  color=a['color'], fontsize=9.5, fontweight='bold', va='top')
    ax_delta.text(0.012, 0.10, f"{b['abbr']} ahead", transform=ax_delta.transAxes,
                  color=b['color'], fontsize=9.5, fontweight='bold', va='bottom')

    # ---- speed ------------------------------------------------------------
    S.panel(ax_speed, 'SPEED  (km/h)')
    for d in drv:
        ax_speed.plot(grid, d['speed'], color=d['color'], lw=1.9,
                      label=d['abbr'], alpha=0.95)
    ax_speed.set_xlim(0, grid[-1])
    ax_speed.set_xticklabels([])
    ax_speed.legend(loc='lower right', ncol=2)

    # ---- pedals -----------------------------------------------------------
    S.panel(ax_pedal, 'THROTTLE  /  BRAKE  (%)')
    for d in drv:
        ax_pedal.plot(grid, d['throttle'], color=d['color'], lw=1.6, alpha=0.95)
        ax_pedal.fill_between(grid, 0, np.where(d['brake'] > 0, 100, 0),
                              color=d['color'], alpha=0.16, lw=0)
    ax_pedal.set_ylim(-4, 108)
    ax_pedal.set_xlim(0, grid[-1])
    ax_pedal.set_xticklabels([])

    # ---- gear -------------------------------------------------------------
    S.panel(ax_gear, 'GEAR')
    for k, d in enumerate(drv):
        ax_gear.step(grid, d['gear'] + (0.0 if k == 0 else 0.06),
                     where='post', color=d['color'], lw=1.5, alpha=0.95)
    ax_gear.set_ylim(0.5, 8.6)
    ax_gear.set_yticks([2, 4, 6, 8])
    ax_gear.set_xlim(0, grid[-1])
    ax_gear.set_xlabel('Lap distance (m)')

    playheads = [ax.axvline(0, color=S.INK, lw=1.1, alpha=0.85, zorder=9)
                 for ax in (ax_delta, ax_speed, ax_pedal, ax_gear)]

    # ---- driver tiles -----------------------------------------------------
    tiles = []
    TILE_Y, TILE_H, TILE_W = 0.792, 0.092, 0.226
    for k, d in enumerate(drv):
        x0 = 0.028 + k * 0.242
        fig.patches.append(plt.Rectangle(
            (x0, TILE_Y), TILE_W, TILE_H, transform=fig.transFigure,
            facecolor=S.PANEL, edgecolor=S.LINE, lw=1, zorder=3))
        fig.add_artist(plt.Line2D([x0, x0], [TILE_Y, TILE_Y + TILE_H],
                                  color=d['color'], lw=5,
                                  transform=fig.transFigure, zorder=4))
        fig.text(x0 + 0.016, TILE_Y + TILE_H - 0.014, d['abbr'], color=S.INK,
                 fontsize=23, fontweight='bold', va='top', zorder=5)
        fig.text(x0 + 0.084, TILE_Y + TILE_H - 0.016, d['team'], color=S.INK_2,
                 fontsize=10, va='top', zorder=5)
        fig.text(x0 + 0.084, TILE_Y + TILE_H - 0.040, S.fmt_laptime(d['laptime']),
                 color=S.INK, fontsize=15, va='top', fontweight='bold', zorder=5)
        tiles.append(fig.text(x0 + 0.016, TILE_Y + 0.020, '', color=S.INK_2,
                              fontsize=11, va='bottom', zorder=5,
                              family='monospace'))

    # Live gap sits in the header band above the delta panel: nothing else is
    # there, and inside the axes it collided with the trace itself.
    fig.text(0.975, 0.975, 'LIVE GAP', color=S.INK_3, fontsize=10,
             ha='right', va='top', fontweight='600')
    gap_txt = fig.text(0.975, 0.947, '', color=S.INK, fontsize=27, ha='right',
                       va='top', fontweight='bold', family='monospace')
    fig.text(0.975, 0.906, f"{b['abbr']} minus {a['abbr']}", color=S.INK_3,
             fontsize=9.5, ha='right', va='top')

    S.footer(fig, f'Source: FastF1 telemetry · {name} qualifying · '
                  f'{MINISECTORS} minisectors · lap replayed at '
                  f'{max(a["t"][-1], b["t"][-1]) / CLIP_SECONDS:.1f}x')

    # ---- animation --------------------------------------------------------
    total_t = float(max(a['t'][-1], b['t'][-1]))
    n_frames = CLIP_SECONDS * FPS
    times = np.linspace(0, total_t, n_frames)

    def frame(i):
        now = times[i]
        for d, car, halo, tile in zip(drv, (car_a, car_b), (halo_a, halo_b), tiles):
            dist = float(np.interp(now, d['t'], grid))
            j = int(np.searchsorted(grid, dist))
            j = min(max(j, 0), GRID - 1)
            xs, ys = (ax_, ay_) if d is a else (bx_, by_)
            car.set_data([xs[j]], [ys[j]])
            halo.set_data([xs[j]], [ys[j]])
            tile.set_text(f"{d['speed'][j]:3.0f} km/h   G{int(d['gear'][j])}")
        # playhead follows the leading car so the traces stay in step
        lead = float(np.interp(now, a['t'], grid))
        for ph in playheads:
            ph.set_xdata([lead, lead])
        gap_txt.set_text(f"{S.fmt_delta(float(np.interp(lead, grid, delta)))} s")
        return ()

    # A frame from partway round the lap doubles as the <video> poster, so the
    # card on the page is never a blank rectangle before playback starts.
    frame(int(n_frames * 0.42))
    poster = OUT / f'{label}_qualifying_duel.jpg'
    fig.savefig(poster, dpi=100, facecolor=S.BG, pil_kwargs={'quality': 88})
    print(f'  {poster.name}  {poster.stat().st_size / 1e3:.0f} KB')

    if still:
        plt.close(fig)
        return poster

    anim = FuncAnimation(fig, frame, frames=n_frames, interval=1000 / FPS, blit=False)
    out = OUT / f'{label}_qualifying_duel.mp4'
    anim.save(str(out), writer=S.writer(fps=FPS), dpi=100)
    plt.close(fig)

    size_mb = out.stat().st_size / 1e6
    print(f'  {out.name}  {size_mb:.2f} MB  '
          f'{a["abbr"]} {S.fmt_laptime(a["laptime"])} vs '
          f'{b["abbr"]} {S.fmt_laptime(b["laptime"])}')
    return out


def main():
    import sys
    still = '--still' in sys.argv
    targets = [
        (2024, 'Monaco', 'monaco'),
        (2024, 'Belgium', 'spa'),
        (2024, 'Italy', 'monza'),
    ]
    print('Building qualifying head-to-head animations')
    for year, gp, label in targets:
        try:
            print(f'{gp} {year} ...')
            build(year, gp, label, still=still)
        except Exception as exc:
            print(f'  SKIPPED {gp}: {type(exc).__name__}: {exc}')


if __name__ == '__main__':
    main()
