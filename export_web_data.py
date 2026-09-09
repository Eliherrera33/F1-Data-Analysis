# -*- coding: utf-8 -*-
"""
Export the data the interactive features read in the browser.

    python export_web_data.py geometry    # track curvature + calibration
    python export_web_data.py duels       # head-to-head telemetry
    python export_web_data.py quiz        # anonymised traces for the blind test
    python export_web_data.py             # all of it

Everything lands in data/web/ as compact JSON. Nothing in here is invented:
geometry comes from the pole lap's position channel, the head-to-heads from
qualifying_duel.load_duel, and the quiz traces from real laps.
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
from scipy.signal import savgol_filter

import fastf1

import cfd_flow as C
import lapsim_core as L

warnings.filterwarnings('ignore')

ROOT = Path(__file__).parent
OUT = ROOT / 'data' / 'web'
OUT.mkdir(parents=True, exist_ok=True)
fastf1.Cache.enable_cache(str(ROOT / 'cache'))

CIRCUITS = [
    ('monaco', 2024, 'Monaco', 'Monaco'),
    ('spa', 2024, 'Belgium', 'Spa-Francorchamps'),
    ('monza', 2024, 'Italy', 'Monza'),
]

DS = 4.0          # m, uniform spacing of the exported track
SMOOTH_M = 90.0   # m, smoothing window for curvature


def _session(year, gp, kind='Q'):
    ses = fastf1.get_session(year, gp, kind)
    ses.load()
    return ses


def _pick(laps, abbr):
    return laps.pick_drivers(abbr) if hasattr(laps, 'pick_drivers') else laps.pick_driver(abbr)


def _round(a, nd):
    return [round(float(v), nd) for v in np.asarray(a).tolist()]


# ==========================================================================
# 1. track geometry
# ==========================================================================

def track_geometry(year, gp, smooth_m=SMOOTH_M):
    """Uniformly spaced, smoothed centreline with curvature, plus the real
    pole lap's speed and DRS state resampled onto the same grid."""
    ses = _session(year, gp)
    abbr = ses.results.iloc[0]['Abbreviation']
    lap = _pick(ses.laps, abbr).pick_fastest()
    tel = lap.get_telemetry()

    # FastF1 position is in tenths of a metre.
    x = tel['X'].to_numpy(float) / 10.0
    y = tel['Y'].to_numpy(float) / 10.0
    v = tel['Speed'].to_numpy(float) / 3.6
    drs = tel['DRS'].to_numpy(float) if 'DRS' in tel else np.zeros(len(tel))

    # Arc length from the positions themselves so geometry is self-consistent.
    seg = np.hypot(np.diff(x), np.diff(y))
    s = np.concatenate([[0.0], np.cumsum(seg)])
    keep = np.concatenate([[True], np.diff(s) > 0])
    x, y, v, drs, s = x[keep], y[keep], v[keep], drs[keep], s[keep]
    length = float(s[-1])

    grid = np.arange(0.0, length, DS)
    xs = np.interp(grid, s, x)
    ys = np.interp(grid, s, y)
    vs = np.interp(grid, s, v)
    drs_open = np.interp(grid, s, (drs >= 10).astype(float)) > 0.5
    drs_source = 'channel'
    if not drs_open.any():
        # Some sessions carry a dead DRS channel (Spa 2024 is a constant 9 for
        # the whole lap). Fall back to the two longest full-throttle straights,
        # skipping the first 150 m of each for the activation point.
        drs_source = 'derived'
        fast = np.interp(grid, s, v) > 230 / 3.6
        runs, i = [], 0
        while i < len(fast):
            if fast[i]:
                j = i
                while j < len(fast) and fast[j]:
                    j += 1
                runs.append((j - i, i, j))
                i = j
            else:
                i += 1
        runs.sort(reverse=True)
        drs_open = np.zeros(len(grid), dtype=bool)
        # A DRS zone is a few hundred metres, not a whole flat-out sector. Left
        # uncapped this covered 42% of Spa, discounted the wing's drag almost
        # to nothing, and had the optimiser asking for maximum wing there.
        for _, a, b in runs[:2]:
            a2 = min(b, a + int(150 / DS))
            b2 = min(b, a2 + int(700 / DS))
            drs_open[a2:b2] = True

    # Curvature needs two derivatives of a noisy signal, which is where the
    # unphysical 8 g peaks came from. A Savitzky-Golay fit over ~90 m gives the
    # derivatives analytically from a local cubic, with wraparound because the
    # lap is closed.
    win = int(smooth_m / DS) | 1
    dx = savgol_filter(xs, win, 3, deriv=1, delta=DS, mode='wrap')
    dy = savgol_filter(ys, win, 3, deriv=1, delta=DS, mode='wrap')
    ddx = savgol_filter(xs, win, 3, deriv=2, delta=DS, mode='wrap')
    ddy = savgol_filter(ys, win, 3, deriv=2, delta=DS, mode='wrap')
    kappa = (dx * ddy - dy * ddx) / np.power(dx * dx + dy * dy, 1.5)
    xs_s = savgol_filter(xs, win, 3, mode='wrap')
    ys_s = savgol_filter(ys, win, 3, mode='wrap')

    rot = 0.0
    corners = []
    try:
        ci = ses.get_circuit_info()
        rot = float(ci.rotation)
        for _, c in ci.corners.iterrows():
            corners.append(dict(n=int(c['Number']), d=float(c['Distance'])))
    except Exception:
        pass

    lat_g = vs * vs * np.abs(kappa) / L.G0
    return dict(
        name=ses.event['EventName'], year=year, abbr=abbr,
        laptime=float(lap['LapTime'].total_seconds()),
        length=length, ds=DS, rotation=rot, corners=corners,
        x=xs_s, y=ys_s, kappa=kappa, v_real=vs, drs=drs_open, drs_source=drs_source,
        lat_g_p99=float(np.percentile(lat_g, 99.5)),
        min_radius=float(np.percentile(1 / np.maximum(np.abs(kappa), 1e-6), 1)),
    )


def aero_for(rw, fw, frh, rrh):
    r = C.solve(250.0, rw, fw, frh, rrh)
    return r['s_cz'], r['s_cx']


def best_setup(track, mu, power, coarse=False):
    """Search over the levers the page exposes. Front wing tracks the rear
    wing to hold balance; rear ride height sits 12 mm above the front."""
    best = None
    rws = range(5, 36, 5) if coarse else range(5, 36, 2)
    frhs = (24, 30, 40) if coarse else (22, 26, 30, 36, 42)
    for rw in rws:
        for frh in frhs:
            scz, scx = aero_for(rw, rw * 0.5, frh, frh + 12)
            t = L.simulate(track['kappa'], track['ds'], track['drs'], scz, scx,
                           mu=mu, power=power)['lap_time']
            if best is None or t < best[0]:
                best = (t, rw, frh)
    return best


def calibrate(tracks):
    """Pick mu and power so the best-found setup lands on the real pole."""
    results = []
    for mu in (1.40, 1.46, 1.52, 1.58, 1.64, 1.70):
        for power in (520e3, 560e3, 600e3, 640e3, 680e3):
            err = 0.0
            for tr in tracks:
                t, _, _ = best_setup(tr, mu, power, coarse=True)
                err += ((t - tr['laptime']) / tr['laptime']) ** 2
            results.append((err, mu, power))
    err, mu, power = min(results)
    return mu, power, float(np.sqrt(err / len(tracks)) * 100)


def export_geometry():
    # The smoothing window trades noise against real corner tightness, so it
    # is swept alongside grip and power and the best joint fit wins.
    best = None
    for smooth_m in (60.0, 90.0, 120.0):
        tracks = []
        for key, year, gp, label in CIRCUITS:
            tr = track_geometry(year, gp, smooth_m)
            tr['key'], tr['label'] = key, label
            tracks.append(tr)
        mu, power, rms = calibrate(tracks)
        print(f'smoothing {smooth_m:.0f} m -> mu={mu:.2f} P={power/1e3:.0f} kW rms={rms:.2f}%', flush=True)
        if best is None or rms < best[0]:
            best = (rms, smooth_m, mu, power, tracks)
    rms, smooth_m, mu, power, tracks = best
    print(f'\nchosen: smoothing {smooth_m:.0f} m, mu={mu:.2f}, power={power/1e3:.0f} kW, rms {rms:.2f}%')
    for tr in tracks:
        print(f'  {tr["label"]:<18s} {tr["length"]:.0f} m, {len(tr["x"])} pts, pole {tr["abbr"]} '
              f'{tr["laptime"]:.3f}s, lat g p99.5 = {tr["lat_g_p99"]:.2f}, '
              f'min radius {tr["min_radius"]:.0f} m, DRS {tr["drs"].mean()*100:.0f}% ({tr["drs_source"]})')

    payload = dict(mu=mu, power=power, mass=L.MASS, smooth_m=smooth_m, rms_pct=round(rms, 2),
                   circuits=[])
    print('\nwith calibrated model:')
    for tr in tracks:
        t, rw, frh = best_setup(tr, mu, power)
        ref_cz, ref_cx = aero_for(20, 10, 40, 50)
        t_ref = L.simulate(tr['kappa'], tr['ds'], tr['drs'], ref_cz, ref_cx,
                           mu=mu, power=power)['lap_time']
        resid = (t - tr['laptime']) / tr['laptime'] * 100
        print(f'  {tr["label"]:<18s} real {tr["laptime"]:7.3f}  best {t:7.3f} '
              f'(rw {rw}, frh {frh})  {resid:+.2f}%   reference setup {t_ref:7.3f}')
        payload['circuits'].append(dict(
            key=tr['key'], label=tr['label'], name=tr['name'], year=tr['year'],
            pole_abbr=tr['abbr'], pole_time=round(tr['laptime'], 3),
            length=round(tr['length'], 1), ds=tr['ds'], rotation=tr['rotation'],
            corners=tr['corners'],
            x=_round(tr['x'], 1), y=_round(tr['y'], 1),
            kappa=_round(tr['kappa'], 6),
            drs=[int(b) for b in tr['drs']], drs_source=tr['drs_source'],
            v_real=_round(tr['v_real'], 2),
            best_rw=rw, best_frh=frh, best_time=round(t, 3),
            residual_pct=round(resid, 2),
        ))

    path = OUT / 'tracks.json'
    path.write_text(json.dumps(payload, separators=(',', ':')), encoding='utf-8')
    print(f'\nwrote {path.name}  {path.stat().st_size/1e3:.0f} KB')

    # keep lapsim_core's constants in step with the calibration
    core = (ROOT / 'lapsim_core.py').read_text(encoding='utf-8')
    import re
    core = re.sub(r'MU = [0-9.]+', f'MU = {mu:.2f}', core)
    core = re.sub(r'POWER_W = [0-9.]+', f'POWER_W = {power:.1f}', core)
    (ROOT / 'lapsim_core.py').write_text(core, encoding='utf-8')


# ==========================================================================
# 2. head-to-head duels
# ==========================================================================

def export_duels(n_points=720):
    import qualifying_duel as Q
    out = []
    for key, year, gp, label in CIRCUITS:
        print(f'{label} duel ...')
        ses, grid, drv = Q.load_duel(year, gp, 'Q')
        edges, winners, margins = Q.minisector_winners(grid, drv)
        rot = 0.0
        corners = []
        try:
            ci = ses.get_circuit_info()
            rot = float(ci.rotation)
            for _, c in ci.corners.iterrows():
                cx, cy = Q.rotate(float(c['X']), float(c['Y']), rot)
                corners.append(dict(n=int(c['Number']), d=float(c['Distance']),
                                    x=round(cx, 1), y=round(cy, 1)))
        except Exception:
            pass

        idx = np.linspace(0, len(grid) - 1, n_points).round().astype(int)
        drivers = []
        for d in drv:
            x, y = Q.rotate(d['x'], d['y'], rot)
            drivers.append(dict(
                abbr=d['abbr'], team=d['team'], color=d['color'],
                laptime=round(float(d['laptime']), 3),
                compound=d['compound'],
                t=_round(d['t'][idx], 3),
                x=_round(x[idx], 1), y=_round(y[idx], 1),
                speed=_round(d['speed'][idx], 1),
                throttle=_round(d['throttle'][idx], 0),
                brake=[int(b > 0) for b in d['brake'][idx]],
                gear=[int(g) for g in d['gear'][idx]],
            ))

        # The biggest swings, so the page can point at where the lap was won.
        swings = []
        for i in range(len(winners)):
            swings.append(dict(i=i, winner=int(winners[i]), margin=round(float(margins[i]), 3),
                               d0=round(float(edges[i]), 0), d1=round(float(edges[i + 1]), 0)))
        swings.sort(key=lambda s: -s['margin'])

        out.append(dict(
            key=key, label=label, name=ses.event['EventName'], year=year,
            length=round(float(grid[-1]), 1),
            dist=_round(grid[idx], 1),
            drivers=drivers,
            minisectors=dict(edges=_round(edges, 0), winners=[int(w) for w in winners],
                             margins=_round(margins, 3)),
            swings=swings[:6],
            corners=corners,
        ))
        a, b = drivers
        print(f'  {a["abbr"]} {a["laptime"]} v {b["abbr"]} {b["laptime"]}  '
              f'biggest swing {swings[0]["margin"]:.3f}s in minisector {swings[0]["i"]+1}')

    path = OUT / 'duels.json'
    path.write_text(json.dumps(out, separators=(',', ':')), encoding='utf-8')
    print(f'wrote {path.name}  {path.stat().st_size/1e3:.0f} KB')


# ==========================================================================
# 3. blind-test traces
# ==========================================================================

# Event names are passed to fastf1's fuzzy matcher, which resolved
# 'Great Britain' to the Austrian Grand Prix. Use the official short names.
# (key, year, name passed to fastf1, label shown, word that must appear in the
# resolved event name)
QUIZ_CIRCUITS = [
    ('monaco', 2024, 'Monaco', 'Monaco', 'Monaco'),
    ('spa', 2024, 'Belgium', 'Spa-Francorchamps', 'Belgian'),
    ('monza', 2024, 'Italy', 'Monza', 'Italian'),
    ('silverstone', 2024, 'British', 'Silverstone', 'British'),
    ('suzuka', 2024, 'Japan', 'Suzuka', 'Japanese'),
    ('redbullring', 2024, 'Austria', 'Red Bull Ring', 'Austrian'),
]


def export_quiz(n_points=320):
    out = []
    for key, year, gp, label, must in QUIZ_CIRCUITS:
        try:
            ses = _session(year, gp)
        except Exception as exc:
            print(f'  skipped {label}: {exc}')
            continue
        ev = str(ses.event['EventName'])
        if must.lower() not in ev.lower():
            print(f'  skipped {label}: fastf1 resolved "{gp}" to "{ev}"')
            continue
        laps = []
        for _, r in ses.results.head(6).iterrows():
            abbr = r['Abbreviation']
            try:
                lap = _pick(ses.laps, abbr).pick_fastest()
                tel = lap.get_telemetry()
                if tel is None or len(tel) < 50:
                    continue
            except Exception:
                continue
            d = tel['Distance'].to_numpy(float)
            d = d - d[0]
            u = d / d[-1]
            g = np.linspace(0, 1, n_points)
            spd = np.interp(g, u, tel['Speed'].to_numpy(float))
            thr = tel['Throttle'].to_numpy(float)
            brk = tel['Brake'].to_numpy(float)
            laps.append(dict(
                abbr=abbr, team=str(lap['Team']),
                laptime=round(float(lap['LapTime'].total_seconds()), 3),
                speed=_round(spd, 0),
                top=round(float(spd.max()), 0), low=round(float(spd.min()), 0),
                full_throttle=round(float((thr > 98).mean() * 100), 0),
                braking=round(float((brk > 0).mean() * 100), 0),
                brake_events=int(np.sum(np.diff((brk > 0).astype(int)) == 1)),
            ))
            if len(laps) == 2:
                break
        if len(laps) < 2:
            print(f'  skipped {label}: fewer than two clean laps')
            continue
        out.append(dict(key=key, label=label, name=ses.event['EventName'], year=year,
                        length=round(float(d[-1]), 0), laps=laps))
        print(f'  {label}: {laps[0]["abbr"]} {laps[0]["laptime"]} / '
              f'{laps[1]["abbr"]} {laps[1]["laptime"]}  top {laps[0]["top"]:.0f}  '
              f'full throttle {laps[0]["full_throttle"]:.0f}%')

    path = OUT / 'quiz.json'
    path.write_text(json.dumps(out, separators=(',', ':')), encoding='utf-8')
    print(f'wrote {path.name}  {path.stat().st_size/1e3:.0f} KB')


if __name__ == '__main__':
    what = sys.argv[1:] or ['geometry', 'duels', 'quiz']
    if 'geometry' in what:
        export_geometry()
    if 'duels' in what:
        export_duels()
    if 'quiz' in what:
        export_quiz()
