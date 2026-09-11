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
import pandas as pd
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
    # key, year, gp, label, must-match word in the resolved event name, kind
    ('monaco', 2024, 'Monaco', 'Monaco', 'Monaco', 'street'),
    ('spa', 2024, 'Belgium', 'Spa-Francorchamps', 'Belgian', 'power'),
    ('monza', 2024, 'Italy', 'Monza', 'Italian', 'power'),
    ('silverstone', 2024, 'British', 'Silverstone', 'British', 'flowing'),
    ('suzuka', 2024, 'Japan', 'Suzuka', 'Japanese', 'flowing'),
    ('redbullring', 2024, 'Austria', 'Red Bull Ring', 'Austrian', 'stop-go'),
    ('singapore', 2024, 'Singapore', 'Singapore', 'Singapore', 'street'),
    ('bahrain', 2024, 'Bahrain', 'Bahrain', 'Bahrain', 'stop-go'),
    ('hungaroring', 2024, 'Hungary', 'Hungaroring', 'Hungarian', 'twisty'),
    ('baku', 2024, 'Azerbaijan', 'Baku', 'Azerbaijan', 'street'),
    ('zandvoort', 2024, 'Netherlands', 'Zandvoort', 'Dutch', 'flowing'),
]

CSV = ROOT / 'data' / 'f1db_csv'
REF = ROOT / 'data' / 'reference'
CONS = ROOT / 'data' / 'consolidated'


def _resample(u, y, g):
    return np.interp(g, u, np.asarray(y, float))


def _quiz_circuit(key, year, gp, label, must, kind, n_points):
    ses = _session(year, gp)
    ev = str(ses.event['EventName'])
    if must.lower() not in ev.lower():
        raise RuntimeError(f'fastf1 resolved "{gp}" to "{ev}"')
    laps, track = [], None
    for _, r in ses.results.head(8).iterrows():
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
        spd = _resample(u, tel['Speed'], g)
        thr = tel['Throttle'].to_numpy(float)
        brk = tel['Brake'].to_numpy(float)
        gear = tel['nGear'].to_numpy(float)
        if track is None:
            gm = np.linspace(0, 1, 360)
            x = _resample(u, tel['X'], gm)
            y = _resample(u, tel['Y'], gm)
            x -= (x.max() + x.min()) / 2
            y -= (y.max() + y.min()) / 2
            scale = max(np.ptp(x), np.ptp(y)) / 2 or 1
            track = dict(x=_round(x / scale, 3), y=_round(y / scale, 3),
                         speed=_round(_resample(u, tel['Speed'], gm), 0),
                         length=round(float(d[-1]), 0))
        laps.append(dict(
            abbr=abbr, team=str(lap['Team']),
            laptime=round(float(lap['LapTime'].total_seconds()), 3),
            speed=_round(spd, 0),
            throttle=_round(_resample(u, thr, g), 0),
            brake=_round(_resample(u, brk, g), 0),
            gear=_round(_resample(u, gear, g), 0),
            top=round(float(spd.max()), 0), low=round(float(spd.min()), 0),
            full_throttle=round(float((thr > 98).mean() * 100), 0),
            braking=round(float((brk > 0).mean() * 100), 0),
            brake_events=int(np.sum(np.diff((brk > 0).astype(int)) == 1)),
            gear_max=int(np.nanmax(gear)),
            gear_min=int(np.nanmin(gear[gear > 0])) if (gear > 0).any() else 1,
        ))
        if len(laps) == 2:
            break
    if len(laps) < 2:
        raise RuntimeError('fewer than two clean laps')
    return dict(key=key, label=label, name=ev, year=year, kind=kind,
                length=track['length'], map=track, laps=laps)


def _names():
    drivers = pd.read_csv(REF / 'drivers.csv')
    cons = pd.read_csv(REF / 'constructors.csv')
    dn = dict(zip(drivers['id'], drivers['name']))
    cn = dict(zip(cons['id'], cons['name']))
    return dn, cn


def _secs(s):
    """'1:29.179' or '25.208' -> seconds; NaN on failure."""
    try:
        s = str(s)
        if ':' in s:
            m, r = s.split(':')
            return int(m) * 60 + float(r)
        return float(s)
    except Exception:
        return float('nan')


def _race_name(r):
    return r.split('-', 1)[1].replace('-', ' ').title().replace(' Of ', ' of ')


def export_history():
    """Season-level facts for the history / strategy questions. Everything
    is derived from the f1db CSVs so the quiz can cite real numbers."""
    dn, cn = _names()

    champions = []
    wc = pd.read_csv(CONS / 'world_champions.csv')
    cc = pd.read_csv(CONS / 'constructor_champions.csv').set_index('year')
    rp = pd.read_csv(CONS / 'races_per_year.csv').set_index('year')
    for _, r in wc.iterrows():
        y = int(r['year'])
        champions.append(dict(
            year=y, driver=dn.get(r['champion'], r['champion']), points=float(r['points']),
            team=cn.get(cc.loc[y, 'champion'], cc.loc[y, 'champion']) if y in cc.index else None,
            team_points=float(cc.loc[y, 'points']) if y in cc.index else None,
            races=int(rp.loc[y, 'races']) if y in rp.index else None))

    seasons = {}
    pit_evolution = []
    summ_all = pd.read_csv(CONS / 'driver_season_summary.csv')
    for y in range(2012, 2026):
        try:
            pit = pd.read_csv(CSV / f'{y}_pit_stops.csv')
        except FileNotFoundError:
            continue
        pit['secs'] = pit['time'].map(_secs)
        # pit-lane time incl. stationary; drops red-flag / penalty stops
        clean = pit[(pit['secs'] > 15) & (pit['secs'] < 60)]
        if len(clean):
            pit_evolution.append(dict(year=y, median=round(float(clean['secs'].median()), 2),
                                      fastest=round(float(clean['secs'].min()), 2), n=int(len(clean))))
        if y < 2022:
            continue
        res = pd.read_csv(CSV / f'{y}_race_results.csv')
        qual = pd.read_csv(CSV / f'{y}_qualifying.csv')
        fl = pd.read_csv(CSV / f'{y}_fastest_laps.csv')
        summ = summ_all[summ_all['year'] == y].sort_values('total_points', ascending=False).head(8)

        winners = []
        for race, grp in res.groupby('race', sort=True):
            w = grp[grp['position'].astype(str) == '1']
            if not len(w):
                continue
            w = w.iloc[0]
            try:
                grid = int(float(w['gridPosition']))
            except Exception:
                grid = None
            winners.append(dict(race=_race_name(race), driver=dn.get(w['driverId'], w['driverId']),
                                team=cn.get(w['constructorId'], w['constructorId']), grid=grid,
                                laps=int(float(w['laps'])) if pd.notna(w['laps']) else None))
        poles = []
        for race, grp in qual.groupby('race', sort=True):
            g = grp.assign(_p=pd.to_numeric(grp['position'], errors='coerce')).dropna(subset=['_p']).sort_values('_p')
            if len(g) < 2:
                continue
            p1, p2 = g.iloc[0], g.iloc[1]
            t1, t2 = _secs(p1['q3']), _secs(p2['q3'])
            if not (np.isfinite(t1) and np.isfinite(t2)):
                continue
            poles.append(dict(race=_race_name(race), driver=dn.get(p1['driverId'], p1['driverId']),
                              team=cn.get(p1['constructorId'], p1['constructorId']),
                              time=round(t1, 3), margin=round(t2 - t1, 3),
                              second=dn.get(p2['driverId'], p2['driverId'])))
        fastest = []
        for race, grp in fl.groupby('race', sort=True):
            g = grp[grp['position'].astype(str) == '1']
            if not len(g):
                continue
            g = g.iloc[0]
            fastest.append(dict(race=_race_name(race), driver=dn.get(g['driverId'], g['driverId']),
                                lap=int(float(g['lap'])) if pd.notna(g['lap']) else None,
                                time=round(_secs(g['time']), 3)))
        by_team = clean.groupby('constructorId')['secs'].median().sort_values()
        quick = clean.nsmallest(5, 'secs')
        seasons[str(y)] = dict(
            standings=[dict(driver=dn.get(r['driver'], r['driver']), team=cn.get(r['constructor'], r['constructor']),
                            points=float(r['total_points']), wins=int(r['wins']), podiums=int(r['podiums']),
                            races=int(r['races'])) for _, r in summ.iterrows()],
            winners=winners, poles=poles, fastest=fastest,
            pit_by_team=[dict(team=cn.get(k, k), median=round(float(v), 2)) for k, v in by_team.items()],
            pit_fastest=[dict(team=cn.get(r['constructorId'], r['constructorId']),
                              driver=dn.get(r['driverId'], r['driverId']),
                              race=_race_name(r['race']), time=round(float(r['secs']), 3), lap=int(r['lap']))
                         for _, r in quick.iterrows()],
            pit_median=round(float(clean['secs'].median()), 2),
            stops_per_car_race=round(float(len(clean)) / max(1, clean['driverId'].nunique()) / max(1, len(winners)), 2),
        )
    return dict(champions=champions, seasons=seasons, pit_evolution=pit_evolution)


def export_quiz(n_points=320):
    circuits = []
    for key, year, gp, label, must, kind in QUIZ_CIRCUITS:
        try:
            c = _quiz_circuit(key, year, gp, label, must, kind, n_points)
        except Exception as exc:
            print(f'  skipped {label}: {exc}')
            continue
        circuits.append(c)
        L = c['laps']
        print(f'  {label}: {L[0]["abbr"]} {L[0]["laptime"]} / {L[1]["abbr"]} {L[1]["laptime"]}  '
              f'top {L[0]["top"]:.0f}  full throttle {L[0]["full_throttle"]:.0f}%  '
              f'gears {L[0]["gear_min"]}-{L[0]["gear_max"]}')
    history = export_history()
    out = dict(version=2, circuits=circuits, history=history)
    path = OUT / 'quiz.json'
    path.write_text(json.dumps(out, separators=(',', ':')), encoding='utf-8')
    print(f'wrote {path.name}  {path.stat().st_size/1e3:.0f} KB  ({len(circuits)} circuits, '
          f'{len(history["seasons"])} seasons)')


if __name__ == '__main__':
    what = sys.argv[1:] or ['geometry', 'duels', 'quiz']
    if 'geometry' in what:
        export_geometry()
    if 'duels' in what:
        export_duels()
    if 'quiz' in what:
        export_quiz()
