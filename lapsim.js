/* ============================================================
   F1 DATA ANALYTICS - RUN A LAP
   ------------------------------------------------------------
   Turns the wind tunnel's setup into a lap time at three circuits.

   The model is a direct port of lapsim_core.py and must stay in
   step with it: grip-limited speed at every point of the track,
   a forward pass limited by the friction circle and by power
   against drag, a backward pass limited by braking, and the lap
   time is the integral of ds / v. Grip and power are the values
   the Python side calibrated against real pole laps; they arrive
   with the track data so the two implementations cannot drift.

   The lesson it exists to teach: there is no universally good
   setup. Wing that wins at Monaco loses at Monza.
   ============================================================ */

(function () {
    'use strict';

    const U = window.F1UI;
    if (!U) return;

    const RHO = 1.225, G0 = 9.81, V_MIN = 8.0;
    const DRS_DF = 0.35, DRS_DRAG = 0.30;
    const DRS_RW_SHARE_CZ = 0.2619, DRS_RW_SHARE_CX = 0.325;

    // ------------------------------------------------------------------
    // physics - mirrors lapsim_core.simulate exactly
    // ------------------------------------------------------------------
    function simulate(track, sCz, sCx, mu, power, mass) {
        const kappa = track.kappaAbs, drs = track.drs, ds = track.ds, n = kappa.length;
        const czOpen = sCz * (1 - DRS_RW_SHARE_CZ * (1 - DRS_DF));
        const cxOpen = sCx * (1 - DRS_RW_SHARE_CX * (1 - DRS_DRAG));

        const vLim = new Float64Array(n);
        for (let i = 0; i < n; i++) {
            const cz = drs[i] ? czOpen : sCz;
            const cx = drs[i] ? cxOpen : sCx;
            const denom = kappa[i] - mu * RHO * cz / (2 * mass);
            let v = denom > 1e-9 ? Math.sqrt(mu * G0 / Math.max(denom, 1e-9)) : Infinity;
            const vTop = Math.cbrt(power / (0.5 * RHO * cx));
            v = Math.min(v, vTop);
            vLim[i] = Math.max(v, V_MIN);
        }

        const aLatMax = (v, i) => mu * (G0 + 0.5 * RHO * v * v * (drs[i] ? czOpen : sCz) / mass);
        const spare = (v, i) => {
            const need = v * v * kappa[i];
            const have = aLatMax(v, i);
            const r = Math.min(need / Math.max(have, 1e-9), 1.0);
            return Math.sqrt(Math.max(0, 1 - r * r));
        };
        const dragAcc = (v, i) => 0.5 * RHO * v * v * (drs[i] ? cxOpen : sCx) / mass;

        const vF = Float64Array.from(vLim);
        let v = vLim[0];
        for (let pass = 0; pass < 2; pass++) {
            for (let i = 0; i < n; i++) {
                const j = (i + 1) % n;
                const aTrac = aLatMax(v, i) * spare(v, i);
                const aPow = power / (mass * Math.max(v, V_MIN));
                const a = Math.min(aTrac, aPow) - dragAcc(v, i);
                const vNext = Math.sqrt(Math.max(v * v + 2 * a * ds, V_MIN * V_MIN));
                v = Math.min(vNext, vLim[j]);
                vF[j] = v;
            }
        }

        const vB = Float64Array.from(vF);
        v = vF[n - 1];
        for (let pass = 0; pass < 2; pass++) {
            for (let i = n - 1; i >= 0; i--) {
                const j = (i - 1 + n) % n;
                const aBrk = aLatMax(v, i) * spare(v, i) + dragAcc(v, i);
                const vPrev = Math.sqrt(v * v + 2 * aBrk * ds);
                v = Math.min(vPrev, vF[j]);
                vB[j] = v;
            }
        }

        const out = new Float64Array(n);
        const sectors = [0, 0, 0];
        const third = Math.floor(n / 3);
        let lap = 0;
        for (let i = 0; i < n; i++) {
            out[i] = Math.max(Math.min(vF[i], vB[i]), V_MIN);
            const dt = ds / out[i];
            lap += dt;
            sectors[i < third ? 0 : i < 2 * third ? 1 : 2] += dt;
        }
        return { v: out, lapTime: lap, sectors };
    }

    // ------------------------------------------------------------------
    // state
    // ------------------------------------------------------------------
    const root = document.getElementById('lapSim');
    if (!root) return;

    const S = { data: null, tracks: [], cards: new Map(), optimum: new Map(), setup: null, aero: null };

    function aeroFor(rw, fw, frh, rrh) {
        const A = window.F1Aero;
        if (!A) return null;
        const r = A.solve({ speed: 250, rearWing: rw, frontWing: fw, frontRH: frh, rearRH: rrh, drs: false });
        return { sCz: r.sCz, sCx: r.sCx };
    }

    /** Same search the Python side ran, so the optimum the page shows is
     *  the optimum the model was calibrated on. */
    function findOptimum(track) {
        let best = null;
        for (let rw = 5; rw <= 35; rw += 2) {
            for (const frh of [22, 26, 30, 36, 42]) {
                const a = aeroFor(rw, rw * 0.5, frh, frh + 12);
                if (!a) return null;
                const r = simulate(track, a.sCz, a.sCx, S.data.mu, S.data.power, S.data.mass);
                if (!best || r.lapTime < best.lapTime) {
                    best = { lapTime: r.lapTime, sectors: r.sectors, v: r.v, rw, fw: rw * 0.5, frh, rrh: frh + 12 };
                }
            }
        }
        return best;
    }

    // ------------------------------------------------------------------
    // build
    // ------------------------------------------------------------------
    function build() {
        root.innerHTML = '';
        const head = document.createElement('div');
        head.className = 'ls-head';
        head.innerHTML =
            '<div><span class="ls-kicker">Run a lap</span>' +
            '<h3 class="ls-title">Does your setup actually work?</h3>' +
            '<p class="ls-desc">The tunnel gives you forces. This runs the same setup round three ' +
            'circuits with a lap-time model calibrated to the real pole laps, and tells you how far ' +
            'you are from the fastest configuration at each one.</p></div>' +
            '<div class="ls-legend"><span class="ls-legend-ramp"></span><span>slow</span><span>fast</span></div>';
        root.appendChild(head);

        const grid = document.createElement('div');
        grid.className = 'ls-grid';
        root.appendChild(grid);

        for (const tr of S.tracks) {
            const card = document.createElement('article');
            card.className = 'ls-card';
            card.dataset.key = tr.key;
            card.innerHTML =
                '<div class="ls-card-head"><span class="ls-circuit">' + tr.label + '</span>' +
                '<span class="ls-badge" hidden>OPTIMUM</span></div>' +
                '<div class="ls-well"><canvas class="ls-map" aria-label="' + tr.label +
                ' track map coloured by simulated speed"></canvas></div>' +
                '<div class="ls-readout">' +
                '<div class="ls-lap"><span class="ls-lap-value">--:--.---</span>' +
                '<span class="ls-lap-unit">simulated lap</span></div>' +
                '<div class="ls-delta"><span class="ls-delta-value">--.---</span>' +
                '<span class="ls-delta-unit">vs fastest setup</span></div></div>' +
                '<div class="ls-sectors" aria-label="Sector time lost to the fastest setup">' +
                [1, 2, 3].map(k => '<div class="ls-sector"><span class="ls-sector-name">S' + k +
                    '</span><span class="ls-sector-bar"><span class="ls-sector-fill"></span></span>' +
                    '<span class="ls-sector-value">--</span></div>').join('') + '</div>' +
                '<div class="ls-foot"><span class="ls-pole">Real pole ' + U.fmtLap(tr.pole_time) +
                ' · ' + tr.pole_abbr + '</span>' +
                '<button type="button" class="ls-opt">Fastest setup here <span aria-hidden="true">&rarr;</span></button></div>';
            grid.appendChild(card);
            S.cards.set(tr.key, card);

            card.querySelector('.ls-opt').addEventListener('click', () => applyOptimum(tr.key));
        }

        const insight = document.createElement('p');
        insight.className = 'ls-insight';
        insight.setAttribute('aria-live', 'polite');
        root.appendChild(insight);
    }

    function applyOptimum(key) {
        const opt = S.optimum.get(key);
        if (!opt) return;
        document.dispatchEvent(new CustomEvent('wt:set', {
            detail: { rearWing: opt.rw, frontWing: Math.round(opt.fw), frontRH: opt.frh, rearRH: opt.rrh, drs: false,
                      source: 'lapsim', key }
        }));
        if (typeof window.gtag === 'function') {
            window.gtag('event', 'wind_tunnel_control', { control: 'lapsim_optimum', value_label: key });
        }
    }

    // ------------------------------------------------------------------
    // draw
    // ------------------------------------------------------------------
    function drawMap(tr, card, result) {
        const canvas = card.querySelector('.ls-map');
        if (!canvas || !canvas.getBoundingClientRect().width) return;
        const bb = tr.bbox;
        const ratio = Math.max(1.05, Math.min(1.9, bb.w / bb.h));
        const { g, w, h } = U.setupCanvas(canvas, ratio);
        g.clearRect(0, 0, w, h);

        const pad = 14;
        const sx = (w - 2 * pad) / bb.w, sy = (h - 2 * pad) / bb.h;
        const sc = Math.min(sx, sy);
        const ox = (w - bb.w * sc) / 2, oy = (h - bb.h * sc) / 2;
        const X = i => ox + (tr.x[i] - bb.x0) * sc;
        const Y = i => h - (oy + (tr.y[i] - bb.y0) * sc);

        // underlay
        g.lineCap = 'round'; g.lineJoin = 'round';
        g.strokeStyle = 'rgba(0,0,0,0.55)'; g.lineWidth = 7;
        g.beginPath();
        for (let i = 0; i < tr.n; i++) (i ? g.lineTo(X(i), Y(i)) : g.moveTo(X(i), Y(i)));
        g.closePath(); g.stroke();

        // speed-coloured segments
        const v = result ? result.v : tr.vRealArr;
        const vmax = tr.vScaleMax, vmin = tr.vScaleMin;
        g.lineWidth = 4;
        for (let i = 0; i < tr.n; i++) {
            const j = (i + 1) % tr.n;
            g.strokeStyle = U.speedColor((v[i] - vmin) / (vmax - vmin));
            g.beginPath(); g.moveTo(X(i), Y(i)); g.lineTo(X(j), Y(j)); g.stroke();
        }

        // start / finish
        g.fillStyle = U.palette.ink;
        g.beginPath(); g.arc(X(0), Y(0), 3.2, 0, Math.PI * 2); g.fill();
        g.strokeStyle = U.palette.bg; g.lineWidth = 1.2; g.stroke();
    }

    function render() {
        if (!S.data || !S.setup) return;
        const { sCz, sCx } = S.setup.result;
        const rw = S.setup.params.rearWing;
        const closest = { key: null, gap: Infinity };
        const notes = [];

        for (const tr of S.tracks) {
            const card = S.cards.get(tr.key);
            const r = simulate(tr, sCz, sCx, S.data.mu, S.data.power, S.data.mass);
            const opt = S.optimum.get(tr.key);
            const gap = opt ? r.lapTime - opt.lapTime : NaN;

            card.querySelector('.ls-lap-value').textContent = U.fmtLap(r.lapTime);
            const dv = card.querySelector('.ls-delta-value');
            const badge = card.querySelector('.ls-badge');
            const isOpt = isFinite(gap) && gap < 0.02;
            dv.textContent = isFinite(gap) ? (isOpt ? '0.000' : U.fmtDelta(gap)) + ' s' : '--';
            dv.classList.toggle('good', isOpt);
            badge.hidden = !isOpt;
            card.classList.toggle('optimal', isOpt);

            if (opt) {
                const maxLoss = Math.max(0.15, ...r.sectors.map((s, k) => s - opt.sectors[k]));
                card.querySelectorAll('.ls-sector').forEach((el, k) => {
                    const loss = r.sectors[k] - opt.sectors[k];
                    el.querySelector('.ls-sector-fill').style.width = U.clamp(loss / maxLoss, 0, 1) * 100 + '%';
                    el.querySelector('.ls-sector-value').textContent = U.fmtDelta(loss, 2);
                    el.classList.toggle('bad', loss > 0.05);
                });
                if (gap < closest.gap) { closest.key = tr.key; closest.gap = gap; }
                if (!isOpt) {
                    const dir = rw < opt.rw - 1 ? 'more wing' : rw > opt.rw + 1 ? 'less wing' :
                        (S.setup.params.frontRH > opt.frh + 2 ? 'a lower floor' : 'a higher floor');
                    notes.push({ key: tr.key, label: tr.label, gap, dir });
                }
            }
            drawMap(tr, card, r);
        }

        const insight = root.querySelector('.ls-insight');
        if (!insight || !closest.key) return;
        const c = S.tracks.find(t => t.key === closest.key);
        if (!notes.length) {
            insight.innerHTML = 'This setup is at the optimum everywhere it can be. ' +
                'That never happens in the real world — the circuits want different cars.';
        } else if (notes.length === S.tracks.length) {
            notes.sort((a, b) => a.gap - b.gap);
            insight.innerHTML = 'Closest to ideal at <b>' + c.label + '</b> (' + U.fmtDelta(closest.gap) +
                ' s). ' + notes.slice(1).map(n => '<b>' + n.label + '</b> wants ' + n.dir + ' (' +
                U.fmtDelta(n.gap) + ' s)').join(', ') + '.';
        } else {
            insight.innerHTML = 'Optimal at <b>' + c.label + '</b>. ' +
                notes.map(n => '<b>' + n.label + '</b> wants ' + n.dir + ' (' + U.fmtDelta(n.gap) + ' s)').join(', ') +
                ' — the same car cannot be right at both.';
        }
    }

    // ------------------------------------------------------------------
    // wiring
    // ------------------------------------------------------------------
    function prepareTrack(c) {
        const n = c.x.length;
        const kappaAbs = new Float64Array(n);
        for (let i = 0; i < n; i++) kappaAbs[i] = Math.abs(c.kappa[i]);
        let x0 = Infinity, x1 = -Infinity, y0 = Infinity, y1 = -Infinity;
        for (let i = 0; i < n; i++) {
            if (c.x[i] < x0) x0 = c.x[i]; if (c.x[i] > x1) x1 = c.x[i];
            if (c.y[i] < y0) y0 = c.y[i]; if (c.y[i] > y1) y1 = c.y[i];
        }
        const vReal = Float64Array.from(c.v_real);
        return Object.assign({}, c, {
            n, kappaAbs, drs: c.drs, vRealArr: vReal,
            bbox: { x0, y0, w: x1 - x0, h: y1 - y0 },
            vScaleMin: 12, vScaleMax: 95,
        });
    }

    function onSetup(detail) {
        S.setup = detail;
        render();
    }

    function init() {
        U.loadJSON('data/web/tracks.json').then(data => {
            S.data = data;
            S.tracks = data.circuits.map(prepareTrack);
            build();

            const seed = () => {
                const WT = window.F1WindTunnel;
                if (WT && WT.getState) onSetup(WT.getState());
            };

            // The optimum search is ~240 simulations; keep it off the first paint.
            setTimeout(() => {
                for (const tr of S.tracks) {
                    const opt = findOptimum(tr);
                    if (opt) {
                        S.optimum.set(tr.key, opt);
                        const drift = Math.abs(opt.lapTime - tr.best_time);
                        if (drift > 0.05) {
                            console.warn('[lapsim] ' + tr.label + ' optimum differs from export by ' + drift.toFixed(3) + ' s');
                        }
                    }
                }
                seed();
            }, 60);

            document.addEventListener('wt:change', e => onSetup(e.detail));
            window.addEventListener('resize', () => render(), { passive: true });
            U.whenVisible(root, () => render());
            seed();
        }).catch(err => {
            console.warn('[lapsim] unavailable:', err.message);
            root.hidden = true;
        });
    }

    if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', init);
    else init();
})();
