/* ============================================================
   F1 DATA ANALYTICS - QUALIFYING HEAD-TO-HEAD PLAYER
   ------------------------------------------------------------
   The two fastest laps of a session, locked to one clock, under
   the visitor's control. Scrub through the lap, watch the gap
   build, jump straight to the minisectors where it was won.

   Data comes from data/web/duels.json (exported by
   export_web_data.py from real FastF1 telemetry). Both drivers
   are positioned by TIME, so the quicker one visibly pulls away.
   ============================================================ */

(function () {
    'use strict';

    const U = window.F1UI;
    if (!U) return;

    const players = [];

    // ------------------------------------------------------------------
    // one player per circuit panel
    // ------------------------------------------------------------------
    function createPlayer(host, data) {
        const [A, B] = data.drivers;
        const n = data.dist.length;
        const delta = new Float64Array(n);
        for (let i = 0; i < n; i++) delta[i] = B.t[i] - A.t[i];
        const tMax = Math.max(A.t[n - 1], B.t[n - 1]);

        // bounding box of the map
        let x0 = Infinity, x1 = -Infinity, y0 = Infinity, y1 = -Infinity;
        for (const d of data.drivers) for (let i = 0; i < n; i++) {
            if (d.x[i] < x0) x0 = d.x[i]; if (d.x[i] > x1) x1 = d.x[i];
            if (d.y[i] < y0) y0 = d.y[i]; if (d.y[i] > y1) y1 = d.y[i];
        }
        const bbox = { x0, y0, w: x1 - x0, h: y1 - y0 };

        host.innerHTML =
            '<div class="duel-stage" tabindex="0" aria-label="Interactive head-to-head player. Space plays and pauses, arrow keys step.">' +
            '<canvas class="duel-map"></canvas>' +
            '<div class="duel-gap"><span class="duel-gap-label">Live gap</span>' +
            '<span class="duel-gap-value">+0.000</span><span class="duel-gap-who"></span></div>' +
            '<div class="duel-hint">Drag the timeline, or press play</div>' +
            '</div>' +
            '<div class="duel-tiles">' + data.drivers.map((d, k) =>
                '<div class="duel-tile" style="--drv:' + d.color + '">' +
                '<div class="duel-tile-head"><span class="duel-abbr">' + d.abbr + '</span>' +
                '<span class="duel-team">' + d.team + '</span>' +
                '<span class="duel-time">' + U.fmtLap(d.laptime) + '</span></div>' +
                '<div class="duel-tile-body">' +
                '<div class="duel-stat"><span class="duel-stat-value duel-speed">0</span><span class="duel-stat-unit">km/h</span></div>' +
                '<div class="duel-stat"><span class="duel-stat-value duel-gear">-</span><span class="duel-stat-unit">gear</span></div>' +
                '<div class="duel-pedals"><span class="duel-thr"><span class="duel-thr-fill"></span></span>' +
                '<span class="duel-brk" title="Brake"></span></div>' +
                '</div></div>').join('') + '</div>' +
            '<canvas class="duel-traces" aria-label="Cumulative delta and speed traces"></canvas>' +
            '<div class="duel-controls">' +
            '<button type="button" class="duel-play" aria-label="Play"><span class="duel-play-icon"></span></button>' +
            '<input type="range" class="duel-scrub" min="0" max="1000" value="0" step="1" aria-label="Lap timeline">' +
            '<span class="duel-clock"><span class="duel-clock-now">0:00.000</span><span class="duel-clock-sep">/</span>' +
            '<span class="duel-clock-end">' + U.fmtLap(tMax) + '</span></span>' +
            '<div class="duel-rate" role="group" aria-label="Playback speed">' +
            '<button type="button" data-rate="0.5">&frac12;&times;</button>' +
            '<button type="button" data-rate="1" class="active">1&times;</button>' +
            '<button type="button" data-rate="2">2&times;</button></div>' +
            '</div>' +
            '<div class="duel-swings"><span class="duel-swings-label">Where it was won</span>' +
            data.swings.slice(0, 5).map(s => {
                const w = data.drivers[s.winner];
                return '<button type="button" class="duel-chip" data-i="' + s.i + '" style="--drv:' + w.color + '">' +
                    '<span class="duel-chip-ms">MS ' + (s.i + 1) + '</span><span class="duel-chip-who">' + w.abbr +
                    '</span><span class="duel-chip-val">+' + s.margin.toFixed(3) + '</span></button>';
            }).join('') + '</div>';

        const el = {
            stage: host.querySelector('.duel-stage'),
            map: host.querySelector('.duel-map'),
            traces: host.querySelector('.duel-traces'),
            gapValue: host.querySelector('.duel-gap-value'),
            gapWho: host.querySelector('.duel-gap-who'),
            hint: host.querySelector('.duel-hint'),
            play: host.querySelector('.duel-play'),
            scrub: host.querySelector('.duel-scrub'),
            clockNow: host.querySelector('.duel-clock-now'),
            tiles: Array.from(host.querySelectorAll('.duel-tile')),
            rates: Array.from(host.querySelectorAll('.duel-rate button')),
            chips: Array.from(host.querySelectorAll('.duel-chip')),
        };

        const P = {
            data, A, B, n, delta, tMax, bbox, el,
            t: 0, rate: 1, playing: false, raf: 0, last: 0,
            highlight: null,                 // { i, until }
            mapStatic: null, tracesStatic: null,
            mapGeom: null, tracesGeom: null,
            touched: false,
        };

        // ---- geometry helpers ---------------------------------------------
        function mapGeom(w, h) {
            const pad = 26;
            const sc = Math.min((w - 2 * pad) / bbox.w, (h - 2 * pad) / bbox.h);
            const ox = (w - bbox.w * sc) / 2, oy = (h - bbox.h * sc) / 2;
            return {
                X: x => ox + (x - bbox.x0) * sc,
                Y: y => h - (oy + (y - bbox.y0) * sc),
                w, h,
            };
        }

        // ---- static map: dominance-coloured track ------------------------
        function buildMapStatic(w, h) {
            const dpr = Math.min(window.devicePixelRatio || 1, 2);
            const off = document.createElement('canvas');
            off.width = Math.round(w * dpr); off.height = Math.round(h * dpr);
            const g = off.getContext('2d');
            g.setTransform(dpr, 0, 0, dpr, 0, 0);
            const G = mapGeom(w, h);
            const ms = data.minisectors;

            g.lineCap = 'round'; g.lineJoin = 'round';
            g.strokeStyle = 'rgba(0,0,0,0.6)'; g.lineWidth = 13;
            g.beginPath();
            for (let i = 0; i < n; i++) (i ? g.lineTo(G.X(A.x[i]), G.Y(A.y[i])) : g.moveTo(G.X(A.x[i]), G.Y(A.y[i])));
            g.stroke();

            g.lineWidth = 8;
            let k = 0;
            for (let i = 0; i < n - 1; i++) {
                while (k < ms.edges.length - 2 && data.dist[i] >= ms.edges[k + 1]) k++;
                g.strokeStyle = data.drivers[ms.winners[k]].color;
                g.beginPath();
                g.moveTo(G.X(A.x[i]), G.Y(A.y[i]));
                g.lineTo(G.X(A.x[i + 1]), G.Y(A.y[i + 1]));
                g.stroke();
            }

            // corners
            for (const c of data.corners) {
                const cx = G.X(c.x), cy = G.Y(c.y);
                g.fillStyle = 'rgba(10,10,15,0.9)';
                g.beginPath(); g.arc(cx, cy, 8.5, 0, Math.PI * 2); g.fill();
                g.strokeStyle = 'rgba(255,255,255,0.25)'; g.lineWidth = 1; g.stroke();
                U.label(g, String(c.n), cx, cy + 0.5, { size: 8.5, color: U.palette.ink2, align: 'center', baseline: 'middle', weight: '600' });
            }
            P.mapStatic = off;
            P.mapGeom = G;
        }

        // ---- static traces: delta + speed --------------------------------
        function buildTracesStatic(w, h) {
            const dpr = Math.min(window.devicePixelRatio || 1, 2);
            const off = document.createElement('canvas');
            off.width = Math.round(w * dpr); off.height = Math.round(h * dpr);
            const g = off.getContext('2d');
            g.setTransform(dpr, 0, 0, dpr, 0, 0);

            const L = 44, R = 12, T = 22, gapRows = 18;
            const rowDelta = { y: T, h: (h - T - gapRows - 22) * 0.44 };
            const rowSpeed = { y: rowDelta.y + rowDelta.h + gapRows, h: (h - T - gapRows - 22) * 0.56 };
            const plotW = w - L - R;
            const X = i => L + (data.dist[i] / data.length) * plotW;

            let dmax = 0;
            for (let i = 0; i < n; i++) dmax = Math.max(dmax, Math.abs(delta[i]));
            dmax = Math.max(0.12, dmax * 1.18);
            const Yd = v => rowDelta.y + rowDelta.h / 2 - (v / dmax) * (rowDelta.h / 2);

            let smax = 0;
            for (const d of data.drivers) for (let i = 0; i < n; i++) smax = Math.max(smax, d.speed[i]);
            const smin = 0;
            const Ys = v => rowSpeed.y + rowSpeed.h - ((v - smin) / (smax * 1.04 - smin)) * rowSpeed.h;

            // panel wells
            for (const row of [rowDelta, rowSpeed]) {
                g.fillStyle = 'rgba(255,255,255,0.025)';
                g.fillRect(L, row.y, plotW, row.h);
                g.strokeStyle = U.palette.line; g.lineWidth = 1;
                g.strokeRect(L + 0.5, row.y + 0.5, plotW - 1, row.h - 1);
            }
            // labels
            U.label(g, 'CUMULATIVE DELTA · ' + B.abbr + ' minus ' + A.abbr + ' (s)', L, T - 8, { size: 9, color: U.palette.ink3, weight: '600' });
            U.label(g, 'SPEED (km/h)', L, rowSpeed.y - 6, { size: 9, color: U.palette.ink3, weight: '600' });
            U.label(g, A.abbr + ' ahead', L + 6, rowDelta.y + 11, { size: 9, color: A.color, weight: '700' });
            U.label(g, B.abbr + ' ahead', L + 6, rowDelta.y + rowDelta.h - 5, { size: 9, color: B.color, weight: '700' });
            U.label(g, '+' + dmax.toFixed(2), L - 6, rowDelta.y + 10, { size: 8.5, color: U.palette.ink3, align: 'right' });
            U.label(g, '−' + dmax.toFixed(2), L - 6, rowDelta.y + rowDelta.h - 2, { size: 8.5, color: U.palette.ink3, align: 'right' });
            U.label(g, String(Math.round(smax)), L - 6, rowSpeed.y + 10, { size: 8.5, color: U.palette.ink3, align: 'right' });
            U.label(g, '0', L - 6, rowSpeed.y + rowSpeed.h - 2, { size: 8.5, color: U.palette.ink3, align: 'right' });

            // minisector ticks along the bottom
            g.strokeStyle = 'rgba(255,255,255,0.07)';
            for (const e of data.minisectors.edges) {
                const x = L + (e / data.length) * plotW;
                g.beginPath(); g.moveTo(x, rowSpeed.y); g.lineTo(x, rowSpeed.y + rowSpeed.h); g.stroke();
            }

            // delta fill + line
            const y0 = Yd(0);
            g.strokeStyle = 'rgba(255,255,255,0.28)'; g.lineWidth = 1;
            g.beginPath(); g.moveTo(L, y0); g.lineTo(L + plotW, y0); g.stroke();
            for (const sign of [1, -1]) {
                g.fillStyle = U.hexToRgba(sign > 0 ? A.color : B.color, 0.28);
                g.beginPath(); g.moveTo(X(0), y0);
                for (let i = 0; i < n; i++) g.lineTo(X(i), Yd(sign > 0 ? Math.max(delta[i], 0) : Math.min(delta[i], 0)));
                g.lineTo(X(n - 1), y0); g.closePath(); g.fill();
            }
            g.strokeStyle = U.palette.ink; g.lineWidth = 1.6;
            g.beginPath();
            for (let i = 0; i < n; i++) (i ? g.lineTo(X(i), Yd(delta[i])) : g.moveTo(X(i), Yd(delta[i])));
            g.stroke();

            // speed lines
            for (const d of data.drivers) {
                g.strokeStyle = d.color; g.lineWidth = 1.6;
                g.beginPath();
                for (let i = 0; i < n; i++) (i ? g.lineTo(X(i), Ys(d.speed[i])) : g.moveTo(X(i), Ys(d.speed[i])));
                g.stroke();
            }

            P.tracesStatic = off;
            P.tracesGeom = { L, plotW, top: rowDelta.y, bottom: rowSpeed.y + rowSpeed.h, X, Ys, Yd, rowDelta, rowSpeed };
        }

        // ---- per-frame --------------------------------------------------
        function stateAt(t) {
            const out = [];
            for (const d of data.drivers) {
                const i = U.indexAt(t, d.t);
                const j = Math.min(i + 1, n - 1);
                const f = d.t[j] > d.t[i] ? U.clamp((t - d.t[i]) / (d.t[j] - d.t[i]), 0, 1) : 0;
                out.push({
                    x: U.lerp(d.x[i], d.x[j], f), y: U.lerp(d.y[i], d.y[j], f),
                    speed: U.lerp(d.speed[i], d.speed[j], f), gear: d.gear[i],
                    throttle: d.throttle[i], brake: d.brake[i],
                    dist: U.lerp(data.dist[i], data.dist[j], f), i,
                });
            }
            return out;
        }

        function draw() {
            // Self-heal: if the static layers were never built (the visibility
            // hook fired while the panel had no box), build them now that it has.
            if (!P.mapStatic || !P.tracesStatic) {
                if (el.map.getBoundingClientRect().width) { resize(); return; }
                return;
            }
            const s = stateAt(P.t);
            const lead = s[0].dist;          // playhead follows driver A
            const gap = U.interp(lead, data.dist, delta);

            // map
            {
                const { g, w, h } = U.setupCanvasKeep(el.map);
                g.clearRect(0, 0, w, h);
                g.drawImage(P.mapStatic, 0, 0, w, h);
                const G = P.mapGeom;
                if (P.highlight && performance.now() < P.highlight.until) {
                    const ms = data.minisectors;
                    const i0 = U.indexAt(ms.edges[P.highlight.i], data.dist);
                    const i1 = U.indexAt(ms.edges[P.highlight.i + 1], data.dist);
                    g.strokeStyle = 'rgba(255,255,255,0.9)'; g.lineWidth = 12; g.lineCap = 'round';
                    g.beginPath();
                    for (let i = i0; i <= i1; i++) (i === i0 ? g.moveTo(G.X(A.x[i]), G.Y(A.y[i])) : g.lineTo(G.X(A.x[i]), G.Y(A.y[i])));
                    g.stroke();
                } else if (P.highlight) P.highlight = null;

                // trailing lead is drawn on top
                const order = gap >= 0 ? [1, 0] : [0, 1];
                for (const k of order) {
                    const d = data.drivers[k], st = s[k];
                    const cx = G.X(st.x), cy = G.Y(st.y);
                    g.beginPath(); g.arc(cx, cy, 11, 0, Math.PI * 2);
                    g.fillStyle = 'rgba(10,10,15,0.85)'; g.fill();
                    g.beginPath(); g.arc(cx, cy, 7.5, 0, Math.PI * 2);
                    g.fillStyle = d.color; g.fill();
                    g.strokeStyle = '#fff'; g.lineWidth = 1.8; g.stroke();
                }
            }

            // traces
            {
                const { g, w, h } = U.setupCanvasKeep(el.traces);
                g.clearRect(0, 0, w, h);
                g.drawImage(P.tracesStatic, 0, 0, w, h);
                const T = P.tracesGeom;
                const x = T.L + (lead / data.length) * T.plotW;
                g.strokeStyle = 'rgba(255,255,255,0.85)'; g.lineWidth = 1.2;
                g.beginPath(); g.moveTo(x, T.top); g.lineTo(x, T.bottom); g.stroke();
                // live dots on the speed lines
                for (let k = 0; k < 2; k++) {
                    const d = data.drivers[k];
                    const v = U.interp(lead, data.dist, d.speed);
                    g.beginPath(); g.arc(x, T.Ys(v), 3.6, 0, Math.PI * 2);
                    g.fillStyle = d.color; g.fill(); g.strokeStyle = U.palette.bg; g.lineWidth = 1.2; g.stroke();
                }
                g.beginPath(); g.arc(x, T.Yd(gap), 3.6, 0, Math.PI * 2);
                g.fillStyle = U.palette.ink; g.fill();
            }

            // readouts
            el.gapValue.textContent = U.fmtDelta(gap) + ' s';
            el.gapValue.style.color = gap >= 0 ? A.color : B.color;
            el.gapWho.textContent = gap >= 0 ? B.abbr + ' behind' : A.abbr + ' behind';
            el.clockNow.textContent = U.fmtLap(P.t);
            el.scrub.value = String(Math.round(P.t / tMax * 1000));
            el.tiles.forEach((tile, k) => {
                const st = s[k];
                tile.querySelector('.duel-speed').textContent = Math.round(st.speed);
                tile.querySelector('.duel-gear').textContent = st.gear;
                tile.querySelector('.duel-thr-fill').style.width = U.clamp(st.throttle, 0, 100) + '%';
                tile.querySelector('.duel-brk').classList.toggle('on', !!st.brake);
            });
        }

        // ---- transport ---------------------------------------------------
        function setTime(t, fromUser) {
            P.t = U.clamp(t, 0, tMax);
            if (fromUser && !P.touched) { P.touched = true; el.hint.classList.add('gone'); track('scrub'); }
            draw();
        }
        function tick(now) {
            if (!P.playing) return;
            const dt = Math.min((now - P.last) / 1000, 0.1);
            P.last = now;
            let t = P.t + dt * P.rate;
            if (t >= tMax) t = 0;          // loop
            setTime(t, false);
            P.raf = requestAnimationFrame(tick);
        }
        function play() {
            if (P.playing) return;
            P.playing = true; P.last = performance.now();
            el.play.classList.add('playing'); el.play.setAttribute('aria-label', 'Pause');
            el.hint.classList.add('gone');
            P.raf = requestAnimationFrame(tick);
            track('play');
        }
        function pause() {
            P.playing = false; cancelAnimationFrame(P.raf);
            el.play.classList.remove('playing'); el.play.setAttribute('aria-label', 'Play');
        }
        function toggle() { P.playing ? pause() : play(); }

        function track(action) {
            if (typeof window.gtag !== 'function') return;
            window.gtag('event', 'duel_control', { control: action, value_label: data.key });
        }

        el.play.addEventListener('click', toggle);
        el.scrub.addEventListener('input', () => { pause(); setTime(parseFloat(el.scrub.value) / 1000 * tMax, true); });
        el.rates.forEach(b => b.addEventListener('click', () => {
            P.rate = parseFloat(b.dataset.rate);
            el.rates.forEach(x => x.classList.toggle('active', x === b));
        }));
        el.chips.forEach(chip => chip.addEventListener('click', () => {
            const i = parseInt(chip.dataset.i, 10);
            const d0 = data.minisectors.edges[i];
            pause();
            P.highlight = { i, until: performance.now() + 1600 };
            setTime(U.interp(d0, data.dist, A.t), true);
            track('swing');
            // keep the highlight visible even where rAF is unavailable
            setTimeout(draw, 1700);
        }));
        el.stage.addEventListener('keydown', e => {
            if (e.key === ' ') { e.preventDefault(); toggle(); }
            else if (e.key === 'ArrowRight') { e.preventDefault(); pause(); setTime(P.t + 0.5, true); }
            else if (e.key === 'ArrowLeft') { e.preventDefault(); pause(); setTime(P.t - 0.5, true); }
        });
        el.stage.addEventListener('click', e => { if (e.target === el.map) toggle(); });

        function resize() {
            const mw = el.map.getBoundingClientRect().width;
            if (!mw) return;
            const ratio = U.clamp(bbox.w / bbox.h, 1.15, 2.1);
            const m = U.setupCanvas(el.map, ratio);
            buildMapStatic(m.w, m.h);
            const t = U.setupCanvas(el.traces, 2.7);
            buildTracesStatic(t.w, t.h);
            if (P.mapStatic && P.tracesStatic) draw();
        }

        P.resize = resize; P.pause = pause; P.draw = draw; P.setTime = setTime; P.host = host;
        players.push(P);
        return P;
    }

    // Size a canvas without resetting its backing store (the static layers
    // were built for the current size; a resize rebuilds them).
    U.setupCanvasKeep = function (canvas) {
        const dpr = Math.min(window.devicePixelRatio || 1, 2);
        const w = canvas.width / dpr, h = canvas.height / dpr;
        const g = canvas.getContext('2d');
        g.setTransform(dpr, 0, 0, dpr, 0, 0);
        return { g, w, h };
    };

    // ------------------------------------------------------------------
    // init
    // ------------------------------------------------------------------
    function init() {
        const hosts = Array.from(document.querySelectorAll('.duel[data-key]'));
        if (!hosts.length) return;

        U.loadJSON('data/web/duels.json').then(list => {
            const byKey = new Map(list.map(d => [d.key, d]));
            for (const host of hosts) {
                const data = byKey.get(host.dataset.key);
                if (!data) { host.hidden = true; continue; }
                const P = createPlayer(host, data);
                U.whenVisible(host, () => P.resize());
            }

            // tab panels: (re)lay out when a panel is revealed, pause the rest
            document.querySelectorAll('.sim-tab').forEach(tab => {
                tab.addEventListener('click', () => setTimeout(() => {
                    for (const P of players) {
                        const hidden = !!P.host.closest('[hidden]');
                        if (hidden) P.pause(); else P.resize();
                    }
                }, 0));
            });

            let rt = 0;
            window.addEventListener('resize', () => {
                clearTimeout(rt);
                rt = setTimeout(() => players.forEach(P => { if (!P.host.closest('[hidden]')) P.resize(); }), 120);
            }, { passive: true });

            document.addEventListener('visibilitychange', () => { if (document.hidden) players.forEach(P => P.pause()); });
        }).catch(err => {
            console.warn('[duel] unavailable:', err.message);
            hosts.forEach(h => { h.hidden = true; });
        });
    }

    if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', init);
    else init();
})();
