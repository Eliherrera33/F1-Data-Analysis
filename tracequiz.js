/* ============================================================
   F1 DATA ANALYTICS - READ THE RACE
   ------------------------------------------------------------
   A ten-question set generated fresh every run from the data on
   this page: real qualifying telemetry, the wind-tunnel force
   model, the lap-time model's setup optima, and thirteen seasons
   of results, qualifying and pit-stop records.

   Five categories, three difficulty levels. Every question is
   drawn from a generator that picks its own circuit, season,
   speed or setting, so the same set does not come round twice.
   Every answer opens with an evidence figure - a track map, a
   force curve, a delta trace - so the visitor sees *why*, not
   just whether.
   ============================================================ */

(function () {
    'use strict';

    const U = window.F1UI;
    const root = document.getElementById('traceQuiz');
    if (!U || !root) return;

    const COUNT = 10;
    const CAR_MASS = 798;

    const CATS = {
        telemetry: { label: 'Telemetry', color: '#00d2be', blurb: 'reading the traces' },
        aero: { label: 'Aerodynamics', color: '#7aa2ff', blurb: 'wings, floors and drag' },
        strategy: { label: 'Race craft', color: '#ff8700', blurb: 'pit stops and grids' },
        history: { label: 'History', color: '#c58bff', blurb: 'seasons and champions' },
        basics: { label: 'Fundamentals', color: '#ff7a70', blurb: 'how the sport works' },
    };
    const LEVELS = { 1: { label: 'Rookie', pts: 1 }, 2: { label: 'Engineer', pts: 2 }, 3: { label: 'Pit wall', pts: 3 } };

    const D = { circuits: [], history: null, tracks: null };
    let questions = [], index = 0, score = 0, maxScore = 0, answered = false;
    let evidenceQ = null;

    // ------------------------------------------------------------------
    // small helpers
    // ------------------------------------------------------------------
    const rnd = n => Math.floor(Math.random() * n);
    const choice = arr => arr[rnd(arr.length)];
    const shuffle = a => { for (let i = a.length - 1; i > 0; i--) { const j = rnd(i + 1); [a[i], a[j]] = [a[j], a[i]]; } return a; };
    const pick = (arr, k) => shuffle(arr.slice()).slice(0, k);
    const fmt = (n, d) => Number(n).toLocaleString('en-GB', { maximumFractionDigits: d === undefined ? 0 : d, minimumFractionDigits: d === undefined ? 0 : d });
    const esc = s => String(s).replace(/[&<>]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;' }[c]));
    const km = m => (m / 1000).toFixed(2) + ' km';
    const minIdx = arr => { let m = 0; for (let i = 1; i < arr.length; i++) if (arr[i] < arr[m]) m = i; return m; };
    const maxIdx = arr => { let m = 0; for (let i = 1; i < arr.length; i++) if (arr[i] > arr[m]) m = i; return m; };
    const mean = arr => arr.reduce((a, b) => a + b, 0) / arr.length;
    const others = (arr, keyFn, exclude, k) => pick(arr.filter(o => keyFn(o) !== exclude), k);

    /** Distractor numbers around a true value: one well below, one well above. */
    function numericOptions(v, unit, digits, spread) {
        const s = spread || 0.5;
        const lo = v * (1 - s), hi = v * (1 + s);
        return shuffle([v, lo, hi]).map(x => fmt(x, digits) + unit);
    }

    /** Per-sample time from speed against lap fraction. Lets a speed trace be
     *  turned into a cumulative delta, which is how engineers actually read a
     *  two-lap comparison. */
    function deltaTrace(length, fast, slow) {
        const n = fast.length, ds = length / (n - 1), out = new Array(n);
        let acc = 0;
        for (let i = 0; i < n; i++) {
            acc += ds * (1 / Math.max(slow[i], 5) - 1 / Math.max(fast[i], 5)) * 3.6;
            out[i] = acc;
        }
        return out;
    }

    // ------------------------------------------------------------------
    // chart kit - every figure in the quiz is one of these
    // ------------------------------------------------------------------
    const P = U.palette;

    function frame(g, L, T, pw, ph) {
        g.fillStyle = 'rgba(255,255,255,0.025)';
        g.fillRect(L, T, pw, ph);
    }

    /** Line series against lap fraction (0..1). */
    function chartTrace(g, w, h, o) {
        const L = 42, R = 14, T = 24, Bm = 26, pw = w - L - R, ph = h - T - Bm;
        const n = o.series[0].y.length;
        let ymax = o.ymax, ymin = o.ymin === undefined ? 0 : o.ymin;
        if (ymax === undefined) { ymax = -Infinity; for (const s of o.series) for (const v of s.y) ymax = Math.max(ymax, v); ymax = Math.ceil(ymax / 50) * 50 + 10; }
        if (o.ymin === undefined && o.allowNegative) { ymin = Infinity; for (const s of o.series) for (const v of s.y) ymin = Math.min(ymin, v); ymin = Math.min(0, ymin); }
        const X = i => L + (i / (n - 1)) * pw;
        const Y = v => T + ph - ((v - ymin) / (ymax - ymin || 1)) * ph;
        frame(g, L, T, pw, ph);
        g.strokeStyle = P.line; g.lineWidth = 1;
        const step = o.ystep || 50;
        for (let v = Math.ceil(ymin / step) * step; v <= ymax; v += step) {
            g.beginPath(); g.moveTo(L, Y(v)); g.lineTo(L + pw, Y(v)); g.stroke();
            U.label(g, o.yfmt ? o.yfmt(v) : String(v), L - 6, Y(v) + 3.5, { size: 8.5, color: P.ink3, align: 'right' });
        }
        for (let p = 0; p <= 100; p += 25) {
            const x = L + (p / 100) * pw;
            g.beginPath(); g.moveTo(x, T); g.lineTo(x, T + ph); g.stroke();
            if (p < 100) U.label(g, p + '%', x, h - 8, { size: 8.5, color: P.ink3, align: 'center' });
        }
        if (ymin < 0) { g.strokeStyle = P.lineStrong; g.beginPath(); g.moveTo(L, Y(0)); g.lineTo(L + pw, Y(0)); g.stroke(); }
        U.label(g, o.ylabel || 'km/h', 6, T - 9, { size: 8.5, color: P.ink3 });
        if (pw > 300) U.label(g, o.xlabel || 'lap distance →', L + pw, h - 8, { size: 8.5, color: P.ink3, align: 'right' });

        // bands (shaded index ranges)
        for (const b of o.bands || []) {
            g.fillStyle = b.color;
            g.fillRect(X(b.i0), T, X(b.i1) - X(b.i0), ph);
        }
        // fill between two series (index a above b)
        if (o.fillBetween) {
            const [a, b] = o.fillBetween.pair;
            g.fillStyle = o.fillBetween.color;
            g.beginPath();
            o.series[a].y.forEach((v, i) => (i ? g.lineTo(X(i), Y(Math.max(v, o.series[b].y[i]))) : g.moveTo(X(i), Y(Math.max(v, o.series[b].y[i])))));
            for (let i = n - 1; i >= 0; i--) g.lineTo(X(i), Y(o.series[b].y[i]));
            g.closePath(); g.fill();
        }
        for (const s of o.series) {
            g.save();
            if (s.area) {
                g.fillStyle = U.hexToRgba(s.color, 0.18);
                g.beginPath(); g.moveTo(X(0), Y(0));
                s.y.forEach((v, i) => g.lineTo(X(i), Y(v)));
                g.lineTo(X(n - 1), Y(0)); g.closePath(); g.fill();
            }
            g.shadowColor = U.hexToRgba(s.color, 0.45); g.shadowBlur = s.thin ? 0 : 8;
            g.strokeStyle = s.color; g.lineWidth = s.thin ? 1.2 : 2; g.lineJoin = 'round';
            if (s.dash) g.setLineDash(s.dash);
            g.beginPath();
            s.y.forEach((v, i) => (i ? g.lineTo(X(i), Y(v)) : g.moveTo(X(i), Y(v))));
            g.stroke();
            g.restore();
        }
        // markers: labelled vertical ticks
        for (const m of o.marks || []) {
            const x = X(m.i), y = Y(o.series[m.s || 0].y[m.i]);
            g.save();
            g.strokeStyle = m.color || P.ink; g.lineWidth = 1; g.setLineDash([3, 3]);
            g.beginPath(); g.moveTo(x, T); g.lineTo(x, T + ph); g.stroke();
            g.setLineDash([]);
            g.fillStyle = m.color || P.ink;
            g.beginPath(); g.arc(x, y, 4, 0, Math.PI * 2); g.fill();
            if (m.label) {
                const bw = 20;
                g.fillStyle = m.color || P.ink; g.beginPath();
                g.roundRect ? g.roundRect(x - bw / 2, T + 4, bw, 16, 4) : g.rect(x - bw / 2, T + 4, bw, 16);
                g.fill();
                U.label(g, m.label, x, T + 16, { size: 10, weight: '700', color: '#0a0a0f', align: 'center' });
            }
            g.restore();
        }
        return { X, Y, L, T, pw, ph };
    }

    /** Track map coloured by speed. */
    function chartMap(g, w, h, o) {
        const m = o.map, n = m.x.length;
        const pad = 22, side = o.stats ? Math.min(150, w * 0.32) : 0;
        const aw = w - pad * 2 - side, ah = h - pad * 2;
        const s = Math.min(aw, ah) / 2 * 0.92;
        const cx = pad + aw / 2, cy = h / 2;
        const X = i => cx + m.x[i] * s, Y = i => cy - m.y[i] * s;
        const vmin = Math.min.apply(null, m.speed), vmax = Math.max.apply(null, m.speed);
        // underlay
        g.save();
        g.lineCap = 'round'; g.lineJoin = 'round';
        g.strokeStyle = 'rgba(255,255,255,0.06)'; g.lineWidth = 11;
        g.beginPath(); for (let i = 0; i < n; i++) (i ? g.lineTo(X(i), Y(i)) : g.moveTo(X(i), Y(i))); g.closePath(); g.stroke();
        for (let i = 1; i < n; i++) {
            const t = (m.speed[i] - vmin) / (vmax - vmin || 1);
            g.strokeStyle = o.mono ? o.mono : U.speedColor(t);
            g.lineWidth = 4.5;
            g.beginPath(); g.moveTo(X(i - 1), Y(i - 1)); g.lineTo(X(i), Y(i)); g.stroke();
        }
        // highlighted range
        if (o.highlight) {
            const [i0, i1] = o.highlight;
            g.strokeStyle = o.highlightColor || P.red; g.lineWidth = 8; g.shadowColor = o.highlightColor || P.red; g.shadowBlur = 12;
            g.beginPath();
            for (let i = i0; i <= i1; i++) (i === i0 ? g.moveTo(X(i), Y(i)) : g.lineTo(X(i), Y(i)));
            g.stroke();
            g.shadowBlur = 0;
        }
        // start/finish
        g.fillStyle = P.ink; g.beginPath(); g.arc(X(0), Y(0), 3.5, 0, Math.PI * 2); g.fill();
        // markers
        for (const mk of o.marks || []) {
            const x = X(mk.i), y = Y(mk.i);
            g.fillStyle = mk.color || P.ink;
            g.beginPath(); g.arc(x, y, 9, 0, Math.PI * 2); g.fill();
            g.strokeStyle = '#0a0a0f'; g.lineWidth = 2; g.stroke();
            if (mk.label) U.label(g, mk.label, x, y + 4, { size: 10, weight: '800', color: '#0a0a0f', align: 'center' });
        }
        g.restore();
        if (o.title) U.label(g, o.title, pad, 16, { size: 11, weight: '700', color: P.ink });
        if (o.sub) U.label(g, o.sub, pad, 30, { size: 9, color: P.ink3 });
        if (o.stats) {
            const x0 = w - side - 4;
            const step = Math.min(36, (h - 34) / o.stats.length), big = step < 30 ? 11 : 13;
            let y = 22;
            for (const st of o.stats) {
                U.label(g, st[0], x0, y, { size: 8.5, color: P.ink3 });
                U.label(g, st[1], x0, y + big + 2, { size: big, weight: '700', color: st[2] || P.ink, family: 'Orbitron, Inter, sans-serif' });
                y += step;
            }
        }
        if (!o.mono && !o.noRamp && w > 420) {
            // ramp
            const rw = 70, rx = w - pad - rw, ry = h - 14;
            for (let i = 0; i < rw; i++) { g.fillStyle = U.speedColor(i / rw); g.fillRect(rx + i, ry, 1.5, 4); }
            U.label(g, 'slow', rx - 4, ry + 4, { size: 8, color: P.ink3, align: 'right' });
            U.label(g, 'fast', rx + rw + 4, ry + 4, { size: 8, color: P.ink3 });
        }
        return { X, Y };
    }

    /** Small multiples of track maps. */
    function chartMaps(g, w, h, o) {
        const k = o.items.length, cw = w / k;
        o.items.forEach((it, j) => {
            g.save(); g.translate(j * cw, 0);
            chartMap(g, cw, h - 20, { map: it.map, mono: it.mono, highlight: it.highlight, highlightColor: it.highlightColor, noRamp: true });
            U.label(g, it.label, cw / 2, h - 22, { size: 10.5, weight: '700', color: it.color || P.ink, align: 'center' });
            if (it.sub) U.label(g, it.sub, cw / 2, h - 8, { size: 9, color: P.ink3, align: 'center' });
            g.restore();
        });
    }

    /** Horizontal bars. items: {label, value, color, hl, note} */
    function chartBars(g, w, h, o) {
        const items = o.items, k = items.length;
        const L = o.labelW || 110, R = 60, T = o.title ? 26 : 12, pw = w - L - R;
        const rowH = Math.min(38, (h - T - 8) / k), bh = Math.min(20, rowH * 0.62);
        const vmax = o.max || Math.max.apply(null, items.map(i => i.value)) * 1.05;
        const vmin = o.min || 0;
        if (o.title) U.label(g, o.title, 8, 16, { size: 10.5, weight: '700', color: P.ink });
        items.forEach((it, i) => {
            const y = T + i * rowH + (rowH - bh) / 2;
            const bw = Math.max(2, ((it.value - vmin) / (vmax - vmin || 1)) * pw);
            g.fillStyle = 'rgba(255,255,255,0.05)'; g.fillRect(L, y, pw, bh);
            g.fillStyle = it.hl ? (it.color || P.red) : U.hexToRgba(it.color || '#868f9e', it.dim ? 0.35 : 0.75);
            g.fillRect(L, y, bw, bh);
            U.label(g, it.label, L - 8, y + bh / 2 + 3.5, { size: 9.5, weight: it.hl ? '700' : '500', color: it.hl ? P.ink : P.ink2, align: 'right' });
            let text = it.text !== undefined ? it.text : fmt(it.value, o.digits);
            g.font = (it.hl ? '700' : '500') + ' 9.5px Inter, system-ui, sans-serif';
            const room = w - 6 - (L + bw + 6);
            if (g.measureText(text).width > room) {
                // too long to sit after the bar: shorten with an ellipsis
                while (text.length > 4 && g.measureText(text + '…').width > room) text = text.slice(0, -1);
                text = text.replace(/[\s,·]+$/, '') + '…';
            }
            U.label(g, text, L + bw + 6, y + bh / 2 + 3.5, { size: 9.5, weight: it.hl ? '700' : '500', color: it.hl ? P.ink : P.ink3 });
        });
    }

    /** x-y curves with a numeric x axis. series: {xs, ys, color, name, dash} */
    function chartCurve(g, w, h, o) {
        const L = 48, R = 14, T = 24, Bm = 28, pw = w - L - R, ph = h - T - Bm;
        let xmin = Infinity, xmax = -Infinity, ymin = 0, ymax = -Infinity;
        for (const s of o.series) { for (const x of s.xs) { xmin = Math.min(xmin, x); xmax = Math.max(xmax, x); } for (const y of s.ys) { ymax = Math.max(ymax, y); ymin = Math.min(ymin, y); } }
        if (o.xmin !== undefined) xmin = o.xmin; if (o.xmax !== undefined) xmax = o.xmax;
        if (o.ymax !== undefined) ymax = o.ymax; if (o.ymin !== undefined) ymin = o.ymin;
        for (const hl of o.hlines || []) ymax = Math.max(ymax, hl.y * 1.05);
        const nice = (span, target) => { const raw = span / target, mag = Math.pow(10, Math.floor(Math.log10(raw))); return [1, 2, 2.5, 5, 10].map(m => m * mag).find(st => span / st <= target + 0.5) || mag * 10; };
        const ystep = o.ystep || nice(ymax - ymin, 4);
        if (o.ymax === undefined) ymax = Math.ceil(ymax / ystep) * ystep;
        if (o.ymin === undefined && ymin !== 0) ymin = Math.floor(ymin / ystep) * ystep;
        const X = x => L + ((x - xmin) / (xmax - xmin || 1)) * pw;
        const Y = y => T + ph - ((y - ymin) / (ymax - ymin || 1)) * ph;
        frame(g, L, T, pw, ph);
        g.strokeStyle = P.line; g.lineWidth = 1;
        for (let v = Math.ceil(ymin / ystep) * ystep; v <= ymax + 1e-9; v += ystep) {
            const y = Y(v);
            g.beginPath(); g.moveTo(L, y); g.lineTo(L + pw, y); g.stroke();
            U.label(g, o.yfmt ? o.yfmt(v) : fmt(v, ystep < 1 ? 1 : 0), L - 6, y + 3.5, { size: 8.5, color: P.ink3, align: 'right' });
        }
        // round tick step: 1, 2, 5 x 10^k that gives about 6 ticks
        const span = xmax - xmin, raw = span / (o.nx || 6), mag = Math.pow(10, Math.floor(Math.log10(raw)));
        const xstep = o.xstep || [1, 2, 5, 10].map(m => m * mag).find(st => span / st <= (o.nx || 6) + 0.5) || mag * 10;
        for (let v = Math.ceil(xmin / xstep) * xstep; v <= xmax + 1e-9; v += xstep) {
            const x = X(v);
            g.beginPath(); g.moveTo(x, T); g.lineTo(x, T + ph); g.stroke();
            U.label(g, o.xfmt ? o.xfmt(v) : fmt(v), x, h - 10, { size: 8.5, color: P.ink3, align: 'center' });
        }
        U.label(g, o.ylabel || '', 6, T - 9, { size: 8.5, color: P.ink3 });
        U.label(g, o.xlabel || '', L + pw, T - 9, { size: 8.5, color: P.ink3, align: 'right' });
        for (const hl of o.hlines || []) {
            g.save(); g.strokeStyle = hl.color || P.ink3; g.setLineDash([4, 4]); g.lineWidth = 1;
            g.beginPath(); g.moveTo(L, Y(hl.y)); g.lineTo(L + pw, Y(hl.y)); g.stroke(); g.restore();
            U.label(g, hl.label, L + 6, Y(hl.y) - 5, { size: 8.5, color: hl.color || P.ink3 });
        }
        for (const s of o.series) {
            g.save();
            g.shadowColor = U.hexToRgba(s.color, 0.45); g.shadowBlur = 8;
            g.strokeStyle = s.color; g.lineWidth = 2; g.lineJoin = 'round';
            if (s.dash) g.setLineDash(s.dash);
            g.beginPath();
            s.xs.forEach((x, i) => (i ? g.lineTo(X(x), Y(s.ys[i])) : g.moveTo(X(x), Y(s.ys[i]))));
            g.stroke(); g.restore();
        }
        for (const m of o.points || []) {
            g.save();
            g.strokeStyle = m.color || P.ink; g.setLineDash([3, 3]); g.lineWidth = 1;
            g.beginPath(); g.moveTo(X(m.x), T + ph); g.lineTo(X(m.x), Y(m.y)); g.lineTo(L, Y(m.y)); g.stroke();
            g.setLineDash([]);
            g.fillStyle = m.color || P.ink; g.beginPath(); g.arc(X(m.x), Y(m.y), 4.5, 0, Math.PI * 2); g.fill();
            if (m.label) {
                g.font = '700 10px Inter, system-ui, sans-serif';
                const tw = g.measureText(m.label).width, right = X(m.x) + 8 + tw > L + pw;
                U.label(g, m.label, right ? X(m.x) - 8 : X(m.x) + 8, Y(m.y) - 8, { size: 10, weight: '700', color: m.color || P.ink, align: right ? 'right' : 'left' });
            }
            g.restore();
        }
        if (o.legend) {
            let x = L + 8;
            for (const s of o.series) {
                if (!s.name) continue;
                g.fillStyle = s.color; g.fillRect(x, T + 8, 14, 3);
                U.label(g, s.name, x + 18, T + 12, { size: 9, color: P.ink2 });
                x += 18 + g.measureText(s.name).width * 0.85 + 30;
            }
        }
        return { X, Y };
    }

    // ------------------------------------------------------------------
    // aero helpers (live force model from windtunnel.js)
    // ------------------------------------------------------------------
    const AERO = () => window.F1Aero;
    const SETUPS = {
        monaco: { label: 'Monaco', rearWing: 32, frontWing: 16, frontRH: 28, rearRH: 46 },
        silverstone: { label: 'Silverstone', rearWing: 20, frontWing: 10, frontRH: 40, rearRH: 50 },
        monza: { label: 'Monza', rearWing: 7, frontWing: 4, frontRH: 46, rearRH: 62 },
    };
    const POWER_W = 470000, RHO = 1.225;
    function solve(setup, speed, drs) {
        return AERO().solve(Object.assign({ speed, drs: !!drs }, setup));
    }
    const kgAt = (setup, speed, drs) => solve(setup, speed, drs).downforce / 9.81;
    const vmaxOf = (setup, drs) => Math.cbrt(POWER_W / (0.5 * RHO * solve(setup, 250, drs).sCx)) * 3.6;

    // ------------------------------------------------------------------
    // question generators
    // each returns { cat, level, prompt, hint, options, answer, draw, evidence, explain, takeaways, link }
    // ------------------------------------------------------------------
    const gens = [];
    const CIRC = () => D.circuits;
    const HIST = () => D.history;
    const seasonYears = () => Object.keys(HIST().seasons);
    const circuitStats = c => [['top speed', fmt(c.laps[0].top) + ' km/h'], ['full throttle', fmt(c.laps[0].full_throttle) + '%'], ['braking zones', String(c.laps[0].brake_events)], ['lap', km(c.length)]];
    const traceColor = P.teal;

    // ---- telemetry ----------------------------------------------------
    gens.push({ cat: 'telemetry', level: 2, make() {
        const c = choice(CIRC()), L = c.laps[0];
        const alts = others(CIRC(), o => o.key, c.key, 2);
        const sorted = CIRC().slice().sort((a, b) => b.laps[0].top - a.laps[0].top);
        const topRank = sorted.indexOf(c) + 1;
        const bits = [];
        if (topRank === 1) bits.push('the highest top speed in the set, <b>' + fmt(L.top) + ' km/h</b>');
        else if (topRank === CIRC().length) bits.push('the lowest top speed of any circuit here, <b>' + fmt(L.top) + ' km/h</b> — only a street circuit produces that');
        else bits.push('a peak of <b>' + fmt(L.top) + ' km/h</b>');
        bits.push('<b>' + fmt(L.full_throttle) + '%</b> of the lap flat out');
        bits.push('<b>' + L.brake_events + '</b> distinct braking zones');
        const alt = alts[0], AL = alt.laps[0];
        return {
            prompt: 'Which circuit produced this speed trace?',
            hint: 'Top speed, how much of the lap is flat out, how many braking zones — the fingerprint is all there.',
            options: shuffle([c.label].concat(alts.map(a => a.label))), answer: c.label,
            draw: (g, w, h, reveal) => chartTrace(g, w, h, { series: [{ y: L.speed, color: traceColor }], marks: reveal ? [{ i: minIdx(L.speed), label: '↓', color: P.red }] : [] }),
            evidence: { caption: c.label + ' — the same lap on the map, coloured by speed. The red dot is the slowest corner.',
                draw: (g, w, h) => chartMap(g, w, h, { map: c.map, title: c.name, sub: L.abbr + ' · ' + U.fmtLap(L.laptime), stats: circuitStats(c), marks: [{ i: Math.round(minIdx(L.speed) / L.speed.length * (c.map.x.length - 1)), color: P.red }] }) },
            explain: 'This is <b>' + esc(c.label) + '</b>: ' + bits.join(', ') + '. Pole was ' + L.abbr + ' in ' + U.fmtLap(L.laptime) + ' over ' + km(c.length) + '.',
            takeaways: ['Compare it with ' + alt.label + ': ' + fmt(AL.top) + ' km/h peak, ' + fmt(AL.full_throttle) + '% full throttle, ' + AL.brake_events + ' braking zones. Same sport, different shape.',
                'Long flat plateaus are straights; every dip is a corner. Count the dips and you have counted the corners that matter.'],
        };
    } });

    gens.push({ cat: 'telemetry', level: 2, make() {
        const c = choice(CIRC());
        const [a, b] = c.laps;
        const order = Math.random() < 0.5 ? [a, b] : [b, a];
        const fast = a.laptime <= b.laptime ? a : b, slow = fast === a ? b : a;
        const labelFast = order[0] === fast ? 'A' : 'B';
        const delta = deltaTrace(c.length, fast.speed, slow.speed);
        const n = fast.speed.length, W = 10;
        let bestI = 0, bestV = -Infinity;
        for (let i = 0; i <= n - W; i++) { let s = 0; for (let k = 0; k < W; k++) s += fast.speed[i + k] - slow.speed[i + k]; if (s / W > bestV) { bestV = s / W; bestI = i; } }
        const pct = Math.round((bestI + W / 2) / n * 100);
        const gap = (slow.laptime - fast.laptime).toFixed(3);
        const minFast = Math.min.apply(null, fast.speed), minSlow = Math.min.apply(null, slow.speed);
        const peakNote = fast.top < slow.top ? ' The slower lap actually had the higher peak (' + fmt(slow.top) + ' vs ' + fmt(fast.top) + ' km/h) — it lost the time in the corners, not on the straights.' : '';
        const colors = [P.teal, '#ff7a70'];
        const mi0 = Math.round(bestI / n * (c.map.x.length - 1)), mi1 = Math.round((bestI + W) / n * (c.map.x.length - 1));
        return {
            prompt: 'Two laps from the same qualifying session at ' + c.label + '. Which one is faster?',
            hint: 'Same track, so the shapes match — look at where the lines separate, and which one stays higher through the corners.',
            options: ['Lap A', 'Lap B'], answer: 'Lap ' + labelFast,
            draw: (g, w, h, reveal) => chartTrace(g, w, h, {
                series: [{ y: order[0].speed, color: colors[0], name: 'Lap A' }, { y: order[1].speed, color: colors[1], name: 'Lap B' }],
                fillBetween: reveal ? { pair: labelFast === 'A' ? [0, 1] : [1, 0], color: U.hexToRgba(labelFast === 'A' ? colors[0] : colors[1], 0.16) } : null,
            }),
            legend: [{ name: 'Lap A', color: colors[0] }, { name: 'Lap B', color: colors[1] }],
            evidence: { caption: 'Cumulative time gap, built from the two speed traces (time = distance ÷ speed). Where the line climbs, Lap ' + labelFast + ' is gaining. The red section on the map is the biggest single gain.',
                draw: (g, w, h) => {
                    g.save(); chartTrace(g, w * 0.6, h, { series: [{ y: delta, color: P.amber, area: true }], ylabel: 'gap, s', ystep: 0.1, yfmt: v => v.toFixed(1), allowNegative: true, ymax: Math.max(0.05, Math.ceil(Math.max.apply(null, delta) * 10) / 10) }); g.restore();
                    g.save(); g.translate(w * 0.6, 0); chartMap(g, w * 0.4, h, { map: c.map, mono: 'rgba(255,255,255,0.28)', highlight: [mi0, mi1] }); g.restore();
                } },
            explain: '<b>Lap ' + labelFast + '</b> — ' + fast.abbr + ' in ' + U.fmtLap(fast.laptime) + ' against ' + slow.abbr + '\'s ' + U.fmtLap(slow.laptime) + ', a gap of <b>' + gap + ' s</b>. The biggest advantage came around <b>' + pct + '%</b> of the lap, where ' + fast.abbr + ' carried about ' + Math.round(bestV) + ' km/h more through the section' + (minFast > minSlow ? ', and had a higher minimum corner speed (' + fmt(minFast) + ' vs ' + fmt(minSlow) + ' km/h)' : '') + '.' + peakNote,
            takeaways: ['Engineers rarely stare at two speed traces. They plot the <i>delta</i> — the running time gap — because a 5 km/h difference through a long corner is worth more than 5 km/h at the end of a straight.',
                'Minimum corner speed decides the lap. Carry 3 km/h more through a corner and you carry it down the whole straight that follows.'],
        };
    } });

    gens.push({ cat: 'telemetry', level: 1, make() {
        const c = choice(CIRC()), L = c.laps[0], n = L.speed.length;
        // three candidate corners: local minima spaced apart
        const mins = [];
        for (let i = 6; i < n - 6; i++) { let ok = true; for (let k = -6; k <= 6; k++) if (L.speed[i + k] < L.speed[i]) { ok = false; break; } if (ok && (!mins.length || i - mins[mins.length - 1] > 25)) mins.push(i); }
        if (mins.length < 3) return null;
        const trio = pick(mins, 3).sort((a, b) => a - b);
        const labels = ['A', 'B', 'C'];
        const ans = labels[minIdx(trio.map(i => L.speed[i]))];
        const marks = trio.map((i, k) => ({ i, label: labels[k], color: P.amber }));
        const mapMarks = trio.map((i, k) => ({ i: Math.round(i / n * (c.map.x.length - 1)), label: labels[k], color: labels[k] === ans ? P.red : P.amber }));
        return {
            prompt: 'Three corners are marked on this ' + c.label + ' lap. Which is the slowest?',
            hint: 'On a speed trace, every dip is a corner. The deeper the dip, the tighter the corner.',
            options: labels.map(l => 'Corner ' + l), answer: 'Corner ' + ans,
            draw: (g, w, h) => chartTrace(g, w, h, { series: [{ y: L.speed, color: traceColor }], marks }),
            evidence: { caption: 'The same three corners on the ' + c.label + ' map. The slowest one is red — it is the tightest radius on the lap.',
                draw: (g, w, h) => chartMap(g, w, h, { map: c.map, title: c.name, marks: mapMarks, stats: trio.map((i, k) => ['corner ' + labels[k], fmt(L.speed[i]) + ' km/h', labels[k] === ans ? P.red : undefined]) }) },
            explain: '<b>Corner ' + ans + '</b> bottoms out at <b>' + fmt(L.speed[trio[labels.indexOf(ans)]]) + ' km/h</b>. The others: ' + trio.map((i, k) => labels[k] + ' ' + fmt(L.speed[i]) + ' km/h').filter((_, k) => labels[k] !== ans).join(', ') + '. Speed through a corner is set by its radius and by grip — the tighter the bend, the lower the dip.',
            takeaways: ['Corner speed scales with the square root of radius: a bend twice as tight is not twice as slow, but about 30% slower.',
                'The slowest corner on the lap is where downforce helps least and mechanical grip — tyres and suspension — matters most.'],
        };
    } });

    gens.push({ cat: 'telemetry', level: 1, make() {
        const c = choice(CIRC()), L = c.laps[0], n = L.speed.length;
        const kinds = [
            { key: 'brake', label: 'Hard on the brakes', test: i => L.brake[i] > 50 },
            { key: 'full', label: 'Flat out on the throttle', test: i => L.throttle[i] > 98 && L.speed[i] > L.speed[i - 3] + 2 },
            { key: 'mid', label: 'Mid-corner, off both pedals', test: i => L.brake[i] < 5 && L.throttle[i] < 40 },
        ];
        const k = choice(kinds);
        const cands = []; for (let i = 8; i < n - 8; i++) if (k.test(i)) cands.push(i);
        if (!cands.length) return null;
        const i = choice(cands);
        const mi = Math.round(i / n * (c.map.x.length - 1));
        return {
            prompt: 'Speed, throttle and brake for one ' + c.label + ' lap. At the marked point, what is the driver doing?',
            hint: 'Read all three channels at the mark: the pedals explain what the speed line is about to do.',
            options: kinds.map(x => x.label), answer: k.label,
            legend: [{ name: 'speed', color: traceColor }, { name: 'throttle %', color: P.amber }, { name: 'brake %', color: P.red }],
            draw: (g, w, h) => chartTrace(g, w, h, {
                series: [{ y: L.speed, color: traceColor }, { y: L.throttle.map(v => v * 3.5), color: P.amber, thin: true }, { y: L.brake.map(v => v * 3.5), color: P.red, thin: true }],
                ymax: 360, marks: [{ i, label: '?', color: P.ink }],
            }),
            evidence: { caption: 'Where that moment sits on the circuit. Speed ' + fmt(L.speed[i]) + ' km/h · throttle ' + fmt(L.throttle[i]) + '% · brake ' + (L.brake[i] > 0 ? 'on' : 'off') + '.',
                draw: (g, w, h) => chartMap(g, w, h, { map: c.map, title: c.name, marks: [{ i: mi, color: P.ink }], stats: [['speed', fmt(L.speed[i]) + ' km/h'], ['throttle', fmt(L.throttle[i]) + '%'], ['brake', L.brake[i] > 0 ? 'ON' : 'off', L.brake[i] > 0 ? P.red : undefined], ['gear', String(L.gear[i])]] }) },
            explain: '<b>' + k.label + '.</b> ' + (k.key === 'brake' ? 'The brake channel is at ' + fmt(L.brake[i]) + '% and the speed line is falling steeply — an F1 car sheds over 100 km/h in under two seconds here. Braking is where the biggest g-forces of the lap occur.'
                : k.key === 'full' ? 'Throttle is pinned at 100% with the brake off and speed still climbing, in ' + L.gear[i] + (L.gear[i] === 1 ? 'st' : L.gear[i] === 2 ? 'nd' : L.gear[i] === 3 ? 'rd' : 'th') + ' gear. The car is traction- or drag-limited, not driver-limited.'
                : 'Throttle at ' + fmt(L.throttle[i]) + '%, brake off, speed near its minimum: the driver is balancing the car through the apex, waiting for the front to bite before feeding the power back in.'),
            takeaways: ['A qualifying lap is only ~' + fmt(L.braking) + '% braking and ~' + fmt(L.full_throttle) + '% full throttle at ' + c.label + '. The rest is the part-throttle "coasting" phase that separates good drivers from great ones.',
                'Throttle and brake never overlap on a clean lap. If they do, the driver is trail-braking into the apex — or the data is telling you something is wrong with the car.'],
        };
    } });

    gens.push({ cat: 'telemetry', level: 3, make() {
        const c = choice(CIRC()), L = c.laps[0];
        const v = L.full_throttle;
        const opts = numericOptions(v, '%', 0, 0.35);
        const ans = fmt(v) + '%';
        if (new Set(opts).size < 3) return null;
        const n = L.throttle.length, bands = [];
        let start = -1;
        for (let i = 0; i < n; i++) { const on = L.throttle[i] > 98; if (on && start < 0) start = i; if (!on && start >= 0) { bands.push({ i0: start, i1: i, color: 'rgba(255,135,0,0.16)' }); start = -1; } }
        if (start >= 0) bands.push({ i0: start, i1: n - 1, color: 'rgba(255,135,0,0.16)' });
        const longest = bands.reduce((b, x) => (x.i1 - x.i0 > b.i1 - b.i0 ? x : b), bands[0]);
        const mi = i => Math.round(i / n * (c.map.x.length - 1));
        const ranked = CIRC().slice().sort((a, b) => b.laps[0].full_throttle - a.laps[0].full_throttle);
        return {
            prompt: 'This is the throttle trace for a ' + c.label + ' qualifying lap. What share of the lap is at full throttle?',
            hint: 'Count the plateaus at 100%. Street circuits sit around 40–50%; the fastest tracks in the world pass 70%.',
            options: opts, answer: ans,
            legend: [{ name: 'throttle %', color: P.amber }],
            draw: (g, w, h, reveal) => chartTrace(g, w, h, { series: [{ y: L.throttle, color: P.amber, area: true }], ymax: 105, ystep: 25, ylabel: '% throttle', bands: reveal ? bands : [] }),
            evidence: { caption: 'Full-throttle share across every circuit in the set. ' + c.label + ' is highlighted; the map shows its longest flat-out stretch.',
                draw: (g, w, h) => {
                    g.save(); chartBars(g, w * 0.58, h, { items: ranked.map(x => ({ label: x.label, value: x.laps[0].full_throttle, hl: x.key === c.key, color: P.amber, text: fmt(x.laps[0].full_throttle) + '%' })), max: 100 }); g.restore();
                    g.save(); g.translate(w * 0.58, 0); chartMap(g, w * 0.42, h, { map: c.map, mono: 'rgba(255,255,255,0.28)', highlight: [mi(longest.i0), mi(longest.i1)], highlightColor: P.amber }); g.restore();
                } },
            explain: '<b>' + ans + '</b> of the ' + c.label + ' lap is at full throttle — ' + (ranked.indexOf(c) === 0 ? 'the highest share in the set' : ranked.indexOf(c) === ranked.length - 1 ? 'the lowest share in the set' : 'rank ' + (ranked.indexOf(c) + 1) + ' of ' + ranked.length + ' here') + '. The longest flat-out stretch runs for about <b>' + fmt((longest.i1 - longest.i0) / n * c.length) + ' m</b>. Full-throttle share is the single number engineers use to decide how much wing to run: the more of the lap is flat out, the more drag costs and the less downforce pays.',
            takeaways: ['This is exactly why the lap-time model above picks a different rear wing for each circuit. Flat-out share is the input; wing angle is the output.',
                'Power-unit engineers use the same number for energy deployment: a lap that is 75% flat out needs the hybrid battery managed very differently from one that is 45%.'],
            link: { href: '#aero', label: 'See how wing angle trades against it in the wind tunnel' },
        };
    } });

    gens.push({ cat: 'telemetry', level: 1, make() {
        const cands = CIRC().filter(c => c.kind === 'street' || c.kind === 'power');
        if (cands.length < 2) return null;
        const c = choice(cands), L = c.laps[0];
        const isStreet = c.kind === 'street';
        const opp = CIRC().filter(x => x.kind === (isStreet ? 'power' : 'street'));
        const alt = opp.length ? choice(opp) : null;
        return {
            prompt: 'No labels: is this speed trace from a street circuit or a high-speed permanent circuit?',
            hint: 'Street circuits are narrow, bumpy and lined with walls. That shows up as a low ceiling and constant dips.',
            options: ['Street circuit', 'High-speed circuit'], answer: isStreet ? 'Street circuit' : 'High-speed circuit',
            draw: (g, w, h) => chartTrace(g, w, h, { series: [{ y: L.speed, color: traceColor }] }),
            evidence: { caption: 'It was ' + c.label + '. ' + (alt ? 'Beside it, ' + alt.label + ' — the opposite kind of circuit — at the same scale.' : ''),
                draw: (g, w, h) => chartMaps(g, w, h, { items: [{ map: c.map, label: c.label, sub: fmt(L.top) + ' km/h peak · ' + fmt(L.full_throttle) + '% flat out', color: P.teal }].concat(alt ? [{ map: alt.map, label: alt.label, sub: fmt(alt.laps[0].top) + ' km/h peak · ' + fmt(alt.laps[0].full_throttle) + '% flat out' }] : []) }) },
            explain: 'This is <b>' + esc(c.label) + '</b>, a ' + (isStreet ? 'street' : 'high-speed permanent') + ' circuit. It peaks at <b>' + fmt(L.top) + ' km/h</b>, spends <b>' + fmt(L.full_throttle) + '%</b> of the lap flat out and has <b>' + L.brake_events + '</b> braking zones' + (alt ? '. ' + alt.label + ', for contrast: ' + fmt(alt.laps[0].top) + ' km/h, ' + fmt(alt.laps[0].full_throttle) + '% flat out, ' + alt.laps[0].brake_events + ' braking zones.' : '.'),
            takeaways: ['Street circuits reward a car with maximum downforce and a driver with maximum confidence in the walls. Peak speed is almost irrelevant.',
                'The trace ceiling is the giveaway. Below 300 km/h all lap and it is almost certainly a street circuit or Hungary.'],
        };
    } });

    // ---- aero ---------------------------------------------------------
    gens.push({ cat: 'aero', level: 2, make() {
        if (!AERO()) return null;
        const speed = choice([150, 200, 250, 300]);
        const s = SETUPS.silverstone;
        const kg = kgAt(s, speed);
        const opts = numericOptions(kg, ' kg', 0, 0.55), ans = fmt(kg) + ' kg';
        const xs = []; for (let v = 50; v <= 340; v += 10) xs.push(v);
        const ys = xs.map(v => kgAt(s, v));
        const crossover = xs.find(v => kgAt(s, v) >= CAR_MASS);
        return {
            prompt: 'The baseline wind-tunnel setup on this page (Silverstone wing levels). Roughly how much downforce does it make at ' + speed + ' km/h?',
            hint: 'The car weighs 798 kg with the driver in it. Somewhere on this curve the wings push down harder than gravity does.',
            options: opts, answer: ans,
            draw: (g, w, h, reveal) => chartCurve(g, w, h, { series: [{ xs, ys, color: '#7aa2ff', name: 'downforce' }], xlabel: 'km/h', ylabel: 'kg of load', hlines: [{ y: CAR_MASS, label: 'car weight 798 kg', color: P.ink3 }], points: reveal ? [{ x: speed, y: kg, label: fmt(kg) + ' kg', color: P.ink }] : [], nx: 6 }),
            evidence: { caption: 'Load on the car at ' + speed + ' km/h: gravity versus aerodynamics. Above about ' + crossover + ' km/h the car could, in principle, drive on the ceiling.',
                draw: (g, w, h) => chartBars(g, w, h, { items: [{ label: 'car weight', value: CAR_MASS, color: '#868f9e', text: '798 kg' }, { label: 'downforce @ ' + speed, value: kg, hl: true, color: '#7aa2ff', text: fmt(kg) + ' kg' }, { label: 'total on tyres', value: CAR_MASS + kg, color: P.teal, text: fmt(CAR_MASS + kg) + ' kg' }], max: Math.max(CAR_MASS + kg, 1) * 1.1, labelW: 120 }) },
            explain: 'About <b>' + ans + '</b>. Downforce is ½·ρ·v²·sCz: with the baseline sCz of ' + solve(s, speed).sCz.toFixed(2) + ' m² the load at ' + speed + ' km/h is ' + fmt(kg) + ' kg, so the tyres are carrying <b>' + fmt(CAR_MASS + kg) + ' kg</b> in total. That extra load is what lets a car corner at 4–5 g when the tyre alone would give under 2 g.',
            takeaways: ['Downforce crosses car weight at roughly ' + crossover + ' km/h on this setup. Every corner taken above that speed is being held down more by air than by gravity.',
                'It scales with the square of speed: half the speed gives a quarter of the load, which is why slow corners feel like a different car.'],
            link: { href: '#aero', label: 'Dial the speed yourself in the wind tunnel' },
        };
    } });

    gens.push({ cat: 'aero', level: 1, make() {
        if (!AERO()) return null;
        const v = choice([100, 120, 150]);
        const s = SETUPS.silverstone;
        const k1 = kgAt(s, v), k2 = kgAt(s, v * 2);
        const xs = []; for (let x = 40; x <= 320; x += 10) xs.push(x);
        return {
            prompt: 'An F1 car doubles its speed from ' + v + ' to ' + v * 2 + ' km/h. What happens to its downforce?',
            hint: 'Aerodynamic force depends on how hard the air hits, and the air hits harder than you would think.',
            options: shuffle(['It doubles', 'It quadruples', 'It stays the same']), answer: 'It quadruples',
            draw: (g, w, h, reveal) => chartCurve(g, w, h, { series: [{ xs, ys: xs.map(x => kgAt(s, x)), color: '#7aa2ff', name: 'downforce' }], xlabel: 'km/h', ylabel: 'kg', points: reveal ? [{ x: v, y: k1, label: fmt(k1) + ' kg', color: P.ink }, { x: v * 2, y: k2, label: fmt(k2) + ' kg', color: P.ink }] : [{ x: v, y: k1, color: P.ink3 }], nx: 7 }),
            evidence: { caption: 'Load at ' + v + ' km/h and at ' + v * 2 + ' km/h on the baseline setup. Drag obeys the same law, which is why the last 20 km/h of top speed costs so much.',
                draw: (g, w, h) => chartBars(g, w, h, { items: [{ label: v + ' km/h', value: k1, color: '#7aa2ff', text: fmt(k1) + ' kg' }, { label: v * 2 + ' km/h', value: k2, hl: true, color: '#7aa2ff', text: fmt(k2) + ' kg  (×' + (k2 / k1).toFixed(1) + ')' }], labelW: 90 }) },
            explain: '<b>It quadruples.</b> Aerodynamic force is proportional to speed <i>squared</i>: ' + fmt(k1) + ' kg at ' + v + ' km/h becomes ' + fmt(k2) + ' kg at ' + v * 2 + ' km/h. The wings do not change; the air simply arrives with four times the energy.',
            takeaways: ['The same square law applies to drag, so the engine power needed to hold a speed rises with the <i>cube</i> of speed. Going from 320 to 340 km/h costs about 20% more power.',
                'This is why low-speed corners are a mechanical-grip problem and high-speed corners are an aero problem. The car is effectively two different cars.'],
            link: { href: '#aero', label: 'Watch the pressure field change with speed' },
        };
    } });

    gens.push({ cat: 'aero', level: 1, make() {
        if (!AERO()) return null;
        const keys = Object.keys(SETUPS);
        const vm = keys.map(k => ({ k, label: SETUPS[k].label, vmax: vmaxOf(SETUPS[k]), sCx: solve(SETUPS[k], 250).sCx, sCz: solve(SETUPS[k], 250).sCz, rw: SETUPS[k].rearWing }));
        const best = vm.reduce((a, b) => (b.vmax > a.vmax ? b : a));
        const worst = vm.reduce((a, b) => (b.vmax < a.vmax ? b : a));
        const askHigh = Math.random() < 0.5;
        const target = askHigh ? best : worst;
        return {
            prompt: 'Three setups from the wind tunnel: Monaco (rear wing 32°), Silverstone (20°) and Monza (7°). Which reaches the highest top speed?'.replace('highest', askHigh ? 'highest' : 'lowest'),
            hint: 'More wing means more downforce, but wings are the draggiest thing on the car.',
            options: vm.map(x => x.label + ' setup'), answer: target.label + ' setup',
            draw: (g, w, h, reveal) => chartBars(g, w, h, { title: reveal ? 'Drag-limited top speed with 470 kW at the wheels' : 'Drag-limited top speed', items: vm.map(x => ({ label: x.label + ' · ' + x.rw + '°', value: reveal ? x.vmax : 0, hl: reveal && x.k === target.k, color: '#7aa2ff', text: reveal ? fmt(x.vmax) + ' km/h' : '?' })), min: 200, max: 380, labelW: 130 }),
            evidence: { caption: 'What each setup pays and what it gets: drag (sCx) against downforce (sCz), both area-inclusive coefficients from the force model.',
                draw: (g, w, h) => chartBars(g, w, h, { items: vm.flatMap(x => [{ label: x.label + ' drag', value: x.sCx, color: P.red, hl: x.k === target.k, text: x.sCx.toFixed(2) + ' m²' }, { label: x.label + ' downforce', value: x.sCz, color: P.teal, dim: x.k !== target.k, text: x.sCz.toFixed(2) + ' m²' }]), max: 4.4, labelW: 130 }) },
            explain: '<b>' + target.label + '.</b> Top speed is where engine power equals drag power, and drag power grows with the cube of speed. The Monza wing (7°) has an sCx of ' + vm[2].sCx.toFixed(2) + ' m² against Monaco\'s ' + vm[0].sCx.toFixed(2) + ' m², which is worth about <b>' + fmt(best.vmax - worst.vmax) + ' km/h</b> of top speed — but it gives up ' + fmt((1 - vm[2].sCz / vm[0].sCz) * 100) + '% of the downforce to get it.',
            takeaways: ['Wing drag grows faster than wing load (induced drag scales with lift squared), so the last few degrees of wing are the most expensive.',
                'Teams do not chase top speed for its own sake. They chase lap time — which is why the lap model above picks a different wing for each circuit.'],
            link: { href: '#lapSim', label: 'See what each setup does to a lap' },
        };
    } });

    gens.push({ cat: 'aero', level: 3, make() {
        if (!AERO()) return null;
        const s = Object.assign({}, SETUPS.silverstone);
        const hs = []; for (let h = 12; h <= 60; h += 2) hs.push(h);
        const floorAt = h => { const r = AERO().solve({ speed: 250, rearWing: s.rearWing, frontWing: s.frontWing, frontRH: h, rearRH: h + 10, drs: false }); return r.downforce / 9.81; };
        const ys = hs.map(floorAt);
        const peakH = hs[maxIdx(ys)];
        const opts = shuffle([peakH + ' mm', '14 mm', '40 mm']);
        if (new Set(opts).size < 3) return null;
        return {
            prompt: 'Lowering the floor makes the underbody work harder. At which front ride height does this model make the most downforce?',
            hint: 'Ground effect rises as the floor drops — until the airflow under the car stalls. 2022 taught every team where that cliff is.',
            options: opts, answer: peakH + ' mm',
            draw: (g, w, h, reveal) => chartCurve(g, w, h, { series: [{ xs: hs, ys, color: '#7aa2ff', name: 'total downforce' }], xlabel: 'front ride height, mm', ylabel: 'kg @ 250 km/h', points: reveal ? [{ x: peakH, y: floorAt(peakH), label: 'peak · ' + peakH + ' mm', color: P.ink }, { x: 14, y: floorAt(14), label: 'stalled', color: P.red }, { x: 40, y: floorAt(40), label: 'reference', color: P.ink3 }] : [], nx: 6, ymin: Math.floor(Math.min.apply(null, ys) * 0.9 / 100) * 100 }),
            evidence: { caption: 'Downforce at three ride heights. Below the peak the venturi tunnels stall and the load collapses — that is the porpoising cliff of 2022.',
                draw: (g, w, h) => chartBars(g, w, h, { items: [14, peakH, 40, 56].map(x => ({ label: x + ' mm' + (x === peakH ? ' · peak' : x === 14 ? ' · stalled' : x === 40 ? ' · reference' : ' · high'), value: floorAt(x), hl: x === peakH, color: '#7aa2ff', text: fmt(floorAt(x)) + ' kg' })), min: Math.floor(Math.min.apply(null, ys) * 0.9 / 100) * 100, labelW: 130 }) },
            explain: '<b>' + peakH + ' mm.</b> The floor generates load by accelerating air through the narrow gap beneath the car (the venturi effect): the lower the floor, the faster the air, the lower the pressure. At ' + peakH + ' mm the model makes <b>' + fmt(floorAt(peakH)) + ' kg</b> at 250 km/h; at 14 mm the flow stalls and it drops to <b>' + fmt(floorAt(14)) + ' kg</b>. When a car bottoms out, loses the load, rises, regains it and slams back down — that oscillation is porpoising.',
            takeaways: ['Teams run as low as the stall cliff allows, then stiffen the suspension to stop the car reaching it. Ride height is the most valuable millimetre on the car.',
                'This is why kerbs and bumps matter so much on a ground-effect car: a 5 mm bounce is a 5 mm change in downforce.'],
            link: { href: '#aero', label: 'Drop the ride height in the wind tunnel and watch the floor' },
        };
    } });

    gens.push({ cat: 'aero', level: 2, make() {
        const T = D.tracks;
        if (!T || !T.circuits || T.circuits.length < 3) return null;
        const askLow = Math.random() < 0.5;
        const sorted = T.circuits.slice().sort((a, b) => a.best_rw - b.best_rw);
        const target = askLow ? sorted[0] : sorted[sorted.length - 1];
        const maps = T.circuits.map(t => ({ t, q: CIRC().find(c => c.key === t.key) })).filter(x => x.q);
        if (maps.length < 3) return null;
        const name = t => t.name.replace(' Grand Prix', '');
        return {
            prompt: 'The lap-time model searched every wing angle at three circuits. Which one wants the ' + (askLow ? 'LEAST' : 'MOST') + ' rear wing?',
            hint: 'The more of the lap that is spent flat out, the more drag hurts and the less downforce helps.',
            options: T.circuits.map(t => name(t)), answer: name(target),
            draw: (g, w, h, reveal) => chartMaps(g, w, h, { items: maps.map(x => ({ map: x.q.map, label: name(x.t), sub: reveal ? 'optimum rear wing ' + x.t.best_rw + '°' : fmt(x.q.laps[0].full_throttle) + '% flat out', color: reveal && x.t.key === target.key ? P.red : undefined, mono: reveal && x.t.key !== target.key ? 'rgba(255,255,255,0.22)' : undefined })) }),
            evidence: { caption: 'Optimum rear wing at each circuit, from the lap-time model calibrated to the real pole laps (' + T.rms_pct.toFixed(1) + '% RMS).',
                draw: (g, w, h) => chartBars(g, w, h, { items: sorted.map(t => ({ label: name(t), value: t.best_rw, hl: t.key === target.key, color: '#7aa2ff', text: t.best_rw + '°  ·  ' + U.fmtLap(t.best_time) })), max: 40, labelW: 90 }) },
            explain: '<b>' + name(target) + '</b> — the model\'s optimum is <b>' + target.best_rw + '°</b> of rear wing, against ' + sorted.filter(t => t !== target).map(t => name(t) + ' ' + t.best_rw + '°').join(' and ') + '. ' + (askLow ? 'With long straights and few corners, every degree of wing costs more on the straight than it returns in the bends.' : 'With the lowest full-throttle share of the three, the extra drag is barely felt and the extra grip is worth seconds.'),
            takeaways: ['Apply the wrong optimum and the model shows the cost: Monza\'s wing round Monaco loses nearly two seconds a lap.',
                'Real teams cannot swap wings between corners, so every setup is a compromise weighted by where the lap time is.'],
            link: { href: '#lapSim', label: 'Run your own setup round all three' },
        };
    } });

    gens.push({ cat: 'aero', level: 3, make() {
        if (!AERO()) return null;
        const v = choice([160, 200, 240]);
        const a = SETUPS.monza, b = SETUPS.monaco;
        const ka = kgAt(a, v), kb = kgAt(b, v);
        const pct = (kb / ka - 1) * 100;
        const opts = numericOptions(pct, '%', 0, 0.5), ans = fmt(pct) + '%';
        if (new Set(opts).size < 3) return null;
        const va = vmaxOf(a), vb = vmaxOf(b);
        const xs = []; for (let x = 60; x <= 340; x += 10) xs.push(x);
        return {
            prompt: 'Going from the Monza wing (7°) to the Monaco wing (32°): by roughly how much does downforce rise at ' + v + ' km/h?',
            hint: 'Rear wing is about a quarter of the car\'s load at baseline, and the front wing moves with it.',
            options: opts, answer: ans,
            draw: (g, w, h, reveal) => chartCurve(g, w, h, { series: [{ xs, ys: xs.map(x => kgAt(b, x)), color: P.red, name: 'Monaco 32°' }, { xs, ys: xs.map(x => kgAt(a, x)), color: '#7aa2ff', name: 'Monza 7°' }], legend: true, xlabel: 'km/h', ylabel: 'kg', points: reveal ? [{ x: v, y: kb, label: fmt(kb) + ' kg', color: P.red }, { x: v, y: ka, label: fmt(ka) + ' kg', color: '#7aa2ff' }] : [], nx: 7 }),
            evidence: { caption: 'The price of that load: drag-limited top speed for each setup with the same 470 kW.',
                draw: (g, w, h) => chartBars(g, w, h, { items: [{ label: 'Monza 7° · top speed', value: va, color: '#7aa2ff', hl: true, text: fmt(va) + ' km/h' }, { label: 'Monaco 32° · top speed', value: vb, color: P.red, hl: true, text: fmt(vb) + ' km/h' }, { label: 'Monza 7° · load @ ' + v, value: ka / 4, color: '#7aa2ff', text: fmt(ka) + ' kg' }, { label: 'Monaco 32° · load @ ' + v, value: kb / 4, color: P.red, text: fmt(kb) + ' kg' }], max: 400, labelW: 150 }) },
            explain: 'About <b>' + ans + '</b>: ' + fmt(ka) + ' kg becomes ' + fmt(kb) + ' kg at ' + v + ' km/h. The wing itself more than doubles its load, but the floor and diffuser — which make most of the downforce and do not care about wing angle — dilute the effect on the whole car. The bill is <b>' + fmt(va - vb) + ' km/h</b> of top speed.',
            takeaways: ['Since 2022 the floor makes over half the downforce. Wing angle is the fine adjustment, not the main event.',
                'In the wind tunnel above, watch the pressure field under the car, not the wings, when you change ride height — that is where the load lives.'],
            link: { href: '#aero', label: 'Compare the two setups in the wind tunnel' },
        };
    } });

    // ---- race craft / strategy -----------------------------------------
    gens.push({ cat: 'strategy', level: 1, make() {
        const H = HIST(); if (!H) return null;
        const y = choice(seasonYears()), S = H.seasons[y];
        const med = S.pit_median;
        const opts = shuffle([fmt(med, 0) + ' s', '2.5 s', '45 s']);
        const teams = S.pit_by_team.slice(0, 8);
        return {
            prompt: 'A tyre change takes about 2.5 seconds. In ' + y + ', how long did a typical pit stop cost from pit entry to pit exit?',
            hint: 'The stationary time is the famous number. The pit lane speed limit is the expensive one.',
            options: opts, answer: fmt(med, 0) + ' s',
            draw: (g, w, h, reveal) => chartBars(g, w, h, { title: 'Median pit-lane time by team, ' + y + (reveal ? '' : ' (values hidden)'), items: teams.map((t, i) => ({ label: t.team, value: reveal ? t.median : 0, color: P.amber, hl: reveal && i === 0, text: reveal ? t.median.toFixed(2) + ' s' : '' })), min: 18, max: 30, labelW: 110 }),
            evidence: { caption: 'Median pit-lane time each season since 2012, from ' + fmt(H.pit_evolution.reduce((a, b) => a + b.n, 0)) + ' recorded stops. The tyre change got quicker; the pit lane did not get shorter.',
                draw: (g, w, h) => chartCurve(g, w, h, { series: [{ xs: H.pit_evolution.map(p => p.year), ys: H.pit_evolution.map(p => p.median), color: P.amber, name: 'median' }, { xs: H.pit_evolution.map(p => p.year), ys: H.pit_evolution.map(p => p.fastest), color: P.teal, name: 'fastest', dash: [4, 4] }], legend: true, ymin: 12, ymax: 30, xfmt: v => String(Math.round(v)), yfmt: v => fmt(v) + ' s', nx: 6, points: [{ x: +y, y: med, color: P.ink }] }) },
            explain: '<b>About ' + fmt(med, 0) + ' seconds</b> (' + med.toFixed(2) + ' s median across the season). The car is stationary for only 2–3 s of that; the rest is driving the length of the pit lane at the 80 km/h limit. That is why strategy is about pit-lane <i>loss</i>, and why a track with a short pit lane makes two-stop strategies viable.',
            takeaways: ['The quickest stop of ' + y + ' was ' + S.pit_fastest[0].time.toFixed(2) + ' s pit-lane time by ' + S.pit_fastest[0].team + ' (' + S.pit_fastest[0].driver + ', ' + S.pit_fastest[0].race + ').',
                'Teams averaged ' + S.stops_per_car_race.toFixed(1) + ' stops per car per race in ' + y + '. Each one is a ' + fmt(med, 0) + '-second bet that fresh rubber pays back.'],
        };
    } });

    gens.push({ cat: 'strategy', level: 2, make() {
        const H = HIST(); if (!H) return null;
        const y = choice(seasonYears()), S = H.seasons[y];
        const teams = S.pit_by_team;
        if (teams.length < 4) return null;
        const best = teams[0];
        const alts = pick(teams.slice(2), 2);
        return {
            prompt: 'Which team had the quickest median pit stop of the ' + y + ' season?',
            hint: 'Consistency, not one heroic stop. The median ignores the outliers.',
            options: shuffle([best.team].concat(alts.map(t => t.team))), answer: best.team,
            draw: (g, w, h, reveal) => chartBars(g, w, h, { title: 'Median pit-lane time, ' + y, items: teams.slice(0, 10).map((t, i) => ({ label: reveal ? t.team : 'Team ' + String.fromCharCode(65 + i), value: t.median, color: P.amber, hl: reveal && t.team === best.team, text: t.median.toFixed(2) + ' s' })), min: 18, max: Math.ceil(teams[Math.min(9, teams.length - 1)].median) + 1, labelW: 100 }),
            evidence: { caption: 'The five quickest individual stops of ' + y + '. A single fast stop is a good crew on a good day; a low median is a good crew every day.',
                draw: (g, w, h) => chartBars(g, w, h, { items: S.pit_fastest.map(p => ({ label: p.team + ' · ' + p.race, value: p.time, color: P.teal, hl: p.team === best.team, text: p.time.toFixed(2) + ' s' })), min: 12, max: Math.ceil(S.pit_fastest[S.pit_fastest.length - 1].time) + 1, labelW: 150 }) },
            explain: '<b>' + esc(best.team) + '</b>, with a median pit-lane time of <b>' + best.median.toFixed(2) + ' s</b> against a field median of ' + S.pit_median.toFixed(2) + ' s. ' + alts.map(t => t.team + ' sat at ' + t.median.toFixed(2) + ' s').join('; ') + '. Over a season of ' + S.stops_per_car_race.toFixed(1) + ' stops per race, a few tenths per stop is a position or two.',
            takeaways: ['Pit-crew performance is one of the few areas where a midfield team can beat a front-runner outright. It costs training, not budget-cap money.',
                'Strategists model the pit-lane loss per circuit to the tenth. A slow crew shifts the whole undercut window.'],
        };
    } });

    gens.push({ cat: 'strategy', level: 2, make() {
        const H = HIST(); if (!H) return null;
        const y = choice(seasonYears()), S = H.seasons[y];
        const W = S.winners.filter(w => w.grid);
        const fromPole = W.filter(w => w.grid === 1).length;
        const opts = shuffle([fromPole, Math.max(0, fromPole - 5), Math.min(W.length, fromPole + 5)]);
        if (new Set(opts).size < 3) return null;
        const byGrid = {}; for (const w of W) byGrid[w.grid] = (byGrid[w.grid] || 0) + 1;
        const grids = Object.keys(byGrid).map(Number).sort((a, b) => a - b);
        const far = W.reduce((a, b) => (b.grid > a.grid ? b : a));
        return {
            prompt: 'In ' + y + ' there were ' + W.length + ' races. How many were won from pole position?',
            hint: 'Track position is king in modern F1 — but a good car on the second row is never far away.',
            options: opts.map(o => o + ' of ' + W.length), answer: fromPole + ' of ' + W.length,
            draw: (g, w, h, reveal) => chartBars(g, w, h, { title: 'Race wins by grid slot, ' + y, items: grids.map(gd => ({ label: 'from P' + gd, value: reveal ? byGrid[gd] : 0, color: P.amber, hl: reveal && gd === 1, text: reveal ? String(byGrid[gd]) : '?' })), max: Math.max(W.length * 0.8, 4), labelW: 80 }),
            evidence: { caption: 'The furthest back anyone won from in ' + y + ': ' + far.driver + ', ' + far.race + ', from P' + far.grid + '.',
                draw: (g, w, h) => chartBars(g, w, h, { items: W.slice().sort((a, b) => b.grid - a.grid).slice(0, 8).map(x => ({ label: x.race, value: x.grid, color: P.amber, hl: x === far, text: 'P' + x.grid + ' · ' + x.driver })), max: Math.max(far.grid + 2, 6), labelW: 100 }) },
            explain: '<b>' + fromPole + ' of ' + W.length + '</b> — ' + fmt(fromPole / W.length * 100) + '% of ' + y + ' races were won from pole. ' + (byGrid[2] ? byGrid[2] + ' came from P2' : '') + (byGrid[3] ? ', ' + byGrid[3] + ' from P3' : '') + '. The biggest comeback was <b>' + esc(far.driver) + '</b> at ' + esc(far.race) + ', from P' + far.grid + '.',
            takeaways: ['Qualifying matters because overtaking costs lap time: following another car in dirty air strips the front wing of load, so the chaser slides and cooks the tyres.',
                'Strategy is the main way to pass without overtaking. An undercut — stopping a lap earlier for fresh tyres — is worth more than a DRS zone at most circuits.'],
        };
    } });

    gens.push({ cat: 'strategy', level: 3, make() {
        const H = HIST(); if (!H) return null;
        const y = choice(seasonYears()), S = H.seasons[y];
        const poles = S.poles.filter(p => p.margin > 0);
        if (poles.length < 5) return null;
        const tight = poles.reduce((a, b) => (b.margin < a.margin ? b : a));
        const wide = poles.reduce((a, b) => (b.margin > a.margin ? b : a));
        const med = poles.slice().sort((a, b) => a.margin - b.margin)[Math.floor(poles.length / 2)].margin;
        const opts = shuffle([tight.margin.toFixed(3) + ' s', (tight.margin * 4 + 0.1).toFixed(3) + ' s', (tight.margin * 12 + 0.25).toFixed(3) + ' s']);
        const sorted = poles.slice().sort((a, b) => a.margin - b.margin);
        return {
            prompt: 'Over a full qualifying lap, what was the smallest gap between pole and second place in ' + y + '?',
            hint: 'A lap is 80–110 seconds. The cars are separated by less than the blink of an eye — literally.',
            options: opts, answer: tight.margin.toFixed(3) + ' s',
            draw: (g, w, h, reveal) => chartBars(g, w, h, { title: 'Pole margin over P2, every race of ' + y + (reveal ? '' : ' — which is smallest?'), items: sorted.map(p => ({ label: reveal ? p.race : '', value: p.margin, color: P.amber, hl: reveal && p === tight, text: reveal ? p.margin.toFixed(3) + ' s' : '' })), max: Math.max(wide.margin * 1.05, 0.5), labelW: reveal ? 100 : 10 }),
            evidence: { caption: tight.race + ' ' + y + ': ' + tight.driver + ' took pole by ' + tight.margin.toFixed(3) + ' s over ' + tight.second + '. At ' + fmt(300 / 3.6 * tight.margin, 1) + ' m of track at 300 km/h, that is less than a car length.',
                draw: (g, w, h) => chartBars(g, w, h, { items: [{ label: 'tightest · ' + tight.race, value: tight.margin, hl: true, color: P.amber, text: tight.margin.toFixed(3) + ' s · ' + tight.driver }, { label: 'season median', value: med, color: P.amber, text: med.toFixed(3) + ' s' }, { label: 'widest · ' + wide.race, value: wide.margin, color: P.amber, text: wide.margin.toFixed(3) + ' s · ' + wide.driver }], max: wide.margin * 1.1, labelW: 150 }) },
            explain: '<b>' + tight.margin.toFixed(3) + ' s</b> at ' + esc(tight.race) + ', where <b>' + esc(tight.driver) + '</b> beat ' + esc(tight.second) + ' to pole with a ' + U.fmtLap(tight.time) + '. Over a lap of that length it is a difference of about ' + fmt(tight.margin / tight.time * 100, 2) + '%. The season median margin was ' + med.toFixed(3) + ' s; the widest was ' + wide.margin.toFixed(3) + ' s at ' + esc(wide.race) + '.',
            takeaways: ['This is why telemetry matters. Nobody can feel three thousandths of a second. The delta trace on the engineer\'s screen can.',
                'Qualifying laps are set on the softest tyre with the engine in its highest mode. The gap in the race, with fuel and tyre management, is usually larger.'],
        };
    } });

    // ---- history --------------------------------------------------------
    gens.push({ cat: 'history', level: 1, make() {
        const H = HIST(); if (!H) return null;
        const ch = choice(H.champions);
        const names = Array.from(new Set(H.champions.map(c => c.driver)));
        const alts = pick(names.filter(n => n !== ch.driver), 2);
        const S = H.seasons[String(ch.year)];
        const pts = H.champions.map(c => c.points);
        return {
            prompt: 'Who won the ' + ch.year + ' Formula 1 drivers\' championship?',
            hint: ch.races + ' races that season. The champion scored ' + fmt(ch.points) + ' points.',
            options: shuffle([ch.driver].concat(alts)), answer: ch.driver,
            draw: (g, w, h, reveal) => S
                ? chartBars(g, w, h, { title: 'Final drivers\' standings, ' + ch.year + (reveal ? '' : ' (names hidden)'), items: S.standings.slice(0, 6).map((s, i) => ({ label: reveal ? s.driver : 'P' + (i + 1), value: s.points, color: '#c58bff', hl: reveal && i === 0, text: fmt(s.points) + ' pts · ' + s.wins + ' wins' })), labelW: 120 })
                : chartCurve(g, w, h, { series: [{ xs: H.champions.map(c => c.year), ys: pts, color: '#c58bff', name: 'champion\'s points' }], xfmt: v => String(Math.round(v)), xstep: 2, ymin: 0, points: reveal ? [{ x: ch.year, y: ch.points, label: ch.driver, color: P.ink }] : [{ x: ch.year, y: ch.points, color: P.ink3 }] }),
            evidence: { caption: 'Drivers\' titles since 2012. ' + ch.driver + ' has ' + H.champions.filter(c => c.driver === ch.driver).length + ' in this span; the years are the tell for which rule era each driver owned.',
                draw: (g, w, h) => { const t = {}; for (const c of H.champions) (t[c.driver] = t[c.driver] || []).push(c.year); const ds = Object.keys(t).sort((a, b) => t[b].length - t[a].length); chartBars(g, w, h, { items: ds.map(d => ({ label: d, value: t[d].length, color: '#c58bff', hl: d === ch.driver, text: t[d].length + (t[d].length === 1 ? ' title · ' : ' titles · ') + t[d].join(', ') })), max: 9, labelW: 120 }); } },
            explain: '<b>' + esc(ch.driver) + '</b>, with ' + fmt(ch.points) + ' points from ' + ch.races + ' races' + (S ? ', ' + S.standings[0].wins + ' wins and ' + S.standings[0].podiums + ' podiums' : '') + '. ' + (ch.team ? 'The constructors\' title went to <b>' + esc(ch.team) + '</b>' + (S && S.standings[0].team !== ch.team ? ' — a different team from the champion\'s, which is rare' : '') + '.' : ''),
            takeaways: ['Points: 25 for a win down to 1 for tenth, plus a point for the fastest lap between 2019 and 2024. A championship is won on Sundays, but it is usually decided by the bad Sundays.',
                'The sequence — Vettel, then Hamilton\'s run, then Verstappen, then Norris — tracks the rule changes: 2014 hybrid engines, 2022 ground-effect floors.'],
        };
    } });

    gens.push({ cat: 'history', level: 2, make() {
        const H = HIST(); if (!H) return null;
        const ch = choice(H.champions.filter(c => c.team));
        const teams = Array.from(new Set(H.champions.map(c => c.team).filter(Boolean)));
        const alts = pick(teams.filter(t => t !== ch.team), Math.min(2, teams.length - 1));
        if (alts.length < 2) return null;
        const runs = []; for (const c of H.champions) { const last = runs[runs.length - 1]; if (last && last.team === c.team) last.years.push(c.year); else runs.push({ team: c.team, years: [c.year] }); }
        return {
            prompt: 'Which team won the ' + ch.year + ' constructors\' championship?',
            hint: 'The constructors\' title counts both cars — it is the one the engineers care about.',
            options: shuffle([ch.team].concat(alts)), answer: ch.team,
            draw: (g, w, h, reveal) => chartBars(g, w, h, { title: 'Constructors\' champions since 2012' + (reveal ? '' : ' — which team took ' + ch.year + '?'), items: H.champions.filter(c => c.team).map(c => ({ label: String(c.year), value: c.team_points, color: '#c58bff', hl: c.year === ch.year, dim: !reveal, text: reveal ? c.team + ' · ' + fmt(c.team_points) : (c.year === ch.year ? '?' : c.team) })), labelW: 50 }),
            evidence: { caption: 'Title runs: how many seasons in a row each team held the constructors\' crown.',
                draw: (g, w, h) => chartBars(g, w, h, { items: runs.map(r => ({ label: r.team, value: r.years.length, color: '#c58bff', hl: r.years.includes(ch.year), text: r.years.length + (r.years.length === 1 ? ' season · ' : ' seasons · ') + r.years[0] + (r.years.length > 1 ? '–' + r.years[r.years.length - 1] : '') })), max: 9, labelW: 90 }) },
            explain: '<b>' + esc(ch.team) + '</b>, with ' + fmt(ch.team_points) + ' points. The drivers\' champion that year was ' + esc(ch.driver) + (H.seasons[String(ch.year)] && H.seasons[String(ch.year)].standings[0].team !== ch.team ? ', driving for a different team — the constructors\' title rewards two consistent cars over one exceptional driver' : '') + '.',
            takeaways: ['Constructors\' position sets prize money and, since 2021, wind-tunnel time: the champion gets the least aerodynamic testing the following year.',
                'Mercedes\' eight straight titles (2014–2021) is the longest run in F1 history. It started with the hybrid engine rules.'],
            link: { href: '#cfd', label: 'See the sliding-scale wind-tunnel allocation' },
        };
    } });

    gens.push({ cat: 'history', level: 3, make() {
        const H = HIST(); if (!H) return null;
        const split = H.champions.filter(c => c.team && H.seasons[String(c.year)] && H.seasons[String(c.year)].standings[0].team !== c.team);
        if (!split.length) return null;
        const ch = choice(split);
        const same = H.champions.filter(c => c.team && !split.includes(c));
        const alts = pick(same, 2).map(c => String(c.year));
        return {
            prompt: 'In which of these seasons did the drivers\' champion NOT drive for the constructors\' champion team?',
            hint: 'It takes a dominant driver in the second-best car, or a team with two drivers taking points off each other.',
            options: shuffle([String(ch.year)].concat(alts)), answer: String(ch.year),
            draw: (g, w, h, reveal) => chartBars(g, w, h, { title: 'Champion driver and champion team by season', items: H.champions.filter(c => c.team).map(c => ({ label: String(c.year), value: 1, color: split.includes(c) ? P.red : '#c58bff', hl: reveal && c.year === ch.year, dim: !reveal || split.includes(c) === false, text: c.driver + '  ·  ' + (reveal || !split.includes(c) ? c.team : '?') })), max: 1.02, labelW: 50 }),
            evidence: { caption: ch.year + ': ' + ch.driver + ' won the drivers\' title with ' + fmt(ch.points) + ' points while ' + ch.team + ' took the constructors\' title with ' + fmt(ch.team_points) + '.',
                draw: (g, w, h) => { const S = H.seasons[String(ch.year)]; chartBars(g, w, h, { items: S.standings.slice(0, 6).map((s, i) => ({ label: s.driver, value: s.points, color: s.team === ch.team ? '#c58bff' : P.red, hl: i === 0 || s.team === ch.team, text: fmt(s.points) + ' · ' + s.team })), labelW: 120 }); } },
            explain: '<b>' + ch.year + '.</b> ' + esc(ch.driver) + ' took the drivers\' championship, but <b>' + esc(ch.team) + '</b> won the constructors\' with ' + fmt(ch.team_points) + ' points — two strong drivers out-scoring one dominant one plus a team-mate. It has happened ' + split.length + ' time' + (split.length > 1 ? 's' : '') + ' since 2012: ' + split.map(c => c.year).join(', ') + '.',
            takeaways: ['The constructors\' championship is decided by the second driver. That is why teams agonise over their number-two seat.',
                'A split title year almost always means the fastest car was not the most reliable one, or the fastest driver was not in it.'],
        };
    } });

    gens.push({ cat: 'history', level: 1, make() {
        const H = HIST(); if (!H) return null;
        const ch = choice(H.champions.filter(c => c.races));
        const opts = shuffle([ch.races, ch.races - 6, ch.races + 6].map(String));
        const minY = H.champions.reduce((a, b) => (b.races < a.races ? b : a));
        return {
            prompt: 'How many Grands Prix were on the ' + ch.year + ' calendar?',
            hint: 'The season has grown steadily — with one very obvious exception.',
            options: opts.map(o => o + ' races'), answer: ch.races + ' races',
            draw: (g, w, h, reveal) => chartBars(g, w, h, { title: 'Races per season', items: H.champions.map(c => ({ label: String(c.year), value: reveal || c.year !== ch.year ? c.races : 0, color: '#c58bff', hl: c.year === ch.year, text: reveal || c.year !== ch.year ? String(c.races) : '?' })), max: 27, labelW: 50 }),
            evidence: { caption: 'Points needed to be champion, per race. More races and a bigger points haul, but a champion still needs to average about a podium a weekend.',
                draw: (g, w, h) => chartCurve(g, w, h, { series: [{ xs: H.champions.map(c => c.year), ys: H.champions.map(c => c.points / c.races), color: '#c58bff' }], xfmt: v => String(Math.round(v)), xstep: 2, ymin: 10, ymax: 26, ylabel: 'champion pts / race', hlines: [{ y: 25, label: 'a win every race', color: P.ink3 }, { y: 18, label: 'second every race' }], points: [{ x: ch.year, y: ch.points / ch.races, label: (ch.points / ch.races).toFixed(1) + ' per race', color: P.ink }] }) },
            explain: '<b>' + ch.races + ' races</b> in ' + ch.year + '. ' + esc(ch.driver) + ' averaged ' + (ch.points / ch.races).toFixed(1) + ' points a weekend on the way to the title. The shortest season in this data is ' + minY.year + ' with ' + minY.races + ' races — the pandemic year — and the longest is 24.',
            takeaways: ['A 24-race season means roughly 30 flyaway trips for the mechanics. The calendar is now the biggest constraint on how many people a team can keep.',
                'More races also means more data. The telemetry archive behind this page grows by about 1.5 TB per team per season.'],
        };
    } });

    gens.push({ cat: 'history', level: 2, make() {
        const H = HIST(); if (!H) return null;
        const y = choice(seasonYears()), S = H.seasons[y];
        const st = S.standings.slice().sort((a, b) => b.wins - a.wins);
        if (st[0].wins === st[1].wins) return null;
        const top = st[0];
        const alts = pick(st.slice(1, 6), 2).map(s => s.driver);
        const wins = {}; for (const w of S.winners) wins[w.driver] = (wins[w.driver] || 0) + 1;
        const winners = Object.keys(wins).sort((a, b) => wins[b] - wins[a]);
        return {
            prompt: 'Who won the most races in ' + y + '?',
            hint: S.winners.length + ' races, ' + winners.length + ' different winners.',
            options: shuffle([top.driver].concat(alts)), answer: top.driver,
            draw: (g, w, h, reveal) => chartBars(g, w, h, { title: 'Race wins, ' + y + (reveal ? '' : ' (names hidden)'), items: winners.map((d, i) => ({ label: reveal ? d : 'Driver ' + String.fromCharCode(65 + i), value: wins[d], color: '#c58bff', hl: reveal && d === top.driver, text: String(wins[d]) })), max: Math.max(wins[winners[0]] + 2, 6), labelW: 120 }),
            evidence: { caption: 'Wins against podiums against points for the top of the ' + y + ' table. Wins get headlines; podiums win championships.',
                draw: (g, w, h) => chartBars(g, w, h, { items: S.standings.slice(0, 5).flatMap(s => [{ label: s.driver + ' · wins', value: s.wins, color: '#c58bff', hl: s.driver === top.driver, text: String(s.wins) }, { label: s.driver + ' · podiums', value: s.podiums, color: P.teal, dim: true, text: String(s.podiums) }]), max: Math.max(...S.standings.slice(0, 5).map(s => s.podiums)) + 2, labelW: 160 }) },
            explain: '<b>' + esc(top.driver) + '</b> with ' + top.wins + ' wins' + (winners[1] ? ', ahead of ' + esc(winners[1]) + ' on ' + wins[winners[1]] : '') + '. ' + (S.standings[0].driver === top.driver ? 'That was also enough for the title, with ' + fmt(top.points) + ' points.' : 'Yet the championship went to ' + esc(S.standings[0].driver) + ' — consistency beat peak performance.'),
            takeaways: [winners.length + ' different drivers won a race in ' + y + '. Depth of competition is the health metric the sport watches most closely.',
                'The fastest lap of the race went to a different driver ' + new Set(S.fastest.map(f => f.driver)).size + ' times that season — it is often set late on fresh tyres by someone with nothing to lose.'],
        };
    } });

    // ---- fundamentals ---------------------------------------------------
    gens.push({ cat: 'basics', level: 1, make() {
        if (!AERO()) return null;
        const s = SETUPS.silverstone;
        const xs = []; for (let v = 60; v <= 320; v += 10) xs.push(v);
        const latG = v => 1.8 * (9.81 + solve(s, v).downforce / CAR_MASS) / 9.81;   // TYRE_MU 1.8, as in windtunnel.js
        const noAero = 1.8;
        const v = choice([200, 250, 300]);
        return {
            prompt: 'Why does an F1 car carry wings and a shaped floor at all?',
            hint: 'The tyres can only push as hard as they are pushed onto the road.',
            options: shuffle(['To press the car onto the track for more grip', 'To make the car lighter', 'To keep the engine cool']), answer: 'To press the car onto the track for more grip',
            draw: (g, w, h, reveal) => chartCurve(g, w, h, { series: [{ xs, ys: xs.map(latG), color: '#7aa2ff', name: 'with aero' }, { xs, ys: xs.map(() => noAero), color: P.ink3, name: 'tyre grip alone', dash: [4, 4] }], legend: true, xlabel: 'km/h', ylabel: 'cornering g', ymin: 0, ymax: 6, nx: 6, points: reveal ? [{ x: v, y: latG(v), label: latG(v).toFixed(1) + ' g at ' + v, color: P.ink }] : [] }),
            evidence: { caption: 'Cornering grip at ' + v + ' km/h with and without downforce. A road car manages about 1 g; a slick tyre alone about 1.8 g.',
                draw: (g, w, h) => chartBars(g, w, h, { items: [{ label: 'road car', value: 1.0, color: '#868f9e', text: '1.0 g' }, { label: 'slick tyre, no wings', value: noAero, color: '#868f9e', text: noAero.toFixed(1) + ' g' }, { label: 'F1 car @ ' + v + ' km/h', value: latG(v), hl: true, color: '#7aa2ff', text: latG(v).toFixed(1) + ' g' }], max: 6, labelW: 140 }) },
            explain: '<b>For grip.</b> A tyre\'s grip is roughly its friction coefficient times the load pressing it down. Wings and the floor add load without adding mass, so at ' + v + ' km/h the car can corner at about <b>' + latG(v).toFixed(1) + ' g</b> instead of the ' + noAero.toFixed(1) + ' g the tyre could manage on its own. Downforce does not make the car heavier in the sense that matters for acceleration — it only presses harder when there is airflow to press with.',
            takeaways: ['Everything in the wind tunnel section is in service of this one curve. More load, less drag, and the balance between front and rear axles.',
                'It is also why F1 cars are so slow in the wet: the tyres lose grip, but the downforce that multiplies it does not care about rain.'],
            link: { href: '#gforce', label: 'See the g-force maps drivers actually experience' },
        };
    } });

    gens.push({ cat: 'basics', level: 1, make() {
        if (!AERO()) return null;
        const s = SETUPS.silverstone;
        const off = solve(s, 250, false), on = solve(s, 250, true);
        const vOff = vmaxOf(s, false), vOn = vmaxOf(s, true);
        const c = CIRC().find(x => x.key === 'monza') || choice(CIRC());
        const L = c.laps[0];
        return {
            prompt: 'DRS — the Drag Reduction System — is the flap that opens on the rear wing. What does opening it actually do?',
            hint: 'It is only allowed on designated straights when a car is within one second of the car ahead.',
            options: shuffle(['Flattens the rear wing to cut drag and raise top speed', 'Adds extra power from the hybrid battery', 'Lowers the car onto the track for more grip']), answer: 'Flattens the rear wing to cut drag and raise top speed',
            draw: (g, w, h, reveal) => chartBars(g, w, h, { title: 'Rear wing with DRS closed and open, 250 km/h', items: [{ label: 'drag · closed', value: off.sCx, color: P.red, text: off.sCx.toFixed(2) + ' m²' }, { label: 'drag · open', value: reveal ? on.sCx : 0, color: P.red, hl: reveal, text: reveal ? on.sCx.toFixed(2) + ' m²  (−' + fmt((1 - on.sCx / off.sCx) * 100) + '%)' : '?' }, { label: 'downforce · closed', value: off.sCz, color: P.teal, text: off.sCz.toFixed(2) + ' m²' }, { label: 'downforce · open', value: reveal ? on.sCz : 0, color: P.teal, dim: true, text: reveal ? on.sCz.toFixed(2) + ' m²  (−' + fmt((1 - on.sCz / off.sCz) * 100) + '%)' : '?' }], max: 4.4, labelW: 130 }),
            evidence: { caption: 'Top speed with the flap closed and open on the baseline setup, and where DRS is used on the ' + c.label + ' lap: the flat plateaus.',
                draw: (g, w, h) => {
                    g.save(); chartBars(g, w * 0.5, h, { items: [{ label: 'DRS closed', value: vOff, color: '#7aa2ff', text: fmt(vOff) + ' km/h' }, { label: 'DRS open', value: vOn, hl: true, color: '#7aa2ff', text: fmt(vOn) + ' km/h  (+' + fmt(vOn - vOff) + ')' }], min: 250, max: 400, labelW: 90 }); g.restore();
                    g.save(); g.translate(w * 0.5, 0); chartTrace(g, w * 0.5, h, { series: [{ y: L.speed, color: traceColor, thin: true }], ymax: 360 }); g.restore();
                } },
            explain: '<b>It cuts drag.</b> Opening the flap dumps most of the rear wing\'s load — the model drops the wing\'s downforce by 65% and its drag by 70%, which takes total drag down ' + fmt((1 - on.sCx / off.sCx) * 100) + '% and is worth about <b>' + fmt(vOn - vOff) + ' km/h</b> of top speed on the baseline setup. The lost downforce does not matter on a straight, which is the only place it is allowed.',
            takeaways: ['DRS exists because following another car in its wake strips your front wing of load. The flap gives the chaser back roughly what the dirty air took away.',
                'The lap-time model above uses the same DRS effect: rear-wing load ×0.35 and drag ×0.30 in the activation zones.'],
            link: { href: '#aero', label: 'Toggle DRS in the wind tunnel' },
        };
    } });

    gens.push({ cat: 'basics', level: 1, make() {
        const H = HIST(); if (!H) return null;
        const y = choice(seasonYears()), S = H.seasons[y];
        const poleWins = S.winners.filter(w => w.grid === 1).length;
        const p = choice(S.poles);
        return {
            prompt: 'Pole position — first place on the starting grid — goes to the driver who…',
            hint: 'Saturday decides where you start. Sunday decides where you finish.',
            options: shuffle(['Sets the fastest single lap in qualifying', 'Won the previous race', 'Leads the championship']), answer: 'Sets the fastest single lap in qualifying',
            draw: (g, w, h, reveal) => chartBars(g, w, h, { title: 'Qualifying margins to P2, ' + y + (reveal ? '' : ' — pole is a lap time, not a race result'), items: S.poles.slice(0, 12).map(x => ({ label: x.race, value: x.margin, color: P.amber, hl: reveal && x === p, text: reveal ? '+' + x.margin.toFixed(3) + ' s · ' + x.driver : '+' + x.margin.toFixed(3) + ' s' })), labelW: 100 }),
            evidence: { caption: 'Pole converted to a win ' + poleWins + ' times in ' + S.winners.length + ' races in ' + y + '. Starting first is the best predictor of finishing first, but it is not a guarantee.',
                draw: (g, w, h) => chartBars(g, w, h, { items: [{ label: 'won from pole', value: poleWins, hl: true, color: P.amber, text: poleWins + ' of ' + S.winners.length }, { label: 'won from elsewhere', value: S.winners.length - poleWins, color: '#868f9e', text: String(S.winners.length - poleWins) }], max: S.winners.length, labelW: 130 }) },
            explain: '<b>The fastest single lap in qualifying.</b> Three knockout sessions on Saturday: Q1 eliminates five cars, Q2 five more, and the last ten fight for pole in Q3. At ' + esc(p.race) + ' ' + y + ', ' + esc(p.driver) + ' took pole with a ' + U.fmtLap(p.time) + ', ' + p.margin.toFixed(3) + ' s clear of ' + esc(p.second) + '. That is one flying lap with the fuel load, engine mode and tyre chosen for a single lap — which is why the telemetry on this page comes from qualifying.',
            takeaways: ['A qualifying lap is the purest telemetry there is: minimum fuel, maximum engine mode, soft tyres, no traffic. Race laps are 3–5 s slower.',
                'Sprint weekends add a second, shorter qualifying on Friday for a 100 km race on Saturday. Grand Prix pole still comes from the main session.'],
        };
    } });

    gens.push({ cat: 'basics', level: 2, make() {
        const c = choice(CIRC()), L = c.laps[0];
        const gmax = L.gear_max, gmin = L.gear_min;
        const n = L.gear.length;
        const iMin = minIdx(L.speed);
        const gAtMin = L.gear[iMin];
        const opts = shuffle([String(gAtMin), String(Math.min(8, gAtMin + 3)), String(Math.max(1, gAtMin - 2))]);
        if (new Set(opts).size < 3) return null;
        const mi = Math.round(iMin / n * (c.map.x.length - 1));
        return {
            prompt: 'Speed and gear for a ' + c.label + ' lap. What gear is the car in through its slowest corner?',
            hint: 'F1 cars have eight forward gears. First is almost only used for the start — and the very tightest hairpins.',
            options: opts.map(o => o + (o === '1' ? 'st' : o === '2' ? 'nd' : o === '3' ? 'rd' : 'th') + ' gear'), answer: String(gAtMin) + (gAtMin === 1 ? 'st' : gAtMin === 2 ? 'nd' : gAtMin === 3 ? 'rd' : 'th') + ' gear',
            legend: [{ name: 'speed', color: traceColor }, { name: 'gear (×40)', color: '#c58bff' }],
            draw: (g, w, h, reveal) => chartTrace(g, w, h, { series: [{ y: L.speed, color: traceColor }, { y: L.gear.map(v => v * 40), color: '#c58bff', thin: true }], ymax: 360, marks: reveal ? [{ i: iMin, label: String(gAtMin), color: P.ink }] : [{ i: iMin, label: '?', color: P.ink }] }),
            evidence: { caption: 'Where the slowest corner sits at ' + c.label + '. Gears run from ' + gmin + ' to ' + gmax + ' on this lap; the ratios are chosen once per season for each circuit\'s top speed.',
                draw: (g, w, h) => chartMap(g, w, h, { map: c.map, title: c.name, marks: [{ i: mi, color: P.ink, label: String(gAtMin) }], stats: [['min speed', fmt(L.speed[iMin]) + ' km/h'], ['gear there', String(gAtMin)], ['top gear used', String(gmax)], ['top speed', fmt(L.top) + ' km/h']] }) },
            explain: '<b>' + gAtMin + (gAtMin === 1 ? 'st' : gAtMin === 2 ? 'nd' : gAtMin === 3 ? 'rd' : 'th') + ' gear</b>, at ' + fmt(L.speed[iMin]) + ' km/h. The gear trace is a staircase: each step down is a downshift under braking, each step up is an upshift on the way out. Drivers use the gear to keep the engine in its power band and, on the way in, to add engine braking.',
            takeaways: ['Gear ratios are fixed for the whole season under the current rules, so a car geared for Monza\'s straights runs the same eight ratios at Monaco.',
                'Engineers watch the gear channel against the throttle channel: an early upshift on corner exit usually means the rear tyres were about to spin.'],
        };
    } });

    // ------------------------------------------------------------------
    // set assembly: 10 questions, balanced across categories and levels
    // ------------------------------------------------------------------
    function buildQuestions() {
        const byCat = {};
        for (const g of gens) (byCat[g.cat] = byCat[g.cat] || []).push(g);
        const cats = Object.keys(CATS);
        const perCat = Math.floor(COUNT / cats.length);        // 2 each
        let extra = COUNT - perCat * cats.length;
        const chosen = [];
        // a generator that throws must never take the whole set down with it
        const attempt = gen => { for (let t = 0; t < 4; t++) { try { const q = gen.make(); if (q) return q; } catch (e) { console.warn('[quiz] generator failed:', gen.cat, gen.level, e.message); return null; } } return null; };
        // aim for roughly 3 rookie / 4 engineer / 3 pit-wall questions per set
        const target = { 1: 3, 2: 4, 3: 3 }, count = { 1: 0, 2: 0, 3: 0 };
        for (const cat of shuffle(cats.slice())) {
            const want = perCat + (extra > 0 ? 1 : 0); if (extra > 0) extra--;
            const pool = shuffle((byCat[cat] || []).slice());
            const usedLevels = new Set();
            let got = 0;
            while (got < want && pool.length) {
                // different levels within a category, and levels the set is short of
                pool.sort((a, b) => ((usedLevels.has(a.level) ? 10 : 0) + count[a.level] - target[a.level]) -
                                    ((usedLevels.has(b.level) ? 10 : 0) + count[b.level] - target[b.level]));
                const gen = pool.shift();
                const q = attempt(gen);
                if (!q) continue;
                q.cat = cat; q.level = gen.level; q.pts = LEVELS[gen.level].pts;
                chosen.push(q); usedLevels.add(gen.level); count[gen.level]++; got++;
            }
        }
        // fill any shortfall from anywhere
        let guard = 0;
        while (chosen.length < COUNT && guard++ < 40) {
            const gen = choice(gens);
            const q = attempt(gen);
            if (q) { q.cat = gen.cat; q.level = gen.level; q.pts = LEVELS[gen.level].pts; chosen.push(q); }
        }
        // rookie questions first, pit wall last, shuffled within a level
        const byLevel = { 1: [], 2: [], 3: [] };
        for (const q of chosen.slice(0, COUNT)) byLevel[q.level].push(q);
        const out = shuffle(byLevel[1]).concat(shuffle(byLevel[2]), shuffle(byLevel[3]));
        maxScore = out.reduce((a, q) => a + q.pts, 0);
        return out;
    }

    // ------------------------------------------------------------------
    // drawing into the two canvases
    // ------------------------------------------------------------------
    function drawMain(q, reveal) {
        const canvas = root.querySelector('.quiz-canvas');
        if (!canvas || !canvas.getBoundingClientRect().width) return;
        const narrow = canvas.getBoundingClientRect().width < 480;
        const { g, w, h } = U.setupCanvas(canvas, narrow ? 1.7 : 2.35);
        g.clearRect(0, 0, w, h);
        q.draw(g, w, h, reveal);
    }
    function drawEvidence(q) {
        const canvas = root.querySelector('.quiz-evidence-canvas');
        if (!canvas || !q.evidence || !canvas.getBoundingClientRect().width) return;
        const narrow = canvas.getBoundingClientRect().width < 480;
        const { g, w, h } = U.setupCanvas(canvas, narrow ? 1.5 : 2.6);
        g.clearRect(0, 0, w, h);
        q.evidence.draw(g, w, h);
    }

    // ------------------------------------------------------------------
    // UI
    // ------------------------------------------------------------------
    function build() {
        root.innerHTML =
            '<div class="quiz-head">' +
            '<div><span class="quiz-kicker">Read the race</span>' +
            '<h3 class="quiz-title">Ten questions. A fresh set every time.</h3>' +
            '<p class="quiz-desc">Telemetry, aerodynamics, race craft, history and the fundamentals — every question is generated ' +
            'from the data on this page, and every answer comes with the evidence.</p></div>' +
            '<div class="quiz-progress" aria-label="Progress"></div></div>' +
            '<div class="quiz-body">' +
            '<div class="quiz-stage-col">' +
            '<div class="quiz-stage"><canvas class="quiz-canvas" aria-label="Question figure"></canvas><div class="quiz-legend"></div></div>' +
            '<div class="quiz-evidence" hidden><div class="quiz-evidence-head"><span class="quiz-evidence-kicker">Evidence</span><span class="quiz-evidence-cat"></span></div>' +
            '<canvas class="quiz-evidence-canvas" aria-label="Supporting figure"></canvas><p class="quiz-evidence-cap"></p></div>' +
            '</div>' +
            '<div class="quiz-panel">' +
            '<div class="quiz-meta"><span class="quiz-badge"></span><span class="quiz-level" aria-label="Difficulty"></span></div>' +
            '<p class="quiz-prompt"></p><p class="quiz-hint"></p>' +
            '<div class="quiz-options" role="group" aria-label="Answers"></div>' +
            '<div class="quiz-reveal" hidden><div class="quiz-verdict"></div><p class="quiz-explain"></p>' +
            '<div class="quiz-takeaways"></div><a class="quiz-link" hidden></a>' +
            '<button type="button" class="quiz-next">Next question <span aria-hidden="true">&rarr;</span></button></div>' +
            '<div class="quiz-summary" hidden><div class="quiz-summary-top"><span class="quiz-score"></span><span class="quiz-tier"></span></div>' +
            '<p class="quiz-summary-text"></p><div class="quiz-breakdown"></div>' +
            '<button type="button" class="quiz-restart">New set of ten</button></div>' +
            '</div></div>';

        root.querySelector('.quiz-next').addEventListener('click', next);
        root.querySelector('.quiz-restart').addEventListener('click', restart);
        window.addEventListener('resize', () => { const q = questions[index]; if (q) { drawMain(q, answered); if (answered) drawEvidence(q); } }, { passive: true });
    }

    function renderProgress() {
        const el = root.querySelector('.quiz-progress');
        el.innerHTML = questions.map((q, i) =>
            '<span class="quiz-dot' + (i < index ? (q.correct ? ' right' : ' wrong') : i === index ? ' now' : '') + '" style="--dot:' + CATS[q.cat].color + '" title="' + CATS[q.cat].label + ' · ' + LEVELS[q.level].label + '"></span>').join('') +
            '<span class="quiz-count">' + Math.min(index + 1, COUNT) + ' / ' + COUNT + '</span>';
    }

    function show() {
        const q = questions[index];
        answered = false;
        root.querySelector('.quiz-reveal').hidden = true;
        root.querySelector('.quiz-summary').hidden = true;
        root.querySelector('.quiz-evidence').hidden = true;
        root.querySelector('.quiz-options').hidden = false;
        const badge = root.querySelector('.quiz-badge');
        badge.textContent = CATS[q.cat].label; badge.style.setProperty('--cat', CATS[q.cat].color);
        root.querySelector('.quiz-level').innerHTML = '<span class="quiz-level-name">' + LEVELS[q.level].label + '</span>' +
            [1, 2, 3].map(l => '<i class="' + (l <= q.level ? 'on' : '') + '"></i>').join('') + '<span class="quiz-pts">' + q.pts + ' pt' + (q.pts > 1 ? 's' : '') + '</span>';
        root.querySelector('.quiz-prompt').textContent = q.prompt;
        root.querySelector('.quiz-hint').textContent = q.hint || '';
        root.querySelector('.quiz-legend').innerHTML = (q.legend || []).map(t =>
            '<span class="quiz-legend-item"><i style="background:' + t.color + '"></i>' + t.name + '</span>').join('');
        const opts = root.querySelector('.quiz-options');
        opts.innerHTML = '';
        for (const o of q.options) {
            const b = document.createElement('button');
            b.type = 'button'; b.className = 'quiz-option'; b.textContent = o;
            b.addEventListener('click', () => answer(o, b));
            opts.appendChild(b);
        }
        renderProgress();
        drawMain(q, false);
    }

    function answer(choiceText, btn) {
        if (answered) return;
        answered = true;
        const q = questions[index];
        q.correct = choiceText === q.answer;
        if (q.correct) score += q.pts;
        root.querySelectorAll('.quiz-option').forEach(b => {
            b.disabled = true;
            if (b.textContent === q.answer) b.classList.add('right');
            else if (b === btn) b.classList.add('wrong');
        });
        const verdict = root.querySelector('.quiz-verdict');
        verdict.innerHTML = (q.correct ? 'Correct' : 'Not quite') + '<span class="quiz-verdict-pts">' + (q.correct ? '+' + q.pts : '0') + ' / ' + q.pts + '</span>';
        verdict.className = 'quiz-verdict ' + (q.correct ? 'right' : 'wrong');
        root.querySelector('.quiz-explain').innerHTML = q.explain;
        root.querySelector('.quiz-takeaways').innerHTML = (q.takeaways || []).map(t => '<div class="quiz-takeaway"><span>Also worth knowing</span><p>' + t + '</p></div>').join('');
        const link = root.querySelector('.quiz-link');
        if (q.link) { link.href = q.link.href; link.textContent = q.link.label + ' →'; link.hidden = false; } else link.hidden = true;
        root.querySelector('.quiz-next').textContent = index === COUNT - 1 ? 'See your result' : 'Next question →';
        root.querySelector('.quiz-reveal').hidden = false;
        renderProgress();
        drawMain(q, true);
        if (q.evidence) {
            const ev = root.querySelector('.quiz-evidence');
            ev.hidden = false;
            root.querySelector('.quiz-evidence-cat').textContent = CATS[q.cat].label + ' · ' + CATS[q.cat].blurb;
            root.querySelector('.quiz-evidence-cap').innerHTML = q.evidence.caption;
            ev.classList.remove('in'); void ev.offsetWidth; ev.classList.add('in');
            drawEvidence(q);
        }
        if (typeof window.gtag === 'function') {
            window.gtag('event', 'trace_quiz_answer', { control: q.cat + '_' + q.level, value_label: q.correct ? 'right' : 'wrong' });
        }
    }

    function next() {
        if (index < COUNT - 1) { index++; show(); root.scrollIntoView({ block: 'nearest', behavior: 'smooth' }); return; }
        // summary
        root.querySelector('.quiz-reveal').hidden = true;
        root.querySelector('.quiz-evidence').hidden = true;
        root.querySelector('.quiz-options').innerHTML = '';
        root.querySelector('.quiz-prompt').textContent = 'That is the set.';
        root.querySelector('.quiz-hint').textContent = 'Every question is regenerated — a new set will not repeat this one.';
        const pct = score / maxScore;
        const tier = pct === 1 ? ['Pit wall', 'Every call right, including the hard ones. You could brief a driver.']
            : pct >= 0.75 ? ['Race engineer', 'You read the data the way the garage does. The misses were on the fine detail — run a new set and go for the Pit wall.']
            : pct >= 0.45 ? ['Garage', 'You have the shapes and the ideas. The numbers — how much, how fast, how many — are what separate a fan from an engineer.']
            : ['Grandstand', 'Every answer came with its evidence. Go back through the wind tunnel and the lap simulator and the traces will start to speak.'];
        root.querySelector('.quiz-score').textContent = score + ' / ' + maxScore;
        root.querySelector('.quiz-tier').textContent = tier[0];
        root.querySelector('.quiz-summary-text').textContent = tier[1];
        const bd = {};
        for (const q of questions) { const b = bd[q.cat] = bd[q.cat] || { got: 0, max: 0 }; b.max += q.pts; if (q.correct) b.got += q.pts; }
        root.querySelector('.quiz-breakdown').innerHTML = Object.keys(bd).map(cat =>
            '<div class="quiz-bd-row"><span class="quiz-bd-label" style="color:' + CATS[cat].color + '">' + CATS[cat].label + '</span>' +
            '<span class="quiz-bd-bar"><i style="width:' + Math.round(bd[cat].got / bd[cat].max * 100) + '%;background:' + CATS[cat].color + '"></i></span>' +
            '<span class="quiz-bd-val">' + bd[cat].got + '/' + bd[cat].max + '</span></div>').join('');
        // keep the last figure on the stage as a memento, but hide the legend
        root.querySelector('.quiz-legend').innerHTML = '';
        root.querySelector('.quiz-summary').hidden = false;
        index = COUNT;
        renderProgress();
        if (typeof window.gtag === 'function') {
            window.gtag('event', 'trace_quiz_complete', { control: 'score', value_label: String(score) + '_of_' + maxScore });
        }
    }

    function restart() {
        questions = buildQuestions(); index = 0; score = 0;
        show();
    }

    function init() {
        Promise.all([
            U.loadJSON('data/web/quiz.json'),
            U.loadJSON('data/web/tracks.json').catch(() => null),
        ]).then(([quiz, tracks]) => {
            const list = Array.isArray(quiz) ? quiz : quiz.circuits;
            D.circuits = (list || []).filter(c => c.laps && c.laps.length >= 2 && c.map);
            D.history = Array.isArray(quiz) ? null : quiz.history;
            D.tracks = tracks;
            if (D.circuits.length < 3) throw new Error('not enough circuits');
            build();
            questions = buildQuestions();
            U.whenVisible(root, show);
        }).catch(err => {
            console.warn('[quiz] unavailable:', err.message);
            root.hidden = true;
        });
    }

    if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', init);
    else init();
})();
