/* ============================================================
   F1 DATA ANALYTICS - SHARED UI HELPERS
   Small things every interactive module needs: crisp canvases,
   palette access, number formatting, and a visibility hook that
   does not depend on IntersectionObserver ever firing.
   ============================================================ */

(function () {
    'use strict';

    const css = getComputedStyle(document.documentElement);
    const token = name => css.getPropertyValue(name).trim();

    const palette = {
        bg: token('--bg-screen') || '#0a0a0f',
        card: token('--bg-card') || '#1e222b',
        line: 'rgba(255,255,255,0.09)',
        lineStrong: 'rgba(255,255,255,0.18)',
        ink: token('--text-primary') || '#f5f7fa',
        ink2: token('--text-secondary') || '#adb6c4',
        ink3: token('--text-muted') || '#868f9e',
        red: token('--accent-primary') || '#e10600',
        teal: token('--accent-secondary') || '#00d2be',
        amber: token('--accent-tertiary') || '#ff8700',
        yellow: token('--accent-yellow') || '#fff200',
    };

    /** Size a canvas to its CSS box at device resolution. Returns the 2D
     *  context already scaled, plus the CSS-pixel width and height. */
    function setupCanvas(canvas, ratio) {
        const rect = canvas.getBoundingClientRect();
        const w = Math.max(1, Math.round(rect.width));
        const h = Math.max(1, Math.round(ratio ? rect.width / ratio : rect.height));
        const dpr = Math.min(window.devicePixelRatio || 1, 2);
        if (ratio) canvas.style.height = h + 'px';
        canvas.width = Math.round(w * dpr);
        canvas.height = Math.round(h * dpr);
        const g = canvas.getContext('2d');
        g.setTransform(dpr, 0, 0, dpr, 0, 0);
        return { g, w, h };
    }

    function fmtLap(seconds) {
        if (!isFinite(seconds)) return '--:--.---';
        const m = Math.floor(seconds / 60);
        const s = seconds - m * 60;
        return m + ':' + s.toFixed(3).padStart(6, '0');
    }

    function fmtDelta(seconds, digits) {
        if (!isFinite(seconds)) return '--.---';
        const d = digits === undefined ? 3 : digits;
        return (seconds >= 0 ? '+' : '−') + Math.abs(seconds).toFixed(d);
    }

    const clamp = (v, lo, hi) => (v < lo ? lo : v > hi ? hi : v);
    const lerp = (a, b, t) => a + (b - a) * t;

    /** Linear interpolation of y at x over sorted xs. */
    function interp(x, xs, ys) {
        const n = xs.length;
        if (x <= xs[0]) return ys[0];
        if (x >= xs[n - 1]) return ys[n - 1];
        let lo = 0, hi = n - 1;
        while (hi - lo > 1) {
            const mid = (lo + hi) >> 1;
            if (xs[mid] <= x) lo = mid; else hi = mid;
        }
        const t = (x - xs[lo]) / (xs[hi] - xs[lo] || 1);
        return ys[lo] + (ys[hi] - ys[lo]) * t;
    }

    /** Index of the last xs <= x. */
    function indexAt(x, xs) {
        let lo = 0, hi = xs.length - 1;
        if (x <= xs[0]) return 0;
        if (x >= xs[hi]) return hi;
        while (hi - lo > 1) {
            const mid = (lo + hi) >> 1;
            if (xs[mid] <= x) lo = mid; else hi = mid;
        }
        return lo;
    }

    /** Speed ramp used across the site: deep blue -> teal -> white. */
    function speedColor(t) {
        t = clamp(t, 0, 1);
        const stops = [
            [0.00, [18, 18, 74]], [0.25, [27, 74, 143]], [0.50, [0, 160, 192]],
            [0.72, [0, 210, 190]], [0.90, [200, 245, 238]], [1.00, [255, 255, 255]],
        ];
        for (let i = 1; i < stops.length; i++) {
            if (t <= stops[i][0]) {
                const [t0, c0] = stops[i - 1], [t1, c1] = stops[i];
                const k = (t - t0) / (t1 - t0 || 1);
                return 'rgb(' + c0.map((v, j) => Math.round(lerp(v, c1[j], k))).join(',') + ')';
            }
        }
        return 'rgb(255,255,255)';
    }

    function hexToRgba(hex, a) {
        const h = hex.replace('#', '');
        const r = parseInt(h.slice(0, 2), 16), g = parseInt(h.slice(2, 4), 16), b = parseInt(h.slice(4, 6), 16);
        return 'rgba(' + r + ',' + g + ',' + b + ',' + a + ')';
    }

    /** Run cb once when el is on screen. Uses the observer when it works and a
     *  geometry check as backup, because some embedded renderers never fire it. */
    function whenVisible(el, cb) {
        let done = false;
        const fire = () => { if (!done) { done = true; cb(); } };
        const check = () => {
            const r = el.getBoundingClientRect();
            if (r.height > 0 && r.bottom > 0 && r.top < window.innerHeight + 200) fire();
        };
        if ('IntersectionObserver' in window) {
            const io = new IntersectionObserver(entries => {
                if (entries.some(e => e.isIntersecting)) { fire(); io.disconnect(); }
            }, { rootMargin: '200px' });
            io.observe(el);
        }
        check();
        window.addEventListener('scroll', check, { passive: true });
        window.addEventListener('resize', check, { passive: true });
    }

    function loadJSON(url) {
        // Revalidate rather than force-cache: the data files change whenever the
        // model is recalibrated, and a stale tracks.json had the page reporting
        // an optimum from a previous fit. ETags keep the revalidation cheap.
        return fetch(url, { cache: 'no-cache' }).then(r => {
            if (!r.ok) throw new Error(url + ' ' + r.status);
            return r.json();
        });
    }

    /** Draw text with a monospace numeric look, right-aligned. */
    function label(g, text, x, y, opts) {
        const o = opts || {};
        g.save();
        g.font = (o.weight || '500') + ' ' + (o.size || 11) + 'px ' + (o.family || 'Inter, system-ui, sans-serif');
        g.fillStyle = o.color || palette.ink2;
        g.textAlign = o.align || 'left';
        g.textBaseline = o.baseline || 'alphabetic';
        g.fillText(text, x, y);
        g.restore();
    }

    window.F1UI = { palette, setupCanvas, fmtLap, fmtDelta, clamp, lerp, interp, indexAt,
                    speedColor, hexToRgba, whenVisible, loadJSON, label };
})();
