/* ============================================================
   F1 DATA ANALYTICS - INTERACTIVE WIND TUNNEL
   ------------------------------------------------------------
   Two independent layers:

   1. FORCES - an empirical model built on the PERRINN 2017 open
      -source F1 CFD dataset (windtunnel_data/perrinn_cfd_data.csv).
      Coefficients are stored area-inclusive (sCz, sCx in m2), so
      force = q * sCz directly. Do NOT multiply by frontal area
      again - that was the bug in the original calculator.

   2. FLOW - a 2D potential-flow approximation used purely for the
      visual: a superposition of doublets (body thickness) and
      Rankine vortices (circulation), each mirrored about y = 0 so
      the ground plane is a streamline. Circulation is split
      between elements using the force model's own component
      breakdown, so the picture and the numbers stay in sync.
      Magnitudes are scaled for legibility - see VIS_GAIN.
   ============================================================ */

(function () {
    'use strict';

    // ---------- physical constants ----------
    const RHO = 1.225;        // air density, kg/m^3
    const G0 = 9.81;          // gravity, m/s^2
    const CAR_MASS = 798;     // kg, 2024 minimum weight incl. driver
    const FRONTAL_AREA = 1.5; // m^2, PERRINN reference area
    const TYRE_MU = 1.8;      // peak slick friction coefficient
    const POWER_W = 470000;   // W at the wheels; calibrated so a Monza DRS setup tops out ~355 km/h

    // ---------- PERRINN baseline (40 mm front / 50 mm rear ride height) ----------
    const BASE_SCZ = 3.25;    // downforce coefficient x area, m^2
    const BASE_SCX = 1.16;    // drag coefficient x area, m^2

    // Component split of downforce, normalised from the source table
    // (underbody 42.5 / rear wing 27.5 / front wing 22.5 / diffuser 12.5).
    const DF_SPLIT = { floor: 0.4048, rearWing: 0.2619, frontWing: 0.2143, diffuser: 0.1190 };
    // Component split of drag (rear wing 32.5 / wheels 35 / body 12.5, remainder apportioned).
    const DR_SPLIT = { rearWing: 0.325, wheels: 0.350, body: 0.125, frontWing: 0.120, floor: 0.080 };

    // Reference setup - every scaling law is relative to this.
    const REF = { rearWing: 20, frontWing: 10, frontRH: 40, rearRH: 50 };

    const PRESETS = {
        monaco: { label: 'Monaco', speed: 160, rearWing: 32, frontWing: 16, frontRH: 28, rearRH: 46, drs: false },
        silverstone: { label: 'Silverstone', speed: 250, rearWing: 20, frontWing: 10, frontRH: 40, rearRH: 50, drs: false },
        monza: { label: 'Monza', speed: 330, rearWing: 7, frontWing: 4, frontRH: 46, rearRH: 62, drs: false }
    };

    const clamp = (v, lo, hi) => v < lo ? lo : (v > hi ? hi : v);

    // ---------- analytics ----------
    // The wind tunnel is the reason this page exists, so the one thing worth
    // measuring is whether visitors actually touch it. Fires once per page
    // view on first interaction, then reports which controls got used.
    let engaged = false;
    function track(action, label) {
        if (typeof window.gtag !== 'function') return;
        if (!engaged) {
            engaged = true;
            window.gtag('event', 'wind_tunnel_engage', { engagement_type: action });
        }
        window.gtag('event', 'wind_tunnel_control', {
            control: action,
            value_label: label === undefined ? '' : String(label)
        });
    }

    // ============================================================
    // FORCE MODEL
    // ============================================================

    /**
     * Ground-effect multiplier for the floor, as a function of front ride
     * height in mm. Rises as the floor is lowered (the venturi works harder),
     * peaks around 24 mm, then collapses below that as the underfloor stalls -
     * the porpoising cliff teams spent 2022 discovering. Normalised to 1.0 at
     * the 40 mm PERRINN reference height.
     */
    function groundEffect(h) {
        const raw = x => x >= 24 ? 1 / (1 + Math.pow((x - 24) / 38, 1.6)) : Math.pow(x / 24, 1.5);
        return raw(h) / raw(REF.frontRH);
    }

    function solve(p) {
        const v = p.speed / 3.6;                 // m/s
        const q = 0.5 * RHO * v * v;             // dynamic pressure, Pa
        const rake = p.rearRH - p.frontRH;       // mm

        // --- per-component multipliers ---
        const fRW = clamp(1 + (p.rearWing - REF.rearWing) * 0.030, 0.45, 1.55);
        const fFW = clamp(1 + (p.frontWing - REF.frontWing) * 0.020, 0.70, 1.35);
        const fGE = groundEffect(p.frontRH);
        const fRake = clamp(1 + (rake - 10) * 0.006, 0.90, 1.16);
        const drsDF = p.drs ? 0.35 : 1;          // DRS dumps most of the rear wing's load
        const drsDrag = p.drs ? 0.30 : 1;

        // --- downforce, area-inclusive coefficients (m^2) ---
        const cz = {
            floor: BASE_SCZ * DF_SPLIT.floor * fGE * fRake,
            diffuser: BASE_SCZ * DF_SPLIT.diffuser * fGE * fRake * fRake,
            frontWing: BASE_SCZ * DF_SPLIT.frontWing * fFW * (1 + (REF.frontRH - p.frontRH) * 0.004),
            rearWing: BASE_SCZ * DF_SPLIT.rearWing * fRW * drsDF
        };
        const sCz = cz.floor + cz.diffuser + cz.frontWing + cz.rearWing;

        // --- drag; wing drag grows faster than load (induced drag ~ Cl^2) ---
        const cx = {
            rearWing: BASE_SCX * DR_SPLIT.rearWing * Math.pow(fRW, 1.6) * drsDrag,
            frontWing: BASE_SCX * DR_SPLIT.frontWing * Math.pow(fFW, 1.5),
            floor: BASE_SCX * DR_SPLIT.floor * fGE,
            wheels: BASE_SCX * DR_SPLIT.wheels,
            body: BASE_SCX * DR_SPLIT.body
        };
        const sCx = cx.rearWing + cx.frontWing + cx.floor + cx.wheels + cx.body;

        // --- forces ---
        const downforce = q * sCz;
        const drag = q * sCx;

        // Aero balance: how much of the load sits on the front axle. Rake
        // pitches load forward, so the floor's front share moves with it.
        const floorFrontShare = clamp(0.44 + (rake - 10) * 0.004, 0.32, 0.56);
        const frontLoad = cz.frontWing + floorFrontShare * (cz.floor + cz.diffuser);
        const balance = frontLoad / sCz * 100;

        // Grip: normal load is weight plus downforce.
        const latG = TYRE_MU * (G0 + downforce / CAR_MASS) / G0;
        const cornerRadius = (v * v) / (latG * G0);

        // Terminal velocity where engine power equals drag power.
        const vMax = Math.cbrt(POWER_W / (0.5 * RHO * sCx)) * 3.6;

        return {
            v, q, sCz, sCx, cz, cx, downforce, drag, balance, latG, cornerRadius, vMax,
            Cz: sCz / FRONTAL_AREA,
            Cx: sCx / FRONTAL_AREA,
            efficiency: sCz / sCx,
            downforceKg: downforce / G0,
            weightRatio: downforce / (CAR_MASS * G0) * 100,
            powerKw: drag * v / 1000
        };
    }

    // ============================================================
    // POTENTIAL FLOW FIELD
    // ============================================================

    const VIS_GAIN = 2.0;   // legibility scaling on circulation (see header)
    const SPAN_EFF = 1.8;   // m, effective span used to turn force into 2D circulation

    // A real ride height is 15-60 mm. Drawn to scale in a 1.8 m tall frame that
    // is two or three pixels, so the underfloor - the most interesting part of
    // the whole car - would be invisible. The car is therefore lifted on an
    // exaggerated vertical scale. Slider values, and every number on the page,
    // remain the true ones; only the picture is stretched.
    const RIDE_BASE = 0.050;  // m of clearance even at the lowest setting
    const RIDE_VIS = 5.0;     // exaggeration factor on the ride height itself
    const WHEEL_R = 0.36;     // m, tyre radius - the contact patch stays at y = 0

    /** Drawn floor offset at station x, in metres (includes rake). */
    function floorY(x, p) {
        const t = clamp(x / 5.0, 0, 1);
        const mm = p.frontRH + (p.rearRH - p.frontRH) * t;
        return RIDE_BASE + mm * RIDE_VIS / 1000;
    }

    function buildElements(p, r) {
        const U = r.v;
        const els = [];

        // Body thickness: doublets sized so the local stagnation radius roughly
        // matches the real cross-section at that station.
        const dbl = (x, y, a) => els.push({ k: 'd', x, y: y + floorY(x, p), mu: 2 * Math.PI * U * a * a, rc: a * 0.85 });
        dbl(0.55, 0.20, 0.16);   // nose
        dbl(2.40, 0.38, 0.23);   // sidepod
        dbl(3.25, 0.42, 0.21);   // engine cover
        // Wheels are bolted to the ground, not to the floor plane, so they do
        // not ride up with the exaggerated ride height.
        els.push({ k: 'd', x: 1.05, y: WHEEL_R, mu: 2 * Math.PI * U * 0.30 * 0.30, rc: 0.30 });
        els.push({ k: 'd', x: 4.05, y: WHEEL_R, mu: 2 * Math.PI * U * 0.30 * 0.30, rc: 0.30 });

        // Circulation, split exactly the way the force model splits downforce.
        const k = VIS_GAIN / (RHO * U * SPAN_EFF) * r.q;
        const vtx = (x, y, scz, rc) => els.push({ k: 'v', x, y: y + floorY(x, p), G: scz * k, rc });
        vtx(0.32, 0.10, r.cz.frontWing, 0.16);
        vtx(2.55, 0.13, r.cz.floor, 0.42);
        vtx(3.95, 0.12, r.cz.diffuser, 0.30);
        vtx(4.85, 0.80, r.cz.rearWing, 0.18);

        return els;
    }

    /**
     * Velocity at (x, y). Every element is evaluated twice: once for itself and
     * once for its mirror below the ground plane, which forces v = 0 at y = 0.
     */
    function velocity(x, y, els, U, out) {
        let u = U, w = 0;
        for (let i = 0; i < els.length; i++) {
            const e = els[i];
            for (let m = 0; m < 2; m++) {
                const ey = m === 0 ? e.y : -e.y;
                const dx = x - e.x, dy = y - ey;
                let r2 = dx * dx + dy * dy;
                if (e.k === 'v') {
                    const G = (m === 0 ? e.G : -e.G);
                    const rc2 = e.rc * e.rc;
                    // Rankine core: solid-body rotation inside rc, free vortex outside.
                    const f = r2 < rc2 ? G / (2 * Math.PI * rc2) : G / (2 * Math.PI * r2);
                    u -= f * dy;
                    w += f * dx;
                } else {
                    const rc2 = e.rc * e.rc;
                    if (r2 < rc2) r2 = rc2;
                    const r4 = r2 * r2;
                    const c = e.mu / (2 * Math.PI);
                    u -= c * (dx * dx - dy * dy) / r4;
                    w -= c * (2 * dx * dy) / r4;
                }
            }
        }
        out[0] = u;
        out[1] = w;
    }

    // ============================================================
    // CAR GEOMETRY
    // ============================================================

    const BODY = [
        [0.00, 0.055], [0.38, 0.085], [0.72, 0.22], [1.12, 0.27], [1.48, 0.34],
        [1.78, 0.44], [1.95, 0.56], [2.20, 0.50], [2.55, 0.52], [3.05, 0.54],
        [3.55, 0.50], [4.05, 0.44], [4.45, 0.36], [4.72, 0.30], [4.72, 0.10],
        [3.20, 0.055], [1.60, 0.045], [0.55, 0.045]
    ];

    function carPath(p, toPx) {
        const path = new Path2D();
        const pt = (x, y) => toPx(x, y + floorY(x, p));

        // main body
        BODY.forEach(([x, y], i) => {
            const [px, py] = pt(x, y);
            i === 0 ? path.moveTo(px, py) : path.lineTo(px, py);
        });
        path.closePath();

        // front wing: main plane plus endplate, tied into the nose
        [[-0.22, 0.015], [0.62, 0.060], [0.62, 0.175], [-0.22, 0.115]].forEach(([x, y], i) => {
            const [px, py] = pt(x, y);
            i === 0 ? path.moveTo(px, py) : path.lineTo(px, py);
        });
        path.closePath();

        [[-0.26, 0.010], [-0.14, 0.010], [-0.14, 0.30], [-0.26, 0.26]].forEach(([x, y], i) => {
            const [px, py] = pt(x, y);
            i === 0 ? path.moveTo(px, py) : path.lineTo(px, py);
        });
        path.closePath();

        // Rear wing. From the side you mostly see the ENDPLATE, so that is the
        // dominant shape; the main plane is drawn inside it, pitched by the
        // selected angle, so the slider still has a visible effect.
        [[4.58, 0.24], [4.72, 0.24], [4.86, 0.70], [4.72, 0.70]].forEach(([x, y], i) => {
            const [px, py] = pt(x, y);
            i === 0 ? path.moveTo(px, py) : path.lineTo(px, py);
        });
        path.closePath();

        [[4.68, 0.64], [5.16, 0.66], [5.16, 1.06], [4.68, 0.98]].forEach(([x, y], i) => {
            const [px, py] = pt(x, y);
            i === 0 ? path.moveTo(px, py) : path.lineTo(px, py);
        });
        path.closePath();

        return path;
    }

    /** The pitched main plane, drawn over the endplate so the angle is legible. */
    function wingPlane(p, toPx) {
        const path = new Path2D();
        const ang = (p.drs ? 3 : p.rearWing) * Math.PI / 180;
        const cx = 4.86, cy = 0.86, c = 0.50, th = 0.035;
        [[-c / 2, -th], [c / 2, -th], [c / 2, th], [-c / 2, th]].forEach(([dx, dy], i) => {
            const x = cx + dx * Math.cos(ang) - dy * Math.sin(ang);
            const y = cy + dx * Math.sin(ang) + dy * Math.cos(ang);
            const [px, py] = toPx(x, y + floorY(x, p));
            i === 0 ? path.moveTo(px, py) : path.lineTo(px, py);
        });
        path.closePath();
        return path;
    }

    /** Cheap solid test so particles do not tunnel through the car. */
    function inside(x, y, p) {
        const fy = floorY(x, p);
        const yy = y - fy;
        if (x < -0.30 || x > 5.20 || y < 0) return false;
        if (x >= -0.28 && x <= 0.62 && yy >= 0.01 && yy <= 0.30) return true;      // front wing
        if (x >= 4.56 && x <= 5.20 && yy >= 0.62 && yy <= 1.10) return true;       // rear wing box
        // wheels, measured from the ground rather than the floor plane
        const w1 = (x - 1.05) ** 2 + (y - WHEEL_R) ** 2;
        const w2 = (x - 4.05) ** 2 + (y - WHEEL_R) ** 2;
        if (w1 < WHEEL_R * WHEEL_R || w2 < WHEEL_R * WHEEL_R) return true;
        // Body: interpolate the upper surface. The lower bound matters just as
        // much - without it everything beneath the floor plane counts as solid
        // and the underfloor flow, the whole point of the picture, is culled.
        if (x < 0 || x > 4.72) return false;
        if (yy < 0.035) return false;
        let top = 0;
        for (let i = 0; i < 13; i++) {
            const a = BODY[i], b = BODY[i + 1];
            if (x >= a[0] && x <= b[0]) {
                top = a[1] + (b[1] - a[1]) * (x - a[0]) / (b[0] - a[0] || 1);
                break;
            }
        }
        return yy < top;
    }

    // ============================================================
    // RENDERER
    // ============================================================

    // Flow domain in metres.
    // Horizontal extent is fixed; how much sky is visible falls out of the
    // canvas aspect ratio, so the stage can be any shape without distortion.
    const X0 = -0.62, X1 = 5.72, Y0 = 0;
    const CAR_SPAN = 5.60;  // m the car actually occupies - never crop into this
    const Y_MAX = 2.30;     // m of sky worth showing before the car gets too small

    function init() {
        const root = document.getElementById('windTunnel');
        if (!root) return;

        const stage = root.querySelector('.wt-stage');
        const cvField = root.querySelector('.wt-field');
        const cvFlow = root.querySelector('.wt-flow');
        const cvCar = root.querySelector('.wt-car');
        if (!stage || !cvField || !cvFlow || !cvCar) return;

        const gField = cvField.getContext('2d');
        const gFlow = cvFlow.getContext('2d');
        const gCar = cvCar.getContext('2d');

        const reduced = window.matchMedia('(prefers-reduced-motion: reduce)').matches;

        const params = Object.assign({}, PRESETS.silverstone);
        let result = solve(params);
        let els = buildElements(params, result);
        let mode = 'streamlines';

        let W = 0, H = 0, scale = 1, Y1 = 1.8, xOff = 0;
        const toPx = (x, y) => [(x - X0 - xOff) * scale, H - (y - Y0) * scale];

        // ---------- particles ----------
        const PCOUNT = 820;
        const px = new Float32Array(PCOUNT), py = new Float32Array(PCOUNT);
        const pAge = new Uint16Array(PCOUNT), pLife = new Uint16Array(PCOUNT);
        const vec = [0, 0];

        function seed(i, spread) {
            px[i] = X0 + (spread ? Math.random() * (X1 - X0) : Math.random() * 0.6);
            if (Math.random() < 0.30) {
                // Aim this one at the gap under the floor.
                const gap = floorY(0, params);
                py[i] = 0.012 + Math.random() * Math.max(gap - 0.024, 0.01);
            } else {
                py[i] = Y0 + Math.pow(Math.random(), 1.9) * (Y1 - Y0);
            }
            pAge[i] = 0;
            pLife[i] = 90 + (Math.random() * 130) | 0;
        }
        for (let i = 0; i < PCOUNT; i++) seed(i, true);

        // ---------- colour ramp ----------
        // Pivots on the freestream: air moving at V-infinity is near-white, slower
        // air goes teal, faster air goes red. Most of the domain sits close to 1.0,
        // so the ramp is deliberately steep either side of it.
        function speedColor(ratio, alpha) {
            let r, g, b;
            if (ratio < 1) {
                const k = clamp((ratio - 0.45) / 0.55, 0, 1);
                r = 0 + k * 205; g = 200 + k * 32; b = 225 + k * 15;
            } else {
                const k = clamp((ratio - 1) / 0.85, 0, 1);
                r = 205 + k * 50; g = 232 - k * 196; b = 240 - k * 226;
            }
            return 'rgba(' + (r | 0) + ',' + (g | 0) + ',' + (b | 0) + ',' + alpha.toFixed(3) + ')';
        }

        // ---------- pressure field ----------
        const FW = 300, FH = 110;
        const fieldCanvas = document.createElement('canvas');
        fieldCanvas.width = FW; fieldCanvas.height = FH;
        const gTmp = fieldCanvas.getContext('2d');
        const img = gTmp.createImageData(FW, FH);

        function renderField() {
            gField.clearRect(0, 0, W, H);
            if (mode === 'streamlines') return;

            const d = img.data;
            const U = result.v;
            for (let j = 0; j < FH; j++) {
                const y = Y1 - (j + 0.5) / FH * (Y1 - Y0);
                for (let i = 0; i < FW; i++) {
                    const x = X0 + (i + 0.5) / FW * (X1 - X0);
                    const o = (j * FW + i) * 4;
                    if (inside(x, y, params)) { d[o + 3] = 0; continue; }
                    velocity(x, y, els, U, vec);
                    const ratio = Math.hypot(vec[0], vec[1]) / U;
                    let r, g, b, a;
                    if (mode === 'pressure') {
                        // Cp = 1 - (V/U)^2. Red = stagnation, teal = suction.
                        const cp = clamp(1 - ratio * ratio, -2.2, 1);
                        if (cp >= 0) {
                            const k = cp;                       // 0 -> 1
                            r = 40 + k * 200; g = 40 + k * 10; b = 55 - k * 40;
                        } else {
                            const k = clamp(-cp / 2.2, 0, 1);
                            r = 30 - k * 30; g = 60 + k * 150; b = 90 + k * 100;
                        }
                        a = 150;
                    } else {
                        const k = clamp(ratio / 1.9, 0, 1);
                        r = 8 + k * 245; g = 25 + k * 195; b = 60 + k * 60;
                        a = 140;
                    }
                    // Nothing interesting happens far above the bodywork; fading
                    // there keeps the eye on the car instead of on doublet haloes.
                    const h = clamp((y - 1.15) / 0.6, 0, 1);
                    d[o] = r; d[o + 1] = g; d[o + 2] = b; d[o + 3] = a * (1 - h * 0.85);
                }
            }
            gTmp.putImageData(img, 0, 0);
            gField.save();
            gField.imageSmoothingEnabled = true;
            gField.imageSmoothingQuality = 'high';
            gField.drawImage(fieldCanvas, 0, 0, W, H);
            gField.restore();
        }

        // ---------- car ----------
        function renderCar() {
            gCar.clearRect(0, 0, W, H);

            // Ground plane: a hairline only. Anything thicker covers the
            // underfloor, which is the most important part of the picture.
            const gy = H - (0 - Y0) * scale;
            gCar.strokeStyle = 'rgba(255,255,255,0.32)';
            gCar.lineWidth = 1;
            gCar.beginPath(); gCar.moveTo(0, gy + 0.5); gCar.lineTo(W, gy + 0.5); gCar.stroke();

            // body
            const path = carPath(params, toPx);
            const [, topY] = toPx(0, 1.0);
            const bodyGrad = gCar.createLinearGradient(0, topY, 0, gy);
            bodyGrad.addColorStop(0, '#3a3a4a');
            bodyGrad.addColorStop(0.55, '#20202b');
            bodyGrad.addColorStop(1, '#0c0c12');
            gCar.fillStyle = bodyGrad;
            gCar.fill(path);

            // rim light picks the silhouette off the flow behind it
            gCar.save();
            gCar.shadowColor = 'rgba(225,6,0,0.55)';
            gCar.shadowBlur = 14;
            gCar.strokeStyle = 'rgba(255,255,255,0.85)';
            gCar.lineWidth = 1.6;
            gCar.stroke(path);
            gCar.restore();

            // main plane, pitched by the rear wing slider
            const plane = wingPlane(params, toPx);
            gCar.fillStyle = params.drs ? 'rgba(0,210,190,0.30)' : 'rgba(225,6,0,0.30)';
            gCar.fill(plane);
            gCar.strokeStyle = params.drs ? 'rgba(0,210,190,0.95)' : 'rgba(255,255,255,0.9)';
            gCar.lineWidth = 1.4;
            gCar.stroke(plane);

            // wheels (drawn after the body so they sit in the foreground)
            [1.05, 4.05].forEach(x => {
                const [cx, cy] = toPx(x, WHEEL_R);
                const rad = WHEEL_R * scale;
                const tyre = gCar.createRadialGradient(cx - rad * 0.3, cy - rad * 0.4, rad * 0.1, cx, cy, rad);
                tyre.addColorStop(0, '#26262f');
                tyre.addColorStop(1, '#0b0b10');
                gCar.fillStyle = tyre;
                gCar.strokeStyle = 'rgba(255,255,255,0.42)';
                gCar.lineWidth = 1.5;
                gCar.beginPath();
                gCar.arc(cx, cy, rad, 0, Math.PI * 2);
                gCar.fill(); gCar.stroke();
                // rim
                gCar.strokeStyle = 'rgba(255,255,255,0.16)';
                gCar.lineWidth = 1;
                gCar.beginPath();
                gCar.arc(cx, cy, rad * 0.52, 0, Math.PI * 2);
                gCar.stroke();
            });

            // ride-height callout
            const [hx, hy] = toPx(2.2, 0);
            const hTop = H - (floorY(2.2, params) - Y0) * scale;
            gCar.strokeStyle = 'rgba(0,210,190,0.8)';
            gCar.lineWidth = 1;
            gCar.setLineDash([3, 3]);
            gCar.beginPath(); gCar.moveTo(hx, hy); gCar.lineTo(hx, hTop); gCar.stroke();
            gCar.setLineDash([]);
        }

        // ---------- streamlines ----------
        function step(dt) {
            const U = result.v;
            gFlow.globalCompositeOperation = 'destination-out';
            gFlow.fillStyle = 'rgba(0,0,0,0.19)';
            gFlow.fillRect(0, 0, W, H);
            gFlow.globalCompositeOperation = 'source-over';
            gFlow.lineWidth = 1.05;
            gFlow.lineCap = 'round';

            for (let i = 0; i < PCOUNT; i++) {
                const x0 = px[i], y0 = py[i];
                velocity(x0, y0, els, U, vec);
                const u1 = vec[0], w1 = vec[1];
                // midpoint (RK2)
                velocity(x0 + u1 * dt * 0.5, y0 + w1 * dt * 0.5, els, U, vec);
                const x1 = x0 + vec[0] * dt, y1 = y0 + vec[1] * dt;

                const spd = Math.hypot(vec[0], vec[1]);
                pAge[i]++;

                if (x1 > X1 || y1 > Y1 || y1 < 0.002 || pAge[i] > pLife[i] ||
                    spd < U * 0.06 || inside(x1, y1, params)) {
                    seed(i, false);
                    continue;
                }

                const [ax, ay] = toPx(x0, y0);
                const [bx, by] = toPx(x1, y1);
                const fade = pAge[i] < 12 ? pAge[i] / 12 : 1;
                const ratio = spd / U;
                // Undisturbed air is dim; the further the flow departs from the
                // freestream, the brighter the trace. This is what makes the
                // suction under the floor and the stagnation at the nose readable.
                const dev = clamp(Math.abs(ratio - 1) / 0.55, 0, 1);
                const base = mode === 'streamlines' ? 0.16 + dev * 0.66 : 0.10 + dev * 0.34;
                gFlow.strokeStyle = speedColor(ratio, base * fade);
                gFlow.beginPath();
                gFlow.moveTo(ax, ay);
                gFlow.lineTo(bx, by);
                gFlow.stroke();

                px[i] = x1; py[i] = y1;
            }
        }

        /** Static streamline set, used when the visitor prefers reduced motion. */
        function drawStatic() {
            gFlow.clearRect(0, 0, W, H);
            const U = result.v;
            gFlow.lineWidth = 1.2;
            for (let n = 0; n < 46; n++) {
                let x = X0, y = Y0 + Math.pow((n + 0.5) / 46, 1.4) * (Y1 - Y0);
                gFlow.beginPath();
                let started = false;
                for (let s = 0; s < 900; s++) {
                    velocity(x, y, els, U, vec);
                    const spd = Math.hypot(vec[0], vec[1]);
                    if (spd < U * 0.05) break;
                    const dt = 0.010 / (spd / U);
                    velocity(x + vec[0] * dt * 0.5, y + vec[1] * dt * 0.5, els, U, vec);
                    x += vec[0] * dt; y += vec[1] * dt;
                    if (x > X1 || y > Y1 || y < 0.004 || inside(x, y, params)) break;
                    const [cx, cy] = toPx(x, y);
                    if (!started) { gFlow.moveTo(cx, cy); started = true; }
                    else gFlow.lineTo(cx, cy);
                }
                gFlow.strokeStyle = 'rgba(0,210,190,0.55)';
                gFlow.stroke();
            }
        }

        // ---------- readouts ----------
        const el = id => root.querySelector('[data-wt="' + id + '"]');
        const nodes = {};
        ['downforce', 'downforceKg', 'weightRatio', 'drag', 'efficiency', 'power', 'hp',
            'cz', 'cx', 'scz', 'scx', 'balance', 'latg', 'radius', 'vmax'].forEach(k => { nodes[k] = el(k); });

        const fmt0 = n => n.toLocaleString('en-US', { maximumFractionDigits: 0 });

        function renderReadouts() {
            const r = result;
            if (nodes.downforce) nodes.downforce.textContent = fmt0(r.downforce);
            if (nodes.downforceKg) nodes.downforceKg.textContent = fmt0(r.downforceKg);
            if (nodes.weightRatio) nodes.weightRatio.textContent = Math.round(r.weightRatio) + '%';
            if (nodes.drag) nodes.drag.textContent = fmt0(r.drag);
            if (nodes.efficiency) nodes.efficiency.textContent = r.efficiency.toFixed(2);
            if (nodes.power) nodes.power.textContent = fmt0(r.powerKw);
            if (nodes.hp) nodes.hp.textContent = fmt0(r.powerKw * 1.341);
            if (nodes.cz) nodes.cz.textContent = r.Cz.toFixed(2);
            if (nodes.cx) nodes.cx.textContent = r.Cx.toFixed(2);
            if (nodes.scz) nodes.scz.textContent = r.sCz.toFixed(2);
            if (nodes.scx) nodes.scx.textContent = r.sCx.toFixed(2);
            if (nodes.latg) nodes.latg.textContent = r.latG.toFixed(1);
            if (nodes.radius) nodes.radius.textContent = fmt0(r.cornerRadius);
            if (nodes.vmax) nodes.vmax.textContent = fmt0(r.vMax);

            // aero balance bar
            const bal = root.querySelector('.wt-balance-fill');
            const balTxt = root.querySelector('.wt-balance-value');
            if (bal) bal.style.width = clamp(result.balance, 20, 70) + '%';
            if (balTxt) balTxt.textContent = result.balance.toFixed(1) + '% front';

            // component bars
            const maxCz = Math.max(result.cz.floor, result.cz.rearWing, result.cz.frontWing, result.cz.diffuser);
            root.querySelectorAll('.wt-comp').forEach(row => {
                const key = row.dataset.comp;
                const val = result.cz[key] || 0;
                const fill = row.querySelector('.wt-comp-fill');
                const out = row.querySelector('.wt-comp-value');
                if (fill) fill.style.width = (val / maxCz * 100).toFixed(1) + '%';
                if (out) out.textContent = fmt0(result.q * val) + ' N';
            });

            // warnings
            const warn = root.querySelector('.wt-warning');
            if (warn) {
                let msg = '';
                if (params.frontRH <= 20) msg = 'Floor is stalling &mdash; below about 22&nbsp;mm the underfloor loses suction and the car porpoises.';
                else if (params.drs) msg = 'DRS open &mdash; rear wing load dumped for straight-line speed.';
                else if (result.balance > 50) msg = 'Front-biased balance &mdash; expect snap oversteer on entry.';
                else if (result.balance < 38) msg = 'Rear-biased balance &mdash; the car will understeer through slow corners.';
                warn.innerHTML = msg;
                warn.classList.toggle('active', !!msg);
            }
        }

        // ---------- recompute ----------
        function recompute() {
            result = solve(params);
            els = buildElements(params, result);
            renderReadouts();
            renderCar();
            renderField();
            if (reduced) drawStatic();
        }

        // ---------- sizing ----------
        function resize() {
            const rect = stage.getBoundingClientRect();
            if (!rect.width) return;
            const dpr = Math.min(window.devicePixelRatio || 1, 2);
            W = rect.width; H = rect.height;
            [cvField, cvFlow, cvCar].forEach(c => {
                c.width = Math.round(W * dpr);
                c.height = Math.round(H * dpr);
                c.style.width = W + 'px';
                c.style.height = H + 'px';
                c.getContext('2d').setTransform(dpr, 0, 0, dpr, 0, 0);
            });
            // Fit the domain width by default. On short, narrow stages that
            // leaves metres of empty sky, so zoom in - but never far enough to
            // crop into the car itself.
            scale = W / (X1 - X0);
            if (H / scale > Y_MAX) scale = Math.min(H / Y_MAX, W / CAR_SPAN);
            xOff = ((X1 - X0) - W / scale) / 2;
            Y1 = Y0 + H / scale;
            gFlow.clearRect(0, 0, W, H);
            recompute();
        }

        // ---------- controls ----------
        root.querySelectorAll('.wt-slider').forEach(input => {
            input.addEventListener('change', () => track('slider', input.dataset.param));
            input.addEventListener('input', () => {
                params[input.dataset.param] = parseFloat(input.value);
                if (params.rearRH < params.frontRH + 2) {
                    params.rearRH = params.frontRH + 2;
                    const rh = root.querySelector('[data-param="rearRH"]');
                    if (rh) rh.value = params.rearRH;
                }
                syncLabels();
                recompute();
                clearPreset();
            });
        });

        const drsBtn = root.querySelector('.wt-drs');
        if (drsBtn) {
            drsBtn.addEventListener('click', () => {
                params.drs = !params.drs;
                track('drs', params.drs ? 'open' : 'closed');
                drsBtn.classList.toggle('active', params.drs);
                drsBtn.setAttribute('aria-pressed', String(params.drs));
                recompute();
            });
        }

        root.querySelectorAll('.wt-mode').forEach(btn => {
            btn.addEventListener('click', () => {
                mode = btn.dataset.mode;
                track('mode', mode);
                root.querySelectorAll('.wt-mode').forEach(b => {
                    const on = b === btn;
                    b.classList.toggle('active', on);
                    b.setAttribute('aria-pressed', String(on));
                });
                root.querySelector('.wt-stage').dataset.mode = mode;
                renderField();
            });
        });

        root.querySelectorAll('.wt-preset').forEach(btn => {
            btn.addEventListener('click', () => {
                const preset = PRESETS[btn.dataset.preset];
                if (!preset) return;
                track('preset', btn.dataset.preset);
                Object.assign(params, preset);
                delete params.label;
                root.querySelectorAll('.wt-slider').forEach(s => { s.value = params[s.dataset.param]; });
                if (drsBtn) {
                    drsBtn.classList.toggle('active', params.drs);
                    drsBtn.setAttribute('aria-pressed', String(params.drs));
                }
                root.querySelectorAll('.wt-preset').forEach(b => {
                    const on = b === btn;
                    b.classList.toggle('active', on);
                    b.setAttribute('aria-pressed', String(on));
                });
                syncLabels();
                recompute();
            });
        });

        function clearPreset() {
            root.querySelectorAll('.wt-preset').forEach(b => {
                b.classList.remove('active');
                b.setAttribute('aria-pressed', 'false');
            });
        }

        function syncLabels() {
            root.querySelectorAll('.wt-slider').forEach(s => {
                const out = root.querySelector('output[for="' + s.id + '"]');
                if (out) out.textContent = s.value;
            });
            const rakeOut = root.querySelector('[data-wt="rake"]');
            if (rakeOut) rakeOut.textContent = (params.rearRH - params.frontRH).toFixed(0);
        }

        // ---------- loop ----------
        let running = false, raf = 0, last = 0;
        function frame(now) {
            if (!running) return;
            const dt = Math.min((now - last) / 1000, 0.05) || 0.016;
            last = now;
            // Advance the flow in scaled time so the picture reads the same at
            // 60 km/h and 350 km/h; the colours carry the speed information.
            step(dt * 0.55 * (60 / Math.max(result.v, 12)));
            raf = requestAnimationFrame(frame);
        }

        function start() {
            if (running || reduced) return;
            running = true; last = performance.now();
            raf = requestAnimationFrame(frame);
        }
        function stop() {
            running = false;
            cancelAnimationFrame(raf);
        }

        if ('IntersectionObserver' in window) {
            new IntersectionObserver(entries => {
                entries[0].isIntersecting ? start() : stop();
            }, { threshold: 0.08 }).observe(stage);
        } else {
            start();
        }
        document.addEventListener('visibilitychange', () => {
            document.hidden ? stop() : start();
        });

        if ('ResizeObserver' in window) {
            new ResizeObserver(resize).observe(stage);
        } else {
            window.addEventListener('resize', resize);
        }

        syncLabels();
        resize();
    }

    // ============================================================
    // ATR SLIDING SCALE EXPLORER
    // ============================================================

    const ATR = [
        { pos: 1, team: 'Red Bull', pct: 70, hours: 224 },
        { pos: 2, team: 'Ferrari', pct: 75, hours: 240 },
        { pos: 3, team: 'McLaren', pct: 80, hours: 256 },
        { pos: 4, team: 'Mercedes', pct: 85, hours: 272 },
        { pos: 5, team: 'Aston Martin', pct: 90, hours: 288 },
        { pos: 6, team: 'RB', pct: 95, hours: 304 },
        { pos: 7, team: 'Haas', pct: 100, hours: 320 },
        { pos: 8, team: 'Alpine', pct: 105, hours: 336 },
        { pos: 9, team: 'Williams', pct: 110, hours: 352 },
        { pos: 10, team: 'Sauber', pct: 115, hours: 368 }
    ];

    function initATR() {
        const root = document.getElementById('atrExplorer');
        if (!root) return;

        const chart = root.querySelector('.atr-chart');
        const detail = root.querySelector('.atr-detail');
        if (!chart) return;

        const max = ATR[ATR.length - 1].hours;

        chart.innerHTML = ATR.map(d => (
            '<button type="button" class="atr-bar" data-pos="' + d.pos + '"' +
            ' aria-label="P' + d.pos + ' ' + d.team + ', ' + d.pct + ' percent, ' + d.hours + ' wind tunnel hours per year">' +
            '<span class="atr-bar-track"><span class="atr-bar-fill" style="height:' +
            (d.hours / max * 100).toFixed(1) + '%"></span></span>' +
            '<span class="atr-bar-pos">P' + d.pos + '</span>' +
            '</button>'
        )).join('');

        function show(pos) {
            const d = ATR.find(a => a.pos === pos);
            if (!d || !detail) return;
            const leader = ATR[0];
            const extra = d.hours - leader.hours;
            detail.innerHTML =
                '<div class="atr-detail-head"><span class="atr-detail-pos">P' + d.pos + '</span>' +
                '<span class="atr-detail-team">' + d.team + '</span></div>' +
                '<div class="atr-detail-grid">' +
                '<div><span class="atr-k">Allocation</span><span class="atr-v">' + d.pct + '%</span></div>' +
                '<div><span class="atr-k">Wind tunnel</span><span class="atr-v">' + d.hours + ' h/yr</span></div>' +
                '<div><span class="atr-k">CFD items</span><span class="atr-v">' + d.pct + '%</span></div>' +
                '<div><span class="atr-k">vs P1</span><span class="atr-v">' +
                (extra > 0 ? '+' + extra + ' h' : 'baseline') + '</span></div>' +
                '</div>' +
                '<p class="atr-detail-note">' + (extra > 0
                    ? 'That is ' + extra + ' extra hours a year &mdash; roughly ' + (extra / leader.hours * 100).toFixed(0) +
                    '% more development time than the championship leader gets.'
                    : 'The championship leader is handed the smallest allowance on the grid.') + '</p>';

            chart.querySelectorAll('.atr-bar').forEach(b => {
                const on = parseInt(b.dataset.pos, 10) === pos;
                b.classList.toggle('active', on);
            });
        }

        chart.addEventListener('click', e => {
            const bar = e.target.closest('.atr-bar');
            if (bar) show(parseInt(bar.dataset.pos, 10));
        });
        chart.addEventListener('mouseover', e => {
            const bar = e.target.closest('.atr-bar');
            if (bar) show(parseInt(bar.dataset.pos, 10));
        });
        chart.addEventListener('focusin', e => {
            const bar = e.target.closest('.atr-bar');
            if (bar) show(parseInt(bar.dataset.pos, 10));
        });

        show(10);
    }

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', () => { init(); initATR(); });
    } else {
        init();
        initATR();
    }
})();
