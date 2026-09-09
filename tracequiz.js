/* ============================================================
   F1 DATA ANALYTICS - READ THE TRACE
   ------------------------------------------------------------
   A blind test on real telemetry. Two kinds of question:

     * which circuit is this?  - one anonymised speed trace
     * which lap is faster?    - two laps from the same session

   The point is not the score. It is that after six questions
   the visitor has learned what a speed trace says: a low top
   speed and a lot of braking is a street circuit, three long
   flat-out stretches is Monza, and the faster lap is usually the
   one that carries more speed through a corner, not the one with
   the higher peak.

   Traces are plotted against lap fraction, not metres, so the
   lap length does not give the circuit away.
   ============================================================ */

(function () {
    'use strict';

    const U = window.F1UI;
    const root = document.getElementById('traceQuiz');
    if (!U || !root) return;

    const COUNT = 6;
    let pool = [];        // circuits from quiz.json
    let questions = [];
    let index = 0;
    let score = 0;
    let answered = false;

    const shuffle = a => { for (let i = a.length - 1; i > 0; i--) { const j = Math.floor(Math.random() * (i + 1)); [a[i], a[j]] = [a[j], a[i]]; } return a; };
    const pick = (arr, k) => shuffle(arr.slice()).slice(0, k);

    // ------------------------------------------------------------------
    // question generation
    // ------------------------------------------------------------------
    function rankIn(circuit, field, dir) {
        const vals = pool.map(c => c.laps[0][field]);
        const v = circuit.laps[0][field];
        const sorted = vals.slice().sort((a, b) => dir === 'max' ? b - a : a - b);
        return sorted.indexOf(v);     // 0 = the most extreme
    }

    function circuitExplanation(c) {
        const L = c.laps[0];
        const bits = [];
        const topRank = rankIn(c, 'top', 'max'), topRankLow = rankIn(c, 'top', 'min');
        const ftRank = rankIn(c, 'full_throttle', 'max'), ftRankLow = rankIn(c, 'full_throttle', 'min');
        if (topRankLow === 0) bits.push('the lowest top speed of any circuit here — ' + L.top + ' km/h, which only a street circuit produces');
        else if (topRank === 0) bits.push('the highest top speed in the set at ' + L.top + ' km/h');
        if (ftRank === 0) bits.push(L.full_throttle + '% of the lap at full throttle, more than anywhere else');
        else if (ftRankLow === 0) bits.push('only ' + L.full_throttle + '% of the lap at full throttle');
        if (!bits.length) bits.push(L.brake_events + ' distinct braking zones and a ' + L.top + ' km/h peak');
        return 'This is <b>' + c.label + '</b>: ' + bits.join(', ') + '. ' +
            'Pole was ' + L.abbr + ' in ' + U.fmtLap(L.laptime) + ' over ' + (c.length / 1000).toFixed(2) + ' km.';
    }

    function fasterExplanation(c, fast, slow, labelFast) {
        const n = fast.speed.length;
        // where did the faster lap carry the most speed? moving average of the difference
        const W = 10;
        let bestI = 0, bestV = -Infinity;
        for (let i = 0; i <= n - W; i++) {
            let s = 0;
            for (let k = 0; k < W; k++) s += fast.speed[i + k] - slow.speed[i + k];
            if (s / W > bestV) { bestV = s / W; bestI = i; }
        }
        const pct = Math.round((bestI + W / 2) / n * 100);
        const gap = (slow.laptime - fast.laptime).toFixed(3);
        const minFast = Math.min.apply(null, fast.speed), minSlow = Math.min.apply(null, slow.speed);
        const peakNote = fast.top < slow.top
            ? ' Note the slower lap actually had the higher peak speed (' + slow.top + ' vs ' + fast.top + ' km/h) — it lost the time in the corners, not on the straights.'
            : '';
        return '<b>Lap ' + labelFast + '</b> — ' + fast.abbr + ', ' + U.fmtLap(fast.laptime) + ' against ' +
            slow.abbr + '\'s ' + U.fmtLap(slow.laptime) + ', a gap of ' + gap + ' s. ' +
            'The biggest advantage was around <b>' + pct + '%</b> of the lap, where ' + fast.abbr +
            ' carried about ' + Math.round(bestV) + ' km/h more through the section' +
            (minFast > minSlow ? ' and a higher minimum corner speed (' + Math.round(minFast) + ' vs ' + Math.round(minSlow) + ' km/h)' : '') +
            '.' + peakNote;
    }

    function buildQuestions() {
        const circuits = shuffle(pool.slice());
        const qs = [];
        const forCircuit = circuits.slice(0, Math.ceil(COUNT / 2));
        const forFaster = circuits.slice(Math.ceil(COUNT / 2), COUNT).concat(
            circuits.slice(0, Math.max(0, COUNT - circuits.length)));

        for (const c of forCircuit) {
            const others = pick(pool.filter(o => o.key !== c.key), 2).map(o => o.label);
            qs.push({
                type: 'circuit', circuit: c,
                prompt: 'Which circuit is this?',
                traces: [{ speed: c.laps[0].speed, name: 'Lap', color: U.palette.teal }],
                options: shuffle([c.label].concat(others)),
                answer: c.label,
                explain: circuitExplanation(c),
            });
        }
        for (const c of forFaster) {
            const [a, b] = c.laps;
            const order = Math.random() < 0.5 ? [a, b] : [b, a];
            const fast = a.laptime <= b.laptime ? a : b, slow = fast === a ? b : a;
            const labelFast = order[0] === fast ? 'A' : 'B';
            qs.push({
                type: 'faster', circuit: c,
                prompt: 'Two laps from the same qualifying session. Which one is faster?',
                traces: [{ speed: order[0].speed, name: 'Lap A', color: U.palette.teal },
                         { speed: order[1].speed, name: 'Lap B', color: '#ff7a70' }],
                options: ['Lap A', 'Lap B'],
                answer: 'Lap ' + labelFast,
                explain: fasterExplanation(c, fast, slow, labelFast),
                hint: 'Same track, so the shapes match — look at where the lines separate.',
            });
        }
        return shuffle(qs).slice(0, COUNT);
    }

    // ------------------------------------------------------------------
    // drawing
    // ------------------------------------------------------------------
    function drawTrace(q, reveal) {
        const canvas = root.querySelector('.quiz-canvas');
        if (!canvas || !canvas.getBoundingClientRect().width) return;
        const { g, w, h } = U.setupCanvas(canvas, 2.35);
        g.clearRect(0, 0, w, h);
        const L = 42, R = 14, T = 22, Bm = 26;
        const pw = w - L - R, ph = h - T - Bm;

        let vmax = 0;
        for (const tr of q.traces) for (const v of tr.speed) vmax = Math.max(vmax, v);
        const vTop = Math.ceil(vmax / 50) * 50 + 10;
        const X = i => L + (i / (q.traces[0].speed.length - 1)) * pw;
        const Y = v => T + ph - (v / vTop) * ph;

        // well + grid
        g.fillStyle = 'rgba(255,255,255,0.025)'; g.fillRect(L, T, pw, ph);
        g.strokeStyle = U.palette.line; g.lineWidth = 1;
        for (let v = 0; v <= vTop; v += 50) {
            g.beginPath(); g.moveTo(L, Y(v)); g.lineTo(L + pw, Y(v)); g.stroke();
            U.label(g, String(v), L - 6, Y(v) + 3.5, { size: 8.5, color: U.palette.ink3, align: 'right' });
        }
        for (let p = 0; p <= 100; p += 25) {
            const x = L + (p / 100) * pw;
            g.beginPath(); g.moveTo(x, T); g.lineTo(x, T + ph); g.stroke();
            // the 100% tick would sit under the axis title
            if (p < 100) U.label(g, p + '%', x, h - 8, { size: 8.5, color: U.palette.ink3, align: 'center' });
        }
        U.label(g, 'km/h', L - 6, T - 8, { size: 8.5, color: U.palette.ink3, align: 'right' });
        U.label(g, 'lap distance →', L + pw, h - 8, { size: 8.5, color: U.palette.ink3, align: 'right' });

        // traces
        for (const tr of q.traces) {
            g.save();
            g.shadowColor = U.hexToRgba(tr.color.startsWith('#') ? tr.color : '#00d2be', 0.45);
            g.shadowBlur = 8;
            g.strokeStyle = tr.color; g.lineWidth = 2;
            g.lineJoin = 'round';
            g.beginPath();
            tr.speed.forEach((v, i) => (i ? g.lineTo(X(i), Y(v)) : g.moveTo(X(i), Y(v))));
            g.stroke();
            g.restore();
        }

        if (reveal && q.type === 'faster') {
            // shade where the faster lap is ahead
            const fast = q.traces[q.answer === 'Lap A' ? 0 : 1], slow = q.traces[q.answer === 'Lap A' ? 1 : 0];
            g.fillStyle = U.hexToRgba(fast.color, 0.16);
            g.beginPath();
            fast.speed.forEach((v, i) => (i ? g.lineTo(X(i), Y(Math.max(v, slow.speed[i]))) : g.moveTo(X(i), Y(Math.max(v, slow.speed[i])))));
            for (let i = slow.speed.length - 1; i >= 0; i--) g.lineTo(X(i), Y(slow.speed[i]));
            g.closePath(); g.fill();
        }
    }

    // ------------------------------------------------------------------
    // UI
    // ------------------------------------------------------------------
    function build() {
        root.innerHTML =
            '<div class="quiz-head">' +
            '<div><span class="quiz-kicker">Read the trace</span>' +
            '<h3 class="quiz-title">Can you read telemetry?</h3>' +
            '<p class="quiz-desc">Six questions on real qualifying laps. No labels, no lap lengths — just the ' +
            'speed trace and what it tells you.</p></div>' +
            '<div class="quiz-progress" aria-label="Progress"></div></div>' +
            '<div class="quiz-body">' +
            '<div class="quiz-stage"><canvas class="quiz-canvas" aria-label="Anonymised speed trace"></canvas>' +
            '<div class="quiz-legend"></div></div>' +
            '<div class="quiz-panel">' +
            '<p class="quiz-prompt"></p><p class="quiz-hint"></p>' +
            '<div class="quiz-options" role="group" aria-label="Answers"></div>' +
            '<div class="quiz-reveal" hidden><div class="quiz-verdict"></div><p class="quiz-explain"></p>' +
            '<button type="button" class="quiz-next">Next question <span aria-hidden="true">&rarr;</span></button></div>' +
            '<div class="quiz-summary" hidden><span class="quiz-score"></span><span class="quiz-tier"></span>' +
            '<p class="quiz-summary-text"></p><button type="button" class="quiz-restart">Run it again</button></div>' +
            '</div></div>';

        root.querySelector('.quiz-next').addEventListener('click', next);
        root.querySelector('.quiz-restart').addEventListener('click', restart);
        window.addEventListener('resize', () => { if (questions[index]) drawTrace(questions[index], answered); }, { passive: true });
    }

    function renderProgress() {
        const el = root.querySelector('.quiz-progress');
        el.innerHTML = questions.map((q, i) =>
            '<span class="quiz-dot' + (i < index ? (q.correct ? ' right' : ' wrong') : i === index ? ' now' : '') + '"></span>').join('') +
            '<span class="quiz-count">' + Math.min(index + 1, COUNT) + ' / ' + COUNT + '</span>';
    }

    function show() {
        const q = questions[index];
        answered = false;
        root.querySelector('.quiz-reveal').hidden = true;
        root.querySelector('.quiz-summary').hidden = true;
        root.querySelector('.quiz-prompt').textContent = q.prompt;
        root.querySelector('.quiz-hint').textContent = q.hint || (q.type === 'circuit'
            ? 'Top speed, how much of the lap is flat out, how many braking zones — the fingerprint is all there.' : '');
        root.querySelector('.quiz-legend').innerHTML = q.traces.map(t =>
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
        drawTrace(q, false);
    }

    function answer(choice, btn) {
        if (answered) return;
        answered = true;
        const q = questions[index];
        q.correct = choice === q.answer;
        if (q.correct) score++;
        root.querySelectorAll('.quiz-option').forEach(b => {
            b.disabled = true;
            if (b.textContent === q.answer) b.classList.add('right');
            else if (b === btn) b.classList.add('wrong');
        });
        const verdict = root.querySelector('.quiz-verdict');
        verdict.textContent = q.correct ? 'Correct' : 'Not quite';
        verdict.className = 'quiz-verdict ' + (q.correct ? 'right' : 'wrong');
        root.querySelector('.quiz-explain').innerHTML = q.explain;
        root.querySelector('.quiz-next').textContent = index === COUNT - 1 ? 'See your result' : 'Next question →';
        root.querySelector('.quiz-reveal').hidden = false;
        renderProgress();
        drawTrace(q, true);
        if (typeof window.gtag === 'function') {
            window.gtag('event', 'trace_quiz_answer', { control: q.type, value_label: q.correct ? 'right' : 'wrong' });
        }
    }

    function next() {
        if (index < COUNT - 1) { index++; show(); return; }
        // summary
        root.querySelector('.quiz-reveal').hidden = true;
        root.querySelector('.quiz-options').innerHTML = '';
        root.querySelector('.quiz-prompt').textContent = 'That is the set.';
        root.querySelector('.quiz-hint').textContent = '';
        const tier = score === 6 ? ['Pit wall', 'You read a trace the way a race engineer does. Every call was right.']
            : score >= 4 ? ['Garage', 'You can tell a street circuit from a power track and see where a lap is won. That is most of the job.']
            : score >= 2 ? ['Paddock', 'You are picking up the shapes. Watch the minimum corner speeds — that is where the time hides.']
            : ['Grandstand', 'The traces still look like noise. Run it again and watch how the low points, not the peaks, decide the lap.'];
        root.querySelector('.quiz-score').textContent = score + ' / ' + COUNT;
        root.querySelector('.quiz-tier').textContent = tier[0];
        root.querySelector('.quiz-summary-text').textContent = tier[1];
        root.querySelector('.quiz-summary').hidden = false;
        index = COUNT;
        renderProgress();
        if (typeof window.gtag === 'function') {
            window.gtag('event', 'trace_quiz_complete', { control: 'score', value_label: String(score) });
        }
    }

    function restart() {
        questions = buildQuestions(); index = 0; score = 0;
        show();
    }

    function init() {
        U.loadJSON('data/web/quiz.json').then(list => {
            pool = list.filter(c => c.laps && c.laps.length >= 2);
            if (pool.length < 3) throw new Error('not enough circuits');
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
