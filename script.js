// ============================================
// F1 DATA ANALYTICS - INTERACTIVE SCRIPTS
// ============================================

document.addEventListener('DOMContentLoaded', function () {
    // Initialize all components
    initNavigation();
    initScrollAnimations();
    initCounters();
    initImageModal();
    initParallax();
    initAeroCalculator();
    initSimTabs();
    initFooterYear();
});

// ============================================
// NAVIGATION
// ============================================

function initNavigation() {
    const navToggle = document.getElementById('navToggle');
    const navLinks = document.querySelector('.nav-links');
    const links = document.querySelectorAll('.nav-link');

    // Mobile toggle
    if (navToggle) {
        navToggle.addEventListener('click', () => {
            navLinks.classList.toggle('active');
            navToggle.classList.toggle('active');
        });
    }

    // Smooth scroll and active state
    links.forEach(link => {
        link.addEventListener('click', (e) => {
            e.preventDefault();
            const targetId = link.getAttribute('href');
            const target = document.querySelector(targetId);

            if (target) {
                target.scrollIntoView({ behavior: 'smooth' });

                // Update active state
                links.forEach(l => l.classList.remove('active'));
                link.classList.add('active');

                // Close mobile menu
                navLinks.classList.remove('active');
            }
        });
    });

    // Update active link on scroll (rAF-throttled)
    let scrollTicking = false;
    window.addEventListener('scroll', () => {
        if (scrollTicking) return;
        scrollTicking = true;
        requestAnimationFrame(() => {
            scrollTicking = false;
            onScroll();
        });
    }, { passive: true });

    function onScroll() {
        const sections = document.querySelectorAll('.section, .hero');
        const scrollPos = window.scrollY + 150;

        sections.forEach(section => {
            if (section.offsetTop <= scrollPos &&
                section.offsetTop + section.offsetHeight > scrollPos) {
                const id = section.getAttribute('id');
                links.forEach(link => {
                    link.classList.remove('active');
                    if (link.getAttribute('href') === `#${id}`) {
                        link.classList.add('active');
                    }
                });
            }
        });

        // Navbar background on scroll
        const navbar = document.querySelector('.navbar');
        if (window.scrollY > 50) {
            navbar.style.background = 'rgba(10, 10, 15, 0.95)';
        } else {
            navbar.style.background = 'rgba(10, 10, 15, 0.9)';
        }
    }
}

// ============================================
// SCROLL ANIMATIONS
// ============================================

function initScrollAnimations() {
    const observerOptions = {
        root: null,
        rootMargin: '0px',
        threshold: 0.1
    };

    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.classList.add('visible');

                // Trigger counter animation if present
                const counters = entry.target.querySelectorAll('[data-count]');
                counters.forEach(counter => animateCounter(counter));
            }
        });
    }, observerOptions);

    // Observe elements
    const animateElements = document.querySelectorAll(
        '.section-header, .analysis-card, .gforce-main, .gallery-item, ' +
        '.explanation-card, .aero-card, .metric-card, .strategy-card, ' +
        '.spec-block, .mfr-card, .tech-item, .stats-list li, ' +
        '.calc-panel, .media-card, .coeff'
    );

    animateElements.forEach(el => {
        el.style.opacity = '0';
        el.style.transform = 'translateY(30px)';
        el.style.transition = 'opacity 0.6s ease, transform 0.6s ease';
        observer.observe(el);
    });

    // Add visible state styles
    const style = document.createElement('style');
    style.textContent = `
        .visible {
            opacity: 1 !important;
            transform: translateY(0) !important;
        }
    `;
    document.head.appendChild(style);
}

// ============================================
// ANIMATED COUNTERS
// ============================================

function initCounters() {
    // Counters will be triggered by scroll observer
}

function animateCounter(element) {
    if (element.classList.contains('counted')) return;

    const target = parseInt(element.getAttribute('data-count'));
    const duration = 2000;
    const step = target / (duration / 16);
    let current = 0;

    element.classList.add('counted');

    const timer = setInterval(() => {
        current += step;
        if (current >= target) {
            current = target;
            clearInterval(timer);
        }
        element.textContent = Math.floor(current);
    }, 16);
}

// ============================================
// IMAGE MODAL
// ============================================

function initImageModal() {
    const modal = document.getElementById('imageModal');
    const modalImg = document.getElementById('modalImage');
    const modalCaption = document.getElementById('modalCaption');
    const closeBtn = document.querySelector('.modal-close');

    // All clickable images
    const images = document.querySelectorAll('.card-image img, .gforce-main img, .gallery-item img, .aero-card img, .atr-visual img, .strategy-card img, .engine-visuals img, .media-card img');

    images.forEach(img => {
        img.style.cursor = 'pointer';
        img.addEventListener('click', () => {
            modal.classList.add('active');
            modalImg.src = img.src;
            modalCaption.textContent = img.alt;
            document.body.style.overflow = 'hidden';
        });
    });

    // Close modal
    function closeModal() {
        modal.classList.remove('active');
        document.body.style.overflow = '';
    }

    if (closeBtn) {
        closeBtn.addEventListener('click', closeModal);
    }

    modal.addEventListener('click', (e) => {
        if (e.target === modal) closeModal();
    });

    document.addEventListener('keydown', (e) => {
        if (e.key === 'Escape') closeModal();
    });
}

// ============================================
// PARALLAX EFFECTS
// ============================================

function initParallax() {
    const hero = document.querySelector('.hero');
    const speedIndicator = document.querySelector('.speed-indicator');

    window.addEventListener('scroll', () => {
        const scrolled = window.scrollY;

        if (speedIndicator && scrolled < window.innerHeight) {
            speedIndicator.style.transform = `translateY(${scrolled * 0.3}px)`;
        }
    }, { passive: true });
}

// ============================================
// SPEED INDICATOR ANIMATION
// ============================================

(function animateSpeedIndicator() {
    const speedValue = document.querySelector('.speed-value');
    if (!speedValue) return;

    let speed = 0;
    let increasing = true;

    setInterval(() => {
        if (increasing) {
            speed += Math.random() * 5;
            if (speed >= 350) {
                speed = 350;
                increasing = false;
            }
        } else {
            speed -= Math.random() * 3;
            if (speed <= 280) {
                speed = 280;
                increasing = true;
            }
        }
        speedValue.textContent = Math.floor(speed);
    }, 100);
})();

// ============================================
// SMOOTH REVEAL ON LOAD
// ============================================

window.addEventListener('load', () => {
    document.body.classList.add('loaded');

    // Animate hero elements
    const heroElements = document.querySelectorAll('.hero-badge, .hero-title, .hero-subtitle, .hero-stats, .hero-cta, .hero-visual');
    heroElements.forEach((el, i) => {
        el.style.opacity = '0';
        el.style.transform = 'translateY(30px)';
        setTimeout(() => {
            el.style.transition = 'opacity 0.8s ease, transform 0.8s ease';
            el.style.opacity = '1';
            el.style.transform = 'translateY(0)';
        }, 200 + (i * 150));
    });
});

// ============================================
// TYPING EFFECT (Optional enhancement)
// ============================================

function typeWriter(element, text, speed = 50) {
    let i = 0;
    element.textContent = '';

    function type() {
        if (i < text.length) {
            element.textContent += text.charAt(i);
            i++;
            setTimeout(type, speed);
        }
    }
    type();
}

// ============================================
// LAZY LOADING FALLBACK
// ============================================

if ('loading' in HTMLImageElement.prototype) {
    // Browser supports native lazy loading
    const images = document.querySelectorAll('img[loading="lazy"]');
    images.forEach(img => {
        img.src = img.src;
    });
} else {
    // Fallback for older browsers
    const script = document.createElement('script');
    script.src = 'https://cdnjs.cloudflare.com/ajax/libs/lazysizes/5.3.2/lazysizes.min.js';
    document.body.appendChild(script);
}

// ============================================
// ERROR HANDLING FOR IMAGES
// ============================================

document.querySelectorAll('img').forEach(img => {
    img.addEventListener('error', function () {
        this.style.display = 'none';
        console.warn('Image failed to load:', this.src);
    });
});

console.log('🏎️ F1 Data Analytics Portfolio - Loaded Successfully');

// ============================================
// INTERACTIVE AERO CALCULATOR
// Physics ported from aero_calculator.html
// ============================================

function initAeroCalculator() {
    const panel = document.querySelector('.calc-panel');
    if (!panel) return;

    const RHO = 1.225;          // air density, kg/m^3
    const FRONTAL_AREA = 1.5;   // m^2
    const CAR_WEIGHT = 798;     // kg, 2024 regulations
    const MAX_DOWNFORCE = 25000; // N, gauge full scale
    const MAX_DRAG = 8000;       // N, gauge full scale
    const MAX_POWER = 750;       // kW, gauge full scale

    const SETUPS = {
        monaco: { wing: 32, ride: 30, front: 15, baseCl: 4.00, baseCd: 1.35, speed: 160 },
        silverstone: { wing: 20, ride: 35, front: 10, baseCl: 3.25, baseCd: 1.16, speed: 250 },
        monza: { wing: 8, ride: 45, front: 5, baseCl: 2.50, baseCd: 0.95, speed: 320 }
    };

    let baseCl = SETUPS.silverstone.baseCl;
    let baseCd = SETUPS.silverstone.baseCd;

    const inputs = {
        speed: document.getElementById('calcSpeed'),
        wing: document.getElementById('calcWing'),
        ride: document.getElementById('calcRide'),
        front: document.getElementById('calcFront')
    };
    const outs = {
        speed: document.getElementById('calcSpeedOut'),
        wing: document.getElementById('calcWingOut'),
        ride: document.getElementById('calcRideOut'),
        front: document.getElementById('calcFrontOut')
    };

    const el = id => document.getElementById(id);
    const fmt = n => n.toLocaleString('en-US', { maximumFractionDigits: 0 });

    function rideHeightFactor(ride) {
        if (ride < 25) return 0.8 + (ride / 25) * 0.2;   // porpoising / stall region
        if (ride < 40) return 1.0 + (40 - ride) * 0.01;  // optimal ground effect
        return 1.0 - (ride - 40) * 0.005;                // floor works less
    }

    function update() {
        const speed = parseFloat(inputs.speed.value);
        const wing = parseFloat(inputs.wing.value);
        const ride = parseFloat(inputs.ride.value);
        const front = parseFloat(inputs.front.value);

        outs.speed.textContent = speed;
        outs.wing.textContent = wing;
        outs.ride.textContent = ride;
        outs.front.textContent = front;

        const Cl = baseCl
            * (1 + (wing - 20) * 0.015)
            * rideHeightFactor(ride)
            * (1 + (front - 10) * 0.008);
        const Cd = baseCd * (1 + (wing - 20) * 0.01) * (1 + front * 0.003);

        const v = speed / 3.6;
        const q = 0.5 * RHO * v * v;
        const downforce = q * FRONTAL_AREA * Cl;
        const drag = q * FRONTAL_AREA * Cd;
        const power = drag * v / 1000;

        el('calcDownforce').textContent = fmt(downforce);
        el('calcDownforceKg').textContent = fmt(downforce / 9.81);
        el('calcDfRatio').textContent = Math.round(downforce / (CAR_WEIGHT * 9.81) * 100) + '%';
        el('calcDrag').textContent = fmt(drag);
        el('calcLd').textContent = (Cl / Cd).toFixed(2);
        el('calcPower').textContent = fmt(power);
        el('calcHp').textContent = fmt(power * 1.341);
        el('calcCl').textContent = Cl.toFixed(2);
        el('calcCd').textContent = Cd.toFixed(2);

        const pct = (value, max) => Math.min(value / max * 100, 100).toFixed(0) + '%';
        el('calcDfBar').style.setProperty('--progress', pct(downforce, MAX_DOWNFORCE));
        el('calcDragBar').style.setProperty('--progress', pct(drag, MAX_DRAG));
        el('calcPowerBar').style.setProperty('--progress', pct(power, MAX_POWER));
    }

    Object.keys(inputs).forEach(key => {
        inputs[key].addEventListener('input', update);
    });

    document.querySelectorAll('.preset-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            const setup = SETUPS[btn.dataset.preset];
            if (!setup) return;

            document.querySelectorAll('.preset-btn').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');

            baseCl = setup.baseCl;
            baseCd = setup.baseCd;
            inputs.speed.value = setup.speed;
            inputs.wing.value = setup.wing;
            inputs.ride.value = setup.ride;
            inputs.front.value = setup.front;
            update();
        });
    });

    update();
}

// ============================================
// RACE SIMULATION TABS
// ============================================

function initSimTabs() {
    const tabs = Array.from(document.querySelectorAll('.sim-tab'));
    if (!tabs.length) return;

    function select(tab) {
        tabs.forEach(t => {
            const isActive = t === tab;
            t.classList.toggle('active', isActive);
            t.setAttribute('aria-selected', String(isActive));
            const panel = document.getElementById(t.getAttribute('aria-controls'));
            if (panel) {
                panel.hidden = !isActive;
                panel.classList.toggle('active', isActive);
            }
        });
    }

    tabs.forEach((tab, i) => {
        tab.addEventListener('click', () => select(tab));
        tab.addEventListener('keydown', e => {
            if (e.key !== 'ArrowRight' && e.key !== 'ArrowLeft') return;
            e.preventDefault();
            const next = tabs[(i + (e.key === 'ArrowRight' ? 1 : tabs.length - 1)) % tabs.length];
            next.focus();
            select(next);
        });
    });
}

// ============================================
// FOOTER YEAR
// ============================================

function initFooterYear() {
    const year = document.getElementById('footerYear');
    if (year) year.textContent = new Date().getFullYear();
}
