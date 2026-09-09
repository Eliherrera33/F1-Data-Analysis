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
    initSimTabs();
    initFooterYear();
    initVideos();
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
        '.media-card'
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
    // The hero counters never ran: animateCounter is only called for elements
    // the reveal observer watches, and .stat-card was not in that selector, so
    // every stat sat at its placeholder 0. They get their own observer here,
    // which also keeps them clear of the reveal animation's opacity handling.
    const counters = document.querySelectorAll('[data-count]');
    if (!counters.length) return;

    // The hero stats are above the fold, so run anything already on screen
    // straight away rather than waiting on an observer callback that may never
    // arrive (embedded webviews and some headless renderers never deliver one).
    function inViewport(el) {
        const r = el.getBoundingClientRect();
        return r.height > 0 && r.bottom > 0 && r.top < window.innerHeight;
    }

    const pending = [];
    counters.forEach(c => (inViewport(c) ? animateCounter(c) : pending.push(c)));
    if (!pending.length) return;

    if (!('IntersectionObserver' in window)) {
        pending.forEach(animateCounter);
        return;
    }

    const observer = new IntersectionObserver((entries, obs) => {
        entries.forEach(entry => {
            if (!entry.isIntersecting) return;
            animateCounter(entry.target);
            obs.unobserve(entry.target);
        });
    }, { threshold: 0.4 });

    pending.forEach(c => observer.observe(c));

    // Belt and braces: a scroll listener catches anything the observer misses.
    function sweep() {
        for (let i = pending.length - 1; i >= 0; i--) {
            if (inViewport(pending[i])) {
                animateCounter(pending[i]);
                observer.unobserve(pending[i]);
                pending.splice(i, 1);
            }
        }
        if (!pending.length) window.removeEventListener('scroll', sweep);
    }
    window.addEventListener('scroll', sweep, { passive: true });
}

function animateCounter(element) {
    if (element.classList.contains('counted')) return;
    element.classList.add('counted');

    const target = parseInt(element.getAttribute('data-count'), 10);
    if (!isFinite(target)) return;

    if (window.matchMedia('(prefers-reduced-motion: reduce)').matches) {
        element.textContent = target.toLocaleString();
        return;
    }

    // Ease out, and drive it off the clock rather than a fixed increment so it
    // always finishes on the exact number.
    const duration = 1600;
    const start = performance.now();

    function tick(now) {
        const t = Math.min((now - start) / duration, 1);
        const eased = 1 - Math.pow(1 - t, 3);
        element.textContent = Math.round(target * eased).toLocaleString();
        if (t < 1) requestAnimationFrame(tick);
    }
    requestAnimationFrame(tick);

    // Guarantee the final value even where rAF never runs - a backgrounded tab,
    // an embedded webview, a headless renderer. The animation is decoration; the
    // number is not optional.
    setTimeout(() => {
        element.textContent = target.toLocaleString();
    }, duration + 250);
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
    const images = document.querySelectorAll('.card-image img, .gforce-main img, .gallery-item img, .atr-visual img, .strategy-card img, .engine-visuals img, .media-card img');

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


// ============================================
// VIDEO FIGURES
// ============================================

function initVideos() {
    const videos = Array.from(document.querySelectorAll('.media-video'));
    if (!videos.length) return;

    const reduced = window.matchMedia('(prefers-reduced-motion: reduce)').matches;

    // The markup carries `autoplay` so the clips still run without JS. Once we
    // are here the script takes over: left to itself, autoplay starts videos
    // inside hidden tab panels and races the visibility check.
    videos.forEach(v => {
        v.removeAttribute('autoplay');
        v.autoplay = false;
    });

    if (reduced) {
        // An autoplaying loop is precisely what this preference asks us not to
        // do. Show the poster, hand over the controls, let the visitor choose.
        videos.forEach(v => {
            v.loop = false;
            v.controls = true;
            v.pause();
        });
        return;
    }

    // Visibility is measured when it is needed rather than cached from the last
    // observer callback: unhiding a tab panel changes layout without firing one,
    // which previously left the newly revealed clip paused.
    function onScreen(v) {
        if (v.closest('[hidden]')) return false;
        const r = v.getBoundingClientRect();
        return r.height > 0 && r.bottom > 0 && r.top < window.innerHeight;
    }

    function update() {
        videos.forEach(v => {
            if (onScreen(v)) {
                if (v.paused) {
                    const p = v.play();
                    if (p && p.catch) p.catch(() => { /* blocked; poster stands */ });
                }
            } else if (!v.paused) {
                v.pause();
            }
        });
    }

    let ticking = false;
    function schedule() {
        if (ticking) return;
        ticking = true;
        requestAnimationFrame(() => { ticking = false; update(); });
    }

    window.addEventListener('scroll', schedule, { passive: true });
    window.addEventListener('resize', schedule, { passive: true });
    document.querySelectorAll('.sim-tab').forEach(tab => {
        tab.addEventListener('click', () => setTimeout(update, 0));
    });
    document.addEventListener('visibilitychange', () => {
        if (document.hidden) videos.forEach(v => v.pause());
        else update();
    });

    videos.forEach(v => v.addEventListener('loadeddata', schedule, { once: true }));
    update();
}
