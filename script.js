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

    // Update active link on scroll
    window.addEventListener('scroll', () => {
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
    });
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
        '.spec-block, .mfr-card, .tech-item, .stats-list li'
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
    const images = document.querySelectorAll('.card-image img, .gforce-main img, .gallery-item img, .aero-card img, .atr-visual img, .strategy-card img, .engine-visuals img');

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
    });
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
