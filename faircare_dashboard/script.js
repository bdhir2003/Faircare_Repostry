/*
========================================================================
   FAIRCARE DASHBOARD — INTERACTION SCRIPT
========================================================================
*/
document.addEventListener('DOMContentLoaded', () => {

    // Mermaid init
    if (typeof mermaid !== 'undefined') {
        mermaid.initialize({
            startOnLoad: true, theme: 'dark', securityLevel: 'loose',
            flowchart: { useMaxWidth: true, htmlLabels: true, curve: 'basis' },
            themeVariables: { background: '#0b0f1a', primaryColor: '#141b2a', primaryTextColor: '#e8ecf4', primaryBorderColor: '#6366f1', lineColor: '#06b6d4', secondaryColor: '#1e293b', tertiaryColor: '#0f172a' }
        });
    }

    // Delegated Tab Switcher
    document.addEventListener('click', (e) => {
        const btn = e.target.closest('.tab-btn');
        if (!btn) return;
        const container = btn.closest('.tab-container');
        if (!container) return;
        const targetId = btn.getAttribute('data-tab');
        container.querySelectorAll(':scope > .tab-buttons > .tab-btn').forEach(b => b.classList.remove('active'));
        container.querySelectorAll(':scope > .tab-pane').forEach(p => p.classList.remove('active'));
        btn.classList.add('active');
        const pane = container.querySelector('#' + targetId);
        if (pane) pane.classList.add('active');
    });

    // Collapsible / Accordion Toggler
    document.addEventListener('click', (e) => {
        const trigger = e.target.closest('.collapsible-trigger');
        if (!trigger) return;
        const content = trigger.nextElementSibling;
        if (!content || !content.classList.contains('collapsible-content')) return;
        trigger.classList.toggle('open');
        content.classList.toggle('open');
    });

    // Scroll-aware navigation
    const sections = document.querySelectorAll('section[id]');
    const navLinks = document.querySelectorAll('.nav-link');
    if (sections.length && navLinks.length) {
        const obs = new IntersectionObserver((entries) => {
            entries.forEach(entry => {
                if (entry.isIntersecting) {
                    const id = entry.target.getAttribute('id');
                    navLinks.forEach(l => {
                        l.classList.remove('active');
                        if (l.getAttribute('href') === '#' + id) l.classList.add('active');
                    });
                }
            });
        }, { rootMargin: '-25% 0px -75% 0px' });
        sections.forEach(s => obs.observe(s));
    }
});
