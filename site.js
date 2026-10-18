const motion = matchMedia('(prefers-reduced-motion: reduce)');
if (!motion.matches && 'IntersectionObserver' in window) {
  const observer = new IntersectionObserver(entries => entries.forEach(entry => {
    if (entry.isIntersecting) { entry.target.classList.add('visible'); observer.unobserve(entry.target); }
  }), {threshold: 0.08});
  document.documentElement.classList.add('motion-ready');
  document.querySelectorAll('.reveal').forEach(el => observer.observe(el));
}
const toggle = document.querySelector('#motion-toggle');
if (toggle) {
  const updateMotion = () => {
    document.body.classList.toggle('paused', motion.matches);
    toggle.hidden = motion.matches;
  };
  updateMotion();
  motion.addEventListener('change', updateMotion);
  toggle.addEventListener('click', () => {
    const paused = document.body.classList.toggle('paused');
    toggle.textContent = paused ? '▶' : 'Ⅱ';
    toggle.setAttribute('aria-label', paused ? 'Play animation' : 'Pause animation');
  });
}
const year = document.querySelector('#year');
if (year) year.textContent = new Date().getFullYear();
