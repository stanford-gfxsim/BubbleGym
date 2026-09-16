/* Theme toggle for the per-figure sub-pages.
 *
 * The choice lives in localStorage under the same key the project page uses,
 * and localStorage is per-origin, so a reader who picked dark on the project
 * page arrives here already dark. The <head> of each page sets data-theme
 * before first paint; this file only handles the button and the live response
 * to a system-preference change. */
(function () {
  var root = document.documentElement;
  var btn = document.getElementById('theme-toggle');
  var icon = document.getElementById('theme-icon');
  var label = document.getElementById('theme-label');
  var media = window.matchMedia
    ? window.matchMedia('(prefers-color-scheme: dark)') : null;

  /* The resolved theme: an explicit choice if one was made, otherwise the
     system preference. The stylesheet follows the same rule. */
  function current() {
    var set = root.getAttribute('data-theme');
    if (set === 'dark' || set === 'light') return set;
    return (media && media.matches) ? 'dark' : 'light';
  }

  function paintButton() {
    if (!btn) return;
    var dark = current() === 'dark';
    /* The button offers the other mode, so it shows the one you'd switch to. */
    icon.innerHTML = dark ? '&#9788;' : '&#9790;';
    label.textContent = dark ? 'Light' : 'Dark';
    btn.setAttribute('aria-pressed', String(dark));
  }

  if (btn) {
    btn.addEventListener('click', function () {
      var next = current() === 'dark' ? 'light' : 'dark';
      root.setAttribute('data-theme', next);
      try { localStorage.setItem('bubblegym-theme', next); } catch (e) {}
      paintButton();
    });
  }

  /* Follow the system while no explicit choice has been made. */
  if (media && media.addEventListener) {
    media.addEventListener('change', function () {
      if (!root.getAttribute('data-theme')) paintButton();
    });
  }

  paintButton();
})();
