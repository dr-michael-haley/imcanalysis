export const defaultReview = () => ({ tab: 'embedding', height: 280 });
export function validateReview(value = defaultReview()) {
  if (!value || !['cells', 'embedding', 'remap', 'plots', 'flags', 'bookmarks', 'figure'].includes(value.tab) || !Number.isFinite(value.height) || value.height < 180 || value.height > 700) throw new Error('Invalid workspace review panel settings.');
  return { tab: value.tab, height: value.height };
}
export function createReviewPanel(state, $, redraw) {
  const tabs = ['cells', 'embedding', 'remap', 'plots', 'flags', 'bookmarks', 'figure'];
  function sync() {
    const spec = state.review;
    $('review-content').style.setProperty('--review-height', `${spec.height}px`);
    $('review-resize').setAttribute('aria-valuenow', String(Math.round(spec.height)));
    for (const tab of tabs) {
      $(`review-tab-${tab}`).setAttribute('aria-selected', String(tab === spec.tab));
      $(`review-tab-${tab}`).tabIndex = tab === spec.tab ? 0 : -1;
      $(`review-${tab}`).hidden = tab !== spec.tab;
    }
    requestAnimationFrame(redraw);
  }
  function select(tab) { state.review.tab = tab; $('review-panels').open = true; sync(); }
  for (const tab of tabs) {
    $(`review-tab-${tab}`).addEventListener('click', () => select(tab));
    $(`review-tab-${tab}`).addEventListener('keydown', event => {
      if (!['ArrowLeft', 'ArrowRight', 'Home', 'End'].includes(event.key)) return;
      event.preventDefault();
      const next = event.key === 'Home' ? tabs[0] : event.key === 'End' ? tabs.at(-1) : tabs[(tabs.indexOf(tab) + (event.key === 'ArrowRight' ? 1 : tabs.length - 1)) % tabs.length];
      select(next); $(`review-tab-${next}`).focus();
    });
  }
  const handle = $('review-resize'); let drag;
  handle.addEventListener('pointerdown', event => { drag = { y: event.clientY, height: state.review.height }; handle.setPointerCapture(event.pointerId); event.preventDefault(); });
  handle.addEventListener('pointermove', event => { if (drag) { state.review.height = Math.max(180, Math.min(700, drag.height + drag.y - event.clientY)); sync(); } });
  for (const name of ['pointerup', 'pointercancel']) handle.addEventListener(name, () => { drag = null; });
  handle.addEventListener('keydown', event => {
    if (!['ArrowUp', 'ArrowDown', 'Home', 'End'].includes(event.key)) return;
    event.preventDefault(); state.review.height = event.key === 'Home' ? 180 : event.key === 'End' ? 700 : Math.max(180, Math.min(700, state.review.height + (event.key === 'ArrowUp' ? 30 : -30))); sync();
  });
  $('review-height-reset').addEventListener('click', () => { state.review.height = 280; sync(); });
  sync(); return { sync, select };
}
