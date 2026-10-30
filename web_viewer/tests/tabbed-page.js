// Existing feature checks now navigate the real sidebar tabs before operating
// a control. Visibility assertions still inspect the current UI without switching.
export function tabbedPage(page) {
  const wrap = locator => new Proxy(locator, {
    get(target, key) {
      const value = target[key];
      if (typeof value !== 'function') return value;
      if (['click', 'fill', 'selectOption', 'setInputFiles', 'check', 'uncheck', 'press', 'scrollIntoViewIfNeeded'].includes(key)) return async (...args) => {
        const panel = await target.first().evaluate(node => node.closest('[role="tabpanel"]')?.id);
        if (panel?.startsWith('panel-')) await page.locator(`#tab-${panel.slice(6)}`).click();
        else if (panel?.startsWith('review-')) {
          if (!await page.locator('#review-panels').evaluate(node => node.open)) await page.locator('#review-panels > summary').click();
          await page.locator(`#review-tab-${panel.slice(7)}`).click();
        }
        if (await target.first().evaluate(node => Boolean(node.closest('#workspace-file-tools')) && node.tagName !== 'SUMMARY') && !await page.locator('#workspace-file-tools').evaluate(node => node.open)) await page.locator('#workspace-file-tools > summary').click();
        return value.apply(target, args);
      };
      if (['first', 'last', 'nth', 'locator', 'filter'].includes(key)) return (...args) => wrap(value.apply(target, args));
      return value.bind(target);
    },
  });
  return new Proxy(page, { get(target, key) { if (key === 'locator') return (...args) => wrap(target.locator(...args)); const value = target[key]; return typeof value === 'function' ? value.bind(target) : value; } });
}
