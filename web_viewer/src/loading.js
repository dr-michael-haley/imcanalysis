// Only the latest operation may update or dismiss the indicator. In particular,
// a cancelled ROI request must not dismiss the next ROI's loading state.
export function createLoadingState(publish) {
  let revision = 0;
  return {
    begin(title, detail) {
      const request = ++revision;
      publish({ title, detail });
      return {
        update(detail) { if (request === revision) publish({ title, detail }); },
        finish() { if (request === revision) { revision++; publish(null); } },
      };
    },
  };
}

// Yield a task, rather than just a microtask, so input, painting and the loading
// animation can proceed during canvas composition, including in background tabs.
export function yieldToBrowser() {
  return globalThis.scheduler?.yield ? globalThis.scheduler.yield() : new Promise(resolve => setTimeout(resolve, 0));
}
