export class ReaderClient {
  constructor() {
    this.worker = new Worker(new URL('./data.worker.js', import.meta.url), { type: 'module' });
    this.pending = new Map();
    this.next = 0;
    this.worker.onmessage = ({ data }) => {
      const pending = this.pending.get(data.id);
      if (!pending) return;
      if (data.progress) { pending.progress?.(data.progress); return; }
      this.pending.delete(data.id);
      if (data.error) pending.reject(new Error(data.error));
      else pending.resolve(data.result);
    };
    this.worker.onerror = event => this.stop(new Error(event.message || 'AnnData worker failed. Reopen the file to retry.'));
    this.worker.onmessageerror = () => this.stop(new Error('Could not receive AnnData results. Reopen the file to retry.'));
  }
  call(method, ...args) {
    return this.callProgress(method, null, ...args);
  }
  callProgress(method, progress, ...args) {
    if (this.failure) return Promise.reject(this.failure);
    return new Promise((resolve, reject) => {
      const id = ++this.next;
      this.pending.set(id, { resolve, reject, progress });
      this.worker.postMessage({ id, method, args });
    });
  }
  stop(error = new Error('Dataset closed.')) {
    this.failure = error;
    this.worker.terminate();
    for (const pending of this.pending.values()) pending.reject(error);
    this.pending.clear();
  }
}
