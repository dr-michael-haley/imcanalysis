import test from "node:test";
import assert from "node:assert/strict";
import { setTimeout as wait } from "node:timers/promises";
import { createReadTracker } from "../src/loading.mjs";

test("tracks pending reads, bytes and ranges while preserving store methods", async () => {
	const events = [];
	let release;
	const gate = new Promise((resolve) => {
		release = resolve;
	});
	class Store {
		#keys = ["one"];
		listKeys() {
			return this.#keys;
		}
		async get() {
			await gate;
			return new Uint8Array(4);
		}
		async getRange(key, range, options) {
			assert.equal(key, "/two");
			assert.equal(options.marker, true);
			return new Uint8Array(range.length);
		}
	}
	const tracker = createReadTracker((e) => events.push(e), 0);
	const store = tracker.wrap(new Store());
	assert.deepEqual(store.listKeys(), ["one"]);
	const first = store.get("/one");
	await wait(10);
	assert.equal(events.at(-1).pending, 1);
	assert.equal(events.at(-1).path, "/one");
	await store.getRange("/two", { offset: 0, length: 3 }, { marker: true });
	release();
	assert.equal((await first).byteLength, 4);
	await wait(10);
	assert.equal(events.at(-1).pending, 0);
	assert.equal(events.at(-1).reads, 2);
	assert.equal(events.at(-1).bytes, 7);
	tracker.dispose();
});

test("cancellation rejects outstanding reads, blocks new reads and ignores late results", async () => {
	const events = [];
	let release,
		calls = 0;
	const tracker = createReadTracker((e) => events.push(e), 0);
	const store = tracker.wrap({
		get: () => {
			calls++;
			return new Promise((resolve) => {
				release = resolve;
			});
		},
	});
	const pending = store.get("/slow");
	const rejection = assert.rejects(pending, { name: "AbortError" });
	await wait(10);
	tracker.dispose();
	await rejection;
	const before = events.length;
	release(new Uint8Array(100));
	await assert.rejects(store.get("/again"), { name: "AbortError" });
	await wait(10);
	assert.equal(calls, 1);
	assert.equal(events.length, before);
});

test("missing metadata and failed reads clear pending without inflating completed reads", async () => {
	let activity;
	const tracker = createReadTracker((e) => {
		activity = e;
	}, 0);
	const store = tracker.wrap({
		async get(key) {
			if (key === "/bad") throw new Error("Unreadable");
		},
	});
	assert.equal(store.getRange, undefined);
	assert.equal(await store.get("/missing"), undefined);
	await assert.rejects(store.get("/bad"), /Unreadable/);
	await wait(10);
	assert.equal(activity.pending, 0);
	assert.equal(activity.reads, 0);
	tracker.dispose();
});
