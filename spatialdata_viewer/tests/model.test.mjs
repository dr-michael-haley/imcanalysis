import test from "node:test";
import assert from "node:assert/strict";
import {
	createWorkspace,
	initialStack,
	validateWorkspace,
} from "../src/model.mjs";

const sdata = {
	coordinateSystems: ["a", "b"],
	images: {
		first: { coordinateSystems: ["a"] },
		second: { coordinateSystems: ["b"] },
	},
	labels: { cells: { coordinateSystems: ["a"] } },
	tables: {},
};
test("coordinate systems keep unrelated ROIs out of a view", () => {
	assert.deepEqual(
		initialStack(sdata, "a").entries.map((entry) => entry.source.elementKey),
		["first", "cells"],
	);
	assert.deepEqual(
		initialStack(sdata, "b").entries.map((entry) => entry.source.elementKey),
		["second"],
	);
});
test("view round trip preserves channels and rejects invalid or foreign settings", () => {
	const stack = initialStack(sdata, "a");
	stack.entries[0].props.channels = {
		colors: [[0, 255, 255]],
		contrastLimits: [[0, 250]],
		selections: [{ c: 2 }],
	};
	const original = createWorkspace(sdata, "a.zarr", "a", stack, {
		target: [100, 200],
		zoom: 2,
	});
	assert.deepEqual(
		validateWorkspace(JSON.parse(JSON.stringify(original)), sdata),
		original,
	);
	const corrupt = (change) => {
		const value = structuredClone(original);
		change(value);
		return value;
	};
	assert.throws(
		() =>
			validateWorkspace(
				corrupt((v) => {
					v.renderStack.entries[0].props.channels.contrastLimits = [[2, 1]];
				}),
				sdata,
			),
		/contrast/,
	);
	assert.throws(
		() =>
			validateWorkspace(
				corrupt((v) => {
					v.renderStack.entries[0].props.channels.selections = [{ c: -1 }];
				}),
				sdata,
			),
		/slice/,
	);
	assert.throws(
		() =>
			validateWorkspace(
				corrupt((v) => {
					v.renderStack.entries[0].kind = "host";
				}),
				sdata,
			),
		/layer/,
	);
	assert.throws(
		() =>
			validateWorkspace(
				corrupt((v) => {
					v.viewState.zoom = Infinity;
				}),
				sdata,
			),
		/camera/,
	);
	assert.throws(
		() => validateWorkspace(original, { ...sdata, images: {} }),
		/different dataset/,
	);
});
