export const elementKinds = ["images", "labels", "shapes", "points"];
export const layerType = (kind) => (kind === "images" ? "image" : kind);
export const elementKind = (type) => (type === "image" ? "images" : type);
export const palette = [
	"#ffffff",
	"#00ffff",
	"#ff00ff",
	"#ffff00",
	"#00ff00",
	"#ff0000",
	"#0000ff",
];
export const rgb = (hex) =>
	hex
		.replace("#", "")
		.match(/../g)
		.map((value) => parseInt(value, 16));
export const hex = (color) =>
	"#" +
	color
		.slice(0, 3)
		.map((value) =>
			Math.max(0, Math.min(255, Math.round(value)))
				.toString(16)
				.padStart(2, "0"),
		)
		.join("");
export const pretty = (value) =>
	value == null
		? "—"
		: typeof value === "object"
			? JSON.stringify(value)
			: String(value);
export function elementsFor(sdata, coordinateSystem) {
	return elementKinds.flatMap((kind) =>
		Object.entries(sdata[kind] || {})
			.filter(([, element]) =>
				element.coordinateSystems.includes(coordinateSystem),
			)
			.map(([key, element]) => ({
				id: `${kind}:${key}`,
				kind,
				type: layerType(kind),
				key,
				element,
			})),
	);
}
export function initialStack(sdata, coordinateSystem) {
	const used = new Set();
	return {
		schemaVersion: 1,
		entries: elementsFor(sdata, coordinateSystem).map(({ id, type, key }) => {
			const visible =
				(type === "image" || type === "labels") && !used.has(type);
			used.add(type);
			return {
				kind: "spatial",
				id,
				source: { elementType: type, elementKey: key },
				visible,
				props: {
					opacity: 1,
					...(type === "labels"
						? {
								channels: {
									colors: [[129, 216, 193]],
									channelsFilled: [false],
									channelStrokeWidths: [1],
								},
							}
						: {}),
					...(type === "shapes"
						? { fillColor: [129, 216, 193, 100], strokeWidth: 1 }
						: {}),
				},
			};
		}),
	};
}
export function inventory(sdata) {
	return JSON.stringify(
		[...elementKinds, "tables"].map((kind) => [
			kind,
			Object.keys(sdata[kind] || {}).sort(),
		]),
	);
}
export function createWorkspace(
	sdata,
	source,
	coordinateSystem,
	renderStack,
	viewState,
) {
	return {
		format: "sbt-spatialdata-explorer",
		version: 1,
		source,
		inventory: inventory(sdata),
		coordinateSystem,
		renderStack,
		viewState,
	};
}
export function validateWorkspace(value, sdata) {
	if (value?.format !== "sbt-spatialdata-explorer" || value.version !== 1)
		throw new Error("This is not a supported SpatialData Explorer view file.");
	if (value.inventory !== inventory(sdata))
		throw new Error(
			"The saved view belongs to a different dataset. Open its SpatialData store first.",
		);
	if (!sdata.coordinateSystems.includes(value.coordinateSystem))
		throw new Error("The saved coordinate system is not in this dataset.");
	const entries = value.renderStack?.entries;
	if (!Array.isArray(entries) || entries.length > 10000)
		throw new Error("The saved layer list is invalid.");
	const allowed = new Map(
		elementsFor(sdata, value.coordinateSystem).map((element) => [
			element.id,
			element,
		]),
	);
	const seen = new Set();
	for (const entry of entries) {
		const expected = allowed.get(entry.id);
		if (
			entry.kind !== "spatial" ||
			!expected ||
			seen.has(entry.id) ||
			entry.source?.elementKey !== expected.key ||
			entry.source?.elementType !== expected.type
		)
			throw new Error("The saved view contains an unknown or duplicate layer.");
		seen.add(entry.id);
		if (
			typeof entry.visible !== "boolean" ||
			!Number.isFinite(entry.props?.opacity) ||
			entry.props.opacity < 0 ||
			entry.props.opacity > 1
		)
			throw new Error("The saved layer settings are invalid.");
		const channels = entry.props.channels;
		if (channels) {
			for (const name of [
				"colors",
				"contrastLimits",
				"selections",
				"channelsVisible",
			]) {
				if (
					channels[name] &&
					(!Array.isArray(channels[name]) || channels[name].length > 6)
				)
					throw new Error(
						"A view can contain at most six image channels per layer.",
					);
			}
			for (const pair of channels.contrastLimits || [])
				if (
					!Array.isArray(pair) ||
					pair.length !== 2 ||
					!pair.every(Number.isFinite) ||
					pair[0] >= pair[1]
				)
					throw new Error("The saved contrast limits are invalid.");
			for (const color of channels.colors || [])
				if (
					!Array.isArray(color) ||
					color.length !== 3 ||
					!color.every((v) => Number.isFinite(v) && v >= 0 && v <= 255)
				)
					throw new Error("The saved channel colour is invalid.");
			for (const selection of channels.selections || [])
				if (
					!selection ||
					Object.entries(selection).some(
						([axis, v]) =>
							!["c", "z", "t"].includes(axis) ||
							!Number.isSafeInteger(v) ||
							v < 0,
					)
				)
					throw new Error("The saved image slice is invalid.");
		}
	}
	const camera = value.viewState;
	if (
		camera !== null &&
		(!Array.isArray(camera?.target) ||
			![2, 3].includes(camera.target.length) ||
			!camera.target.every(Number.isFinite) ||
			!Number.isFinite(camera.zoom) ||
			Math.abs(camera.zoom) > 40)
	)
		throw new Error("The saved camera is invalid.");
	return value;
}
export function download(name, data, mime = "application/json") {
	const url = URL.createObjectURL(
		data instanceof Blob ? data : new Blob([data], { type: mime }),
	);
	const link = document.createElement("a");
	link.href = url;
	link.download = name;
	link.click();
	setTimeout(() => URL.revokeObjectURL(url), 1000);
}
