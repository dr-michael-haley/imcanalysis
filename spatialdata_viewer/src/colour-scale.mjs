// Share IMC Explorer's palettes so both viewers use the same colours.
export {
	COLOURMAPS,
	colourmapGradient,
} from "../../web_viewer/src/colourmaps.js";
import { COLOURMAPS, colourmap } from "../../web_viewer/src/colourmaps.js";
export function validateColourSettings(value) {
	for (const key of ["vmin", "vmax"])
		if (value[key] != null && !Number.isFinite(value[key]))
			throw new Error("Colour limits must be finite numbers.");
	if (value.vmin != null && value.vmax != null && value.vmin >= value.vmax)
		throw new Error("vmax must be greater than vmin.");
	if (
		value.colourmap !== undefined &&
		!Object.hasOwn(COLOURMAPS, value.colourmap)
	)
		throw new Error("Unknown colour map.");
	if (value.reverse !== undefined && typeof value.reverse !== "boolean")
		throw new Error("Invalid colour-map direction.");
}
export const validNumber = (v) =>
	v != null && v !== "" && Number.isFinite(Number(v));
export function continuousScale(values, options = {}) {
	validateColourSettings(options);
	let dataLow = Infinity,
		dataHigh = -Infinity;
	for (const v of values)
		if (validNumber(v)) {
			dataLow = Math.min(dataLow, Number(v));
			dataHigh = Math.max(dataHigh, Number(v));
		}
	if (!Number.isFinite(dataLow)) dataLow = dataHigh = 0;
	const rawLow = options.vmin ?? dataLow,
		rawHigh = options.vmax ?? dataHigh;
	if (
		rawLow > rawHigh ||
		(rawLow === rawHigh && (options.vmin != null || options.vmax != null))
	)
		throw new Error(
			"vmax must be greater than vmin. Reset the range or adjust the limits.",
		);
	const transform = (v) => (options.scale === "log1p" ? Math.log1p(v) : v);
	if (options.scale === "log1p" && Math.min(dataLow, rawLow, rawHigh) < 0)
		throw new Error(
			"Log1p requires non-negative values. Choose the linear scale.",
		);
	const low = transform(rawLow),
		high = transform(rawHigh);
	return {
		low,
		high,
		dataLow,
		dataHigh,
		rawLow,
		rawHigh,
		colour(value) {
			if (!validNumber(value)) return [110, 118, 130, 130];
			const t =
				high === low
					? high === 0
						? 0
						: 1
					: (transform(Number(value)) - low) / (high - low);
			return [
				...colourmap(t, options.colourmap || "viridis", options.reverse),
				255,
			];
		},
	};
}
