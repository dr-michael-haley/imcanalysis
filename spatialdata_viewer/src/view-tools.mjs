import {
	defaultFigure as tissueFigure,
	validateFigure as tissueValidate,
	defaultScaleBar,
	validateScaleBar,
	gridShape,
	figureLayout,
	scaleBarGeometry,
	xml,
} from "../../web_viewer/src/figure.js";
export { gridShape, figureLayout, scaleBarGeometry, xml };
export {
	figureFilename,
	validateFigureFilename,
} from "../../web_viewer/src/figure-filename.js";
export const defaultFigure = () => ({
	...tissueFigure(),
	calibrated: false,
	showLayerNames: true,
});
export function validateFigure(value) {
	const s = { ...defaultFigure(), ...value, ...tissueValidate(value) };
	if (
		typeof s.calibrated !== "boolean" ||
		typeof s.showLayerNames !== "boolean"
	)
		throw new Error("Invalid figure calibration or layer names.");
	if (s.physicalLock && !s.calibrated)
		throw new Error(
			"Confirm coordinate-system calibration before locking a physical field of view.",
		);
	return s;
}
export const defaultViewTools = () => ({
	legend: {
		visible: true,
		position: "top-right",
		fontSize: 11,
		maxRows: 24,
		title: "",
		layers: {},
	},
	layerNames: { visible: true, position: "bottom-left", fontSize: 10 },
	scaleBar: { ...defaultScaleBar(), position: "bottom-right" },
	display: { images: true, cells: true, points: true },
});
const positions = ["top-left", "top-right", "bottom-left", "bottom-right"];
export function viewTools(pane) {
	const defaults = defaultViewTools();
	return Object.fromEntries(
		Object.entries(defaults).map(([key, value]) => [
			key,
			{ ...value, ...pane[key] },
		]),
	);
}
export function validateViewTools(pane) {
	const s = viewTools(pane);
	s.scaleBar = validateScaleBar(s.scaleBar);
	for (const key of ["legend", "layerNames"]) {
		const v = s[key];
		if (
			typeof v.visible !== "boolean" ||
			!positions.includes(v.position) ||
			!Number.isFinite(v.fontSize) ||
			v.fontSize < 8 ||
			v.fontSize > 36
		)
			throw new Error(`Invalid ${key} overlay.`);
	}
	if (
		!Number.isInteger(s.legend.maxRows) ||
		s.legend.maxRows < 1 ||
		s.legend.maxRows > 100 ||
		typeof s.legend.title !== "string" ||
		s.legend.title.length > 120
	)
		throw new Error("Invalid legend settings.");
	if (
		!s.legend.layers ||
		typeof s.legend.layers !== "object" ||
		Array.isArray(s.legend.layers) ||
		Object.keys(s.legend.layers).length > 1000
	)
		throw new Error("Invalid legend layers.");
	for (const v of Object.values(s.legend.layers)) {
		if (
			!v ||
			(v.show !== undefined && typeof v.show !== "boolean") ||
			(v.title !== undefined &&
				(typeof v.title !== "string" || v.title.length > 160))
		)
			throw new Error("Invalid legend layer label.");
		if (
			v.labels &&
			(typeof v.labels !== "object" ||
				Array.isArray(v.labels) ||
				Object.entries(v.labels).some(
					([k, text]) =>
						k.length > 1000 || typeof text !== "string" || text.length > 160,
				))
		)
			throw new Error("Invalid legend entry labels.");
		if (
			v.hidden &&
			(!Array.isArray(v.hidden) ||
				v.hidden.some((key) => typeof key !== "string"))
		)
			throw new Error("Invalid hidden legend entries.");
	}
	if (Object.values(s.display).some((v) => typeof v !== "boolean"))
		throw new Error("Invalid view display toggles.");
	return s;
}
export const displayType = (type) =>
	type === "image" ? "images" : type === "points" ? "points" : "cells";
export function displayedEntries(pane) {
	const { display } = viewTools(pane);
	return pane.renderStack.entries.filter(
		(e) => e.visible && display[displayType(e.source.elementType)],
	);
}
export const addedLayers = (pane) =>
	pane.activeLayers ||
	pane.renderStack.entries
		.filter((e) => e.visible || e.id === pane.selectedId)
		.map((e) => e.id);
export function addLayer(pane, id) {
	if (!pane.renderStack.entries.some((e) => e.id === id)) return pane;
	const entries = pane.renderStack.entries.map((e) =>
		e.id === id ? { ...e, visible: true } : e,
	);
	return {
		...pane,
		activeLayers: [...new Set([...addedLayers(pane), id])],
		selectedId: id,
		renderStack: {
			...pane.renderStack,
			entries: addedLayers(pane).includes(id)
				? entries
				: [
						...entries.filter((e) => e.id !== id),
						entries.find((e) => e.id === id),
					],
		},
	};
}
export function removeLayer(pane, id) {
	const activeLayers = addedLayers(pane).filter((key) => key !== id);
	return {
		...pane,
		activeLayers,
		selectedId:
			pane.selectedId === id ? activeLayers[0] || "" : pane.selectedId,
		renderStack: {
			...pane.renderStack,
			entries: pane.renderStack.entries.map((e) =>
				e.id === id ? { ...e, visible: false } : e,
			),
		},
	};
}
export function reorderLayer(pane, id, step) {
	const order = [...addedLayers(pane)],
		i = order.indexOf(id),
		j = i + step;
	if (i < 0 || j < 0 || j >= order.length) return pane;
	[order[i], order[j]] = [order[j], order[i]];
	const inactive = pane.renderStack.entries.filter(
		(e) => !order.includes(e.id),
	);
	return {
		...pane,
		activeLayers: order,
		renderStack: {
			...pane.renderStack,
			entries: [
				...inactive,
				...order.map((key) =>
					pane.renderStack.entries.find((e) => e.id === key),
				),
			],
		},
	};
}
export function exportDimensions(count, rect, settings) {
	const layout = figureLayout(count, rect.width, rect.height, settings);
	const paneHeight = Math.round(
		(layout.outputHeight - (layout.rows - 1) * settings.gap) / layout.rows,
	);
	const width = Math.round(layout.outputWidth),
		height = paneHeight * layout.rows + (layout.rows - 1) * settings.gap;
	if (
		width < 1 ||
		height < 1 ||
		width > 16384 ||
		height > 16384 ||
		width * height > 32_000_000
	)
		throw new Error(
			"Figure exceeds 32 million pixels or 16,384 pixels per side. Reduce pane width or change the aspect ratio.",
		);
	return {
		...layout,
		outputWidth: width,
		outputHeight: height,
		paneWidth: settings.paneWidth,
		paneHeight,
	};
}
export function figureCamera(camera, width, sourceWidth, settings) {
	if (!camera)
		throw new Error("Wait for the view to finish framing before exporting.");
	return {
		...camera,
		zoom: settings.physicalLock
			? Math.log2((width * settings.umPerPixel) / settings.physicalWidth)
			: camera.zoom + Math.log2(width / sourceWidth),
	};
}
// Calibration is per coordinate-system unit after SpatialData transformations.
// Unit labels on image pixels alone do not establish the units of a named target system.
export function initialCalibration() {
	return { umPerPixel: 1, calibrated: false };
}
