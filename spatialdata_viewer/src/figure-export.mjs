import {
	displayedEntries,
	exportDimensions,
	figureCamera,
	xml,
} from "./view-tools.mjs";
import { paneOverlaySvg } from "./pane-overlays.mjs";
export async function createFigureExport({
	session,
	dataset,
	groups,
	overlays,
	rect,
	sizes,
	render,
	signal,
	progress,
}) {
	const settings = session.figure,
		dimensions = exportDimensions(session.panes.length, rect, settings);
	const width = dimensions.paneWidth,
		height = dimensions.paneHeight,
		outputWidth = dimensions.outputWidth,
		outputHeight = dimensions.outputHeight;
	const panes = [],
		metadata = [];
	for (const [index, pane] of session.panes.entries()) {
		signal.throwIfAborted();
		const camera = figureCamera(
			pane.viewState,
			width,
			session.figureReturn?.[pane.id]?.width || sizes[pane.id]?.width || width,
			settings,
		);
		const entries = displayedEntries(pane).map((e) => ({
			...e,
			props: {
				...e.props,
				...(e.props.geneListId !== undefined
					? {
							featureNames:
								session.geneLists.find((g) => g.id === e.props.geneListId)
									?.genes || [],
						}
					: {}),
			},
		}));
		for (const e of entries) {
			const overlay = overlays[`${pane.id}/${e.id}`];
			if (overlay?.loading)
				throw new Error(`${pane.name}: cell colours are still loading.`);
			if (overlay?.error) throw new Error(`${pane.name}: ${overlay.error}`);
		}
		const prefix = `pane-${index + 1}`,
			content = [
				`<rect width="${width}" height="${height}" fill="${settings.background}"/>`,
			];
		const raster = async (items, label, id) => {
			if (!items.length) return;
			progress(`Rendering ${pane.name} · ${label}`);
			const href = await render({
				dataset,
				system: session.coordinateSystem,
				pane,
				overlays,
				stack: { schemaVersion: 1, entries: items },
				camera,
				width,
				height,
				signal,
			});
			signal.throwIfAborted();
			content.push(
				`<g id="${prefix}-${id}" inkscape:groupmode="layer" inkscape:label="${xml(label)}"><image width="${width}" height="${height}" href="${href}"/></g>`,
			);
		};
		if (settings.format === "svg")
			for (const [i, entry] of entries.entries())
				await raster([entry], entry.source.elementKey, `layer-${i}`);
		else await raster(entries, "Spatial data", "data");
		content.push(
			paneOverlaySvg(
				pane,
				groups[pane.id] || [],
				settings,
				width,
				height,
				camera.zoom,
				prefix,
			),
		);
		panes.push(
			`<g id="${prefix}" inkscape:groupmode="layer" inkscape:label="${xml(pane.name)}" transform="translate(${(index % dimensions.columns) * (width + settings.gap)} ${Math.floor(index / dimensions.columns) * (height + settings.gap)})"><clipPath id="${prefix}-clip"><rect width="${width}" height="${height}"/></clipPath><g clip-path="url(#${prefix}-clip)">${content.join("")}</g></g>`,
		);
		metadata.push({
			name: pane.name,
			coordinateSystem: session.coordinateSystem,
			centre: camera.target,
			coordinateWidth: width / 2 ** camera.zoom,
			coordinateHeight: height / 2 ** camera.zoom,
			umPerCoordinateUnit: settings.calibrated ? settings.umPerPixel : null,
			layers: entries.map((e) => e.source),
		});
	}
	const svg = `<svg xmlns="http://www.w3.org/2000/svg" xmlns:inkscape="http://www.inkscape.org/namespaces/inkscape" width="${outputWidth}" height="${outputHeight}" viewBox="0 0 ${outputWidth} ${outputHeight}"><title>MERSCOPE Explorer · ${xml(session.coordinateSystem)}</title><metadata>${xml(JSON.stringify(metadata))}</metadata><rect width="100%" height="100%" fill="${settings.background}"/>${panes.join("")}</svg>`;
	if (svg.length > 128 * 1024 * 1024)
		throw new Error(
			"The figure exceeds 128 MiB. Reduce pane width or use PNG.",
		);
	if (settings.format === "svg")
		return new Blob([svg], { type: "image/svg+xml;charset=utf-8" });
	progress("Encoding figure…");
	const url = URL.createObjectURL(new Blob([svg], { type: "image/svg+xml" }));
	try {
		const picture = new Image();
		picture.src = url;
		await picture.decode();
		signal.throwIfAborted();
		const canvas = document.createElement("canvas");
		canvas.width = outputWidth;
		canvas.height = outputHeight;
		canvas.getContext("2d").drawImage(picture, 0, 0);
		return await new Promise((resolve, reject) =>
			canvas.toBlob(
				(blob) => {
					canvas.width = canvas.height = 0;
					blob
						? resolve(blob)
						: reject(new Error("Image encoding failed. Reduce pane width."));
				},
				`image/${settings.format}`,
				0.95,
			),
		);
	} finally {
		URL.revokeObjectURL(url);
	}
}
