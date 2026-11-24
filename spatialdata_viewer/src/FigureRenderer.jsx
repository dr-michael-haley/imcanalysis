import React, { useEffect, useMemo, useRef } from "react";
import { SpatialViewer, useSpatialCanvasRenderer } from "@spatialdata/vis";

// Render export pixels through the same engine, waiting for metadata, Viv tiles
// and PointsLayer's async batch/filter readiness via Deck's isLoaded contract.
// The public renderer and Deck ref avoid resizing the user's live canvases.
export default function FigureRenderer({ job }) {
	const deck = useRef(null),
		finished = useRef(false),
		statusError = useRef(null),
		current = useRef(null);
	const resolver = useMemo(
		() =>
			({ layerId, featureIds }) =>
				featureIds
					? job.overlays[`${job.pane.id}/${layerId}`]?.resolve?.(featureIds)
					: undefined,
		[job],
	);
	const renderer = useSpatialCanvasRenderer({
		spatialData: job.dataset,
		coordinateSystem: job.system,
		renderStack: job.stack,
		viewState: job.camera,
		width: job.width,
		height: job.height,
		autoFit: false,
		featureColorResolver: resolver,
		onEntryStatusChange: (id, status) => {
			if (status?.errors.length)
				statusError.current = `${id}: ${status.errors.map((e) => e.error.message).join("; ")}`;
		},
	});
	current.current = renderer;
	const finish = (error, value) => {
		if (finished.current) return;
		finished.current = true;
		error ? job.reject(error) : job.resolve(value);
	};
	const props = useMemo(
		() => ({
			controller: false,
			useDevicePixels: false,
			style: { backgroundColor: "transparent" },
			onError: (e) => finish(e),
			onAfterRender: () => {
				if (finished.current) return;
				if (statusError.current) return finish(new Error(statusError.current));
				const instance = deck.current?.deck,
					state = current.current;
				if (
					!instance ||
					state.isLoading ||
					state.isBlocking ||
					!state.hasLayersDrawn
				)
					return;
				const layers = (instance.props.layers || [])
					.flat(Infinity)
					.filter(Boolean);
				if (!layers.length || !layers.every((layer) => layer.isLoaded)) return;
				const canvas = instance.getCanvas();
				if (canvas?.width === job.width && canvas.height === job.height) {
					try {
						finish(null, canvas.toDataURL("image/png"));
					} catch (e) {
						finish(e);
					}
				}
			},
		}),
		[job],
	);
	useEffect(() => {
		const timer = setTimeout(
			() =>
				finish(
					new Error(
						"Figure rendering timed out. Wait for the data to load or reduce the export size.",
					),
				),
			120000,
		);
		const redraw = setInterval(() => {
			if (!finished.current) deck.current?.deck?.redraw("figure export");
		}, 100);
		const abort = () => finish(new Error("Figure export cancelled."));
		job.signal.addEventListener("abort", abort);
		if (job.signal.aborted) abort();
		return () => {
			clearTimeout(timer);
			clearInterval(redraw);
			job.signal.removeEventListener("abort", abort);
		};
	}, [job]);
	return (
		<div
			className="figure-render-host"
			style={{ width: job.width, height: job.height }}
			aria-hidden="true"
		>
			<SpatialViewer
				deckRef={deck}
				width={job.width}
				height={job.height}
				viewState={job.camera}
				onViewStateChange={() => {}}
				layers={renderer.deckLayers}
				layerOrder={renderer.layerOrder}
				vivLayerProps={renderer.vivLayerProps}
				deckProps={props}
			/>
		</div>
	);
}
