import React, { useEffect } from "react";
import { useImageLayerContext, mergeLayerChannelState } from "@spatialdata/vis";
import {
	featureCodeToRgb,
	assignFeatureColors,
	resolveFeatureFillColorMode,
	DEFAULT_FEATURE_NUMERIC_RAMP,
} from "@spatialdata/layers";
import { loadAssociatedTableFeatureRows } from "@spatialdata/core";
import { hex } from "./model.mjs";
import { colourmapGradient } from "./colour-scale.mjs";
import { displayedEntries } from "./view-tools.mjs";

function ImageLegend({ entry, onGroup }) {
	const context = useImageLayerContext(entry.source.elementKey);
	let group;
	if (context) {
		const merged = mergeLayerChannelState(
			entry.props.channels || {},
			context.defaults,
			entry.id,
		);
		group = {
			id: entry.id,
			title: entry.source.elementKey,
			rows: Array.from({ length: merged.channelCount }, (_, i) => ({
				key: `channel:${i}`,
				name:
					context.channelNames[merged.selections[i]?.c || 0] ||
					`Channel ${i + 1}`,
				colour: merged.colors[i],
				visible: merged.channelsVisible[i] !== false,
			})).filter((r) => r.visible),
		};
	}
	const signature = JSON.stringify(group || null);
	useEffect(() => {
		onGroup(entry.id, group);
		return () => onGroup(entry.id, undefined);
	}, [entry.id, signature, onGroup]);
	return null;
}
function PointsLegend({ dataset, entry, geneLists, onGroup }) {
	const spec = JSON.stringify([entry.props, geneLists]);
	useEffect(() => {
		let live = true;
		onGroup(entry.id, undefined);
		const names =
			entry.props.geneListId !== undefined
				? geneLists.find((g) => g.id === entry.props.geneListId)?.genes || []
				: entry.props.featureNames;
		dataset.points[entry.source.elementKey]
			.listFeatures()
			.then((catalog) => {
				if (!live) return;
				const rows =
					entry.props.colorByFeature !== false && catalog?.entries?.length
						? (catalog?.entries || [])
								.filter((r) => names === undefined || names.includes(r.name))
								.map((r) => ({
									key: r.name,
									name: r.name,
									colour:
										entry.props.featureColorOverrides?.[r.code] ||
										featureCodeToRgb(r.code),
								}))
						: [
								{
									key: "points",
									name: "Transcripts",
									colour: entry.props.color || [255, 255, 255],
								},
							];
				onGroup(entry.id, {
					id: entry.id,
					title: entry.source.elementKey,
					rows,
				});
			})
			.catch(() => {
				if (live)
					onGroup(entry.id, {
						id: entry.id,
						title: entry.source.elementKey,
						rows: [{ key: "unavailable", name: "Gene legend unavailable" }],
					});
			});
		return () => {
			live = false;
			onGroup(entry.id, undefined);
		};
	}, [dataset, entry.id, spec, onGroup]);
	return null;
}
function LabelsLegend({ dataset, entry, onGroup }) {
	const signature = JSON.stringify(entry.props.fillColorByColumn);
	useEffect(() => {
		let live = true;
		const spec = entry.props.fillColorByColumn;
		onGroup(entry.id, undefined);
		if (!spec?.columnName) return;
		loadAssociatedTableFeatureRows({
			spatialData: dataset,
			kind: "labels",
			key: entry.source.elementKey,
			extraColumnNames: [spec.columnName],
		})
			.then((rows) => {
				if (!live) return;
				const values = (rows.rowIds || []).map((id) => {
					const value =
						rows.extraColumns?.[0]?.[rows.rowIndexByFeatureId?.get(id)];
					return value == null ? "" : String(value).trim();
				});
				const columnKind = rows.extraColumnKinds?.[0];
				const colors = assignFeatureColors({
					...spec,
					values,
					columnKind,
					alpha: 255,
				});
				const mode = resolveFeatureFillColorMode(
					spec.mode,
					values.filter(Boolean),
					columnKind,
				);
				const group = {
					id: entry.id,
					title: `${entry.source.elementKey} · ${spec.columnName}`,
				};
				if (mode === "continuous") {
					let low = Infinity,
						high = -Infinity;
					values.forEach((v, i) => {
						if (v && colors[i] && Number.isFinite(Number(v))) {
							low = Math.min(low, Number(v));
							high = Math.max(high, Number(v));
						}
					});
					if (Number.isFinite(low)) {
						group.range = spec.numericDomain || [low, high];
						group.gradient = (spec.numericRamp || DEFAULT_FEATURE_NUMERIC_RAMP)
							.map(hex)
							.join(",");
						if (spec.numericScale === "symlog") group.title += " · symlog";
					}
				} else {
					const unique = new Map();
					values.forEach((v, i) => {
						if (v && colors[i])
							unique.set(v, { key: v, name: v, colour: colors[i] });
					});
					group.rows = [...unique.values()].sort((a, b) =>
						a.name.localeCompare(b.name, "en", { numeric: true }),
					);
				}
				if (!group.range && !group.rows?.length)
					group.rows = [{ key: "empty", name: "No associated colour values" }];
				onGroup(entry.id, group);
			})
			.catch(() => {
				if (live)
					onGroup(entry.id, {
						id: entry.id,
						title: entry.source.elementKey,
						rows: [{ key: "error", name: "Label legend unavailable" }],
					});
			});
		return () => {
			live = false;
			onGroup(entry.id, undefined);
		};
	}, [dataset, entry.id, signature, onGroup]);
	return null;
}
export default function PaneLegendData({ pane, dataset, geneLists, onGroup }) {
	return displayedEntries(pane).map((entry) =>
		entry.source.elementType === "image" ? (
			<ImageLegend key={entry.id} entry={entry} onGroup={onGroup} />
		) : entry.source.elementType === "points" ? (
			<PointsLegend
				key={entry.id}
				entry={entry}
				dataset={dataset}
				geneLists={geneLists}
				onGroup={onGroup}
			/>
		) : entry.source.elementType === "labels" &&
			entry.props.fillColorByColumn ? (
			<LabelsLegend
				key={entry.id}
				entry={entry}
				dataset={dataset}
				onGroup={onGroup}
			/>
		) : null,
	);
}
export function paneLegendGroups(pane, overlays, loaded) {
	return displayedEntries(pane).flatMap((e) => {
		if (e.source.elementType === "labels" && e.props.fillColorByColumn)
			return loaded[e.id]
				? [loaded[e.id]]
				: [
						{
							id: e.id,
							title: e.source.elementKey,
							rows: [{ key: "loading", name: "Reading label colours…" }],
						},
					];
		if (["image", "points"].includes(e.source.elementType))
			return loaded[e.id] ? [loaded[e.id]] : [];
		const data = overlays[`${pane.id}/${e.id}`];
		if (data?.values && data.matched && !data.error && !data.loading) {
			return [
				{
					id: e.id,
					title: `${e.source.elementKey} · ${data.label}`,
					...(data.mode === "continuous"
						? {
								range: [data.low, data.high],
								gradient: colourmapGradient(
									data.binding.colourmap,
									data.binding.reverse,
								),
								log: data.binding.scale === "log1p",
							}
						: {
								rows: data.categories.map((r) => ({
									key: r.label,
									name: r.label,
									colour: r.colour,
								})),
							}),
				},
			];
		}
		return [
			{
				id: e.id,
				title: e.source.elementKey,
				rows: [
					{
						key: "single",
						name: data?.binding?.field
							? data.error
								? "Colour data unavailable"
								: data.loading
									? "Reading colours…"
									: "Unpaired boundaries"
							: e.props.fillColorByColumn?.columnName ||
								(e.source.elementType === "labels"
									? "Cell labels"
									: "Boundaries"),
						colour: e.props.fillColor ||
							e.props.channels?.colors?.[0] || [129, 216, 193],
					},
				],
			},
		];
	});
}
