import {
	viewTools,
	displayedEntries,
	scaleBarGeometry,
	xml,
} from "./view-tools.mjs";
const colour = (c) =>
	Array.isArray(c) ? `rgb(${c.slice(0, 3).join(",")})` : c || "#ffffff";
export function customiseLegend(groups, legend) {
	return groups
		.filter((g) => legend.layers[g.id]?.show !== false)
		.map((g) => {
			const custom = legend.layers[g.id] || {};
			return {
				...g,
				title: custom.title || g.title,
				rows: g.rows
					?.filter((r) => !custom.hidden?.includes(r.key))
					.map((r) => ({ ...r, name: custom.labels?.[r.key] || r.name })),
			};
		});
}
export function paneOverlaySvg(
	pane,
	rawGroups,
	settings,
	width,
	height,
	zoom,
	prefix = "pane",
) {
	const tools = viewTools(pane),
		figure = settings.mode === "figure",
		unit = figure ? width / settings.paneWidth : 1;
	const pad = 12 * unit,
		fg = figure ? settings.foreground : "#eef5f5",
		background = figure ? settings.background : "#101b22";
	const elements = [],
		offsets = Object.fromEntries(
			["top-left", "top-right", "bottom-left", "bottom-right"].map((p) => [
				p,
				pad,
			]),
		);
	function block(position, w, h, content, id, backdrop = true) {
		const x = position.endsWith("right") ? width - w - pad : pad;
		const y = position.startsWith("bottom")
			? height - offsets[position] - h
			: offsets[position];
		offsets[position] += h + pad;
		if (y < 0 || y + h > height) return;
		elements.push(
			`<g id="${prefix}-${id}" transform="translate(${x} ${y})">${backdrop && (!figure || settings.overlayBackground) ? `<rect x="-${pad / 2}" y="-${pad / 2}" width="${w + pad}" height="${h + pad}" fill="${background}" fill-opacity="${figure ? 1 : 0.88}" rx="${figure ? 0 : 4}"/>` : ""}${content}</g>`,
		);
	}
	const text = (x, y, value, size, fill = fg, extra = "") =>
		`<text x="${x}" y="${y}" font-family="Arial, sans-serif" font-size="${size}" fill="${xml(fill)}" ${extra}>${xml(value)}</text>`;
	const short = (value, available, size) => {
		const limit = Math.max(4, Math.floor(available / (size * 0.57)));
		return String(value).length > limit
			? String(value).slice(0, limit - 1) + "…"
			: String(value);
	};
	if (figure && settings.showNames) {
		const size = settings.fontSize * unit;
		block(
			"top-left",
			Math.min(width * 0.45, pane.name.length * size * 0.6),
			size * 1.25,
			text(
				0,
				size,
				short(pane.name, width * 0.45, size),
				size,
				fg,
				'font-weight="bold"',
			),
			"name",
		);
	}
	if (tools.legend.visible && (!figure || settings.showLegends)) {
		const groups = customiseLegend(rawGroups, tools.legend),
			size = (figure ? settings.legendSize : tools.legend.fontSize) * unit,
			line = size * 1.5,
			w = Math.min(width * 0.46, size * 20),
			maxH = height * 0.65;
		let y = 0,
			shown = 0,
			omitted = 0;
		const content = [];
		if (tools.legend.title) {
			y += line;
			content.push(
				text(
					0,
					y,
					short(tools.legend.title, w, size),
					size,
					fg,
					'font-weight="bold"',
				),
			);
		}
		for (const [i, group] of groups.entries()) {
			if (y + line * 3 > maxH) {
				omitted += group.rows?.length || 1;
				continue;
			}
			y += line;
			content.push(
				text(0, y, short(group.title, w, size), size, fg, 'font-weight="bold"'),
			);
			for (const row of group.rows || []) {
				if (
					shown >= (figure ? settings.legendRows : tools.legend.maxRows) ||
					y + line > maxH
				) {
					omitted++;
					continue;
				}
				y += line;
				shown++;
				if (row.colour)
					content.push(
						`<rect x="0" y="${y - size * 0.8}" width="${size * 0.7}" height="${size * 0.7}" fill="${xml(colour(row.colour))}"/>`,
					);
				content.push(
					text(
						row.colour ? size * 1.1 : 0,
						y,
						short(row.name, w - size * 1.1, size),
						size,
					),
				);
			}
			if (group.range) {
				const stops = group.gradient.match(/#[a-f0-9]{6}/gi) || [
						"#000000",
						"#ffffff",
					],
					id = `${prefix}-ramp-${i}`;
				y += line;
				content.push(
					`<defs><linearGradient id="${id}">${stops.map((c, n) => `<stop offset="${n / (stops.length - 1)}" stop-color="${c}"/>`).join("")}</linearGradient></defs><rect x="0" y="${y - size * 0.7}" width="${w}" height="${size * 0.7}" fill="url(#${id})"/>`,
				);
				y += line;
				content.push(
					text(
						0,
						y,
						group.range.map((v) => Number(v.toPrecision(4))).join(" – ") +
							(group.log ? " · log1p" : ""),
						size,
					),
				);
			}
			y += size * 0.4;
		}
		if (omitted) {
			y += line;
			content.push(text(0, y, `+ ${omitted} more`, size));
		}
		if (content.length)
			block(
				tools.legend.position,
				w,
				y + size * 0.3,
				content.join(""),
				"legend",
			);
	}
	if (tools.layerNames.visible && (!figure || settings.showLayerNames)) {
		const size = tools.layerNames.fontSize * unit,
			w = width * 0.6,
			line = size * 1.45;
		const names = displayedEntries(pane).map(
			(e) =>
				`${{ image: "Image", shapes: "Shapes", points: "Points", labels: "Labels" }[e.source.elementType]} · ${e.source.elementKey}`,
		);
		const limit = Math.max(1, Math.floor((height * 0.25) / line)),
			shown = names.slice(0, limit);
		if (names.length > limit)
			shown[limit - 1] = `+ ${names.length - limit + 1} more layers`;
		if (shown.length)
			block(
				tools.layerNames.position,
				Math.min(w, Math.max(...shown.map((s) => s.length)) * size * 0.58),
				shown.length * line,
				shown
					.map((s, i) => text(0, (i + 1) * line, short(s, w, size), size))
					.join(""),
				"layer-names",
			);
	}
	if (tools.scaleBar.visible && (!figure || settings.showScaleBars)) {
		const measured = scaleBarGeometry(
				tools.scaleBar,
				settings.calibrated ? settings.umPerPixel : 1,
				2 ** zoom,
				width,
			),
			size = (figure ? settings.scaleFontSize : tools.scaleBar.fontSize) * unit,
			col = figure ? fg : tools.scaleBar.colour;
		const label = `${Number(measured.length.toPrecision(4))} ${settings.calibrated ? "µm" : "units"}`;
		const w = Math.max(measured.pixels, label.length * size * 0.6);
		block(
			tools.scaleBar.position,
			Math.min(w, width - 2 * pad),
			size * 2,
			measured.fits
				? text(w / 2, size, label, size, col, 'text-anchor="middle"') +
						`<path d="M${(w - measured.pixels) / 2} ${size * 1.8}h${measured.pixels}" stroke="${xml(col)}" stroke-width="${tools.scaleBar.thickness * unit}"/>`
				: text(0, size, "Scale bar too long", size, col),
			"scale-bar",
		);
	}
	return elements.join("");
}
