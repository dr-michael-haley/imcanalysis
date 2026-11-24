import React, { useEffect, useRef, useState } from "react";
import { createPortal } from "react-dom";
import Icon from "./icons.jsx";
import ViewOverlayControls from "./ViewOverlayControls.jsx";
export default function ViewerToolbar({
	dataset,
	session,
	mode,
	onMode,
	onSystem,
	onZoom,
	onFit,
	onFullscreen,
	onBookmark,
	onFigure,
	onSidebar,
	onPane,
	onFigureChange,
	onClear,
	onSelectDisplayed,
	selectionCount,
	selectionScope,
}) {
	const [menu, setMenu] = useState(""),
		[position, setPosition] = useState({}),
		trigger = useRef(null),
		panel = useRef(null);
	const pane = session.panes.find((p) => p.id === session.activePaneId),
		systems = dataset?.coordinateSystems || [],
		index = systems.indexOf(session.coordinateSystem),
		isFigure = session.figure.mode === "figure",
		locked = isFigure && session.figure.physicalLock;
	useEffect(() => {
		const close = (e) => {
			if (e.key === "Escape") {
				setMenu("");
				onMode("inspect");
				trigger.current?.focus();
			} else if (
				e.type !== "keydown" &&
				!panel.current?.contains(e.target) &&
				!trigger.current?.contains(e.target)
			)
				setMenu("");
		};
		document.addEventListener("pointerdown", close);
		document.addEventListener("keydown", close);
		return () => {
			document.removeEventListener("pointerdown", close);
			document.removeEventListener("keydown", close);
		};
	}, [onMode]);
	function open(name, e) {
		trigger.current = e.currentTarget;
		if (menu === name) return setMenu("");
		const r = e.currentTarget.getBoundingClientRect();
		const above = r.top - 16,
			below = window.innerHeight - r.bottom - 16;
		const openBelow = above < 260 && below > above;
		setPosition({
			right: Math.max(10, window.innerWidth - r.right),
			...(openBelow
				? { top: r.bottom + 6 }
				: { bottom: window.innerHeight - r.top + 6 }),
			maxHeight: Math.max(80, Math.min(540, openBelow ? below : above)),
		});
		setMenu(name);
		if (e.type === "keydown") {
			e.preventDefault();
			requestAnimationFrame(() =>
				panel.current
					?.querySelector("button:not(:disabled), input:not(:disabled)")
					?.focus(),
			);
		}
	}
	return (
		<>
			<div
				className="selection-strip"
				hidden={mode === "inspect" && !selectionCount}
			>
				<strong>{selectionCount.toLocaleString()} selected</strong>
				<span>
					{selectionScope === "embedding"
						? "Embedding · plotted cells"
						: "Tissue · displayed shape centres"}
				</span>
				<button disabled={!dataset || isFigure} onClick={onSelectDisplayed}>
					Select displayed
				</button>
				<button disabled={!selectionCount} onClick={onClear}>
					Clear selected cells
				</button>
				{mode !== "inspect" && (
					<button onClick={() => onMode("inspect")}>Done · Esc</button>
				)}
			</div>
			<nav className="viewer-toolbar" aria-label="Viewer controls">
				<div className="coordinate-navigation">
					<button
						aria-label="Previous coordinate system"
						disabled={index <= 0}
						onClick={() => onSystem(systems[index - 1])}
					>
						<Icon name="left" />
					</button>
					<button
						aria-label="Next coordinate system"
						disabled={index < 0 || index >= systems.length - 1}
						onClick={() => onSystem(systems[index + 1])}
					>
						<Icon name="right" />
					</button>
					<span title={session.coordinateSystem}>
						{session.coordinateSystem || "No dataset"}
					</span>
					<small>{index >= 0 ? `${index + 1} / ${systems.length}` : ""}</small>
				</div>
				<div className="toolbar-actions">
					<button
						aria-label={
							session.sidebar?.hidden ? "Show sidebar" : "Hide sidebar"
						}
						onClick={onSidebar}
					>
						<Icon name="sidebar" />
					</button>
					<button
						aria-expanded={menu === "mode"}
						disabled={!dataset || isFigure}
						onClick={(e) => open("mode", e)}
						onKeyDown={(e) => {
							if (e.key === "ArrowDown") open("mode", e);
						}}
					>
						<Icon name={mode === "click" ? "select" : mode} />
						{mode === "inspect"
							? "Inspect"
							: mode === "click"
								? "Select cells"
								: "Lasso"}
						<Icon name="up" />
					</button>
					<button
						aria-expanded={menu === "view"}
						disabled={!dataset}
						onClick={(e) => open("view", e)}
					>
						<Icon name="view" />
						View
						<Icon name="up" />
					</button>
					<button disabled={!dataset} onClick={onBookmark}>
						<Icon name="bookmark" />
						Bookmark
					</button>
					<button disabled={!dataset} onClick={onFigure}>
						<Icon name="export" />
						Export
					</button>
				</div>
			</nav>
			{menu &&
				createPortal(
					<div
						className="toolbar-popover"
						ref={panel}
						style={position}
						role="group"
						aria-label={menu === "mode" ? "Mouse mode" : "View options"}
						onKeyDown={(e) => {
							if (
								!["ArrowDown", "ArrowUp", "Home", "End"].includes(e.key) ||
								e.target.matches("input,select")
							)
								return;
							const buttons = [
									...e.currentTarget.querySelectorAll("button:not(:disabled)"),
								],
								i = buttons.indexOf(document.activeElement);
							e.preventDefault();
							buttons[
								e.key === "Home"
									? 0
									: e.key === "End"
										? buttons.length - 1
										: (i + (e.key === "ArrowDown" ? 1 : buttons.length - 1)) %
											buttons.length
							]?.focus();
						}}
					>
						<strong>
							{menu === "mode" ? "Mouse mode" : `Display · ${pane.name}`}
						</strong>
						{menu === "mode" ? (
							<>
								{[
									[
										"inspect",
										"Inspect & pan",
										"Click for details. Drag to move the tissue.",
									],
									["click", "Select cells", "Click to add or remove a cell."],
									[
										"lasso",
										"Lasso cells",
										"Draw around shape centres to add cells.",
									],
								].map(([value, title, hint]) => (
									<button
										key={value}
										aria-pressed={mode === value}
										onClick={() => {
											onMode(value);
											setMenu("");
											trigger.current?.focus();
										}}
									>
										<Icon name={value === "click" ? "select" : value} />
										<span>
											{title}
											<small>{hint}</small>
										</span>
									</button>
								))}
								<p className="muted">
									Works in tissue and embeddings. Escape returns to Inspect.
									Label masks support click selection.
								</p>
							</>
						) : (
							<>
								<ViewOverlayControls
									compact
									pane={pane}
									figure={session.figure}
									onFigure={onFigureChange}
									onChange={(patch) => onPane(pane.id, patch)}
								/>
								<div className="toolbar-zoom">
									<button
										aria-label="Zoom out"
										disabled={locked || !pane.viewState}
										onClick={() => onZoom(-0.5)}
									>
										<Icon name="minus" />
									</button>
									<span>
										{pane.viewState
											? `${(2 ** pane.viewState.zoom).toPrecision(3)} px / unit`
											: "Framing…"}
									</span>
									<button
										aria-label="Zoom in"
										disabled={locked || !pane.viewState}
										onClick={() => onZoom(0.5)}
									>
										<Icon name="plus" />
									</button>
								</div>
								<div className="two-columns">
									<button disabled={locked} onClick={onFit}>
										<Icon name="fit" />
										Fit view
									</button>
									<button onClick={onFullscreen}>Fullscreen</button>
								</div>
							</>
						)}
					</div>,
					document.body,
				)}
		</>
	);
}
