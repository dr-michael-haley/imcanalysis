import React from "react";
const paths = {
	image: "M3 3h18v18H3z M3 17l6-6 4 4 3-3 5 5 M8 7h.01",
	shapes: "M6 3l13 3 2 11-12 4-6-9Z",
	points: "M5 5h.01M14 4h.01M20 10h.01M10 12h.01M4 18h.01M17 20h.01",
	labels: "M4 4h7v7H4zM13 4h7v7h-7zM4 13h7v7H4zM13 13h7v7h-7z",
	eye: "M2 12s4-7 10-7 10 7 10 7-4 7-10 7S2 12 2 12Z M9 12a3 3 0 1 0 6 0a3 3 0 1 0-6 0",
	eyeOff:
		"M3 3l18 18M5 7c-2 2-3 5-3 5s4 7 10 7c2 0 4-1 6-2M10 5h2c6 0 10 7 10 7l-2 3",
	up: "m6 15 6-6 6 6",
	down: "m6 9 6 6 6-6",
	left: "m15 6-6 6 6 6",
	right: "m9 6 6 6-6 6",
	plus: "M12 4v16M4 12h16",
	minus: "M4 12h16",
	close: "m6 6 12 12M6 18 18 6",
	inspect: "M5 3v17l5-5 4 7 3-2-4-7h8Z",
	select: "M12 3v18M3 12h18 M5 5h14v14H5z",
	lasso: "M7 18C-3 15 2 3 12 3s16 13 5 15c-8 2-14-4-10-6s9 5 4 10",
	view: "M3 4h18v16H3zM12 4v16",
	bookmark: "M6 3h12v18l-6-4-6 4ZM9 9h6m-3-3v6",
	export: "M12 3v12m-4-4 4 4 4-4M4 15v6h16v-6",
	fit: "M3 9V3h6m6 0h6v6m0 6v6h-6m-6 0H3v-6",
	sidebar: "M3 4h18v16H3zM8 4v16",
	ruler: "M3 8h18v8H3zM7 8v4m5-4v6m5-6v4",
};
export default function Icon({ name }) {
	return (
		<svg
			width="18"
			height="18"
			viewBox="0 0 24 24"
			aria-hidden="true"
			fill="none"
			stroke="currentColor"
			strokeWidth={name === "points" ? 4 : 1.6}
			strokeLinecap="round"
			strokeLinejoin="round"
		>
			<path d={paths[name] || paths.view} />
		</svg>
	);
}
