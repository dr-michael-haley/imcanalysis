import React, { useEffect, useState } from "react";

function elapsed(start) {
	const seconds = Math.max(0, Math.floor((Date.now() - start) / 1000));
	return seconds < 60
		? `${seconds}s`
		: `${Math.floor(seconds / 60)}m ${seconds % 60}s`;
}
function size(bytes) {
	return bytes < 1024 ** 2
		? `${(bytes / 1024).toFixed(0)} KiB`
		: `${(bytes / 1024 ** 2).toFixed(1)} MiB`;
}
export default function LoadingIndicator({ loading, activity, onCancel }) {
	const [, tick] = useState(0);
	const busy = Boolean(loading || activity?.pending);
	useEffect(() => {
		if (!busy) return;
		const timer = setInterval(() => tick((n) => n + 1), 1000);
		return () => clearInterval(timer);
	}, [busy]);
	if (!busy) return null;
	const title = loading?.title || "Reading layer data…";
	const picker = loading?.phase === "picker";
	return (
		<div
			className={loading ? "dataset-loading" : "read-activity"}
			data-testid={loading ? "dataset-loading" : "read-activity"}
		>
			<div className="loading-content">
				<div className="loading-heading">
					<strong role="status">{title}</strong>
					<span className="loading-elapsed" aria-label="Elapsed time">
						{elapsed(loading?.startedAt || activity.startedAt)}
					</span>
				</div>
				<div className="loading-track" role="progressbar" aria-label={title}>
					<span />
				</div>
				{loading?.name && <p className="loading-source">{loading.name}</p>}
				{loading && (
					<p className="loading-detail">
						{picker
							? loading.detail
							: `${loading.detail} Files on a network drive can take longer to read.`}
					</p>
				)}
				{!picker && activity && (
					<p className="loading-counts">
						{activity.pending
							? `${activity.pending.toLocaleString()} pending · `
							: ""}
						{activity.reads.toLocaleString()} reads completed ·{" "}
						{size(activity.bytes)} read
					</p>
				)}
				{!picker && activity?.path && (
					<p className="loading-path" title={activity.path}>
						{activity.path}
					</p>
				)}
				{loading && <button onClick={onCancel}>Cancel loading</button>}
			</div>
		</div>
	);
}
