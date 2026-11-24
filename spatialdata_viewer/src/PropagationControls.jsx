import React from "react";
import { propagationSource, propagationTargets } from "./propagation.mjs";

export default function PropagationControls({
	dataset,
	session,
	paneId,
	entryId,
	state,
	onApply,
}) {
	let source, error;
	try {
		source = propagationSource(dataset, session, paneId, entryId);
	} catch (e) {
		error = e.message;
	}
	const fromPoints = !source?.binding;
	const cells = source
		? propagationTargets(dataset, session, source, "cells").length
		: 0;
	const points = source
		? propagationTargets(dataset, session, source, "points").length
		: 0;
	const report =
		state?.paneId === paneId && state?.entryId === entryId ? state : null;
	const outcomes = report?.plan?.outcomes || [];
	const applied = outcomes.filter((r) => r.applied).length;
	const warnings = outcomes.filter((r) => !r.applied || r.missing?.length);
	return (
		<section className="propagation-controls" aria-label="Apply across views">
			<strong>Apply across views</strong>
			<p className="muted">
				Visible layers in all open views. Cell tables and ID links stay as
				configured.
			</p>
			<div className="propagation-actions">
				<button
					disabled={!source || !cells || source.kind === "all" || state?.busy}
					onClick={() => onApply(paneId, entryId, "cells")}
				>
					{fromPoints ? "Apply genes to cell layers" : "Apply to cell layers"} (
					{cells})
				</button>
				<button
					disabled={!source || !points || source.kind === "obs" || state?.busy}
					onClick={() => onApply(paneId, entryId, "points")}
				>
					Apply genes to Points layers ({points})
				</button>
			</div>
			<p className="muted">
				{error ||
					(source.kind === "obs"
						? "Obs fields apply to cells only. Missing fields are skipped."
						: source.kind === "all"
							? "All genes can be copied to Points. Select genes or a list to apply to cells."
							: fromPoints
								? "Multiple selected genes use a workspace list and colour cells by their sum. Cell colour maps are retained; limits reset to auto."
								: "Cell copies include the expression scale, colour map and limits. Points receive the gene selection.")}
			</p>
			{report?.busy && (
				<p role="status">Checking target tables and Points genes…</p>
			)}
			{report?.error && (
				<p className="error" role="alert">
					{report.error}
				</p>
			)}
			{report?.plan && (
				<div className="propagation-report" role="status" aria-live="polite">
					<p className={warnings.length ? "warning" : "link-status"}>
						Applied to {applied}{" "}
						{report.plan.target === "cells" ? "cell" : "Points"}{" "}
						{applied === 1 ? "layer" : "layers"}
						{outcomes.length > applied
							? ` · ${outcomes.length - applied} skipped`
							: ""}
						{warnings.some((r) => r.applied) ? " · some genes missing" : ""}.
					</p>
					{report.plan.listName && applied > 0 && (
						<p className="muted">Workspace gene list: {report.plan.listName}</p>
					)}
					{warnings.length > 0 && (
						<ul>
							{warnings.map((r) => (
								<li key={`${r.paneId}/${r.entryId}`}>
									<strong>{r.label}</strong>
									{r.table && <span> · {r.table}</span>}
									<br />
									{r.applied
										? `Applied ${r.matched} present genes; missing: ${r.missing.join(", ")}.`
										: `Skipped: ${r.reason}`}
								</li>
							))}
						</ul>
					)}
				</div>
			)}
		</section>
	);
}
