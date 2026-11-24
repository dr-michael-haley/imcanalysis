import { useEffect, useRef, useState } from "react";
import {
	applyPropagation,
	preparePropagation,
	propagationSignature,
} from "./propagation.mjs";
import { tableGenes } from "./table-genes.mjs";

export function usePropagation(dataset, session, setSession) {
	const latest = useRef({ dataset, session }),
		request = useRef(0);
	latest.current = { dataset, session };
	const [state, setState] = useState(null);
	useEffect(() => {
		request.current++;
		setState(null);
		return () => {
			request.current++;
		};
	}, [dataset]);
	async function propagate(paneId, entryId, target) {
		const token = ++request.current;
		const snapshot = latest.current;
		setState({ paneId, entryId, busy: true });
		try {
			const plan = await preparePropagation(
				snapshot.dataset,
				snapshot.session,
				paneId,
				entryId,
				target,
				(key) => tableGenes(snapshot.dataset, key),
			);
			if (
				request.current !== token ||
				latest.current.dataset !== snapshot.dataset
			)
				return;
			// Check before committing; the functional update also protects queued edits.
			applyPropagation(latest.current.session, plan);
			setSession((current) =>
				propagationSignature(current) === plan.signature
					? applyPropagation(current, plan)
					: current,
			);
			setState({ paneId, entryId, plan });
		} catch (error) {
			if (request.current === token)
				setState({ paneId, entryId, error: error.message });
		}
	}
	return { state, propagate };
}
