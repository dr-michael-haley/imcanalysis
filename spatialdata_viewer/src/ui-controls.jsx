import React, { useEffect, useState } from "react";
export function Check({ label, value, onChange, disabled, ...rest }) {
	return (
		<label className="checkbox-label">
			<input
				type="checkbox"
				checked={!!value}
				disabled={disabled}
				onChange={(e) => onChange(e.target.checked)}
				{...rest}
			/>
			{label}
		</label>
	);
}
export function NumberField({
	label,
	value,
	onChange,
	min,
	max,
	step = "any",
	disabled,
}) {
	const [draft, setDraft] = useState(String(value)),
		[error, setError] = useState("");
	useEffect(() => {
		setDraft(String(value));
		setError("");
	}, [value]);
	function save() {
		const n = Number(draft);
		if (
			!draft ||
			!Number.isFinite(n) ||
			n < min ||
			n > max ||
			(step === 1 && !Number.isInteger(n))
		) {
			setError(`Use ${min}–${max}${step === 1 ? " (whole numbers)" : ""}.`);
			return;
		}
		setError("");
		onChange(n);
	}
	return (
		<label>
			{label}
			<input
				aria-label={label}
				type="number"
				value={draft}
				min={min}
				max={max}
				step={step}
				disabled={disabled}
				onChange={(e) => setDraft(e.target.value)}
				onBlur={save}
				onKeyDown={(e) => {
					if (e.key === "Enter") e.currentTarget.blur();
				}}
			/>
			{error && (
				<span className="error" role="alert">
					{error}
				</span>
			)}
		</label>
	);
}
export function PositionField({ label, value, onChange, bottom = false }) {
	return (
		<label>
			{label}
			<select
				aria-label={label}
				value={value}
				onChange={(e) => onChange(e.target.value)}
			>
				{(bottom
					? ["bottom-left", "bottom-right"]
					: ["top-left", "top-right", "bottom-left", "bottom-right"]
				).map((p) => (
					<option key={p} value={p}>
						{p.replace("-", " ")}
					</option>
				))}
			</select>
		</label>
	);
}
