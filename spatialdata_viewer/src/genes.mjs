export function cleanGenes(values) {
	const genes = [
		...new Set(values.map((v) => String(v).trim()).filter(Boolean)),
	];
	if (genes.length > 5000 || genes.some((v) => v.length > 200))
		throw new Error(
			"Use at most 5,000 gene names per list, each under 200 characters.",
		);
	return genes;
}

export const genesFromText = (text) => cleanGenes(text.split(/[\n,;\t]+/));

// RFC 4180 fields, including escaped quotes, CRLF and a UTF-8 BOM.
export function csvRows(text) {
	const rows = [];
	let row = [],
		field = "",
		quoted = false;
	text = text.replace(/^\uFEFF/, "");
	for (let i = 0; i < text.length; i++) {
		const c = text[i];
		if (c === '"') {
			if (quoted && text[i + 1] === '"') {
				field += '"';
				i++;
			} else quoted = !quoted;
		} else if (!quoted && (c === "," || c === "\n" || c === "\r")) {
			row.push(field.trim());
			field = "";
			if (c !== ",") {
				if (row.some(Boolean)) rows.push(row);
				row = [];
				if (c === "\r" && text[i + 1] === "\n") i++;
			}
		} else field += c;
	}
	if (quoted) throw new Error("The CSV has an unclosed quoted field.");
	row.push(field.trim());
	if (row.some(Boolean)) rows.push(row);
	return rows;
}

export function importGeneCsv(text, fallbackName = "Imported genes") {
	const rows = csvRows(text);
	if (!rows.length) throw new Error("The CSV is empty.");
	const header = rows[0].map((v) => v.toLowerCase().replace(/[ -]/g, "_"));
	const geneColumn = header.findIndex((v) =>
		["gene", "genes", "gene_symbol", "symbol"].includes(v),
	);
	const listColumn = header.findIndex((v) =>
		["list", "gene_list", "list_name"].includes(v),
	);
	if (geneColumn < 0 && rows.some((row) => row.length !== 1))
		throw new Error(
			"Use one gene per row, or a CSV with a gene column and optional list column.",
		);
	const groups = new Map();
	for (const row of geneColumn < 0 ? rows : rows.slice(1)) {
		const name =
			listColumn < 0 ? fallbackName : row[listColumn] || fallbackName;
		const gene = row[geneColumn < 0 ? 0 : geneColumn];
		if (gene) {
			if (!groups.has(name)) groups.set(name, []);
			groups.get(name).push(gene);
		}
	}
	const result = [...groups].map(([name, genes]) => ({
		name: name.slice(0, 120),
		genes: cleanGenes(genes),
	}));
	if (!result.length || result.length > 100)
		throw new Error("Import between 1 and 100 non-empty gene lists.");
	return result;
}

export function matchGenes(genes, available) {
	const names = new Set(available);
	return {
		matched: genes.filter((v) => names.has(v)),
		missing: genes.filter((v) => !names.has(v)),
	};
}
