import { getChildNode, getArrayMetadata, getNodeAttrs } from "zarrextra";

// Validate the gene axis without reading or densifying expression values.
export async function tableGenes(dataset, key) {
	const table = dataset.tables[key];
	if (!table) throw new Error("The linked table is unavailable.");
	const node = getChildNode(dataset.rootStore.tree, "tables", key, "X");
	if (!node) throw new Error("This table has no X matrix.");
	const shape = getArrayMetadata(node)?.shape || getNodeAttrs(node)?.shape;
	const names = await table.loadVarIndex();
	if (!shape || shape.length !== 2 || shape[1] !== names.length)
		throw new Error("X columns do not match the gene index.");
	return names;
}
