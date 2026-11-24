export const MAX_VIEWS = 6;

export function gridShape(count) {
  const columns = count <= 3 ? count : count === 4 ? 2 : 3;
  return { columns, rows: Math.ceil(count / columns) };
}
