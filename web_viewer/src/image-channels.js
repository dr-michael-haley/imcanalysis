const identity = item => JSON.stringify([item.sourceId ?? '', item.name]);

export function imageCatalogue(index) {
  const channels = new Map();
  for (const { images } of index.values()) for (const item of images) {
    const id = identity(item);
    if (!channels.has(id)) channels.set(id, { name: item.name, sourceId: item.sourceId ?? '', key: `unavailable:${id}`, file: null });
  }
  return [...channels.values()];
}

export function channelChoices(index, catalogue, roi) {
  const present = index.get(roi)?.images ?? [], byName = new Map(present.map(item => [identity(item), item]));
  return catalogue.map(item => byName.get(identity(item)) ?? item);
}
