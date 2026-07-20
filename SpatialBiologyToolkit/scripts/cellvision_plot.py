"""Generate CellVision UMAP, confusion, projection, and H5SC gallery reports."""

from __future__ import annotations

import logging


def _category_order(values) -> list[str]:
    labels = {str(value) for value in values if str(value) not in {"nan", "<NA>"}}

    def key(value: str):
        try:
            return (0, float(value))
        except ValueError:
            return (1, value)

    return sorted(labels, key=key)


def main() -> None:
    import anndata as ad
    import numpy as np

    from SpatialBiologyToolkit.cellvision import (
        confusion_tables,
        image_channel_metadata,
        leiden_key,
        open_h5sc_images,
        plot_categorical_embedding,
        plot_cell_gallery,
        plot_confusion_matrix,
        safe_slug,
    )
    from SpatialBiologyToolkit.reporting import category_output_path
    from SpatialBiologyToolkit.scripts._cellvision_common import (
        input_paths,
        load_runtime,
        reporter,
    )
    from SpatialBiologyToolkit.scripts.config_and_utils import read_h5ad_compat

    config, paths = load_runtime("plot")
    cellvision = config.cellvision
    source_path, _images_folder, _masks_folder = input_paths(config)
    for path in (paths.clustered, paths.h5sc, source_path):
        if not path.is_file():
            raise FileNotFoundError(f"CellVision plotting input does not exist: {path}")
    clustered = read_h5ad_compat(paths.clustered)
    source = ad.read_h5ad(source_path)
    if "X_cellvision_umap" not in clustered.obsm:
        raise KeyError("CellVision clustered AnnData lacks obsm['X_cellvision_umap'].")
    source_ids = source.obs_names.astype(str)
    if not source_ids.is_unique:
        raise ValueError("Source AnnData observation IDs must remain unique for CellVision projection.")
    source_lookup = {value: index for index, value in enumerate(source_ids)}
    missing_source = [value for value in clustered.obs_names.astype(str) if value not in source_lookup]
    if missing_source:
        raise ValueError(
            "Clustered CellVision cells are missing from the current source AnnData; "
            f"examples: {missing_source[:10]}"
        )

    images, h5sc_obs, h5sc_var = open_h5sc_images(paths.h5sc)
    channel_indices, channel_names = image_channel_metadata(h5sc_var, images.shape)
    h5sc_lookup = {value: index for index, value in enumerate(h5sc_obs.index.astype(str))}
    missing_h5sc = [value for value in clustered.obs_names.astype(str) if value not in h5sc_lookup]
    if missing_h5sc:
        images.file.close()
        raise ValueError(
            "Clustered CellVision cells are missing from the H5SC identity index; "
            f"examples: {missing_h5sc[:10]}"
        )

    figures_root = category_output_path("figures", stage="cellvision") / "cellvision"
    tables_root = category_output_path("tables", stage="cellvision") / "cellvision"
    figures_root.mkdir(parents=True, exist_ok=True)
    tables_root.mkdir(parents=True, exist_ok=True)
    stage_reporter = reporter()
    if stage_reporter is not None:
        stage_reporter.add_input("cellvision_clustered", paths.clustered, "CellVision UMAP and Leiden annotations.")
        stage_reporter.add_input("cellvision_h5sc", paths.h5sc, "Exact H5SC images used to train VICReg.")
        stage_reporter.add_input("anndata", source_path, "Source labels and original UMAP coordinates.")

    generated_figures = 0
    generated_tables = 0
    rng = np.random.default_rng(cellvision.seed)
    try:
        for resolution in cellvision.leiden_resolutions:
            cluster_key = leiden_key(resolution)
            if cluster_key not in clustered.obs.columns:
                raise KeyError(f"CellVision clustered AnnData lacks obs[{cluster_key!r}].")
            resolution_slug = safe_slug(cluster_key)
            resolution_dir = figures_root / resolution_slug
            resolution_dir.mkdir(parents=True, exist_ok=True)
            learned_labels = clustered.obs[cluster_key].astype("string")
            learned_umap = np.asarray(clustered.obsm["X_cellvision_umap"])

            learned_path = resolution_dir / "umap_cellvision_leiden.png"
            plot_categorical_embedding(
                learned_umap,
                learned_labels,
                title=f"CellVision UMAP: {cluster_key}",
                output_path=learned_path,
                dpi=cellvision.figure_dpi,
            )
            generated_figures += 1
            if stage_reporter is not None:
                stage_reporter.add_file("figure", learned_path, f"VICReg-space UMAP colored by {cluster_key}.")

            if cellvision.population_obs is not None:
                population_obs = cellvision.population_obs
                if population_obs not in clustered.obs.columns:
                    raise KeyError(
                        f"CellVision clustered AnnData lacks configured population obs {population_obs!r}."
                    )
                population_labels = clustered.obs[population_obs].astype("string")
                population_path = resolution_dir / f"umap_original_{safe_slug(population_obs)}.png"
                plot_categorical_embedding(
                    learned_umap,
                    population_labels,
                    title=f"CellVision UMAP: original {population_obs}",
                    output_path=population_path,
                    dpi=cellvision.figure_dpi,
                )
                counts, normalized = confusion_tables(population_labels, learned_labels)
                counts_path = tables_root / f"confusion_counts_{resolution_slug}.csv"
                normalized_path = tables_root / f"confusion_row_normalized_{resolution_slug}.csv"
                counts.to_csv(counts_path)
                normalized.to_csv(normalized_path)
                confusion_path = resolution_dir / "confusion_row_normalized.png"
                plot_confusion_matrix(
                    normalized,
                    title=f"Original {population_obs} vs {cluster_key}",
                    output_path=confusion_path,
                    dpi=cellvision.figure_dpi,
                    colorbar_label="Fraction within original population",
                )
                generated_figures += 2
                generated_tables += 2
                if stage_reporter is not None:
                    stage_reporter.add_file("figure", population_path, f"VICReg-space UMAP colored by original {population_obs}.")
                    stage_reporter.add_file("figure", confusion_path, f"Row-normalized original-population comparison for {cluster_key}.")
                    stage_reporter.add_file("table", counts_path, f"Cell counts comparing {population_obs} with {cluster_key}.")
                    stage_reporter.add_file("table", normalized_path, f"Row-normalized comparison of {population_obs} with {cluster_key}.")

            if cellvision.source_umap_key in source.obsm:
                positions = np.asarray([source_lookup[value] for value in clustered.obs_names.astype(str)])
                source_coordinates = np.asarray(source.obsm[cellvision.source_umap_key])
                projection_path = resolution_dir / "source_umap_cellvision_leiden.png"
                plot_categorical_embedding(
                    source_coordinates[positions],
                    learned_labels,
                    title=f"Original UMAP with {cluster_key}",
                    output_path=projection_path,
                    dpi=cellvision.figure_dpi,
                    background_coordinates=source_coordinates,
                )
                generated_figures += 1
                if stage_reporter is not None:
                    stage_reporter.add_file("figure", projection_path, f"Partial CellVision labels projected onto source {cellvision.source_umap_key}.")
            elif stage_reporter is not None:
                stage_reporter.add_warning(
                    f"Source AnnData lacks obsm[{cellvision.source_umap_key!r}]; skipped source-UMAP projections."
                )

            gallery_dir = resolution_dir / "galleries"
            categories = _category_order(learned_labels.dropna().tolist())
            if cellvision.gallery_max_clusters is not None:
                categories = categories[: cellvision.gallery_max_clusters]
            for category in categories:
                cell_ids = clustered.obs_names[learned_labels.astype(str).eq(category)].astype(str).tolist()
                sample_size = min(cellvision.gallery_cells_per_cluster, len(cell_ids))
                selected_ids = rng.choice(cell_ids, size=sample_size, replace=False).tolist()
                row_indices = [h5sc_lookup[value] for value in selected_ids]
                gallery_path = gallery_dir / f"cluster_{safe_slug(category)}.png"
                plot_cell_gallery(
                    images,
                    row_indices=row_indices,
                    cell_ids=selected_ids,
                    channel_indices=channel_indices,
                    channel_names=channel_names,
                    title=f"{cluster_key} = {category}",
                    output_path=gallery_path,
                    dpi=cellvision.figure_dpi,
                )
                generated_figures += 1
                if stage_reporter is not None:
                    stage_reporter.add_file("figure", gallery_path, f"H5SC cell gallery for {cluster_key} cluster {category}.")
    finally:
        images.file.close()

    logging.info(
        "CellVision plotting complete: %d figures and %d tables.",
        generated_figures,
        generated_tables,
    )
    if stage_reporter is not None:
        stage_reporter.add_metric("plot_figures", generated_figures)
        stage_reporter.add_metric("plot_tables", generated_tables)
        stage_reporter.add_metric("plotted_resolutions", len(cellvision.leiden_resolutions))
        stage_reporter.add_metric("gallery_channels", len(channel_names))


if __name__ == "__main__":
    main()
