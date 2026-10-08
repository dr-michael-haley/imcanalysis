"""Spatial environments of saved image-patch embeddings; no image-model fitting.

CellCharter performs centre/one-hop concatenation on explicit physical-radius
graphs. sklearn GMM provides seeded CPU fits and reloadable probabilities.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.spatial import cKDTree
from scipy.sparse.csgraph import connected_components


def radius_graph(coords, rois, radius, pitch):
    """Radius graph restricted to components of retained edge-adjacent tiles.

    No cross-ROI links or self edges. Component restriction prevents bridging
    disconnected islands, but does not constitute segmentation of internal holes.
    """
    coords = np.asarray(coords, dtype=float)
    rois = np.asarray(rois).astype(str)
    if coords.shape != (len(rois), 2) or not np.isfinite(coords).all():
        raise ValueError("Coordinates must be finite N x 2 micrometres")
    if radius < 0 or pitch <= 0:
        raise ValueError("Radius must be nonnegative and patch pitch positive")
    rows, cols = [], []
    for roi in np.unique(rois):
        ids = np.flatnonzero(rois == roi)
        xy = coords[ids]
        if len(np.unique(xy, axis=0)) != len(xy):
            raise ValueError(f"Duplicate patch centres in ROI {roi}")
        # This graph contract is for regular, non-overlapping square tiles.
        grid = (xy - xy.min(axis=0)) / pitch
        if not np.allclose(grid, np.rint(grid), atol=1e-4):
            raise ValueError(f"Patch centres in {roi} do not form the configured grid")
        tree = cKDTree(xy)
        pairs = tree.query_pairs(pitch * (1 + 1e-6), output_type="ndarray")
        edge = sparse.csr_matrix(
            (np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])),
            shape=(len(ids), len(ids)),
        )
        _, component = connected_components(edge + edge.T, directed=False)
        if radius == 0:
            continue
        candidates = tree.query_pairs(radius * (1 + 1e-6), output_type="ndarray")
        for a, b in candidates:
            if component[a] == component[b]:
                rows.extend([ids[a], ids[b]])
                cols.extend([ids[b], ids[a]])
    return sparse.csr_matrix(
        (np.ones(len(rows), dtype=np.float32), (rows, cols)),
        shape=(len(rois), len(rois)),
    )


def neighbour_means(graph, values):
    """Finite-value means; isolates are NaN, never fabricated zeros."""
    values = np.asarray(values, dtype=float)
    valid = np.isfinite(values)
    sums = graph @ np.where(valid, values, 0)
    counts = graph @ valid.astype(float)
    return np.divide(sums, counts, out=np.full_like(sums, np.nan), where=counts > 0)


def aggregate_features(features, obs, graph):
    """Reuse CellCharter, checking its output against explicit graph means."""
    import anndata as ad
    import cellcharter as cc

    data = ad.AnnData(X=np.asarray(features, dtype=np.float32), obs=obs.copy())
    data.obsp["spatial_connectivities"] = graph
    cc.gr.aggregate_neighbors(
        data, n_layers=1, aggregations="mean", sample_key="roi", out_key="X_environment"
    )
    result = np.asarray(data.obsm["X_environment"])
    n = features.shape[1]
    nonempty = np.asarray(graph.sum(axis=1)).ravel() > 0
    expected = neighbour_means(graph, features)
    if not np.allclose(result[nonempty, n:], expected[nonempty], atol=1e-5):
        raise RuntimeError(
            "CellCharter aggregation differs from expected one-hop means"
        )
    if not np.allclose(result[:, :n], features, atol=1e-5):
        raise RuntimeError("CellCharter changed focal features")
    result[~nonempty, n:] = np.nan
    return result


def aligned_metrics(path, obs_names, prefix):
    import anndata as ad

    a = ad.read_h5ad(path)
    if not a.obs_names.is_unique or set(a.obs_names) != set(obs_names):
        raise ValueError(f"Patch identity mismatch: {path}")
    a = a[obs_names]
    values = a.X.toarray() if sparse.issparse(a.X) else np.asarray(a.X)
    return pd.DataFrame(
        values, index=obs_names, columns=[prefix + str(x) for x in a.var_names]
    )


def summarize_metrics(values, labels, obs, reference):
    """Distribution and ROI/case-balanced means of one observation per centre."""
    frames = []
    for group in sorted(set(labels) - {"unassigned"}):
        mask = labels == group
        v = values.loc[mask]
        frame = pd.DataFrame(
            {
                "mean": v.mean(),
                "median": v.median(),
                "q25": v.quantile(0.25),
                "q75": v.quantile(0.75),
                "n_valid": v.count(),
            }
        )
        frame["roi_balanced_mean"] = (
            v.groupby(obs.loc[mask, "roi"], observed=True).mean().mean()
        )
        frame["case_balanced_mean"] = (
            v.groupby(obs.loc[mask, "case_id"], observed=True).mean().mean()
        )
        frame["environment"] = group
        frame.index.name = "metric"
        frames.append(frame.reset_index())
    result = pd.concat(frames, ignore_index=True)
    strat = values.copy()
    strat["environment"] = labels
    strat["reference_cluster"] = obs[reference].astype(str)
    strat = strat[strat.environment != "unassigned"].groupby(
        ["environment", "reference_cluster"], observed=True
    )
    return result, strat.mean().join(strat.size().rename("n_patches"))


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def atomic_json(path, value):
    p = Path(path)
    temp = p.with_suffix(p.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, default=str), encoding="utf-8")
    temp.replace(p)


def fit_gmm(x, k, seed, cfg):
    from sklearn.mixture import GaussianMixture

    model = GaussianMixture(
        n_components=k,
        covariance_type=cfg.covariance_type,
        reg_covar=cfg.reg_covar,
        max_iter=cfg.max_iter,
        n_init=1,
        random_state=seed,
    )
    model.fit(x)
    return model


def label_support(labels, obs, reference):
    rows = []
    for label in sorted(set(labels) - {"unassigned"}):
        sub = obs.loc[labels == label]
        counts = sub[reference].astype(str).value_counts(normalize=True)
        rows.append(
            {
                "environment": label,
                "n_patches": len(sub),
                "fraction_all_patches": len(sub) / len(obs),
                "n_rois": sub.roi.nunique(),
                "n_cases": sub.case_id.nunique(),
                "reference_purity": counts.max(),
                "reference_entropy": float(-(counts * np.log(counts)).sum()),
                "max_roi_fraction": sub.roi.value_counts(normalize=True).max(),
            }
        )
    return pd.DataFrame(rows)


def run(config, output, report):
    """Run bounded radius/K scan; checkpoints are bound to input hashes/config."""
    import anndata as ad
    import joblib
    from sklearn.decomposition import PCA
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
    from threadpoolctl import threadpool_limits
    from SpatialBiologyToolkit.reporting import project_asset_path
    from SpatialBiologyToolkit.anndata_io import write_h5ad_compat

    cfg = config.hyperstac_environments
    root = project_asset_path(config.hyperstac.asset_folder)
    if not cfg.reference_cluster or not cfg.case_mapping_csv:
        raise ValueError(
            "reference_cluster and case_mapping_csv must be explicitly configured"
        )
    if output.resolve() == root.resolve() or output.resolve() in root.resolve().parents:
        raise ValueError(
            "Choose a separate environment output folder below or outside the source asset root"
        )
    paths = [
        root / "imc_hyperstac_representations.h5ad",
        root / "imc_hyperstac_patch_metrics.h5ad",
        root / "permutation_sensitivity/imc_permutation_sensitivity.h5ad",
    ]
    metadata = project_asset_path(cfg.case_mapping_csv)
    for p in [*paths, metadata]:
        if not p.is_file():
            raise FileNotFoundError(p)
    fingerprint = {
        "inputs": {str(p): file_hash(p) for p in [*paths, metadata]},
        "config": cfg.model_dump(mode="json"),
        "implementation_sha256": file_hash(__file__),
    }
    signature = hashlib.sha256(
        json.dumps(fingerprint, sort_keys=True).encode()
    ).hexdigest()
    output.mkdir(parents=True, exist_ok=True)
    marker = output / "input_signature.json"
    if not marker.exists() and any(output.iterdir()):
        raise ValueError(
            "Refusing a nonempty output directory without an analysis signature"
        )
    if marker.exists() and json.loads(marker.read_text())["signature"] != signature:
        raise ValueError(
            "Output contains a different analysis; choose a new output_folder"
        )
    atomic_json(marker, {"signature": signature, **fingerprint})
    source = ad.read_h5ad(paths[0])
    if not source.obs_names.is_unique or cfg.reference_cluster not in source.obs:
        raise ValueError(
            "Unique patch IDs and configured reference clustering are required"
        )
    obs = source.obs.copy()
    if obs[cfg.reference_cluster].isna().any():
        raise ValueError("Reference clustering contains missing labels")
    mapping = pd.read_csv(
        metadata, dtype={cfg.mapping_roi_column: str, cfg.mapping_case_column: str}
    )
    pairs = (
        mapping[[cfg.mapping_roi_column, cfg.mapping_case_column]]
        .dropna()
        .drop_duplicates()
    )
    if pairs[cfg.mapping_roi_column].duplicated().any():
        raise ValueError("Conflicting ROI to case mapping")
    obs["roi"] = obs.roi.astype(str)
    obs["case_id"] = obs.roi.map(
        pairs.set_index(cfg.mapping_roi_column)[cfg.mapping_case_column]
    )
    # Missing outcome cases remain in spatial discovery, but are separate groups.
    obs["resampling_group"] = obs.case_id.fillna("unmapped_roi:" + obs.roi)
    coords = obs[["center_col_um", "center_row_um"]].to_numpy(float)
    raw = source.X.toarray() if sparse.issparse(source.X) else np.asarray(source.X)
    if not np.isfinite(raw).all() or cfg.n_pcs >= min(raw.shape):
        raise ValueError("Invalid embedding values or PCA dimension")
    metrics = aligned_metrics(paths[1], source.obs_names, "metric__")
    perm = aligned_metrics(paths[2], source.obs_names, "sensitivity__")
    # Keep individual shuffle repeats in the source; report their mean per channel.
    for channel in sorted(
        {c.split("__")[2] for c in perm if c.startswith("sensitivity__shuffle__")}
    ):
        cols = [
            c for c in perm if c.startswith("sensitivity__shuffle__" + channel + "__")
        ]
        perm["sensitivity__shuffle_mean__" + channel] = perm[cols].mean(axis=1)
    perm = perm[
        [
            c
            for c in perm
            if c.startswith(("sensitivity__zero__", "sensitivity__shuffle_mean__"))
        ]
    ]
    metrics = pd.concat([metrics, perm], axis=1)
    metrics.to_csv(output / "patch_metrics.csv.gz")
    obs.to_csv(output / "patch_observations.csv.gz")
    report.add_note(
        "Sensitivity summaries describe existing patch encoder perturbations, not environment-classifier attribution. Missing neighbours are unassigned. Case resampling holds the encoder fixed."
    )
    for p in [*paths, metadata]:
        report.add_input("source", p, "Read-only environment discovery input.")
    scores, repeat_rows = [], []
    result = ad.AnnData(obs=obs.copy())
    result.obs["case_id"] = pd.Categorical(result.obs.case_id)
    result.obsm["spatial"] = coords
    groups = obs.resampling_group.to_numpy(str)
    unique_groups = np.unique(groups)
    with threadpool_limits(limits=cfg.cpu_threads):
        pca = PCA(
            n_components=cfg.n_pcs, svd_solver="randomized", random_state=cfg.seed
        )
        base = pca.fit_transform(raw)
        joblib.dump(pca, output / "pca.joblib")
        result.obsm["X_pca"] = base
        if len(unique_groups) < 3:
            raise ValueError(
                "Patient stability requires at least three patient/ROI groups"
            )
        rng = np.random.default_rng(cfg.seed)
        resamples = []
        sample_record = []
        for repeat in range(cfg.patient_repeats):
            n_groups = min(
                len(unique_groups) - 1,
                max(2, int(len(unique_groups) * cfg.patient_fraction)),
            )
            chosen = rng.choice(unique_groups, size=n_groups, replace=False)
            train = np.isin(groups, chosen)
            pp = PCA(
                n_components=cfg.n_pcs,
                svd_solver="randomized",
                random_state=cfg.seed + repeat,
            )
            pp.fit(raw[train])
            resamples.append((train, pp.transform(raw)))
            joblib.dump(pp, output / f"patient_pca_{repeat}.joblib")
            sample_record.append({"repeat": repeat, "training_groups": chosen.tolist()})
        atomic_json(output / "patient_subsamples.json", sample_record)
        for radius in cfg.radii_um:
            key = f"r{radius:g}"
            folder = output / key
            folder.mkdir(exist_ok=True)
            graph = radius_graph(coords, obs.roi, radius, cfg.patch_pitch_um)
            counts = np.asarray(graph.sum(axis=1)).ravel()
            eligible = np.ones(len(obs), dtype=bool) if radius == 0 else counts > 0
            if eligible.sum() <= max(cfg.n_clusters):
                raise ValueError(f"Too few supported patch centres at radius {radius}")
            x = base if radius == 0 else aggregate_features(base, obs, graph)
            resampled_features = [
                b
                if radius == 0
                else np.concatenate([b, neighbour_means(graph, b)], axis=1)
                for _, b in resamples
            ]
            result.obsm["X_environment_" + key] = x
            result.obsp["environment_graph_" + key] = graph
            result.obs["neighbours_" + key] = counts
            coverage = obs[["roi", "case_id", "center_col_um", "center_row_um"]].copy()
            coverage["neighbours"] = counts
            coverage["distance_to_roi_boundary_um"] = np.minimum.reduce(
                [
                    coords[:, 0],
                    coords[:, 1],
                    obs.roi_width_um.to_numpy() - coords[:, 0],
                    obs.roi_height_um.to_numpy() - coords[:, 1],
                ]
            )
            coverage["radius_truncated_by_roi"] = (
                coverage.distance_to_roi_boundary_um < radius
            )
            coverage.to_csv(folder / "neighbour_coverage.csv")
            neighbour_metrics = (
                pd.DataFrame(
                    neighbour_means(graph, metrics.to_numpy()),
                    index=obs.index,
                    columns=metrics.columns,
                )
                if radius
                else None
            )
            for k in cfg.n_clusters:
                name = f"env_{key}_k{k}"
                target = folder / f"k{k}"
                target.mkdir(exist_ok=True)
                checkpoint = target / "complete.json"
                if checkpoint.exists():
                    saved = json.loads(checkpoint.read_text())
                    model = joblib.load(target / "model.joblib")
                    score = saved["score"]
                    local_repeats = saved["repeats"]
                else:
                    print(f"Fitting {name}", flush=True)
                    models = [
                        fit_gmm(x[eligible], k, cfg.seed + i, cfg)
                        for i in range(cfg.fit_repeats)
                    ]
                    good = [m for m in models if m.converged_]
                    if not good:
                        raise RuntimeError(f"No converged GMM fit for {name}")
                    model = max(good, key=lambda m: m.lower_bound_)
                    predictions = [m.predict(x[eligible]) for m in good]
                    agreements = [
                        adjusted_rand_score(predictions[i], predictions[j])
                        for i in range(len(good))
                        for j in range(i)
                    ]
                    local_repeats = []
                    ref = model.predict(x[eligible])
                    # Refit PCA and mixture after sampling whole patients (all their ROIs).
                    for repeat, (train, _) in enumerate(resamples):
                        z = resampled_features[repeat]
                        gm = fit_gmm(
                            z[train & eligible], k, cfg.seed + 100 + repeat, cfg
                        )
                        pred = gm.predict(z[eligible])
                        heldout = ~train[eligible]
                        local_repeats.append(
                            {
                                "setting": name,
                                "repeat": repeat,
                                "converged": bool(gm.converged_),
                                "all_patch_ari": adjusted_rand_score(ref, pred),
                                "heldout_patch_ari": adjusted_rand_score(
                                    ref[heldout], pred[heldout]
                                ),
                                "n_train_groups": len(
                                    sample_record[repeat]["training_groups"]
                                ),
                                "n_heldout_patches": int(heldout.sum()),
                            }
                        )
                    labs = np.full(len(obs), "unassigned", dtype=object)
                    labs[eligible] = model.predict(x[eligible]).astype(str)
                    support = label_support(labs, obs, cfg.reference_cluster)
                    probabilities = model.predict_proba(x[eligible])
                    score = {
                        "setting": name,
                        "radius_um": radius,
                        "n_clusters": k,
                        "n_assigned": int(eligible.sum()),
                        "n_unassigned": int((~eligible).sum()),
                        "converged_fits": len(good),
                        "attempted_fits": len(models),
                        "seed_ari_mean": float(np.mean(agreements))
                        if agreements
                        else None,
                        "patient_heldout_ari_mean": float(
                            np.mean(
                                [
                                    r["heldout_patch_ari"]
                                    for r in local_repeats
                                    if r["converged"]
                                ]
                            )
                        )
                        if any(r["converged"] for r in local_repeats)
                        else None,
                        "valid_patient_repeats": sum(
                            r["converged"] for r in local_repeats
                        ),
                        "min_rois": int(support.n_rois.min()),
                        "min_cases": int(support.n_cases.min()),
                        "min_cluster_fraction": float(
                            support.fraction_all_patches.min()
                        ),
                        "reference_ari": adjusted_rand_score(
                            obs.loc[eligible, cfg.reference_cluster], labs[eligible]
                        ),
                        "reference_nmi": normalized_mutual_info_score(
                            obs.loc[eligible, cfg.reference_cluster], labs[eligible]
                        ),
                        "mean_assignment_probability": float(
                            probabilities.max(axis=1).mean()
                        ),
                        "bic": float(model.bic(x[eligible])),
                    }
                    joblib.dump(model, target / "model.joblib")
                    atomic_json(
                        checkpoint,
                        {
                            "signature": signature,
                            "score": score,
                            "repeats": local_repeats,
                        },
                    )
                labels = np.full(len(obs), "unassigned", dtype=object)
                labels[eligible] = model.predict(x[eligible]).astype(str)
                probs = np.full((len(obs), k), np.nan, dtype=np.float32)
                probs[eligible] = model.predict_proba(x[eligible])
                result.obs[name] = pd.Categorical(labels)
                result.obsm[name + "_probability"] = probs
                scores.append(score)
                repeat_rows.extend(local_repeats)
                support = label_support(labels, obs, cfg.reference_cluster)
                support.to_csv(target / "support.csv", index=False)
                focal, strat = summarize_metrics(
                    metrics, labels, obs, cfg.reference_cluster
                )
                focal.to_csv(target / "focal_metrics.csv", index=False)
                strat.to_csv(target / "focal_metrics_by_reference.csv")
                if neighbour_metrics is not None:
                    surrounding, _ = summarize_metrics(
                        neighbour_metrics, labels, obs, cfg.reference_cluster
                    )
                    surrounding.to_csv(target / "surrounding_metrics.csv", index=False)
                pd.crosstab(
                    pd.Series(labels, index=obs.index, name="environment"),
                    obs[cfg.reference_cluster],
                    normalize="index",
                ).to_csv(target / "reference_composition.csv")
                onehot = pd.get_dummies(
                    obs[cfg.reference_cluster].astype(str), dtype=float
                )
                if radius:
                    nc = pd.DataFrame(
                        neighbour_means(graph, onehot.to_numpy()),
                        index=obs.index,
                        columns=onehot.columns,
                    )
                    nc.groupby(
                        pd.Series(labels, index=obs.index, name="environment")
                    ).mean().to_csv(target / "surrounding_reference_composition.csv")
                fractions = pd.crosstab(
                    obs.roi, pd.Series(labels, index=obs.index), normalize="index"
                )
                fractions.to_csv(target / "roi_fractions.csv")
                pd.crosstab(
                    obs.case_id, pd.Series(labels, index=obs.index), normalize="index"
                ).to_csv(target / "case_patch_fractions.csv")
                roi_case = (
                    obs[["roi", "case_id"]].drop_duplicates().set_index("roi").case_id
                )
                fractions.groupby(roi_case).mean().to_csv(
                    target / "case_equal_roi_fractions.csv"
                )
                pd.DataFrame(scores).to_csv(output / "scan_scorecard.csv", index=False)
                pd.DataFrame(repeat_rows).to_csv(
                    output / "patient_stability.csv", index=False
                )
                pd.DataFrame(
                    {"environment": labels, "probability": np.nanmax(probs, axis=1)},
                    index=obs.index,
                ).to_csv(target / "assignments.csv.gz")
                print(f"Completed {name}", flush=True)
    result.uns["environment_config"] = cfg.model_dump_json()
    result.uns["input_signature"] = signature
    write_h5ad_compat(result, output / "spatial_environments.h5ad")
    make_figures(output, obs, cfg)
    atomic_json(
        output / "completed.json",
        {
            "signature": signature,
            "settings": len(scores),
            "patches": len(obs),
            "reference": cfg.reference_cluster,
        },
    )
    report.add_asset(
        "hyperstac_environments",
        output,
        "Spatial environment AnnData, radius graphs, models and summaries.",
    )
    report.add_metric("settings", len(scores))
    report.add_metric("patches", len(obs))
    return output


def make_figures(output, obs, cfg):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    score = pd.read_csv(output / "scan_scorecard.csv")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for radius, sub in score.groupby("radius_um"):
        axes[0].plot(sub.n_clusters, sub.seed_ari_mean, "o-", label=f"{radius:g} um")
        axes[1].plot(
            sub.n_clusters, sub.patient_heldout_ari_mean, "o-", label=f"{radius:g} um"
        )
    for ax, title in zip(
        axes,
        ["Repeated-fit agreement", "Patient-subsample agreement on omitted patients"],
    ):
        ax.set(
            xlabel="Environment count", ylabel="ARI", title=title, ylim=(-0.05, 1.05)
        )
        ax.legend()
    fig.tight_layout()
    fig.savefig(output / "stability.png", dpi=160)
    plt.close(fig)
    # Bounded maps: deterministic first ROI per TMA, plus lowest-support examples.
    selected = list(
        obs.groupby(obs.roi.str.split("_").str[0], observed=True).roi.first().unique()
    )
    for row in score.itertuples():
        target = output / f"r{row.radius_um:g}" / f"k{row.n_clusters}"
        assignments = pd.read_csv(target / "assignments.csv.gz", index_col=0)
        fig, axes = plt.subplots(
            1, len(selected), figsize=(4 * len(selected), 4), squeeze=False
        )
        for ax, roi in zip(axes[0], selected):
            mask = obs.roi == roi
            lab = pd.to_numeric(assignments.loc[mask, "environment"], errors="coerce")
            ax.scatter(
                obs.loc[mask, "center_col_um"],
                obs.loc[mask, "center_row_um"],
                c=lab,
                cmap="tab20",
                vmin=0,
                vmax=19,
                marker="s",
                s=65,
            )
            ax.set_title(roi)
            ax.set_aspect("equal")
            ax.invert_yaxis()
        fig.suptitle(row.setting)
        fig.tight_layout()
        fig.savefig(target / "maps.png", dpi=140)
        plt.close(fig)
