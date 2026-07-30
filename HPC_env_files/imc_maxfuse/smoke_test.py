"""Tiny end-to-end MaxFuse 0.0.2 environment smoke test."""

from __future__ import annotations

import maxfuse as mf
import numpy as np


def main() -> None:
    print("MaxFuse smoke test: constructing synthetic inputs", flush=True)
    rng = np.random.default_rng(7)
    latent = rng.normal(size=(40, 6)).astype(np.float32)
    reference_shared = (
        latent + rng.normal(scale=0.05, size=latent.shape)
    ).astype(np.float32)
    target_shared = (
        np.repeat(latent, 2, axis=0)
        + rng.normal(
            scale=0.08,
            size=(80, 6),
        )
    ).astype(np.float32)
    reference_active = np.c_[reference_shared, rng.normal(size=(40, 4))].astype(
        np.float32
    )
    target_active = np.c_[target_shared, rng.normal(size=(80, 4))].astype(
        np.float32
    )
    fusor = mf.model.Fusor(
        shared_arr1=reference_shared,
        shared_arr2=target_shared,
        active_arr1=reference_active,
        active_arr2=target_active,
        labels1=np.repeat(np.arange(4), 10),
        labels2=np.repeat(np.arange(4), 20),
    )
    print("MaxFuse smoke test: splitting batches", flush=True)
    fusor.split_into_batches(
        max_outward_size=40,
        matching_ratio=2,
        metacell_size=1,
        batching_scheme="cyclic",
        seed=7,
        verbose=False,
    )
    print("MaxFuse smoke test: constructing graphs", flush=True)
    fusor.construct_graphs(
        n_neighbors1=5,
        n_neighbors2=5,
        svd_components1=4,
        svd_components2=4,
        resolution1=2,
        resolution2=2,
        leiden_seed=7,
        verbose=False,
    )
    print("MaxFuse smoke test: finding initial pivots", flush=True)
    fusor.find_initial_pivots(
        wt1=0.3,
        wt2=0.3,
        svd_components1=4,
        svd_components2=4,
        randomized_svd=True,
        verbose=False,
    )
    print("MaxFuse smoke test: refining pivots", flush=True)
    fusor.refine_pivots(
        wt1=0.3,
        wt2=0.3,
        svd_components1=4,
        svd_components2=4,
        cca_components=3,
        n_iters=1,
        randomized_svd=True,
        verbose=False,
    )
    fusor.filter_bad_matches(target="pivot", filter_prop=0.2)
    print("MaxFuse smoke test: propagating matches", flush=True)
    fusor.propagate(
        svd_components1=4,
        svd_components2=4,
        wt1=0.7,
        wt2=0.7,
        randomized_svd=True,
        verbose=False,
    )
    fusor.filter_bad_matches(target="propagated", filter_prop=0.1)
    matching = fusor.get_matching(order=(2, 1), target="full_data")
    if len(matching[0]) == 0:
        raise RuntimeError("Tiny MaxFuse smoke test returned no matches")
    print(f"MaxFuse smoke test: retained {len(matching[0])} matches", flush=True)


if __name__ == "__main__":
    main()
