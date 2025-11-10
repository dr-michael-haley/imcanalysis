#!/usr/bin/env python3
"""
Test script for the enhanced backgating differential expression functionality.
"""

import numpy as np
import pandas as pd
import scanpy as sc
from pathlib import Path
import sys

# Add the current directory to the path so we can import the backgating module
sys.path.insert(0, str(Path(__file__).parent))

from backgating import perform_differential_expression

def create_test_data():
    """Create synthetic test data for testing differential expression."""
    
    # Set up test parameters
    n_cells = 1000
    n_markers = 20
    n_populations = 4
    
    # Create marker names
    marker_names = [f'Marker_{i:02d}' for i in range(n_markers)]
    
    # Generate synthetic expression data
    np.random.seed(42)  # For reproducible results
    
    # Base expression levels (log-normal distribution)
    base_expression = np.random.lognormal(mean=1.0, sigma=0.5, size=(n_cells, n_markers))
    
    # Create population labels
    pop_labels = np.random.choice([f'Pop_{i}' for i in range(n_populations)], size=n_cells)
    
    # Add population-specific expression patterns
    for i, pop in enumerate([f'Pop_{i}' for i in range(n_populations)]):
        pop_mask = pop_labels == pop
        pop_indices = np.where(pop_mask)[0]
        
        # Give each population higher expression of specific markers
        marker_indices = [(i * 3 + j) % n_markers for j in range(3)]  # 3 markers per population
        
        for marker_idx in marker_indices:
            # Increase expression 2-5x for population-specific markers
            multiplier = np.random.uniform(2, 5)
            base_expression[pop_indices, marker_idx] *= multiplier
    
    # Create AnnData object
    adata = sc.AnnData(X=base_expression)
    adata.var_names = marker_names
    adata.obs['population'] = pd.Categorical(pop_labels)
    adata.obs['ROI'] = np.random.choice(['ROI_A', 'ROI_B', 'ROI_C'], size=n_cells)
    adata.obs['X_loc'] = np.random.randint(0, 1000, size=n_cells)
    adata.obs['Y_loc'] = np.random.randint(0, 1000, size=n_cells)
    adata.obs['Master_Index'] = range(n_cells)
    
    return adata

def test_differential_expression():
    """Test the differential expression functionality."""
    
    print("Creating synthetic test data...")
    adata = create_test_data()
    
    print(f"Test data created:")
    print(f"  - {adata.n_obs} cells")
    print(f"  - {adata.n_vars} markers")
    print(f"  - Populations: {list(adata.obs['population'].cat.categories)}")
    
    # Test differential expression for each population
    print("\nTesting differential expression analysis...")
    
    for pop in adata.obs['population'].cat.categories:
        print(f"\n--- Testing population: {pop} ---")
        
        # Test with differential expression
        top_markers_de = perform_differential_expression(
            adata=adata,
            pop_obs='population',
            target_population=pop,
            markers_exclude=['Marker_00', 'Marker_01'],  # Exclude first two markers
            method='wilcoxon',
            n_top_markers=3,
            min_logfc_threshold=0.2,
            max_pval_adj=0.1,
            verbose=True
        )
        
        print(f"Top markers identified: {top_markers_de}")
        
        # Verify that excluded markers are not included
        excluded_markers = ['Marker_00', 'Marker_01']
        included_excluded = [m for m in top_markers_de if m in excluded_markers]
        if included_excluded:
            print(f"WARNING: Excluded markers found in results: {included_excluded}")
        else:
            print("✓ Excluded markers properly filtered out")
    
    # Test with only_use_markers parameter
    print(f"\n--- Testing with only_use_markers parameter ---")
    limited_markers = ['Marker_05', 'Marker_06', 'Marker_07', 'Marker_08', 'Marker_09']
    
    top_markers_limited = perform_differential_expression(
        adata=adata,
        pop_obs='population',
        target_population='Pop_0',
        only_use_markers=limited_markers,
        method='wilcoxon',
        n_top_markers=3,
        min_logfc_threshold=0.1,
        max_pval_adj=0.1,
        verbose=True
    )
    
    print(f"Top markers from limited set: {top_markers_limited}")
    
    # Verify that only specified markers are included
    unexpected_markers = [m for m in top_markers_limited if m not in limited_markers]
    if unexpected_markers:
        print(f"WARNING: Unexpected markers found: {unexpected_markers}")
    else:
        print("✓ Only specified markers used in analysis")
    
    print("\n--- Test completed successfully! ---")

if __name__ == "__main__":
    try:
        test_differential_expression()
    except Exception as e:
        print(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()