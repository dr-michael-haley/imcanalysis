#!/usr/bin/env python3
"""
Test script to debug differential expression analysis issues.
"""

import sys
import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
from pathlib import Path

# Add current directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

try:
    from backgating import perform_differential_expression
    print("Successfully imported perform_differential_expression", flush=True)
except ImportError as e:
    print(f"Failed to import perform_differential_expression: {e}", flush=True)
    sys.exit(1)

def create_test_data():
    """Create simple test AnnData with known populations."""
    print("Creating test AnnData...", flush=True)
    
    # Create synthetic data
    n_cells = 1000
    n_markers = 10
    
    # Random expression data
    np.random.seed(42)
    X = np.random.lognormal(0, 1, (n_cells, n_markers))
    
    # Create marker names
    marker_names = [f"Marker_{i+1}" for i in range(n_markers)]
    
    # Create populations
    populations = np.random.choice(['Pop_A', 'Pop_B', 'Pop_C'], n_cells)
    
    # Make Pop_A express Marker_1 and Marker_2 highly
    pop_a_mask = populations == 'Pop_A'
    X[pop_a_mask, 0] *= 3  # Marker_1
    X[pop_a_mask, 1] *= 2.5  # Marker_2
    
    # Create AnnData
    adata = ad.AnnData(X)
    adata.var_names = marker_names
    adata.obs['test_population'] = pd.Categorical(populations)
    
    print(f"Created test data: {adata.n_obs} cells, {adata.n_vars} markers", flush=True)
    print(f"Population distribution: {dict(adata.obs['test_population'].value_counts())}", flush=True)
    
    return adata

def test_differential_expression():
    """Test the differential expression function."""
    print("="*60, flush=True)
    print("TESTING DIFFERENTIAL EXPRESSION", flush=True)
    print("="*60, flush=True)
    
    # Create test data
    adata = create_test_data()
    
    # Test differential expression
    try:
        print("\n1. Testing Pop_A vs rest...", flush=True)
        result_a = perform_differential_expression(
            adata=adata,
            pop_obs='test_population',
            target_population='Pop_A',
            markers_exclude=[],
            n_top_markers=3,
            verbose=True
        )
        print(f"Result for Pop_A: {result_a}", flush=True)
        
        print("\n2. Testing Pop_B vs rest...", flush=True)
        result_b = perform_differential_expression(
            adata=adata,
            pop_obs='test_population',
            target_population='Pop_B',
            markers_exclude=[],
            n_top_markers=3,
            verbose=True
        )
        print(f"Result for Pop_B: {result_b}", flush=True)
        
        print("\n3. Testing with marker exclusion...", flush=True)
        result_c = perform_differential_expression(
            adata=adata,
            pop_obs='test_population',
            target_population='Pop_A',
            markers_exclude=['Marker_1'],
            n_top_markers=2,
            verbose=True
        )
        print(f"Result for Pop_A (excluding Marker_1): {result_c}", flush=True)
        
        print("\n" + "="*60, flush=True)
        print("ALL TESTS COMPLETED SUCCESSFULLY", flush=True)
        print("="*60, flush=True)
        
    except Exception as e:
        print(f"\nERROR during testing: {e}", flush=True)
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    success = test_differential_expression()
    sys.exit(0 if success else 1)