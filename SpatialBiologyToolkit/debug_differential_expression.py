#!/usr/bin/env python3
"""
Simple test to debug differential expression analysis in backgating.
"""

import sys
from pathlib import Path

# Add the current directory to the path so we can import the backgating module
sys.path.insert(0, str(Path(__file__).parent))

def test_differential_expression_debug():
    """Test if differential expression is being called correctly."""
    
    try:
        import anndata as ad
        import scanpy as sc
        import pandas as pd
        import numpy as np
        from backgating import perform_differential_expression, backgating_assessment
        
        print("=== Testing Differential Expression Analysis ===")
        
        # Create simple test data
        n_cells = 200
        n_markers = 10
        
        # Create marker names
        markers = [f'Marker_{i}' for i in range(n_markers)]
        
        # Generate expression data
        np.random.seed(42)
        X = np.random.lognormal(mean=1.0, sigma=0.5, size=(n_cells, n_markers))
        
        # Create populations with differential expression
        pop_labels = np.random.choice(['Pop_A', 'Pop_B', 'Pop_C'], size=n_cells)
        
        # Make Pop_A express Marker_0 and Marker_1 highly
        pop_a_mask = pop_labels == 'Pop_A'
        X[pop_a_mask, 0] *= 5.0  # Marker_0 high in Pop_A
        X[pop_a_mask, 1] *= 3.0  # Marker_1 high in Pop_A
        
        # Create AnnData
        adata = ad.AnnData(X=X)
        adata.var_names = markers
        adata.obs['population'] = pd.Categorical(pop_labels)
        adata.obs['ROI'] = np.random.choice(['ROI_1', 'ROI_2'], size=n_cells)
        adata.obs['X_loc'] = np.random.randint(0, 1000, size=n_cells)
        adata.obs['Y_loc'] = np.random.randint(0, 1000, size=n_cells)
        adata.obs['Master_Index'] = range(n_cells)
        
        print(f"Created test data: {adata.n_obs} cells, {adata.n_vars} markers")
        print(f"Populations: {list(adata.obs['population'].cat.categories)}")
        
        # Test direct call to perform_differential_expression
        print("\n=== Testing direct call to perform_differential_expression ===")
        
        top_markers = perform_differential_expression(
            adata=adata,
            pop_obs='population',
            target_population='Pop_A',
            markers_exclude=[],
            method='wilcoxon',
            n_top_markers=3,
            min_logfc_threshold=0.1,
            max_pval_adj=0.05,
            verbose=True
        )
        
        print(f"Direct call result: {top_markers}")
        
        # Test via backgating_assessment with save_markers mode
        print("\n=== Testing via backgating_assessment (save_markers mode) ===")
        
        try:
            backgating_assessment(
                adata=adata,
                image_folder="dummy_folder",  # Won't be used in save_markers mode
                pop_obs='population',
                output_folder='test_backgating_debug',
                mode='save_markers',
                use_differential_expression=True,
                de_method='wilcoxon',
                min_logfc_threshold=0.1,
                max_pval_adj=0.05,
                number_top_markers=3,
                markers_exclude=[],
                pops_list=['Pop_A']  # Only test one population
            )
            
            # Check if settings file was created
            settings_file = Path('test_backgating_debug') / 'backgating_settings_population.csv'
            if settings_file.exists():
                settings_df = pd.read_csv(settings_file, index_col=0)
                print(f"Settings file created successfully!")
                print("Marker assignments:")
                print(settings_df[['Red', 'Green', 'Blue']])
            else:
                print("ERROR: Settings file was not created!")
                
        except Exception as e:
            print(f"ERROR in backgating_assessment: {e}")
            import traceback
            traceback.print_exc()
        
    except ImportError as e:
        print(f"Import error: {e}")
        print("Some required packages are not available for testing")
    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_differential_expression_debug()