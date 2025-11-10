#!/usr/bin/env python3
"""
Example script demonstrating the different backgating modes for manual marker customization.

This example shows the typical workflow:
1. Use 'save_markers' mode to compute optimal markers
2. Manually edit the settings file  
3. Use 'load_markers' mode to generate images with custom markers
"""

import anndata as ad
from pathlib import Path
import sys
import pandas as pd

# Add the current directory to the path so we can import the backgating module
sys.path.insert(0, str(Path(__file__).parent))

from backgating import backgating_assessment

def example_backgating_workflow():
    """Example of using different backgating modes for marker customization."""
    
    # Load your processed AnnData (replace with your actual path)
    adata_path = "anndata_processed.h5ad"
    if not Path(adata_path).exists():
        print(f"Please update adata_path to point to your processed AnnData file")
        return
    
    adata = ad.read_h5ad(adata_path)
    
    # Configuration
    image_folder = "processed"  # Your denoised images folder
    pop_obs = "leiden_res_0.5"  # Your population column
    output_folder = "Backgating_Example"
    
    print("=== STEP 1: Compute optimal markers (save_markers mode) ===")
    
    # Step 1: Compute differential expression and save marker assignments
    # This will create backgating_settings_*.csv files but no images
    backgating_assessment(
        adata=adata,
        image_folder=image_folder,
        pop_obs=pop_obs,
        output_folder=output_folder,
        mode='save_markers',  # Only compute and save marker assignments
        use_differential_expression=True,
        de_method='wilcoxon',
        min_logfc_threshold=0.2,
        max_pval_adj=0.05,
        markers_exclude=['DNA1', 'DNA3'],
        number_top_markers=3,
        cells_per_group=25,  # Smaller for example
        radius=10
    )
    
    print(f"\n=== STEP 2: Review and edit settings file ===")
    
    # Show the user where to find the settings file
    settings_file = Path(output_folder) / f"backgating_settings_{pop_obs}.csv"
    
    if settings_file.exists():
        print(f"Settings file created: {settings_file}")
        
        # Load and display the settings for user review
        settings_df = pd.read_csv(settings_file, index_col=0)
        print("\nCurrent marker assignments:")
        print(settings_df[['Red', 'Green', 'Blue']].head())
        
        print(f"\nTo customize markers:")
        print(f"1. Edit {settings_file}")
        print(f"2. Modify Red/Green/Blue columns as desired")
        print(f"3. Save the file")
        print(f"4. Run step 3 below to generate images with your custom markers")
        
        # Example of programmatic modification (you could also edit manually)
        print(f"\n=== Example: Programmatically modify settings ===")
        
        # Example: Force all populations to use DNA1 for blue channel
        settings_df['Blue'] = 'DNA1'
        
        # Example: Set specific markers for a population if it exists
        if 'T_cells' in settings_df.index:
            settings_df.loc['T_cells', 'Red'] = 'CD3'
            settings_df.loc['T_cells', 'Green'] = 'CD8'
            settings_df.loc['T_cells', 'Blue'] = 'DNA1'
        
        # Save modified settings
        settings_df.to_csv(settings_file)
        print(f"Modified settings saved to {settings_file}")
        print("\nModified marker assignments:")
        print(settings_df[['Red', 'Green', 'Blue']].head())
    
    else:
        print(f"Settings file not found: {settings_file}")
        return
    
    print(f"\n=== STEP 3: Generate images with custom markers (load_markers mode) ===")
    
    # Step 3: Load the (possibly modified) settings and generate images
    backgating_assessment(
        adata=adata,
        image_folder=image_folder,
        pop_obs=pop_obs,
        output_folder=output_folder,
        mode='load_markers',  # Load existing settings and generate images
        cells_per_group=25,  # Smaller for example  
        radius=10,
        # Note: DE parameters are ignored in 'load_markers' mode
        # since we're using pre-computed marker assignments
    )
    
    print(f"\n=== Workflow Complete ===")
    print(f"Check {output_folder} for:")
    print(f"  - backgating_settings_{pop_obs}.csv (marker assignments)")
    print(f"  - {pop_obs}/ (backgating images)")
    print(f"  - Cells.png files (cell thumbnails)")

def example_single_run():
    """Example of standard single-run backgating (full mode)."""
    
    print("=== Single-run backgating example (full mode) ===")
    
    # Load your processed AnnData (replace with your actual path)
    adata_path = "anndata_processed.h5ad"
    if not Path(adata_path).exists():
        print(f"Please update adata_path to point to your processed AnnData file")
        return
    
    adata = ad.read_h5ad(adata_path)
    
    # This is the standard approach - compute markers and generate images in one step
    backgating_assessment(
        adata=adata,
        image_folder="processed",
        pop_obs="leiden_res_0.5", 
        output_folder="Backgating_Standard",
        mode='full',  # Complete analysis in one step
        use_differential_expression=True,
        markers_exclude=['DNA1', 'DNA3'],
        number_top_markers=2,
        cells_per_group=50,
        radius=15
    )

if __name__ == "__main__":
    print("Backgating Mode Examples")
    print("=" * 50)
    
    try:
        # Run the multi-step workflow example
        example_backgating_workflow()
        
        # Uncomment to also run single-step example
        # example_single_run()
        
    except Exception as e:
        print(f"Example failed: {e}")
        import traceback
        traceback.print_exc()
        
    print("\n" + "=" * 50)
    print("Examples completed!")