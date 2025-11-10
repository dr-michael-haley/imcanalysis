# Updated Clustering and Clustermap Functionality in create_spoox_heatmaps

## Summary of Changes

The `create_spoox_heatmaps` function has been updated with several major enhancements:

1. **Flexible clustering options**: Instead of the previous `cluster_mh` boolean parameter, you can now specify any column for clustering
2. **Seaborn clustermap support**: Option to generate clustermaps instead of regular heatmaps with full customization
3. **Dictionary-based color mapping**: Intuitive color specification using dictionaries that map cell types to colors
4. **Robust error handling**: Automatic fallback to regular heatmaps when clustering fails due to data issues
5. **Tuple figsize**: Consistent use of tuple format for figure dimensions

## New Parameters

### Clustering Parameters
- **`cluster_by`** (Optional[str]): Column name to use for clustering heatmaps. 
  - Default: `'Morueta-Holme_All'`
  - Set to `None` to disable clustering
  - Can be any valid column name from the SpOOx output data

### Clustermap Parameters
- **`use_clustermap`** (bool): If True, use seaborn clustermap instead of regular heatmap. Default is False.
- **`row_colors`** (Optional[Union[pd.Series, pd.DataFrame]]): Colors to use for row annotations in clustermap.
- **`col_colors`** (Optional[Union[pd.Series, pd.DataFrame]]): Colors to use for column annotations in clustermap.
- **`row_cluster`** (bool): Whether to cluster rows in clustermap. Default is True.
- **`col_cluster`** (bool): Whether to cluster columns in clustermap. Default is True.
- **`dendrogram_ratio`** (float): Ratio of dendrogram size to main plot in clustermap. Default is 0.2.
- **`colors_ratio`** (float): Ratio of color annotation size to main plot in clustermap. Default is 0.03.

### Enhanced Features
- **Error Handling**: Automatic fallback to regular heatmaps when clustering fails
- **Figsize**: Now uses tuple format (width, height) for consistent sizing

## Examples

### 1. Regular heatmaps with PCF clustering
```python
create_spoox_heatmaps(
    data_input=spoox_data,
    cluster_by='gr20',  # Cluster using PCF gr20 values
    save_folder='heatmaps_pcf_clustered'
)
```

### 2. Basic clustermaps
```python
create_spoox_heatmaps(
    data_input=spoox_data,
    use_clustermap=True,  # Use clustermaps instead of regular heatmaps
    save_folder='clustermaps_basic'
)
```

### 3. Customized clustermaps with annotations
```python
# Create color annotations using dictionary (recommended approach)
row_colors = {
    'T_cells': 'red',
    'B_cells': 'blue',
    'Macrophages': 'green',
    'Neutrophils': 'orange',
    'NK_cells': 'purple'
}

create_spoox_heatmaps(
    data_input=spoox_data,
    use_clustermap=True,
    row_colors=row_colors,
    col_colors=row_colors,  # Same mapping for columns
    row_cluster=True,
    col_cluster=True,
    dendrogram_ratio=0.3,  # Larger dendrograms
    colors_ratio=0.05,     # Larger color bars
    save_folder='clustermaps_annotated'
)
```

### 4. Clustermaps with no row clustering
```python
create_spoox_heatmaps(
    data_input=spoox_data,
    use_clustermap=True,
    row_cluster=False,  # Don't cluster rows
    col_cluster=True,   # But do cluster columns
    save_folder='clustermaps_col_only'
)
```

### 5. No clustering with clustermap layout
```python
create_spoox_heatmaps(
    data_input=spoox_data,
    use_clustermap=True,
    cluster_by=None,     # No pre-clustering
    row_cluster=False,   # No row clustering
    col_cluster=False,   # No column clustering
    save_folder='clustermaps_no_clustering'
)
```

### 6. Custom figure size
```python
create_spoox_heatmaps(
    data_input=spoox_data,
    use_clustermap=True,
    figsize=(12, 8),  # Width=12, Height=8
    save_folder='clustermaps_custom_size'
)
```

### 7. Robust clustering with error handling
```python
# If clustering fails due to data issues, the function will automatically
# fall back to regular heatmaps and show a warning message
create_spoox_heatmaps(
    data_input=problematic_data,  # Data that might cause clustering issues
    use_clustermap=True,
    cluster_by='gr20',
    save_folder='clustermaps_robust'
)
# Output: Warning messages if clustering fails, fallback heatmaps created
```

### 8. Backward compatibility
```python
# This still works but will show a deprecation warning
create_spoox_heatmaps(
    data_input=spoox_data,
    cluster_mh=True,  # Equivalent to cluster_by='Morueta-Holme_All'
    save_folder='heatmaps_old_style'
)
```

## Available Clustering Columns

You can cluster by any of these SpOOx output columns:
- `'gr10 PCF lower'`
- `'gr20'`
- `'gr20 PCF lower'`
- `'gr20 PCF upper'`
- `'gr20 PCF combined'`
- `'Morueta-Holme_Significant'`
- `'Morueta-Holme_All'`
- `'contacts'`
- `'%contacts'`
- `'Network'`
- `'Network(%)'`

## Key Differences: Regular Heatmaps vs Clustermaps

### Regular Heatmaps (`use_clustermap=False`)
- Multiple states plotted side-by-side in a single figure
- Consistent ordering across all states when clustering is enabled
- Compact visualization for comparing conditions
- File naming: `heatmap_{column_name}.png`

### Clustermaps (`use_clustermap=True`)
- Separate clustermap for each state
- Individual dendrograms and clustering for each state
- More detailed clustering information
- Support for color annotations
- File naming: `clustermap_{column_name}_{state}.png`

## Color Annotation Formats

The `row_colors` and `col_colors` parameters accept three different formats:

### 1. Dictionary (Recommended)
Maps cell type names directly to colors:
```python
colors = {
    'T_cells': 'red',
    'B_cells': 'blue',
    'Macrophages': 'green'
}
```

### 2. pandas Series
Index represents cell types, values are colors:
```python
colors = pd.Series(['red', 'blue', 'green'], 
                  index=['T_cells', 'B_cells', 'Macrophages'])
```

### 3. pandas DataFrame (for multiple annotation tracks)
Each column represents a different annotation type:
```python
colors = pd.DataFrame({
    'Cell_Type': pd.Series(['red', 'blue', 'green'], index=cell_types),
    'Function': pd.Series(['dark', 'light', 'medium'], index=cell_types)
})
```

## Color Annotation Examples

### Simple categorical colors (Dictionary approach - Recommended)
```python
# Create simple color mapping using dictionaries
row_colors = {
    'T_cells': 'red',
    'B_cells': 'blue', 
    'Macrophages': 'green',
    'Neutrophils': 'orange'
}

create_spoox_heatmaps(
    data_input=spoox_data,
    use_clustermap=True,
    row_colors=row_colors,
    col_colors=row_colors,  # Same colors for rows and columns
    save_folder='clustermaps_colored'
)
```

### Simple categorical colors (pandas Series approach)
```python
# Alternative: Create color mapping using pandas Series
import pandas as pd

cell_types = ['T_cells', 'B_cells', 'Macrophages', 'Neutrophils']
colors = ['red', 'blue', 'green', 'orange']
row_colors = pd.Series(colors, index=cell_types)

create_spoox_heatmaps(
    data_input=spoox_data,
    use_clustermap=True,
    row_colors=row_colors,
    col_colors=row_colors,
    save_folder='clustermaps_colored_series'
)
```

### Multiple annotation tracks (DataFrame approach)
```python
# Create multiple color tracks using DataFrame
import pandas as pd

cell_types = ['T_cells', 'B_cells', 'Macrophages', 'Neutrophils']
multi_colors = pd.DataFrame({
    'Tissue': pd.Series(['lightblue', 'lightgreen', 'lightcoral', 'lightyellow'], index=cell_types),
    'Function': pd.Series(['darkred', 'darkblue', 'darkgreen', 'darkorange'], index=cell_types)
})

create_spoox_heatmaps(
    data_input=spoox_data,
    use_clustermap=True,
    row_colors=multi_colors,
    col_colors=multi_colors,
    colors_ratio=0.08,  # Increase to accommodate multiple tracks
    save_folder='clustermaps_multi_annotated'
)
```

## Benefits

### Clustering Flexibility
- **Biological relevance**: Choose the most appropriate metric for your specific analysis
- **Consistency**: All visualizations use the same ordering when clustering is enabled
- **Comparative analysis**: Easy comparison between different clustering strategies

### Clustermap Advantages
- **Detailed clustering**: Individual dendrograms show hierarchical relationships
- **Annotation support**: Add color tracks to highlight cell type categories
- **Publication ready**: High-quality clustermaps suitable for publications
- **Customizable layout**: Control dendrogram and annotation sizes

### Error Handling and Robustness
- **Automatic fallback**: If clustering fails, regular heatmaps are created automatically
- **Informative warnings**: Clear error messages explain what went wrong
- **Graceful degradation**: Analysis continues even when some visualizations can't be clustered
- **File naming**: Fallback files are clearly marked (e.g., `heatmap_fallback_gr20_Tumor.png`)

### Backward Compatibility
- Existing code continues to work without modification
- Deprecation warnings guide users to new parameter names
- Smooth transition path for legacy analyses
- Figsize parameter now accepts tuples but maintains compatibility