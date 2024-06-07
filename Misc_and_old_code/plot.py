import anndata as ad
import pandas as pd

import scanpy as sc
import anndata as ad
import scipy as sp
from scipy import stats


import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests

from pathlib import Path
import os

import warnings

# Matplotlib and seaborn for plotting
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, Normalize
import seaborn as sb

from utils import _cleanstring, _save, _check_input_type, _to_list, subset, adlog, print_full, compare_lists

import sklearn.preprocessing
import umap
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


def count_summary(data,
                 pop_col=None,
                 levels=['ROI'],
                 mean_over=['ROI'],
                 crosstab_normalize=False,
                 mode='population_counts'
                 ):
    '''
    This function takes an input, then converts it into long and wide formats ready for plotting
    
    data - Can be an AnnData, a Pandas dataframe, or a path to a .csv. which will be loaded as a dataframe
    pop_col (optional) - Column that identifies a column
    levels - The levels at which the data is structured, usually just ['ROI'] or ['Case', 'ROI']
    mean_over - At what level you want to calculate a mean, for example if you wanted to calculate case averages
    crosstab_normalize - Whether, and how, to normalize crosstab results
    mode - 'population_counts' crosstabulates for counts, 'numeric' summarises numeric data
    
    returns:
    wide (wide form data), long (long form data)
    
    '''
                 
    import anndata as ad
    from pandas.api.types import is_numeric_dtype
    
    # Check the data input is in correct format
    data = _check_input_type(data)
    
    # Make sure data are lists
    levels = _to_list(levels)
    mean_over = _to_list(mean_over)
    
    # Ideally data will be returned in long and wide formats if appropriate
    wide=None
    long=None
    
    if mean_over != []:
        assert all([x in levels for x in mean_over]), 'Observation to mean over should also be in levels list'    
    
    if mode=='numeric':
        
        if pop_col:
            levels.append(pop_col)
            mean_over.append(pop_col)
            
        long = data.groupby(levels, observed=True).mean(numeric_only=True)
    
        if mean_over != []:
            long = long.groupby(mean_over, observed=True).mean(numeric_only=True)
            levels = mean_over
                        
        if pop_col:
            
            levels.remove(pop_col)
            
            wide = pd.pivot(data=long.reset_index(), 
                     index=levels,
                     columns=pop_col)
        
    elif mode=='population_counts':
        
        assert not(is_numeric_dtype(data[pop_col])), 'Column is numeric, cannot calculation population-level counts'
        
        # Crosstabulate the counts data for the given population
        crosstab_df = pd.crosstab([data[x] for x in levels], columns=data[pop_col], normalize=crosstab_normalize)

        wide = crosstab_df

        long = crosstab_df.reset_index().melt(id_vars=levels)#.set_index(levels)

        if mean_over != []:
            wide = wide.groupby(mean_over).mean(numeric_only=True)

        long = long.groupby((mean_over+[pop_col]), observed=True).mean(numeric_only=True).reset_index()

        #long = long.groupby(([x for x in levels if x not in mean_over]+[value_col])).mean()


        # For long form, first step is getting data in its most granular
        #data = data.groupby(['Case','ROI','group_2','sc3s_10'], observed=True).count().reset_index()
           
    return wide, long

                 
def bargraph(data,
             pop_col=None,
             value_col=None,
             hue=None,
             hue_order=None,
             specify_populations=[],
             levels=None,
             mean_over=None,             
             confidence_interval=68,
             figsize=(5,5),
             crosstab_normalize=False,
             cells_per_mm=False,
             palette=None,
             hide_grid=False,
             legend=True,
             save_data=True,
             save_figure=False,
             return_data=True,
             rotate_x_labels=90,
             case_col_name='Case',
             ROI_col_name='ROI',
            ):
    
    '''
    This plots bargraphs, either for population abundances, or for measured values. Will figure out what plot you want based upon whether you supply a population column (pop_col_) and/or a value column (value_col)
    
    data - Can be an AnnData (in which case the adata.obs will be retrieved), a Pandas dataframe, or a path to a .csv. which will be loaded as a dataframe
    specify_populations -  List of specific populations from pop_col to plot, by default will just plot everything
    pop_col - Column that identifies a population (categorical) column
    value_col - Column that identifies the result of a measurement 
    hue - Column to subgroup the graph
    hue_order - List in which to order of hues
    levels - The levels at which the data is structured, by default will assume several ROIs within a Case
    mean_over -At what level you want to calculate a mean, by default will average over ROIs
    confidence_interval = 68 is standard error
    crosstab_normalize - Whether, and how, to normalize crosstab results
    cells_per_mm - Normalises values by mm2 taken from the sample dataframe stored in adata.uns['sample']
    palette - Will colour based on values stored in adata.uns if no colour map supplied
    rotate_x_labels - Angle of x labels
    return_data - Will return the dataframe used to create the figure, useful for doing statistics on!
    
    returns:
    fig object
    Saves figures and raw data
    
    '''
    
    from pandas.api.types import is_numeric_dtype

    # Check the data input is in correct format
    data = _check_input_type(data)
        
    # If user doesn't specify levels or what to mean over, then figure it out for them
    if not levels and not mean_over:
       
        # Check ROI column found in data
        assert ROI_col_name in data.columns.tolist(), f'{ROI_col_name} column not found in data'
        
        if case_col_name in data.columns.tolist():
            levels = ['Case','ROI']
            mean_over = ['Case','ROI']
        else:            
            print(f'Case column ({case_col}) not found in data, just using ROI column ({ROI_col_name})')
            levels = ['ROI']
            mean_over = ['ROI']    
    
    # Make sure data are lists
    levels = _to_list(levels)
    mean_over = _to_list(mean_over)
    specify_populations = _to_list(specify_populations)
    
    # Plot specific populations
    if specify_populations != []:
        data = data[data[pop_col].isin(specify_populations)]
    
    # If using a hue, must also be used in levels and in mean_over
    if hue != None and hue not in levels:
        levels.append(hue)
    
    if hue != None and hue not in mean_over:
        mean_over.append(hue)
        
    # Retrieve default colour map from AnnData if not specified
    if not palette:
        if hue != None:
            try:
                palette = adata.uns[f'{hue}_colormap']
            except:
                print(f'No colour map found for {hue}')
        else:
            try:
                palette = adata.uns[f'{pop_col}_colormap']
            except:
                print(f'No colour map found for {pop_col}')            
        
    
    if pop_col and value_col:
        print('Numeric broken down by pop')
    
        #Summarise at the appropriate level
        _, long_form_data = count_summary(data=data,
                             pop_col=pop_col,
                             levels=levels,
                             mean_over=mean_over,
                             mode='numeric',
                             crosstab_normalize=crosstab_normalize)
        
        plot_data = long_form_data.reset_index()
        y_plot = value_col
        x_plot = pop_col
        
    elif pop_col==None and value_col:
        print('Numeric only')
        
        #Summarise at the appropriate level
        _, long_form_data = count_summary(data=data,
                             pop_col=pop_col,
                             levels=levels,
                             mean_over=mean_over,
                             mode='numeric',
                             crosstab_normalize=crosstab_normalize)
                
        plot_data = long_form_data.reset_index()
        y_plot = value_col
        x_plot = levels[-1]
                       
    
    # Population-level data
    elif pop_col and value_col==None:
        print('Population counts')
        
        #Summarise at the appropriate level
        _, long_form_data = count_summary(data=data,
                             pop_col=pop_col,
                             levels=levels,
                             mean_over=mean_over,
                             crosstab_normalize=crosstab_normalize)
        
        long_form_data.rename(columns={'value':'Cells'}, inplace=True)        
                
        plot_data = long_form_data 
        y_plot='Cells' 
        x_plot=pop_col 

    if cells_per_mm:
        
        try:
            size_dict = adata.uns['sample']['mm2'].to_dict()
            
            plot_data['mm2'] = plot_data['ROI'].map(size_dict)
            
            new_y = str(y_plot + " per mm2")

            plot_data[new_y] = plot_data[y_plot] / plot_data['mm2']
            
            y_plot = new_y
            
        except:
            print('Could not convert to mm2')
    
    
    # Display plotting data for QC check    
    #utils.print_full(plot_data)
    display(plot_data)
    
    # Plotting
    fig, ax = plt.subplots(figsize=figsize)

    sb.barplot(data = plot_data, 
               y=y_plot, 
               x=x_plot, 
               hue=hue,
               hue_order=hue_order,
               palette=palette,
               ax=ax)
    
    # Rotate x labels
    ax.set_xticklabels(ax.get_xticklabels(),rotation = rotate_x_labels)

    if hide_grid:
        ax.grid(False)
        
    if legend:
        ax.legend(bbox_to_anchor=(1.01, 1))
    else:
        try:
            ax.get_legend().remove()
        except:
            pass
        
    filename = f'Bargraph_{_cleanstring(pop_col)}_{_cleanstring(levels)}_{_cleanstring(mean_over)}'
    
    if save_data:
        
        file_path = _save(('Figures','Barcharts','Raw'), (filename + '.csv'))
        print(f'Saving raw data: {file_path}')
        long_form_data.to_csv(file_path)
    
    if save_figure:
        
        for ext in ['.png','.svg']:
            file_path = _save(('Figures','Barcharts'), (filename + ext))
            print(f'Saving figure: {file_path}')
            fig.savefig(file_path, bbox_inches='tight', dpi=300)

    if return_data:
        return plot_data
    else:
        return fig
        
        
def mlm_stats(data, pop_col, group_col, case_col='Case', value_col='Cells', roi_col='ROI', method='holm-sidak', average_cases=False, show_t_values=False, run_t_tests=True):
    """
    Conducts a mixed linear model analysis on the given pandas DataFrame for each unique entry in the 'pop_col' column, and optionally performs a t-test on the means of the groups.

    Parameters:
    - data: a pandas DataFrame containing the data
    - pop_col: the name of the column in 'data' that identifies different populations
    - group_col: the name of the column in 'data' that identifies 2 groups
    - case_col: the name of the column in 'data' that identifies cases (default 'Case')
    - value_col: the name of the column in 'data' that contains the values to be analyzed (default 'Cells')
    - roi_col: the name of the column in 'data' that identifies ROIs (default 'ROI')
    - method: method used to correct for multiple comparisons (default 'holm-sidak')
    - average_cases: whether to average the results over 'case_col' before performing the t-test (default False)
    - show_t_values: whether to include the t-values in the output (default False)
    - run_t_tests: whether to run t-tests (default True)

    Returns: a pandas DataFrame containing the p-values of the mixed linear model analysis and, if 'run_t_tests' is True, t-test for each population, both raw and corrected for multiple comparisons. If 'show_t_values' is True, the t-values from the t-test will also be included.
    """
    assert set([pop_col, group_col, case_col, value_col, roi_col]).issubset(data.columns), "Some required columns are missing from the input DataFrame."
    assert data[group_col].nunique() == 2, "'group_col' must identify exactly two groups."

    # Clean population names for stats
    data[pop_col] = [_cleanstring(x) for x in data[pop_col]]

    pop_list = data[pop_col].unique().tolist()

    results = []

    for i in pop_list:
        subset = data[data[pop_col]==i]
        
        formula = f"{value_col} ~ {group_col}"
        
        # Check if there are multiple rows with the same case and ROI
        roi_counts = subset.groupby(case_col)[roi_col].nunique()
        case_roi_counts = subset.groupby([case_col, roi_col], observed=True).size()
        
        # Capture any warnings for running the mixed linear model
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            if (case_roi_counts  > 1).any():
                print('Multiple values per ROI detected, running ROI nested in Case...')
                vc_formula = {roi_col: f'0 + C({roi_col})'}
                re_formula = '1'

                md = smf.mixedlm(formula=formula, 
                                  data=subset,
                                  groups=subset[case_col],
                                  vc_formula=vc_formula,
                                  re_formula=re_formula)
            else:
                md = smf.mixedlm(formula=formula, 
                                 data=subset,
                                 groups=subset[case_col])

            mdf = md.fit()
        
        warning_messages = [str(warn.message) for warn in w]
        
        result = {pop_col: i, 'mlm_p_value': mdf.pvalues[1], 'mlm_warnings': str(warning_messages)}

        if run_t_tests:
            # Average the results over 'case_col' if specified before the t-test
            if average_cases:
                subset = subset.groupby([group_col, case_col], observed=True).mean(numeric_only=True).reset_index()

            # Get data for the groups
            group1 = subset[subset[group_col] == subset[group_col].unique()[0]][value_col]
            group2 = subset[subset[group_col] == subset[group_col].unique()[1]][value_col]
            
            # T-test
            t_stat, t_pval = stats.ttest_ind(group1, group2)
            
            # Mann-Whitney
            u_stat, mwu_pval = stats.mannwhitneyu(group1, group2)

            result['t_test_p_value'] = t_pval
            result['mannwhitneyu_p_value'] = mwu_pval
            
            if show_t_values:
                result['t_value'] = t_stat
                result['u_value'] = u_stat

        results.append(result)

    results_df = pd.DataFrame(results)

    # Correct for multiple comparisons
    reject_mlm, pvals_corrected_mlm, _, _ = multipletests(results_df['mlm_p_value'], method=method)
    results_df['mlm_p_value_corrected'] = pvals_corrected_mlm

    if 't_test_p_value' in results_df.columns:
        _, pvals_corrected_ttest, _, _ = multipletests(results_df['t_test_p_value'], method=method)
        results_df['t_test_p_value_corrected'] = pvals_corrected_ttest
        
        _, pvals_corrected_mwu, _, _ = multipletests(results_df['mannwhitneyu_p_value'], method=method)
        results_df['mannwhitneyu_p_value_corrected'] = pvals_corrected_mwu
    
    # MLM is not appropriate if averaged over cases
    if average_cases:
        results_df = results_df.drop(columns=['mlm_p_value','mlm_warnings','mlm_p_value_corrected'])
    
    
    return results_df
        


def cellabundance_UMAP(adata, ROI_id, population, colour_by=None, annotate=True, save=None, normalize=False, dim_red='UMAP', ax=None, cmap='tab20', figsize=(3, 3), point_size=50):
    """
    Visualizes the abundance of cell populations within regions of interest (ROI) using UMAP, PCA, or tSNE with Seaborn's scatterplot.

    Parameters:
    - adata: AnnData object containing the single-cell data.
    - ROI_id: String, the name of the column in adata.obs that contains the ROI identifiers.
    - population: String, the name of the column in adata.obs that contains cell population identifiers.
    - colour_by: String (optional), the name of the column to color the points by. If None, points are colored by ROI_id.
    - annotate: Boolean, whether to annotate the points with ROI identifiers.
    - save: String or None, file path to save the plot. If None, the plot is not saved.
    - normalize: Boolean, whether to normalize the cell counts across ROIs when using Pandas Crosstabulate function.
    - dim_red: String, the dimensionality reduction technique to use ('UMAP', 'PCA', 'tSNE').
    - ax: matplotlib.axes.Axes object (optional), the axes on which to draw the plot. If None, a new figure and axes object are created.
    - cmap: String, the name of the Matplotlib colormap to use. Defaults to 'tab20'.
    - figsize: Tuple of integers, the size of the figure to create. Ignored if 'ax' is not None.
    - point_size: Integer, the size of the points in the scatter plot.

    Returns:
    The matplotlib figure or axes object containing the plot.
    """
    
    # Create cells table based on whether colour_by is specified
    if colour_by:
        cells = pd.crosstab([adata.obs[ROI_id], adata.obs[colour_by]], adata.obs[population], normalize=normalize).reset_index().copy()
    else:
        cells = pd.crosstab(adata.obs[ROI_id], adata.obs[population], normalize=normalize).reset_index().copy()
    
    # Prepare data for dimensionality reduction
    summary_data = cells.iloc[:, 2:] if colour_by else cells.iloc[:, 1:]
    scaled_summary_data = sklearn.preprocessing.StandardScaler().fit_transform(summary_data)

    # Select and apply dimensionality reduction method
    if dim_red == 'UMAP':
        reducer = umap.UMAP()
        embedding = reducer.fit_transform(scaled_summary_data)
    elif dim_red == 'PCA':
        pca = PCA(n_components=2)
        embedding = pca.fit_transform(scaled_summary_data)
    elif dim_red == 'tSNE':
        tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
        embedding = tsne.fit_transform(scaled_summary_data)
    
    # Prepare the DataFrame for Seaborn
    plot_data = pd.DataFrame(embedding, columns=[f'{dim_red} Dimension 1', f'{dim_red} Dimension 2'])
    plot_data['color'] = cells[colour_by] if colour_by else cells[ROI_id]

    # Determine if we need to create a new figure
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
        created_new_fig = True
    else:
        created_new_fig = False
    
    # Plot using Seaborn
    sb.scatterplot(data=plot_data, x=f'{dim_red} Dimension 1', y=f'{dim_red} Dimension 2', hue='color', ax=ax, palette=cmap, s=point_size, legend="full")

    # Adjust legend to not overlap with data
    ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

    # Annotation (optional)
    if annotate:
        for _, row in plot_data.iterrows():
            ax.text(row[f'{dim_red} Dimension 1'], row[f'{dim_red} Dimension 2'], row['color'], ha='right', size='small')
    
    # Save plot if requested
    if save:
        fig.savefig(save, bbox_inches='tight')
    
    # Show plot if a new figure was created
    if created_new_fig:
        plt.show()
        return fig
    else:
        return ax

