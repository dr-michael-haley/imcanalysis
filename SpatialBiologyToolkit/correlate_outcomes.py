import os

import pandas as pd
import numpy as np
import anndata as ad
import seaborn as sb
import statsmodels as sm
import matplotlib.pyplot as plt
from lifelines import CoxPHFitter
from sklearn.decomposition import PCA
from matplotlib import colormaps
from matplotlib.colors import rgb2hex

# Set up output figure settings
plt.rcParams['figure.figsize'] = (3, 3)  # rescale figures, increase size here

def adata_population_survival(
    adata_input: ad.AnnData,
    timeline: pd.DataFrame = None,
    case_obs: str = 'Case',
    survival_obs: str = 'Survival_diagnosis',
    cell_obs: str = 'spatial_cluster',
    crosstab_normalize: str = 'index',
    quartiles: int = 2,
    top_quarts_only: bool = False,
    kmf: bool = True,
    linear: bool = True,
    cox: bool = True,
    xlim: tuple = None,
    prefix: str = '',
    scale_cell_fraction: float = None,
    drop_columns: list = [],
    save_directory: str = 'Survival_Figures'
) -> tuple:
    """
    Analyze and visualize the relationship between cell populations and survival data.

    Parameters
    ----------
    adata_input : ad.AnnData
        Input annotated data matrix.
    timeline : pd.DataFrame, optional
        Precomputed timeline DataFrame. If not provided, it will be computed.
    case_obs : str
        Column in adata_input.obs representing case IDs.
    survival_obs : str
        Column in adata_input.obs representing survival time.
    cell_obs : str
        Column in adata_input.obs representing cell clusters.
    crosstab_normalize : str
        Normalization method for crosstabulation.
    quartiles : int
        Number of quartiles to use.
    top_quarts_only : bool
        Whether to only consider the top and bottom quartiles.
    kmf : bool
        Whether to plot Kaplan-Meier survival curves.
    linear : bool
        Whether to perform linear regression analysis.
    cox : bool
        Whether to perform Cox proportional hazards regression.
    xlim : tuple, optional
        Limits for the x-axis in plots.
    prefix : str
        Prefix for saved file names.
    scale_cell_fraction : float, optional
        Factor to scale cell fractions.
    drop_columns : list
        List of columns to drop from the timeline DataFrame.
    save_directory : str
        Directory to save output figures.

    Returns
    -------
    tuple
        Tuple containing timeline DataFrame, cox p-values DataFrame, ols p-values DataFrame, and cox summary DataFrame.
    """
    # Create save directory
    os.makedirs(save_directory, exist_ok=True)
    
    if not isinstance(timeline, pd.DataFrame):
        timeline = pd.crosstab(
            index=[adata_input.obs[case_obs], adata_input.obs[survival_obs]],
            columns=adata_input.obs[cell_obs],
            normalize=crosstab_normalize
        )

    timeline.columns = timeline.columns.astype('str')
    timeline.columns = np.where(timeline.columns.str.isnumeric(), 'Cluster_' + timeline.columns, timeline.columns)
    
    for char in ['-', ' ', '(', ')', '+']:
        timeline.columns = timeline.columns.str.replace(char, '')

    timeline = timeline.drop(columns=drop_columns)

    if scale_cell_fraction:
        timeline *= scale_cell_fraction

    clusters = timeline.columns.tolist()
    print(clusters)
    
    for c in clusters:
        qs = pd.qcut(timeline[c], q=quartiles, labels=False, duplicates='drop')
        timeline[f'{c}_quartile'] = qs + 1

    timeline = timeline.reset_index().rename(columns={survival_obs: 'Days'})
    timeline['Censor'] = True

    if kmf:
        for cluster in [x + '_quartile' for x in clusters]:
            T = timeline['Days']
            E = timeline['Censor']
            groups = timeline[cluster]
            try:
                quarts = np.sort(timeline[cluster].unique()).tolist()
                if top_quarts_only:
                    quarts = [quarts[0]] + [quarts[-1]]
                kmfs = [ep.tl.kmf(T[groups == x], E[groups == x], label='Quartile' + str(x)) for x in quarts]
                ax = ep.pl.kmf(
                    kmfs,
                    color=["k", "r", 'b', 'g', 'y'],
                    xlabel="Days",
                    ylabel=f"{cluster}",
                    xlim=xlim,
                    show=False
                )
                fig = ax.get_figure()
                fig.savefig(f'{save_directory}/kmf_{prefix}_{cluster}.png', bbox_inches='tight', dpi=300)
                fig.savefig(f'{save_directory}/kmf_{prefix}_{cluster}.svg', bbox_inches='tight', dpi=300)
            except BaseException as e:
                print(f'KMF Failed for {cluster}: ' + str(e))

    if linear:
        ols_pval_list = []
        timeline_endpoint = timeline[timeline.Censor == True]
        for cluster in clusters:
            try:
                adata = ad.AnnData(timeline_endpoint[['Days', cluster]])
                formula = f'{cluster} ~ Days'
                ols = ep.tl.ols(adata, var_names=['Days', cluster], formula=formula, missing="drop")
                lm_result = ols.fit()
                lm_result.summary()
                ols_pval_list.append(float(lm_result.pvalues[1]))
                ax = ep.pl.ols(
                    adata,
                    x="Days",
                    y=cluster,
                    ols_results=[lm_result],
                    ols_color=["red"],
                    xlabel="Days survival",
                    ylabel=cluster,
                    size=30,
                    show=False
                )
                fig = ax.get_figure()
                fig.savefig(f'{save_directory}/ols_{prefix}_{cluster}.png', bbox_inches='tight', dpi=300)
                fig.savefig(f'{save_directory}/ols_{prefix}_{cluster}.svg', bbox_inches='tight', dpi=300)
            except BaseException as e:
                print('OLS Failed: ' + str(e))
                ols_pval_list.append(f'Error: {str(e)}')
        ols_pval_list = pd.DataFrame(zip(clusters, ols_pval_list), columns=['Cluster', 'OLS_pval']).style.set_caption("OLS_pval")

    if cox:
        cox_pval_list = []
        cox_summary = []
        for cluster in clusters:
            cox_data = timeline[['Days', 'Censor', cluster]]
            coxph = CoxPHFitter()
            try:
                coxph.fit(cox_data, duration_col="Days", event_col="Censor")
                cox_pval_list.append(coxph.summary['p'][0].copy())
                cox_summary.append(coxph.summary.copy())
            except BaseException as e:
                print('COX Failed: ' + str(e))
                cox_pval_list.append(f'Error: {str(e)}')
        cox_pval_list = pd.DataFrame(zip(clusters, cox_pval_list), columns=['Cluster', 'COX_pval'])
        cox_summary = pd.concat(cox_summary)
        cox_summary['P (corr)'] = sm.stats.multitest.multipletests(cox_summary['p'], alpha=0.05, method='holm-sidak')[1]
        cox_summary.rename(columns={'exp(coef)': 'Hazard Ratio for % present of cluster'}, inplace=True)

    return timeline, cox_pval_list, ols_pval_list, cox_summary

def create_pca_timeline(
    adata_input: ad.AnnData,
    n_components: int = 2,
    case_obs: str = 'Case',
    survival_obs: str = 'Survival_diagnosis',
    cell_obs: str = 'spatial_cluster',
    crosstab_normalize: str = 'index',
    show_loadings: bool = True
) -> pd.DataFrame:
    """
    Create a PCA timeline from cell population data and survival information.

    Parameters
    ----------
    adata_input : ad.AnnData
        Input annotated data matrix.
    n_components : int
        Number of PCA components.
    case_obs : str
        Column in adata_input.obs representing case IDs.
    survival_obs : str
        Column in adata_input.obs representing survival time.
    cell_obs : str
        Column in adata_input.obs representing cell clusters.
    crosstab_normalize : str
        Normalization method for crosstabulation.
    show_loadings : bool
        Whether to show PCA loadings.

    Returns
    -------
    pd.DataFrame
        Timeline DataFrame with PCA components and survival data.
    """
    df = pd.crosstab(index=[adata_input.obs[case_obs], adata_input.obs[survival_obs]], columns=adata_input.obs[cell_obs], normalize=crosstab_normalize)

    # Perform PCA on dataframe
    pca = PCA(n_components=n_components)
    pca.fit(df)
    pca_result = pca.transform(df)
    variance_ratio = pca.explained_variance_ratio_

    # Plot PCA embedding
    plt.scatter(pca_result[:, 0], pca_result[:, 1])
    plt.xlabel('PCA component 1')
    plt.ylabel('PCA component 2')
    plt.show()

    # Plot percentage of variance explained by each component
    plt.bar(range(len(variance_ratio)), variance_ratio)
    plt.xlabel('PCA component')
    plt.ylabel('Percentage of variance explained')
    plt.show()

    timeline = pd.DataFrame(pca_result, index=df.reset_index()['Case'], columns=[f'PC{x+1}' for x in range(n_components)])
    timeline['Days'] = timeline.index.map(adata_input.obs[~adata_input.obs.duplicated(case_obs)].set_index(case_obs)[survival_obs].to_dict())
    timeline_file = timeline.reset_index().set_index(['Case', 'Days'])

    if show_loadings:
        cs = colormaps['Dark2'].colors
        cs = [rgb2hex(x) for x in cs][0:len(df.columns)]
        loadings = pca.components_.T * (pca.explained_variance_ ** 0.5)
        fig, ax = plt.subplots(figsize=(3, 3))
        for i, feature in enumerate(zip(df.columns, cs)):
            arrow_color = feature[1]
            feature = feature[0]
            ax.arrow(0, 0, loadings[i, 0] * 2, loadings[i, 1] * 2, head_width=0.02, linewidth=2, head_length=0.06, color=arrow_color)
            ax.text(loadings[i, 0] * 1.2, loadings[i, 1] * 1.2, feature, color='black', ha='center', va='center')
        ax.set_xlim([-.2, .2])
        ax.set_ylim([-.2, .2])
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        ax.set_title('PCA Loadings')
        plt.show()
        fig.savefig('Figures/survival_pca_loadings.svg', bbox_inches='tight')

    return timeline_file