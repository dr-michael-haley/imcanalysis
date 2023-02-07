import pandas as pd
import scanpy as sc
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import colorcet as cc
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# from shapely.ops import polygonize,unary_union
from shapely.geometry import MultiPoint, Point, Polygon
from scipy.spatial import Voronoi
import networkx as nx
import math
import statsmodels.api as sm
from copy import copy


from sklearn.utils._testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning
    
def celltable_to_adata(column_properties,cell_table,dictionary,misc_table=False,dict_index='ROI',quiet=False,marker_normalisation=None,xy_as_obs=True):
    
    """This function will load any cell_table irrespective of which pipeline used, as long as the 
    Args:
        column_properties (.csv):
            Annotated list of all the columns in the file. Valid labels and how the are handled:
                marker - Will be added as a maker
                roi - Unique image identifier, e.g, name of the region of interest or image
                x - the location of the cell in the x axis
                y - the location of the cell i the y axis
                observation - the column will be added as observation, eg a categorial variable such as patient or treatment
        cell_table:
            The master cell_table .csv file as produced by whatever pipeline you are using
        dictionary:
            This will be used to add extra columns for each ROI based upon other characteristics of your ROI. This will allow you to easily group analyses later, for example by specifying that specific ROIs came from the same patient, or are grouped in some other way.
        misc_table:
            If this is set as True, then a separate DataFrame with all the misc columns will also be returned
        dict_index:
            The name of the column used as the index in the dictionary file, defaults to 'ROI'
    Returns:
        AnnData:
            Completely annotated dataset
        
        If misc_table=True, then will also return a separate DataFrame with all the misc columns
    """
    import pandas as pd
    import scanpy as sc
    global markers
    global markers_normalised
    
    #This stops a warning getting returned that we don't need to worry about
    pd.set_option('mode.chained_assignment',None)
    
    #Load in the data to Pandas dataframe
    master = pd.read_csv(cell_table, low_memory=False)
    columns_table = pd.read_csv(column_properties, low_memory=False)

    # Reset inputs
    markers = []
    ROIs = []
    misc = []
    observation = []
    X_loc = ""
    Y_loc = ""

    # This will run through the columns table and sort out which columns from the raw table go where in the resulting adata table
    for index, row in columns_table.iterrows():
        
        # Make all lower case
        row['Category'] = row['Category'].lower()
        
        if row['Category']=='marker':
            markers.append(row['Column'])
            if quiet==False:
                print(row['Column'] + ' --- added as MARKER')
                
        elif row['Category']=='roi':
            if ROIs == []:
                ROIs = row['Column']
                if quiet==False:
                    print(row['Column'] + ' --- added as ROI information')
            else:
                stop('ERROR: Multiple ROI columns')
                
        elif row['Category']=='x':
            if quiet==False:    
                print(row['Column'] + ' ---  added as X location')
            X_loc = row['Column']
            
        elif row['Category']=='y':
            if quiet==False:
                print(row['Column'] + ' ---  added as Y location')
            Y_loc = row['Column']
            
        elif row['Category']=='observation':
            if quiet==False:
                print(row['Column'] + ' --- added as OBSERVATION') 
            observation.append(row['Column'])
            
        elif row['Category']=='misc':
            if quiet==False:
                print(row['Column'] + ' --- added to MISC dataframe')
            misc.append(row['Column'])
            
        else:
            if quiet==False:
                print(row['Column'] + " ---  DISCARDED")

    #Error catching to make sure markers or ROIs were identified
    if markers==[]:
        stop("ERROR: No markers were identified")
    if ROIs==[]:
        stop("ERROR: No ROI identifiers were found")    
    
    # Create the anndata firstly with the marker information
    if marker_normalisation==None:
        adata = sc.AnnData(master[markers])
    elif marker_normalisation=='99th':
        raw_markers = master[markers]
        markers_normalised = raw_markers.div(raw_markers.quantile(q=.99)).clip(upper=1)        
        markers_normalised = np.nan_to_num(markers_normalised, nan=0)
        adata = sc.AnnData(markers_normalised)
        adata.var_names=markers
        print('\nData normalised to 99th percentile')

    # Add in a master index to uniquely identify each cell over the entire dataset    
    adata.obs['Master_Index']=master.index.copy()

    # Add ROI information
    adata.obs['ROI']= master[ROIs].values.tolist()    
 
    # Add in other observations if found
    if not observation==[]:
        for a in observation:
            adata.obs[a]=master[a].tolist()

    # Add in spatial data if it's provided
    if not X_loc=="" and not Y_loc=="":
        adata.obsm['spatial'] = master[[X_loc, Y_loc]].to_numpy()
        
        #Adds in the x and y a observations. These wont be used by scanpy, but can be useful for easily exporting later.
        if xy_as_obs==True:
            print('\nX and Y locations also stored as observations (and in .obsm[spatial])')
            adata.obs['X_loc']=master[X_loc].values.tolist()
            adata.obs['Y_loc']=master[Y_loc].values.tolist()       
        
    else:
        print("No or incomplete spatial information found")

    #If a dictionary was specified, then it will be used here to populate extra columns as observations in the anndata
    if not dictionary==None:
            #Read dictionary from file
            master_dict = pd.read_csv(dictionary, low_memory=False)

            # Setup dictionary
            m = master_dict.set_index(dict_index).to_dict()

            # Add the new columns based off the dictionary file
            for i in master_dict.columns:
                if not i==dict_index:
                    master[i]=master[ROIs].map(m[i])
                    adata.obs[i]=master[i].values.tolist()
    else:
        print("No dictionary found")

    # Add any misc data to spare dataframe
    if not misc==[]:
        misc_data = master[misc]
        misc_data['Master_Index']=misc_data.index.copy()
    
    if misc_table==True:
        return adata, misc_data
    else:
        return adata
    
def remove_list(full_list,to_remove):
    """This function will remove all items in one list from another list, returning a filtered list """
    
    filtered =  [m for m in full_list if m not in to_remove]
    return filtered


def return_adata_xy(adata):
    """This function will retrieve X and X co-ordinates from an AnnData object """
    
    import numpy as np
    X, Y = np.split(adata.obsm['spatial'],[-1],axis=1)
    return X, Y

   

def filter_anndata(adata_source,markers_to_remove=[],obs_to_filter_on=None,obs_to_remove=[]):
    """This function will filter the anndata
    Args:
        adata_source:
            The complete anndata object
        markers_to_remove:
            The list of markers (variables) to remove
        obs_to_filter_on:
            The name of the obs that we will filter on
        obs_to_remove:
            The list of observations to remove
    Returns:
        AnnData:
            Filtered
        
        If misc_table=True, then will also return a separate DataFrame with all the misc columns
    """

    #Change this with a list of the markers you'd like to be remove entirely from the dataset, e.g. DNA stains
    #markers_to_remove = ['DNA1', 'DNA2','PanCytokeratin','iNOS']

    #Change this to the name of the obs you'd like to use to identify samples to remove
    #obs_to_filter_on = 'Type'

    #Change to the items from the above obs you'd like to remove
    #obs_to_remove = ['Tonsil','BrainCtrl','Test']

    # Make a list of all markers found
    all_markers = adata_source.var_names.tolist()

    #Make a new list that only has the markers we're interested in
    markers_limited = [m for m in all_markers if m not in markers_to_remove]
    
    if obs_to_filter_on==None:
        return adata_source[:,markers_limited]
    else:
        return adata_source[~adata_source.obs[obs_to_filter_on].isin(obs_to_remove),markers_limited]

    
    
def population_description(adata,groupby_obs='pheno_leiden',distribution_obs=[]):
    """ This function gives a few readouts of a specific population, including a heatmap with a dendogram, the abundance of the populations in raw numbers, and their distribution relative to other observations
    Args:
        adata
            The adata source- will be copied and therefore isn't modified
            
        groupby_obs
            The adata.obs reference for the population
        
        distribution_obs
            List of adata.obs to cross tabulate with the population of interest
    Returns:
        Just graphs!
            
        """
    import pandas as pd
    import scanpy as sc
    
    adata_working = adata.copy()
    
    sc.tl.dendrogram(adata_working, groupby = groupby_obs)
    sc.pl.matrixplot(adata_working, adata_working.var_names, groupby=groupby_obs, dendrogram=True, title='Marker expression grouped by '+groupby_obs)
    
    adata_working.obs[groupby_obs].value_counts().plot.bar(title='Absolute number of cells in each '+groupby_obs+' population')

    if isinstance(distribution_obs, str):
        distribution_obs = [distribution_obs]
    
    for i in distribution_obs:
        tmp = pd.crosstab(adata_working.obs[groupby_obs],adata_working.obs[i], normalize='index')
        tmp.plot.bar(stacked=True,figsize=(16, 6), title='Proportion of each '+groupby_obs+' population in '+i).legend(bbox_to_anchor=(1.1, 1))
    
    
    
def squidpy_nhood_enrichment_hyperion(adata, cluster_identifier, ROI_column_name, ROIs_to_exclude=[],n_neighbours=10,run_initial=True,average_over_rois=True):

    """This function perform neighbourhood enrichment using Squidpy for each individual ROI, then combine them and then add them back into the original adata. This function will first perform the analysis on all the samples together, and so will overwrite any existing analyses from spatial_neighbors and nhood_enrichment. T
    Args:
        adata
            AnnData object with single cell information for all ROIs
        
        cluster_identifier
            adata.obs that specifies the clustering information/populations
        
        ROI_column_name
            adata.obs that specifies the ROI/image identifier
            
        ROIs_to_exclude
            List of ROI names to exclude
            
        n_neighs
            Number of nearest neighbours, default is 10
            
        average_over_rois
            True by default - will take of mean of counts of interactions over the ROIs. If False, will just sum.
            
        run_initial
            By default will run an initital analysis on the whole adata to setup the data structure to store the new results

    Returns:
        Adds the 'corrected' neighborhood information onto the adata
        
        If misc_table=True, then will also return a separate DataFrame with all the misc columns
    """

    import pandas as pd
    import squidpy as sq
    import scipy as sp
        
    if run_initial:
        sq.gr.spatial_neighbors(adata, n_neighs=n_neighbours, coord_type="generic")
        sq.gr.nhood_enrichment(adata, cluster_key=cluster_identifier)   
    
    
    #Load all the ROIs in the adata
    ROI_list = adata.obs[ROI_column_name].unique().tolist()
    
    #If specified, remove ROIs
       
    if ROIs_to_exclude!=[]:
        
        if isinstance(ROIs_to_exclude, str):
            ROIs_to_exclude = [ROIs_to_exclude]

        for x in ROIs_to_exclude:
            ROI_list.remove(x) 

  
    # Create empty lists which will be populated at end and made into a data frame
    pop_1 = []
    pop_2 = []
    roi = []
    value_result = []

    for i in ROI_list:
        
        print('Calculating for '+i)
        
        working_adata = adata[adata.obs[ROI_column_name]==i].copy()    

        sq.gr.spatial_neighbors(working_adata, n_neighs=n_neighbours, coord_type="generic")
        sq.gr.nhood_enrichment(working_adata, cluster_key=cluster_identifier)

        pops = working_adata.obs[cluster_identifier].cat.categories
        ne_counts = working_adata.uns[(cluster_identifier+'_nhood_enrichment')]['count']
        num_pops = range(len(ne_counts))

        #Loop across every row in the array
        for k,x in zip(ne_counts,num_pops):

                #Loop for every item in the list
                for b in num_pops:

                    value_result.append(k[b])
                    pop_1.append(pops[x])
                    pop_2.append(pops[b])
                    roi.append(i)

    #Make a dataframe of the lists        
    df = pd.DataFrame(list(zip(roi,pop_1,pop_2,value_result)),
                   columns =['ROI','Population 1', 'Population 2', 'Count'])

    if average_over_rois==True:
         #Average the counts over the different ROIs
        summary_df = df.groupby(['Population 1','Population 2']).mean()
    else:
         #Sum the counts over the different ROIs
        summary_df = df.groupby(['Population 1','Population 2']).sum()   
    
    #Calculate the z-scores of the counts
    summary_df['z-score']=sp.stats.zscore(summary_df['Count'])

    summary_df.reset_index(inplace=True)

    #Final dataframes that you can use to look at data
    final_array_zscore = summary_df.pivot(index='Population 1',columns='Population 2',values='z-score')
    final_array_count = summary_df.pivot(index='Population 1',columns='Population 2',values='Count')

    #Put back into original adata
    adata.uns[(cluster_identifier+'_nhood_enrichment')]['zscore']=final_array_zscore.to_numpy()
    adata.uns[(cluster_identifier+'_nhood_enrichment')]['count']=final_array_count.to_numpy()
    
def astir_adata(adata,astir_markers_yml,id_threshold=0.7,max_epochs = 1000,learning_rate = 2e-3,initial_epochs = 3,use_hierarchy=False,diagnostics=True,cell_type_label="astir_cell_type", hierarchy_label='astir_hierarchy'):
    """This function does an Astir analyis on an adata object, then adds the results back into the adata.obs
    Args:
        adata
            AnnData object with single cell information for all ROIs
        
        astir_markers_yml
            Path to the .yml file that details what markers we expect to be expressed where. See the astir documentation for how this should be formatted.
        
        id_threshold (see Astir documentation)
            The confidence threshold at which cells will be identified - lowering this will mean more cells are given identities, but the certainty of these predictions will be lower.
            
        max_epochs, learning_rate, initial_epochs (see Astir documentation)
            These are the default values used by Astir
            
        use_hierarchy
            Whether or not to use a hierarchy, which should be detailed inthe .yml file
        
        cell_type_label
            The label that will be added to adata.obs that defines the cell types
            
        hierarchy_label
            The label that will be added to adata.obs that defines the cell hierarchy (if being used)
                        
        diagnostics (see Astir documentation)
            Whether or not to return some diagnostic measures that Astir can perform which give feedback on the performance of the Astir process

    Returns:
        Adds the Astir-calculated cell types (and hierarchies) to the adata.obs

    """
    # Import the function   
    import os
    import astir as ast
    import numpy as np
    
    from astir.data import from_anndata_yaml    


    # Save the adata to a file
    adata.write(filename='adata_astir.temp')

    # Create the astir object
    ast = from_anndata_yaml('adata_astir.temp', astir_markers_yml, create_design_mat=True, batch_name=None)

    # Delete the temporary adata file
    os.remove('adata_astir.temp')

    # Create batch size proportional to the number of cells
    N = ast.get_type_dataset().get_exprs_df().shape[0]
    batch_size = int(N/100)

    #Run the cell type identification
    ast.fit_type(max_epochs = max_epochs,
             batch_size = batch_size,
             learning_rate = learning_rate,
             n_init_epochs = initial_epochs)

    #Map the cell types back to the original adata
    adata.obs[cell_type_label] = ast.get_celltypes(threshold=id_threshold)['cell_type']
    
    #If using a hierarchy, then will add this data also
    if use_hierarchy:
        hierarchy_table = ast.assign_celltype_hierarchy(depth = 1)

        cell_types = hierarchy_table.columns.tolist()

        #Start a new list that will store the hierarchy data
        hierarchy = []

        #This will work down each row and figure out which hierarchy type have the highest probability
        for index, row in hierarchy_table.iterrows():
            row_values = row.values
            max_prob = np.max(row_values)

            if max_prob < id_threshold:
                #If the cell doesn't fit into any category, return Unknown
                hierarchy.append('Other')
            else:
                #Add to the list the 
                hierarchy.append(cell_types[np.argmax(row_values)])

        adata.obs[hierarchy_label] = hierarchy
        
    if diagnostics:
        print('Diagnostiscs and results for Astir....\n')
        print(ast.get_celltypes().value_counts())
        ast.diagnostics_celltype()
        
    
    
def grouped_astir_adata(adata,group_analysis_by,astir_markers_yml,adata_cell_index,id_threshold=0.7,max_epochs = 1000,learning_rate = 2e-3,initial_epochs = 3,cell_type_label="astir_cell_type", hierarchy_label='astir_hierarchy'):
    """This function i largely experimental - the point is to do each Astir analyis separately. Does not currently do hierarchy.

    """
    # Import the function   
    import os
    import astir as ast
    from astir.data import from_anndata_yaml    
    
    # Create blank lists which we will add to
    celltype =[]
    masterindex = []


    #This will loop for each case
    for case in tqdm(adata.obs[group_analysis_by].cat.categories.tolist()):
        print('Running for '+case)
        # Create a working adata object that only has the cells from one case
        adata_astir = adata[adata.obs[group_analysis_by]==case].copy()

        # Save the adata to a file
        adata_astir.write(filename='adata_astir.temp')

        # Create the astir object
        ast = from_anndata_yaml('adata_astir.temp', astir_markers_yml, create_design_mat=True, batch_name=None)

        # Delete the temporary adata file
        os.remove('adata_astir.temp')

        # Create batch size proportional to the number of cells
        N = ast.get_type_dataset().get_exprs_df().shape[0]
        batch_size = int(N/100)

        #Run the cell type identification
        ast.fit_type(max_epochs = max_epochs,
                 batch_size = batch_size,
                 learning_rate = learning_rate,
                 n_init_epochs = initial_epochs)

        adata_astir.obs[cell_type_label] = ast.get_celltypes(threshold=id_threshold)['cell_type']

        celltype.extend(adata_astir.obs[cell_type_label])
        masterindex.extend(adata_astir.obs[adata_cell_index])


    adata_astir_dict = pd.DataFrame(list(zip(masterindex,celltype)),columns = [adata_cell_index,cell_type_label]).set_index(adata_cell_index).to_dict()

    adata.obs[cell_type_label]=adata.obs[adata_cell_index].map(adata_astir_dict[cell_type_label]).astype('category')

    
    
    
    
def grouped_graph(adata_plotting, ROI_id, group_by_obs, x_axis, display_tables=True, fig_size=(5,5), confidence_interval=68, save=False, log_scale=True, order=False,scale_factor=False,crosstab_norm=False):

    import seaborn as sb
    import pandas as pd
    import statsmodels as sm
    import scipy as sp
    import matplotlib.pyplot as plt 

    # Create cells table    
    cells = pd.crosstab([adata_plotting.obs[group_by_obs], adata_plotting.obs[ROI_id]],adata_plotting.obs[x_axis],normalize=crosstab_norm)
    cells.columns=cells.columns.astype('str')        

    # Creat long form data
    cells_long = cells.reset_index().melt(id_vars=[group_by_obs,ROI_id])
    
    if scale_factor:
        cells_long['value'] = cells_long['value'] / scale_factor
    
    fig, ax = plt.subplots(figsize=fig_size)
    
    if order:
        sb.barplot(data = cells_long, y = "value", x = x_axis, hue = group_by_obs, ci=confidence_interval, order=order, ax=ax)
    else:
        sb.barplot(data = cells_long, y = "value", x = x_axis, hue = group_by_obs, ci=confidence_interval, ax=ax)

    ax.set_xticklabels(ax.get_xticklabels(),rotation = 90, fontsize = 10)
    
    if scale_factor:
        ax.set_ylabel('Cells/mm2')
    else:
        ax.set_ylabel('Cells')        
                  
    if log_scale:
        ax.set_yscale("log")
        
    ax.set_xlabel(x_axis)
    ax.legend(bbox_to_anchor=(1.01, 1))

    #fig = ax.get_figure()


    if save:
        fig.savefig(save, bbox_inches='tight',dpi=200)

    col_names = adata_plotting.obs[group_by_obs].unique().tolist()

    data_frame = cells.reset_index()

    celltype = []
    ttest = []
    mw = []

    for i in cells.columns.tolist():
        celltype.append(i)
        ttest.append(sp.stats.ttest_ind(cells.loc[col_names[0]][i], cells.loc[col_names[1]][i]).pvalue) 
        mw.append(sp.stats.mannwhitneyu(cells.loc[col_names[0]][i], cells.loc[col_names[1]][i]).pvalue)

    stats = pd.DataFrame(list(zip(celltype,ttest,mw)),columns = ['Cell Type','T test','Mann-Whitney'])

    import statsmodels as sm

    #Multiple comparissons correction
    for stat_column in ['T test','Mann-Whitney']:
        corrected_stats = sm.stats.multitest.multipletests(stats[stat_column],alpha=0.05,method='holm-sidak')
        stats[(stat_column+' Reject null?')]=corrected_stats[0]
        stats[(stat_column+' Corrected Pval')]=corrected_stats[1]

    if display_tables:
        print('Raw data:')
        display(cells)

        print('Statistics:')
        display(stats)
        
    grouped_graph.cells = cells     
    grouped_graph.stats = stats                     
    
def pop_stats(adata_plotting,groups,Case_id,ROI_id,x_axis,display_tables=True,fig_size=(5,5), confidence_interval=68,save=False, log_scale=True, scale_factor=False):

    import seaborn as sb
    import pandas as pd
    import statsmodels as sm
    import scipy as sp
    import matplotlib.pyplot as plt 
    
    cells = pd.crosstab([adata_plotting.obs[groups], adata_plotting.obs[Case_id], adata_plotting.obs[ROI_id]],adata_plotting.obs[x_axis])
    cells.columns=cells.columns.astype('str')

    cells_long = cells.reset_index().melt(id_vars=[groups,Case_id,ROI_id])
    cells_long.columns=cells_long.columns.astype('str')
    
    if scale_factor:        
        if isinstance(scale_factor, dict):       
            for g in scale_factor:                    
                cells_long.loc[cells_long[groups]==g, 'value'] = cells_long.loc[cells_long[groups]==g, 'value'] / scale_factor[g]
        else:
            cells_long['value'] = cells_long['value'] / scale_factor      

    #Use this for plotting
    case_average_long = cells_long.groupby([groups,Case_id,x_axis],observed=True).mean().reset_index()

    #Use this for stats
    case_average_wide = cells.groupby([groups,Case_id],observed=True).mean()

    fig, ax = plt.subplots(figsize=fig_size)
    
    #Plotting
    sb.barplot(data = case_average_long, y = "value", x = x_axis, hue = groups, ci=confidence_interval, ax=ax)

    #Plotting settings
    ax.set_xticklabels(ax.get_xticklabels(),rotation = 90, fontsize = 10)
    ax.set_ylabel('Cells')
              
    if log_scale:
        ax.set_yscale("log")
   
    ax.set_xlabel(x_axis)
    ax.legend(bbox_to_anchor=(1.01, 1))
    
    if save:
        fig.savefig(save, bbox_inches='tight',dpi=200)

    col_names = adata_plotting.obs[groups].unique().tolist()

    celltype = []
    ttest = []
    mw = []

    for i in case_average_wide.columns.tolist():
        celltype.append(i)
        ttest.append(sp.stats.ttest_ind(case_average_wide.loc[col_names[0]][i], case_average_wide.loc[col_names[1]][i]).pvalue) 
        mw.append(sp.stats.mannwhitneyu(case_average_wide.loc[col_names[0]][i], case_average_wide.loc[col_names[1]][i]).pvalue)

    stats = pd.DataFrame(list(zip(celltype,ttest,mw)),columns = ['Cell Type','T test','Mann-Whitney'])

    

    #Multiple comparissons correction
    for stat_column in ['T test','Mann-Whitney']:
        corrected_stats = sm.stats.multitest.multipletests(stats[stat_column],alpha=0.05,method='holm-sidak')
        stats[(stat_column+' Reject null?')]=corrected_stats[0]
        stats[(stat_column+' Corrected Pval')]=corrected_stats[1]
    
    if display_tables:
        print('ROI totals:')
        display(cells)
        
        print('Cases averages:')
        display(case_average_wide)
    
        print('Statistics:')
        display(stats)
    
    pop_stats.cells = cells


    
    
def reset_plt():
    import matplotlib
    matplotlib.pyplot.rcParams.update(matplotlib.rcParamsDefault)
    
    
def contact_graph(so, spl):
    """ Used to multithread contact graph creation"""
    import athena as sh
    try:
        sh.graph.build_graph(so, spl, builder_type='contact', mask_key='cellmasks', inplace=False)
        return so.G[spl]['contact']
    except KeyboardInterrupt:
        pass
    except BaseException as err:
        print("An exception occurred in calculating contact graph for " + spl)
        print(f"Unexpected {err=}, {type(err)=}")        
        
def neigh_int(so,m,o,s,g):
    """ Used to multithread neighbourhood interaction"""
    import athena as sh
    try:
        so_copy = so.copy()
        sh.neigh.interactions(so_copy, s, o, mode=m, prediction_type='diff', graph_key=g,  n_permutations=1000)
        key = f'{o}_{m}_diff_{g}'
        return so_copy.uns[s]['interactions'][key]      
        
    except KeyboardInterrupt:
        pass  
    except BaseException as err:
        print(f'Error caclculating sample:{s}, graph:{g}, observation:{o}, mode:{m}')
        print(f"Unexpected {err=}, {type(err)=}")
        
def cell_metrics(so,s,o,g):
    """ Used to multithread cell metrics"""
    import athena as sh
    try:
        so_copy=so.copy()

        sh.metrics.richness(so_copy, s, o, local=True, graph_key=g)
        sh.metrics.shannon(so_copy, s, o, local=True, graph_key=g)
        sh.metrics.quadratic_entropy(so_copy, s, o, local=True, graph_key=g, metric='cosine')
        
        cols = [f"{metric}_{o}_{g}" for metric in ['richness','shannon','quadratic']]
        
        return so_copy.obs[s][cols]
        
    except KeyboardInterrupt:
        pass  
    except BaseException as err:
        print(f'Error caclculating sample:{s}, graph:{g}, observation:{o}')
        print(f"Unexpected {err=}, {type(err)=}")

def analyse_cell(raw_image, size0, size1, radius, cell_axis0, cell_axis1, return_props=None):
    """ Used to multithread extraction of regions around cells""" 
    
    import numpy as np
    from skimage.draw import disk
    from skimage.measure import label, regionprops, regionprops_table

    try:       
        # Create a circle mask
        cell_mask = np.zeros((size0, size1))
        rr, cc = disk((cell_axis0,cell_axis1),radius)
        cell_mask[rr, cc] = 1

        # Convert to a label
        cell_label = label(cell_mask)
        cell_properties = regionprops(cell_label,raw_image, cache=False)


        # Return the area intensity
        if return_props:
            return cell_properties
        else:
            return cell_properties[0].intensity_mean

    except:
        
        #Return NaN if can't perform, usually if the circle goes over edge of image
        return np.nan

def analyse_cell_features(raw_image, size0, size1, cell_id, marker, quant_value, cell_index_id, radius, cell_axis0, cell_axis1):
    """ Used to multithread extraction of regions around cells""" 
    
    import numpy as np
    from skimage.draw import rectangle
    from skimage.measure import label, regionprops, regionprops_table

    # Round the cell location and radius to whole pixels
    cell_axis0 = int(round(cell_axis0))    
    cell_axis1 = int(round(cell_axis1))    
    
    # Check if the square will go over the edge
    try:   
        assert cell_axis0-radius>=0
        assert cell_axis0+radius<=size0
        assert cell_axis1-radius>=0
        assert cell_axis1+radius<=size1    
    except:
        #Return NaN if the square goes over edge of image - we can drop these later
        return None         
        
    img_square = raw_image[(cell_axis0-radius):(cell_axis0+radius), (cell_axis1-radius):(cell_axis1+radius)]

    cell_properties = extract_cell_features(img_square,cell_id,marker,quant_value, cell_index_id)

    return cell_properties


    
def extract_cell_features(image, cell_id, marker, quant_value, cell_index_id):

    from skimage.feature import graycomatrix, graycoprops
    from skimage.util import img_as_ubyte,img_as_int, img_as_uint
    from scipy.stats import kurtosis, skew
        
    import pathlib
    from copy import copy
    import numpy as np
    import itertools
    from tqdm import tqdm
    from skimage.measure import label, regionprops, regionprops_table
    import skimage.io as skio
    import os                           

    results_df_list=[]
        
    # Create blank lists for measurements
    cell_id_measurement_list =[]
    marker_measurement_list=[]

    contrast_list = []
    dissimilarity_list  = []
    homogeneity_list = []
    energy_list = []
    correlation_list = []
    ASM_list = []
    kurtosis_list = []
    skew_list=[]

    mean_list=[]
    max_list=[]
    min_list=[]
    median_list=[]
    std_list=[]    

    # Clip the data from 0 to the 99th percentile of all the data for that marker
    raw_img_norm = image.clip(0, quant_value)

    # Scale each image from 0 to the quant value
    raw_img_norm = raw_img_norm / quant_value

    # Convert to 8 bit image
    img = img_as_ubyte(raw_img_norm)           

    # Image identifiers
    cell_id_measurement_list.append(copy(cell_id))
    marker_measurement_list.append(copy(marker))

    # Pixel intensity features
    mean_list.append(np.mean(img))
    #max_list.append(np.max(img))
    #min_list.append(np.min(img))
    median_list.append(np.median(img))
    std_list.append(np.std(img))

    # Scipy stats
    kurtosis_list.append(kurtosis(img.flat))
    skew_list.append(skew(img.flat))

    # Perform GLCM
    glcm = graycomatrix(img, distances=[5], angles=[0], symmetric=True, normed=True)

    # GLCM features
    contrast_list.append(graycoprops(glcm, 'contrast')[0,0])
    dissimilarity_list.append(graycoprops(glcm, 'dissimilarity')[0,0])
    homogeneity_list.append(graycoprops(glcm, 'homogeneity')[0,0])
    energy_list.append(graycoprops(glcm, 'energy')[0,0])
    correlation_list.append(graycoprops(glcm, 'correlation')[0,0])
    ASM_list.append(graycoprops(glcm, 'ASM')[0,0])
 

    zipped_list=zip(cell_id_measurement_list,
                #marker_measurement_list,
                mean_list,
                #max_list,
                #min_list,
                median_list,
                std_list,
                contrast_list,
                dissimilarity_list,
                homogeneity_list,
                energy_list,
                correlation_list,
                ASM_list,
                kurtosis_list,
                skew_list)

    column_names=[str(cell_index_id),
                #'Channel',
                'Mean',
                #'Max',
                #'Min',
                'Median',
                'Std',
                'Contrast',
                'Dissimilarity',
                'Homogeneity',
                'Energy',
                'Correlation',
                'ASM',
                'Kurtosis',
                'Skew']

    column_names = [f"{marker}_{x}" for x in column_names]

    column_names[0]=cell_index_id

    results_df = pd.DataFrame(zipped_list, columns=column_names).set_index(cell_index_id)
    
    return results_df
    
    
def interactions_summary(so, #Define spatial heterogeneity object
                        samples_list, #Specify list of samples to combine
                        interaction_reference, #Specify which interaction we want to combine
                        num_pops=None, #Specify number of populations in analyses - will calculate if not specified
                        var='diff', #The variable we want to return from the interactions table
                        population_dictionary=False,#This will be used to give labels
                        save=False,
                        aggregate_function='mean',
                        ax=None,
                        show=True,
                        title=True,
                        reindex=False, #List of populations in the order they should appear
                        calc_ttest_p_value=False,
                        test='ttest', 
                        cmap='coolwarm',
                        vmax=None,
                        vmin=None,
                        mult_comp=None,
                        mult_comp_alpha=0.05,
                        cluster_map=False):

    import seaborn as sb
    import matplotlib.pyplot as plt 
    from scipy.stats import ttest_1samp, wilcoxon
    import statsmodels as sm
    #print(interaction_reference + ' - ' + var + ' - ' + aggregate_function)    
    
    
    
    
    #Gets a list of columns, then makes an empty dataframe ready to add to
    columns_list =so.uns[samples_list[0]]['interactions'][interaction_reference].columns.tolist()
    columns_list.append('sample')
    results_df = pd.DataFrame(columns=columns_list)

    # Concatenate all the dataframes, adding a new column 'ROI'
    for i in samples_list:
        new = so.uns[i]['interactions'][interaction_reference].copy()
        new['sample']=i

        results_df=pd.concat([results_df, new])

    
    ####################### Define how we aggregate accross samples
    
    if aggregate_function=='mean':
        # Get mean of each interaction - in this case 'index' identifies the pops interacting
        summary = results_df.reset_index().dropna().groupby('index').mean()
    elif aggregate_function=='std':
        summary = results_df.reset_index().dropna().groupby('index').std()
    elif aggregate_function=='sum':
        summary = results_df.reset_index().dropna().groupby('index').sum()

    
    ####################### Does a 1 sample t test, comparing against a theoretical mean of 0 
    
    if calc_ttest_p_value:
        
        pvalues = []

        for count, i in enumerate(summary.index.values):            
            stats_row = results_df.reset_index()[results_df.index==i][var]
            
            if test=='wilcoxon':
                pvalue = wilcoxon(stats_row, zero_method='wilcox', correction=False).pvalue
            elif test=='ttest':
                pvalue = ttest_1samp(stats_row,0).pvalue

            pvalues.append(pvalue)

        
        ######## Correct for multiple comparissons using statsmodels
        
        if not mult_comp:
            summary['pvalue']=pvalues
        else:
            summary['pvalue']=sm.stats.multitest.multipletests(pvalues,alpha=mult_comp_alpha,method=mult_comp)[1]
        
        

    ####################### Calculate number of pops if not specifed
    
    if not num_pops:
        num_pops = int(np.sqrt(len(summary)))
    
    
    ####################### Makes sure populations are appropriately ordered
    pop_ids_ordered = []
    for i in range(num_pops):
        pop_ids_ordered.append(summary[0:num_pops].index.tolist()[i][1])
     
    
    
    ####################### Reshape into an array withs shape pops X pops
    results_array = np.array(summary[var]).reshape((num_pops,num_pops))
    
    
    ####################### Goes through the pvalues, and if less than the defined value, will return a * 
    if calc_ttest_p_value:
        stats_array = np.array(summary['pvalue']).reshape((num_pops,num_pops))
        sig_array = np.where(stats_array<calc_ttest_p_value,'*', "")
        
        
        
    # Convert array into a dataframe
    df1 = pd.DataFrame(results_array)
    
    ####################### Makes sure populations are appropriately ordered
    df1.columns=pop_ids_ordered
    df1.index=pop_ids_ordered     

    # Rename the columns and rows with pop names, if not will just use numbers
    if population_dictionary:
        df1.rename(columns=population_dictionary,index=population_dictionary,inplace=True)

    
    ####################### If using reindex, will now order them based upon their appearance on the given list
    if reindex:
        df1 = df1.reindex(reindex, columns=reindex)

    #### Functionality for working with subplots
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()
        show = False
    
    # Alter the key words for the * annotations showing significant results
    annot_kws={'fontsize':'x-large', 'fontweight':'extra bold','va':'center','ha':'center'}
    
    # Colour map keywords to make sure the color bars are a sensible shape and size  
    cbar_kws={'fraction':0.046, 'pad':0.04}
    
    if var=='p':
        sb.heatmap(data=df1, cmap=cmap, robust=True, vmax=0.05,vmin=0,ax=ax,linewidths=.5, cbar_kws=cbar_kws)
    
    elif calc_ttest_p_value:
        
        if not cluster_map:        
            sb.heatmap(data=df1, cmap=cmap, vmax=vmax, vmin=vmin, robust=True,ax=ax,linewidths=.5,annot=sig_array, fmt="",
                       annot_kws=annot_kws,
                      cbar_kws=cbar_kws)
        else:
            sb.clustermap(data=df1, cmap=cmap, vmax=vmax, vmin=vmin)            
        
    else:
        sb.heatmap(data=df1, cmap=cmap, robust=True,ax=ax,linewidths=.5, cbar_kws=cbar_kws)
    
    if not cluster_map:
        ax.set_aspect(1)
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    

        ####################### Set a title, or use a default title
        if title==True:
            ax.set_title(interaction_reference + ' - ' + var + ' - ' + aggregate_function)
        elif title==False:
            'Nothing'
        else:
            ax.set_title(title)

        if save:    
            fig = ax.get_figure()
            fig.savefig(save, bbox_inches='tight',dpi=200)

        if show:
            fig.show()
    
    interactions_summary.new = new    
    interactions_summary.roi_data = results_df
    interactions_summary.summary_data = summary
    interactions_summary.heatmap_data = df1
    try:
        interactions_summary.stats_array = stats_array
        interactions_summary.sig_array = sig_array
    except:
        None
    
def interactions_table(so, #Define spatial heterogeneity object
                        samples_list, #Specify list of samples to combine
                        interaction_reference, #Specify which interaction we want to combine
                        var='score', #The variable we want to return from the interactions table
                        population_dictionary=False,#This will be used to give labels
                        mode='mean',
                        remap=False,
                        remap_agg='sum'): #sum, mean or individual)

        ##################### Create blank dataframe to add results to
        columns_list =so.uns[samples_list[0]]['interactions'][interaction_reference].columns.tolist()
        columns_list.append('sample')
        results_df = pd.DataFrame(columns=columns_list)

        ##################### Concatenate the data for all the different ROIs 
        for i in samples_list:
            new = so.uns[i]['interactions'][interaction_reference].copy()
            new['sample']=i
            new.reset_index(inplace=True)

            ##################### If doing interactions, add in a new column that is total interactions per mm2    
            if var=='interactions_per_mm2':
                mapping_dict = so.obs[i].groupby('cell_type_id').size().to_dict()   
                new['pop_counts'] = new['source_label'].map(mapping_dict)
                new[var] = new['pop_counts'].astype(float)*new['score'].astype(float)/(len(samples)*2.25)

            results_df=pd.concat([results_df, new])

        ##################### Remap IDs with actual names of populations if they are given
        if population_dictionary:    
            results_df['source_label']=results_df['source_label'].map(population_dictionary)
            results_df['target_label']=results_df['target_label'].map(population_dictionary)

        ##################### If replacing any populations, do so and then add them together
        if remap:
            results_df['target_label'].replace(remap,inplace=True)
            
            ##################### Decide how the populations will be combined together
            if remap_agg == 'sum':
                results_df = results_df.groupby(['source_label','target_label','sample']).sum().reset_index()
            elif remap_agg == 'mean':
                results_df = results_df.dropna().groupby(['source_label','target_label','sample']).mean().reset_index()
                
        ##################### Decide how to aggregate over ROIs
        if mode=='sum':
            summary = results_df.groupby(['source_label','target_label']).sum().reset_index()
        elif mode=='mean':
            summary = results_df.groupby(['source_label','target_label']).mean().reset_index() 
        elif mode=='individual':
            summary = results_df

        ##################### Store results so they can be retrieved externally for error checking or saving
        interactions_table.results = results_df
        
        return summary
    
    
def interactions_summary_UMAP(so, #Define spatial heterogeneity object
                            samples_list, #Specify list of samples to combine
                            interaction_reference,#Specify which interaction we want to combine
                            category_columns,#Categorical columns
                            var='score', #The variable we want to return from the interactions table
                            annotate=False,
                            save=True,
                            dim_red='UMAP'):

    import sklearn
    import umap
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    

    #Gets a list of columns, then makes an empty dataframe ready to add to
    columns_list =so.uns[samples_list[0]]['interactions'][interaction_reference].columns.tolist()

    #Add columns for categorical variables
    for i in category_columns:
        columns_list.append(i)

    #Make a blank dataframe
    results_df = pd.DataFrame(columns=columns_list)

    # Concatenate all the dataframes, adding a new column for each categorical variable
    for i in samples_list:
        new = so.uns[i]['interactions'][interaction_reference].copy()

        for x in category_columns:
            new[x]=so.obs[i][x].tolist()[0]
            
        results_df=pd.concat([results_df, new])

    #Create a summary table of all the interactions for each sample
    df2 = results_df.reset_index().pivot(index=category_columns, columns='index', values=var).reset_index()

    for i in category_columns:
        df2[i]=df2[i].astype('category')

    #Specify the columns we will use to compute the UMAP, in this case it's only the values for cell interactions
    data_columns = df2.columns[len(category_columns):].tolist()

    #Create data frame for UMAP
    summary_data = df2[data_columns]

    #Fill in NaNs - these will be where there was no interaction. I'm unsure if zeros are the best way to interpolate the missing values though!
    summary_data.fillna(0, inplace=True)

    reducer = umap.UMAP()

    #Transform into Zscores
    scaled_summary_data = sklearn.preprocessing.StandardScaler().fit_transform(summary_data)

    #Perform embedding on scaled data
    if dim_red=='UMAP':
        embedding = reducer.fit_transform(scaled_summary_data)
    elif dim_red=='PCA':
        pca = PCA(n_components=2)
        embedding = pca.fit_transform(scaled_summary_data)
    elif dim_red=='tSNE':
        tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
        embedding = tsne.fit_transform(scaled_summary_data)

    #Create the graphs
    for i in category_columns:
        
        fig, ax = plt.subplots() 
        
        ax.scatter(
            embedding[:, 0],
            embedding[:, 1],
            #c=[sb.color_palette()[x] for x in df2[i].cat.codes],
            c=[cc.glasbey_warm[x] for x in df2[i].cat.codes],
            s=150)
        
        if annotate:
            for loc, txt in zip(embedding,df2[annotate].cat.categories):
                ax.annotate(txt, loc)
        
        fig.gca().set_aspect('equal', 'datalim')
        plt.title(i)
        
        if save:
            plt.savefig('figures/'+i+'_UMAPsummary.svg')
        
        plt.show()
     
    interactions_summary_UMAP.summary = df2
    interactions_summary_UMAP.embedding = embedding

    
    


def cellabundance_UMAP(adata,
                       ROI_id,
                        population,
                        colour_by=False,
                        annotate=True,
                        save=False,
                      normalize=False,
                      dim_red='UMAP'):


    import sklearn
    import umap
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE

    # Create cells table

    #Specify the columns we will use to compute the UMAP, in this case it's only the values for cell interactions
    if colour_by:
        cells = pd.crosstab([adata.obs[ROI_id],adata.obs[colour_by]],adata.obs[population],normalize=normalize).reset_index().copy()
        summary_data = cells[cells.columns[2:].tolist()]
    else:
        cells = pd.crosstab(adata.obs[ROI_id],adata.obs[population],normalize=normalize).reset_index().copy()        
        summary_data = cells[cells.columns[1:].tolist()]

    reducer = umap.UMAP()

    #Transform into Zscores
    scaled_summary_data = sklearn.preprocessing.StandardScaler().fit_transform(summary_data)

    #Perform embedding on scaled data
    if dim_red=='UMAP':
        embedding = reducer.fit_transform(scaled_summary_data)
    elif dim_red=='PCA':
        pca = PCA(n_components=2)
        embedding = pca.fit_transform(scaled_summary_data)
    elif dim_red=='tSNE':
        tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
        embedding = tsne.fit_transform(scaled_summary_data)


    #Declare colour maps
    if colour_by:
        c=[cc.glasbey_warm[x] for x in cells[colour_by].cat.codes]
    else:
        c=[cc.glasbey_warm[x] for x in cells[ROI_id].cat.codes]       
    
    #Create the graphs                       
    fig, ax = plt.subplots() 
    ax.scatter(
        embedding[:, 0],
        embedding[:, 1],
        c=c,
        s=150)

    if annotate:
        for loc, txt in zip(embedding,cells[ROI_id].cat.categories):
            ax.annotate(txt, loc)

    fig.gca().set_aspect('equal', 'datalim')
    ax.set_xlabel(dim_red+"1")
    ax.set_ylabel(dim_red+"2")
    
    if colour_by:
        plt.title(colour_by)

    if save:
        plt.savefig(save)
    

    plt.show()

    cellabundance_UMAP.cells = cells
    cellabundance_UMAP.embedding = embedding
    
def lisa_import(adata,
                LISA_file,
                LISA_col_title,
                remove_R=True):
    
    import pandas as pd
    
    #Import from CSV, add to adata and store as category
    LISA_import = pd.read_csv(LISA_file)

    if remove_R:
        #Removes the 'R' from LISA region
        LISA_import['region']=LISA_import['region'].str.replace('R','').astype('int')
    
    #Gets the cell number from the cellId column so we can make sure the cells are in the right order
    LISA_import['cell_number']=list(zip(*LISA_import.cellID.str.split('_')))[1]
    LISA_import['cell_number']=LISA_import['cell_number'].astype('float64')
    LISA_import.sort_values(['cell_number'],inplace=True)
    
    adata.obs[LISA_col_title]=list(LISA_import['region'])
    adata.obs[LISA_col_title]=adata.obs[LISA_col_title].astype('category')



#########################################################
''' The following functions and code for voronoi plots have been taken entirely from the Nolan lab (https://github.com/nolanlab) '''
#########################################################
    
        
def voronoi_finite_polygons_2d(vor, radius=None):
    """
    adapted from https://stackoverflow.com/questions/20515554/colorize-voronoi-diagram/20678647#20678647 3.18.2019
    
    
    Reconstruct infinite voronoi regions in a 2D diagram to finite
    regions.

    Parameters
    ----------
    vor : Voronoi
        Input diagram
    radius : float, optional
        Distance to 'points at infinity'.

    Returns
    -------
    regions : list of tuplesy
    
        Indices of vertices in each revised Voronoi regions.
    vertices : list of tuples
        Coordinates for revised Voronoi vertices. Same as coordinates
        of input vertices, with 'points at infinity' appended to the
        end.

    """

    if vor.points.shape[1] != 2:
        raise ValueError("Requires 2D input")

    new_regions = []
    new_vertices = vor.vertices.tolist()

    center = vor.points.mean(axis=0)
    if radius is None:
        radius = vor.points.ptp().max()

    # Construct a map containing all ridges for a given point
    all_ridges = {}
    for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
        all_ridges.setdefault(p1, []).append((p2, v1, v2))
        all_ridges.setdefault(p2, []).append((p1, v1, v2))

    # Reconstruct infinite regions
    for p1, region in enumerate(vor.point_region):
        vertices = vor.regions[region]

        if all(v >= 0 for v in vertices):
            # finite region
            new_regions.append(vertices)
            continue

        # reconstruct a non-finite region
        ridges = all_ridges[p1]
        new_region = [v for v in vertices if v >= 0]

        for p2, v1, v2 in ridges:
            if v2 < 0:
                v1, v2 = v2, v1
            if v1 >= 0:
                # finite ridge: already in the region
                continue

            # Compute the missing endpoint of an infinite ridge

            t = vor.points[p2] - vor.points[p1] # tangent
            t /= np.linalg.norm(t)
            n = np.array([-t[1], t[0]])  # normal

            midpoint = vor.points[[p1, p2]].mean(axis=0)
            direction = np.sign(np.dot(midpoint - center, n)) * n
            far_point = vor.vertices[v2] + direction * radius

            new_region.append(len(new_vertices))
            new_vertices.append(far_point.tolist())

        # sort region counterclockwise
        vs = np.asarray([new_vertices[v] for v in new_region])
        c = vs.mean(axis=0)
        angles = np.arctan2(vs[:,1] - c[1], vs[:,0] - c[0])
        new_region = np.array(new_region)[np.argsort(angles)]

        # finish
        new_regions.append(new_region.tolist())

    return new_regions, np.asarray(new_vertices)

def plot_voronoi(points,colors,invert_y = True,edge_color = 'facecolor',line_width = .1,alpha = 1,size_max=np.inf):
    
# spot_samp = spot#.sample#(n=100,random_state = 0)
# points = spot_samp[['X:X','Y:Y']].values
# colors = [sns.color_palette('bright')[i] for i in spot_samp['neighborhood10']]

    if invert_y:
        points[:,1] = max(points[:,1])-points[:,1]
    vor = Voronoi(points)

    regions, vertices = voronoi_finite_polygons_2d(vor)

    pts = MultiPoint([Point(i) for i in points])
    mask = pts.convex_hull
    new_vertices = []
    if type(alpha)!=list:
        alpha = [alpha]*len(points)
    areas = []
    for i,(region,alph) in enumerate(zip(regions,alpha)):
        polygon = vertices[region]
        shape = list(polygon.shape)
        shape[0] += 1
        p = Polygon(np.append(polygon, polygon[0]).reshape(*shape)).intersection(mask)
        areas+=[p.area]
        if p.area <size_max:
            poly = np.array(list(zip(p.boundary.coords.xy[0][:-1], p.boundary.coords.xy[1][:-1])))
            new_vertices.append(poly)
            if edge_color == 'facecolor':
                plt.fill(*zip(*poly), alpha=alph,edgecolor=  colors[i],linewidth = line_width , facecolor = colors[i])
            else:
                plt.fill(*zip(*poly), alpha=alph,edgecolor=  edge_color,linewidth = line_width, facecolor = colors[i])
        # else:

        #     plt.scatter(np.mean(p.boundary.xy[0]),np.mean(p.boundary.xy[1]),c = colors[i])
    return areas


def draw_voronoi_scatter(spot,c,voronoi_palette = sns.color_palette('bright'),scatter_palette = 'voronoi',X = 'X:X', Y = 'Y:Y',voronoi_hue = 'neighborhood10',scatter_hue = 'ClusterName',
        figsize = (5,5),
         voronoi_kwargs = {},
         scatter_kwargs = {}):
    if scatter_palette=='voronoi':
        scatter_palette = voronoi_palette
        scatter_hue = voronoi_hue
    '''
    plot voronoi of a region and overlay the location of specific cell types onto this
    
    spot:  cells that are used for voronoi diagram
    c:  cells that are plotted over voronoi
    palette:  color palette used for coloring neighborhoods
    X/Y:  column name used for X/Y locations
    hue:  column name used for neighborhood allocation
    figsize:  size of figure
    voronoi_kwargs:  arguments passed to plot_vornoi function
    scatter_kwargs:  arguments passed to plt.scatter()

    returns sizes of each voronoi to make it easier to pick a size_max argument if necessary
    '''
    if len(c)>0:
        neigh_alpha = .3
    else:
        neigh_alpha = 1
        
    voronoi_kwargs = {**{'alpha':neigh_alpha},**voronoi_kwargs}
    scatter_kwargs = {**{'s':50,'alpha':1,'marker':'.'},**scatter_kwargs}
    
    plt.figure(figsize = figsize)
    colors  = [voronoi_palette[i] for i in spot[voronoi_hue]]
    a = plot_voronoi(spot[[X,Y]].values,
                 colors,#[{0:'white',1:'red',2:'purple'}[i] for i in spot['color']],
                 **voronoi_kwargs)
    
    if len(c)>0:
        if 'c' not in scatter_kwargs:
            colors  = [scatter_palette[i] for i in c[scatter_hue]]
            scatter_kwargs['c'] = colors
            
        plt.scatter(x = c[X],y = (max(spot[Y])-c[Y].values),
                  **scatter_kwargs
                   )
    plt.axis('off');
    return a

def graph_simplify_once(g, attr, use_xy=False, x='x', y='y'):
    import numpy as np
    import networkx as nx
    global new_node
    global start_len
    global end_len
    import copy as copy
    
    graph=g.copy()
    
    # To enable first loop
    update=False
    
    # Itterate through all nodes        
    nodes = list(graph.nodes)
    for n in nodes:

        if update==False:

            # Itterate through each nodes neighbours
            neighs = list(graph.neighbors(n))
            for m in neighs:

                if update==False:

                    # If one of the neighbours is in the same type
                    if graph.nodes[n][attr]==graph.nodes[m][attr]:

                        attr_value=graph.nodes[n][attr]

                        # Add a new (blank) node
                        #new_node = np.max(graph.nodes)+1        
                        new_node = np.max(len(graph.nodes))+1     


                        #graph.add_nodes_from([(new_node, dict(graph.nodes(data=True)[n]))])
                        graph.add_nodes_from([(new_node, {attr:attr_value})])


                        # Add in links between the original node..
                        n_neigh = list(graph.neighbors(n))
                        for n_edge in n_neigh: 
                            graph.add_edge(new_node, n_edge)

                        # And the neighbour with which it had a shared attribute....
                        m_neigh = list(graph.neighbors(m))
                        for m_edge in m_neigh: 
                            graph.add_edge(new_node, m_edge)

                        # Add the number of cells together
                        n_cells = graph.nodes[n]['cells']
                        m_cells = graph.nodes[m]['cells']
                        
                        if use_xy:
                            new_x = graph.nodes[n][x] + graph.nodes[m][x]
                            new_y = graph.nodes[n][y] + graph.nodes[m][y]
                            nx.set_node_attributes(graph, {new_node: {x:new_x, y:new_y}})                       
                        
                        nx.set_node_attributes(graph, {new_node: {'cells':(n_cells+m_cells)}})

                        # Remove self referrential edges            
                        try:                           
                            graph.remove_edges_from(new_node,new_node)
                        except:
                            pass

                        # Remove the original nodes
                        graph.remove_node(n)

                        try:
                            graph.remove_node(m)
                        except:
                            pass

                        # Update
                        update = True        
                    else:
                        update = False

                else:
                    break

        else:
            break

    # Final pruning of self edges
    graph.remove_edges_from(nx.selfloop_edges(graph))
    
    return graph


def graph_simplify(g, attr, use_xy=False, x='x', y='y', progress=None):
    
    import networkx as nx
    
    graph=g.copy()
    nx.set_node_attributes(graph,1,name='cells')
    
    if use_xy:
        for node in graph.nodes:
            graph.nodes[node][x]=[graph.nodes[node][x]]
            graph.nodes[node][y]=[graph.nodes[node][y]]

    if progress:
        original_length=len(graph.nodes)
        print('Starting size of graph: '+str(len(graph.nodes)))
        prog_count=0
            
            
    while True:
        start=len(graph.nodes)
                
        graph = graph_simplify_once(g=graph, attr=attr, use_xy=use_xy, x=x, y=y)
        prog_count+=1
        
        end=len(graph.nodes)
        
        if progress:
            if prog_count==round(original_length/progress):
                print('Current size:' + str(end))
                prog_count=0
        
        if start==end:
            break
            
    
    if use_xy:
        for node in graph.nodes:
            graph.nodes[node][x]=np.mean(graph.nodes[node][x])
            graph.nodes[node][y]=np.mean(graph.nodes[node][y])
                    
    
    return graph



def graph_simplify_BACKUP(g, attr):
    import numpy as np
    import networkx as nx
    global new_node
    global start_len
    global end_len
    import copy as copy
    
    graph=g.copy()
    
    # Set all cells to 1 cell - we will use this to keep track of node/cell aggregation
    # nx.set_node_attributes(graph,1,name='cells')
    
    
    # Keep iterating through all the nodes until there are no nodes that are connected with the same attribute        
    end_len=0   
    start_len=len(graph.nodes)

    
    while True:
        
        update=False
    
        #Get initial size of network
        start_len=len(graph.nodes)
        
        # Itterate through all nodes        
        nodes = list(graph.nodes)
        
        for n in nodes:

            if update==False:
            
                # Itterate through each nodes neighbours
                neighs = list(graph.neighbors(n))
                for m in neighs:

                    if update==False:

                        # If one of the neighbours is in the same type
                        if graph.nodes[n][attr]==graph.nodes[m][attr]:

                            attr_value=graph.nodes[n][attr]
                            
                            # Add a new (blank) node
                            #new_node = np.max(graph.nodes)+1        
                            new_node = np.max(len(graph.nodes))+1     
                            
                            
                            #graph.add_nodes_from([(new_node, dict(graph.nodes(data=True)[n]))])
                            graph.add_nodes_from([(new_node, {attr:attr_value})])
                            

                            # Add in links between the original node..
                            n_neigh = list(graph.neighbors(n))
                            for n_edge in n_neigh: 
                                graph.add_edge(new_node, n_edge)

                            # And the neighbour with which it had a shared attribute....
                            m_neigh = list(graph.neighbors(m))
                            for m_edge in m_neigh: 
                                graph.add_edge(new_node, m_edge)
                                
                            # Add the number of cells together
                            n_cells = graph.nodes[n]['cells']
                            m_cells = graph.nodes[m]['cells']                                                       
                            nx.set_node_attributes(graph, {new_node: {'cells':(n_cells+m_cells)}})

                            # Remove self referrential edges            
                            try:                           
                                graph.remove_edges_from(new_node,new_node)
                            except:
                                pass

                            # Remove the original nodes
                            graph.remove_node(n)
                                                 
                            try:
                                graph.remove_node(m)
                            except:
                                print('Could not remove node '+str(m))

                            # Update
                            update = True        
                        else:
                            update = False

                    else:
                        break
                
            else:
                break

        #Get end length of network        
        end_len=len(graph.nodes)
        
        #print('START:'+str(start_len)+' END:'+str(end_len))
        
        if start_len==end_len:
            #print('DONE')
            break
        
    
    # Final pruning of self edges
    graph.remove_edges_from(nx.selfloop_edges(graph))
    
    return graph


def load_single_img(filename):
    
    import tifffile as tp

    """
    Loading single image from directory.
    Parameters
    ----------
    filename : The image file name, must end with .tiff.
        DESCRIPTION.
    Returns
    -------
    Img_in : int or float
        Loaded image data.
    """
    if filename.endswith('.tiff') or filename.endswith('.tif'):
        Img_in = tp.imread(filename).astype('float32')
    else:
        raise ValueError('Raw file should end with tiff or tif!')
    if Img_in.ndim != 2:
        raise ValueError('Single image should be 2d!')
    return Img_in

def load_imgs_from_directory(load_directory,channel_name,quiet=False):
    
    import os
    from os import listdir
    from pathlib import Path

    from os.path import isfile, join, abspath, exists
    from glob import glob
    
    Img_collect = []
    Img_file_list=[]
    img_folders = glob(join(load_directory, "*", ""))
    
    # If only one folder is returned, make into a list anyway
    #if not isinstance(img_folders, list):
    #    img_folders = [img_folders]

    if not quiet:
        print('Image data loaded from ...\n')
    
    if img_folders==[]:
        img_folders=[load_directory]
        
    
    for sub_img_folder in img_folders:
        Img_list = [f for f in listdir(sub_img_folder) if isfile(join(sub_img_folder, f)) & (f.endswith(".tiff") or f.endswith(".tif"))]
        
        # If only one image is returned, make into a list anyway      
        if not isinstance(Img_list, list):
            Img_list = [Img_list]        
        
        for Img_file in Img_list:
            if channel_name.lower() in Img_file.lower():
                #Img_read = load_single_img(sub_img_folder + Img_file)
                Img_read = load_single_img(join(sub_img_folder,Img_file))

                
                if not quiet:
                    print(sub_img_folder + Img_file)
                
                Img_file_list.append(Img_file)
                Img_collect.append(Img_read)
                break

    if not quiet:
        print('\n' + 'Image data loaded completed!')
    
    if not Img_collect:
        print(f'No such channel as {channel_name}. Please check the channel name again!')
        return

    return Img_collect, Img_file_list, img_folders


def processed_folder_rename(acquisition_metadata='acquisition_metadata.csv',
                            folder='processed',
                            new_title_column='description'):

                            
    ''' This function goes through a folder, renaming the subfolders with a matched list in a .csv file. I've been using this to rename the weird folder names produced by the Bodenmiller pipeline into actual ROI names. '''                            
    
    import os as os
    import pandas as pd

    roi_folder_names = pd.read_csv(acquisition_metadata)

    t = [x[0] for x in os.walk(folder)]

    for folder, name in zip(t[1:],list(roi_folder_names[new_title_column])): 

        os.rename(folder,os.path.join(folder, name))
        
def slice_adata_rois(adata,
              divide_by,
              ROI_size,
              ROI_col='ROI',
              X='X_loc',
              Y='Y_loc'):
    
    '''This will add a new adata.obs column that slices the ROI into chunks'''
    
    global sliced
    
    bins = np.array(range(0, dimension, int(ROI_size/divide_by)))

    y_bins = np.digitize(adata.obs[X], bins)
    x_bins = np.digitize(adata.obs[Y], bins)

    combined =[r+"_x"+str(m)+"_y"+str(n) for r,m,n in zip(adata.obs[ROI_col] ,x_bins, y_bins)]
    
    sliced = 'sliced_'+str(divide_by)
    
    adata.obs[sliced]=combined
    adata.obs[sliced]=adata.obs[sliced].astype('category')
    
    
def pos_dict(g, x='X_loc', y='Y_loc'):
    
    ''' Generates a position dictionary for draw_networkx '''
    
    pos_dict={}
    for n in g.nodes:
       
        #try:
            x_val=g.nodes[n][x]
            y_val=g.nodes[n][y]
            pos_dict.update({n:np.array([x_val,y_val])})
       # except:
       #     print(f'Error in node {n}')
    return pos_dict


def display_networkx(g, colour_attr, figsize=(20,20), dpi=75, cmap=cc.glasbey_category10, pos=None, ax=None, save=None, show_edges=True, labels=None, node_size=300):

    import networkx as nx
    
    pop_list=[]
    for i in g.nodes:
        pop_list.append(g.nodes[i][colour_attr])

    colours = [cmap[x] for x in pop_list]
        
    if ax==None:   
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    if show_edges:
        nx.draw_networkx(
            g,
            pos,
            node_color=colours,
            ax=ax,
            labels=labels, 
            node_size=node_size)
    else:
        nx.draw_networkx(
            g,
            pos,
            node_color=colours,
            ax=ax,
            edgelist=[],
            labels=labels, 
            node_size=node_size)
        
    if save:
        fig.savefig(save)

def extract_athena_graph(so, sample_id, graph_type, obs, cell_id='cell_id'):

    if type(obs) != list:
        obs=[obs]
    
    import networkx as nx
    
    # Extract networkx graph from SO object
    g = so.G[sample_id][graph_type].copy()

    # Create a new data frame with node annotations
    df_nodes = pd.DataFrame()
    df_nodes[cell_id]=so.obs[sample_id].index
    
    for o in obs:
        df_nodes[o]=so.obs[sample_id][o].reset_index()[o]

    
    # Set attributes
    pops_dict = df_nodes.set_index(cell_id).to_dict('index')
    nx.set_node_attributes(g, pops_dict)

    # This removes self referential loops
    g.remove_edges_from(nx.selfloop_edges(g))
    
    return g
 
    
    
def athena_networkx_display(so, sample_id, graph_type, obs_colour, obs, cell_id='cell_id', cmap=cc.glasbey_category10, palette=None, figsize=(20,20), dpi=75, return_graph=False):

    if type(obs) != list:
        obs=[obs]    
    
    
    import networkx as nx
    # Extract networkx graph from SO object
    g = so.G[sample_id][graph_type].copy()


    # Create a new data frame with node annotations
    df_nodes = pd.DataFrame()
    df_nodes[cell_id]=so.obs[sample_id].index

    
    for o in obs:
        df_nodes[o]=so.obs[sample_id][o].reset_index()[o]

    if palette:
        df_nodes['colour']=df_nodes['population'].map(palette)                 
    else:                 
        df_nodes['colour']=[cmap[x] for x in df_nodes[obs_colour].cat.codes]

    # Set attributes from a dictionary made from dataframe
    pops_dict = df_nodes.set_index(cell_id).to_dict('index')
    nx.set_node_attributes(g, pops_dict)

    # This removes self referential loops
    g.remove_edges_from(nx.selfloop_edges(g))

    #for n in g.nodes:
    #    del g.nodes[n]['X_loc']
    #    del g.nodes[n]['Y_loc']    

    athena_networkx_display.nodes = df_nodes
    
    node_attributes = (obs,)

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    nx.draw_networkx(
        g,
        node_color=df_nodes['colour'],
        ax=ax
    )

    plt.show()
    
    if return_graph==True:
        return g

    
def add_graph_back_to_so(so, sample, graph, graph_title, node_attr, add_to_obs=True):

    import pandas as pd
    
    from copy import copy
    
    global new_df
    global cell_id
    global attr
    
    so.G[sample].update({graph_title:graph})

    cell_id=[]
    attr=[]    
    for n in graph.nodes:
        
        if not n in so.obs[sample].index:
            cell_id.append(copy(n))
            attr.append(copy(graph.nodes[n][node_attr]))
            
    
    new_df = pd.DataFrame(zip(cell_id,attr), columns=['cell_id', node_attr]).set_index('cell_id')
    
    so.obs[sample]=pd.concat([new_df, so.obs[sample]])
    
    
def mlm_pops(adata_plotting,
                 groups,
                 Case_id,
                 ROI_id,
                 x_axis,
                 display_tables=True,
             fig_size=(5,5), 
             confidence_interval=68,
             save=False, 
             log_scale=True,
            crosstab_norm=False,
            col_remap=None,
            hue_order=None,
            boxplot=True,
            mult_comp='holm-sidak',
            order=None,
            scale_factor=None):

    import seaborn as sb
    import pandas as pd
    import statsmodels.api as sm
    import scipy as sp
    import matplotlib.pyplot as plt 
    #import statsmodels.api as sm
    import statsmodels.formula.api as smf
    
    global cells
    global cells_long
    global case_average_long
    
    cells = pd.crosstab([adata_plotting.obs[groups], adata_plotting.obs[Case_id], adata_plotting.obs[ROI_id]],adata_plotting.obs[x_axis],normalize=crosstab_norm)
    cells.columns=cells.columns.astype('str')

    cells_long = cells.reset_index().melt(id_vars=[groups,Case_id,ROI_id])
    cells_long.columns=cells_long.columns.astype('str')
    
    if scale_factor:
        
        if isinstance(scale_factor, dict):       
            for g in scale_factor:                    
                cells_long.loc[cells_long[groups]==g, 'value'] = cells_long.loc[cells_long[groups]==g, 'value'] / scale_factor[g]
        else:
            cells_long['value'] = cells_long['value'] / scale_factor        
            

    fig, ax = plt.subplots(figsize=fig_size)
    
    #Plotting
    #
    if boxplot:
        sb.boxplot(data = cells_long, y = "value", x = x_axis, hue = groups, hue_order=hue_order, ax=ax, showfliers=False)       
    else:
        sb.barplot(data = cells_long, y = "value", x = x_axis, hue = groups, hue_order=hue_order, ci=confidence_interval, ax=ax, order=order)
    
    #Plotting settings
    ax.set_xticklabels(ax.get_xticklabels(),rotation = 90, fontsize = 10)
    ax.set_ylabel('Cells')
              
    if log_scale:
        ax.set_yscale("log")
   
    ax.set_xlabel(x_axis)
    ax.legend(bbox_to_anchor=(1.01, 1))
    
    if save:
        fig.savefig(save, bbox_inches='tight',dpi=200)

    col_names = adata_plotting.obs[groups].unique().tolist()

    celltype = []
    ttest = []
    mw = []
    mlm = []

    stats_df = cells
    
    if col_remap:
        stats_df.rename(columns=col_remap, inplace=True)
    
    for i in [' ', ',', '+', '/']:
        stats_df.columns = stats_df.columns.str.replace(i, '_')
 
    
    mlm_pops.stats = stats_df
    
    for i,s in zip(cells.columns.tolist(), stats_df.columns.tolist()):
        
        formula = f"{s} ~ {groups}"
        print(formula)
        md = smf.mixedlm(formula, stats_df.reset_index(), groups=stats_df.reset_index()[Case_id])
        mdf = md.fit()

        celltype.append(i)
        ttest.append(sp.stats.ttest_ind(cells.loc[col_names[0]][i], cells.loc[col_names[1]][i]).pvalue) 
        mw.append(sp.stats.mannwhitneyu(cells.loc[col_names[0]][i], cells.loc[col_names[1]][i]).pvalue)
        mlm.append(mdf.pvalues[1])

    stats = pd.DataFrame(list(zip(celltype,ttest,mw, mlm)),columns = ['Cell Type','T test','Mann-Whitney', 'Mixed linear model'])   

    #Multiple comparissons correction
    for stat_column in ['T test','Mann-Whitney', 'Mixed linear model']:
        corrected_stats = sm.stats.multitest.multipletests(stats[stat_column],alpha=0.05,method=mult_comp)
        stats[(stat_column+' Reject null?')]=corrected_stats[0]
        stats[(stat_column+' Corrected Pval')]=corrected_stats[1]
    
    if display_tables:
        print('ROI totals:')
        display(cells)
    
        print('Statistics:')
        display(stats)
        


def rgb_to_hex_colourmap(rgb_colour_map):
    
    import matplotlib

    for p, c in rgb_colour_map.items():
        colours_hex[p] = matplotlib.colors.to_hex(c)
    
    return colours_hex


def mlm_table(data_frame,
               value,
                 groups,
                 Case_id,
                 ROI_id,
                 x_axis,
                 display_tables=True,
             fig_size=(5,5), 
             confidence_interval=68,
             save=False, 
             log_scale=True,
            crosstab_norm=False,
            col_remap=None):

    import seaborn as sb
    import pandas as pd
    import statsmodels as sm
    import scipy as sp
    import matplotlib.pyplot as plt 
    #import statsmodels.api as sm
    import statsmodels.formula.api as smf
    
    global cells
    global cells_long
    global case_average_long
    
    fig, ax = plt.subplots(figsize=fig_size)

    sb.boxplot(data = data_frame, y = value, x = x_axis, hue = groups, ax=ax)
    
    #Plotting settings
    ax.set_xticklabels(ax.get_xticklabels(),rotation = 90, fontsize = 10)
    ax.set_ylabel('Cells')
              
    if log_scale:
        ax.set_yscale("log")
   
    ax.set_xlabel(x_axis)
    ax.legend(bbox_to_anchor=(1.01, 1))
    
    if save:
        fig.savefig(save, bbox_inches='tight',dpi=200)

    col_names = data_frame[groups].unique().tolist()

    celltype = []
    ttest = []
    mw = []
    mlm = []

    stats_df = pd.crosstab([data_frame[groups], data_frame[Case_id], data_frame[ROI_id]],data_frame[x_axis],normalize=crosstab_norm)
    
    if col_remap:
        stats_df.rename(columns=col_remap, inplace=True)
    
    for i in [' ', ',', '+', '/']:
        stats_df.columns = stats_df.columns.str.replace(i, '_')
 
    
    mlm_pops.stats = stats_df
    
    for s in stats_df.columns.tolist():
        
        formula = f"{s} ~ {groups}"
        print(formula)
        md = smf.mixedlm(formula, stats_df.reset_index(), groups=stats_df.reset_index()[Case_id])
        mdf = md.fit()

        celltype.append(i)
        #ttest.append(sp.stats.ttest_ind(cells.loc[col_names[0]][s], cells.loc[col_names[1]][s]).pvalue) 
        #mw.append(sp.stats.mannwhitneyu(cells.loc[col_names[0]][s], cells.loc[col_names[1]][s]).pvalue)
        mlm.append(mdf.pvalues[1])

    #stats = pd.DataFrame(list(zip(celltype,ttest,mw, mlm)),columns = ['Cell Type','T test','Mann-Whitney', 'Mixed linear model'])   
    stats = pd.DataFrame(list(zip(celltype, mlm)),columns = ['Cell Type', 'Mixed linear model'])   

    #Multiple comparissons correction
    #for stat_column in ['T test','Mann-Whitney', 'Mixed linear model']:
    for stat_column in ['Mixed linear model']:

        corrected_stats = sm.stats.multitest.multipletests(stats[stat_column],alpha=0.05,method='holm-sidak')
        stats[(stat_column+' Reject null?')]=corrected_stats[0]
        stats[(stat_column+' Corrected Pval')]=corrected_stats[1]
    
    if display_tables:
    
        print('Statistics:')
        display(stats)
        
        
def print_full(x):
    pd.set_option('display.max_rows', len(x))
    display(x)
    pd.reset_option('display.max_rows')
    
    
def environmental_analysis_texture(adata, #The adata object where the cell locations are stored
                           samples_list, #The list of samples
                           marker_list, 
                           radius=20, #This is the 'radius' of the square, so by default a 40 x 40 area of pixels
                           num_cores=4, #The number of cores to use for multithreading
                           folder_dir='images',
                           roi_id='ROI',
                           x_loc = 'X_loc', 
                           y_loc = 'Y_loc', 
                           cell_index_id = 'Master_Index',
                           quantile=0.999, #The quantile at which to take as the maximum stain intensity to scale all images in each marker at
                           return_quant_table=False,
                           return_markers_concatenated=True):
    
    import pathlib
    #from mikeimc_v2 import analyse_cell_features
    import numpy as np
    from multiprocessing import Pool
    import itertools
    from tqdm import tqdm
    from skimage.measure import label, regionprops, regionprops_table
    import skimage.io as skio
    import os
    from copy import copy

    master_list = []
    
    quant_list=[]

            
    for marker in marker_list:
        
               
        # Make blank lists which will be re-used to make dictionaries
        all_marker_data=[]

        print('Marker: ' +marker)

        Img_collect, Img_file_list, img_folders = load_imgs_from_directory(folder_dir,marker,quiet=True)
        roi_list = [os.path.basename(pathlib.Path(x)) for x in img_folders]

        
        # Calculate the value at which to cap off the staining
        quant_value=[]
        for i in Img_collect:
            quant_value.append(np.quantile(i, quantile))
        quant_value = np.array(quant_value).mean()
        
        quant_list.append(quant_value)
        
        
        for image, img_file_name, roi in tqdm(zip(Img_collect, Img_file_list, roi_list), total=len(samples_list)):

            # Check that the image found is in the samples list we want to analyse
            if roi in samples_list:        

                #print('ROI: '+roi)

                adata_roi = adata.obs[adata.obs[roi_id]==roi]
                number_of_cells = range(len(adata_roi))

                raw_image = image
                size0 = [len(raw_image[0]) for i in number_of_cells]
                size1 = [len(raw_image[1]) for i in number_of_cells]                   
                
                raw_image_list = [raw_image for i in number_of_cells]
                cells_id_list = adata_roi[cell_index_id]
                mark_list = [marker for i in number_of_cells]
                qvalue_list = [quant_value for i in number_of_cells]
                cell_index_id_list = [cell_index_id for i in number_of_cells]
                
                                           
                radius_list = [radius for i in number_of_cells]
                roi_cells_x = adata_roi[x_loc]
                roi_cells_y = adata_roi[y_loc]

                analyse_cells_inputs=list(zip(raw_image_list,
                                              size0,
                                              size1,
                                              cells_id_list,
                                              mark_list,
                                              qvalue_list,
                                              cell_index_id_list,
                                              radius_list,
                                              roi_cells_x,
                                              roi_cells_y))

                #analyse_cell(raw_image, size0, size1, radius, cell_axis0, cell_axis1)

                #This will return a list of dataframes, one for each cell
                with Pool(processes = num_cores) as pool:
                    marker_data = pool.starmap(analyse_cell_features, analyse_cells_inputs)
                
                
                #Concat into one dataframe, with all the data for this marker in this ROI
                all_marker_data.extend(copy(marker_data))
                                
            else:
                pass

        # Remove any 'none', which is usually because the cell was on the edge of the ROI and couldn't be measured
        all_marker_data = [i for i in all_marker_data if i is not None]
        
        # Concatenate all the data into a dataframe        
        all_marker_data = pd.concat(all_marker_data)
        
        # Add this marker data on
        master_list.append(copy(all_marker_data))
        
    # Join all the markers together (will be done by default, but can be disabled if you want markers separately) 
    if return_markers_concatenated:
        master_list = pd.concat(master_list,axis=1)
    
    if not return_quant_table:
        return master_list
    else:       
        quant_table = pd.DataFrame(zip(marker_list,quant_list),columns=['Marker','Max value images scaled to']) 
        return master_list, quant_table
                                   
        # Create a mapping dictionary for all the cells
        #mapping_dict = dict(zip(cell_id,intensity))

        # Map into a new obs in the original adata
        #adata.obs[(marker+"_"+str(radius))]=adata.obs[cell_index_id].map(mapping_dict)
        
    
def clean_text(text):
    for ch in ['\\','`','*','_','{','}','[',']','(',')','>','#','+','-','.','!','$','\'',',',' ', '/']:
        if ch in text:
            text = text.replace(ch,'')
            
    return text

def load_channel(load_directory,channel_name):
    
    import os
    from os import listdir
    from pathlib import Path

    from os.path import isfile, join, abspath, exists
    from glob import glob
   
    Img_collect=[]
    
    Img_list = [f for f in listdir(load_directory) if isfile(join(load_directory, f)) & (f.endswith(".tiff") or f.endswith(".tif"))]

    # If only one image is returned, make into a list anyway      
    if not isinstance(Img_list, list):
        Img_list = [Img_list]

    for Img_file in Img_list:
        if channel_name.lower() in Img_file.lower():
            Img_read = load_single_img(join(load_directory,Img_file))
            Img_collect.append(Img_read)
            break

    return Img_collect


def make_images(image_folder, 
                samples_list,
                output_folder,
                name_prefix='',
                minimum=0.2,
                max_quantile='q0.97',
                red=None,
                red_range=None,
                green=None,
                green_range=None,
                blue=None, 
                blue_range=None,
                magenta=None,
                magenta_range=None,
                cyan=None,
                cyan_range=None,
                yellow=None,
                yellow_range=None,
                white=None,
                white_range=None,
                roi_folder_save=False,
                simple_file_names=False,
                save_subfolder=''
                ):
    
    """This function will create RGB images using up to 7 channels, similar to MCD Viewer or ImageJ 
    Args:
        image_folder:
            The folder with the source images. In this folder, each ROI should have its own subfolder, with the folder named after the ROI.
        samples_list:
            List of samples to use from the image_folder. Only these samples will be used for auto-exposure if using quatiles (which is the default)
        output_folder:
            Output folder for images
        name_prefix:
            A prefix that will be put at the front of the file names
        minimum:
            The default minimum value (see 'range' below')
        max_quantile:
            The default max staining, which is the 97th quantile
        {colour}:
            The name of the marker to use for that colour
        {colour_range}:
            You can specify the range that should be the maximum and minimum, using the format (minimum, maximum). Both are specified by raw counts by default. However, if you put in a 'q' before a number, it will use that quantile range instead. e.g. 'q0.99' will use the 99th percentile as the maximum.
    Returns:
        Saves a .png per roi in the specified output directory
    """
    
    from pathlib import Path
    import tifffile as tp
    from skimage import io, exposure, data, img_as_ubyte
    from itertools import compress
    import numpy as np
    import os

    if not isinstance(samples_list, list):
        samples_list = [samples_list]

    global red_imgs, blue_imgs, green_imgs, red_master, green_master, blue_master, red_rois, green_rois, blue_rois, white_imgs, red_summary, green_summary, blue_summary, rum_rois
    
    # Create output directory if doesn't exist
    output = Path(output_folder)
    try:
        os.makedirs(output_folder)
    except:
        pass
    
    # Define empty lists
    red_imgs, green_imgs, blue_imgs,magenta_imgs,cyan_imgs,yellow_imgs,white_imgs = [], [], [], [], [], [], []
    red_rois, green_rois, blue_rois,magenta_rois,cyan_rois,yellow_rois,white_rois = [], [], [], [], [], [], []    
    roi_master=[]


    # Read in all the different colours (if used), including appropriate scaling
    # This is very sloppy and not very pythonic, but it works!

    if red is not None:
        
        if red_range is not None:
            min_v=red_range[0]
            max_q=red_range[1]
        else:
            min_v=minimum
            max_q=max_quantile
            
        red_imgs, red_rois = load_rescale_images(image_folder, samples_list, red, min_v, max_q)
        roi_master.append(red_rois)
        red="r_"+red+"_"

    if green is not None:
        
        if green_range is not None:
            min_v=green_range[0]
            max_q=green_range[1]
        else:
            min_v=minimum
            max_q=max_quantile
            
        green_imgs, green_rois = load_rescale_images(image_folder, samples_list, green, min_v, max_q)
        roi_master.append(green_rois)
        green='g_'+green+"_"
    
    if blue is not None:
        
        if blue_range is not None:
            min_v=blue_range[0]
            max_q=blue_range[1]
        else:
            min_v=minimum
            max_q=max_quantile
        
        blue_imgs, blue_rois = load_rescale_images(image_folder, samples_list, blue, min_v, max_q)
        roi_master.append(blue_rois)
        blue='b_'+blue+"_"
                
    if magenta is not None:
        
        if magenta_range is not None:
            min_v=magenta_range[0]
            max_q=magenta_range[1]
        else:
            min_v=minimum
            max_q=max_quantile
                       
        magenta_imgs, magenta_rois = load_rescale_images(image_folder, samples_list, magenta, min_v, max_q)
        roi_master.append(magenta_rois)
        magenta='m_'+magenta+"_"
        
    if cyan is not None:
        
        if cyan_range is not None:
            min_v=cyan_range[0]
            max_q=cyan_range[1]
        else:
            min_v=minimum
            max_q=max_quantile
        
        cyan_imgs, cyan_rois = load_rescale_images(image_folder, samples_list, cyan, min_v, max_q)
        roi_master.append(cyan_rois)
        cyan='c_'+cyan+"_"
       
    if yellow is not None:
        
        if yellow_range is not None:
            min_v=yellow_range[0]
            max_q=yellow_range[1]
        else:
            min_v=minimum
            max_q=max_quantile
        
        yellow_imgs, yellow_rois = load_rescale_images(image_folder, samples_list, yellow, min_v, max_q)
        roi_master.append(yellow_rois)
        yellow = 'y_'+yellow+"_"   

    if white is not None:
                        
        if white_range is not None:
            min_v=white_range[0]
            max_q=white_range[1]
        else:
            min_v=minimum
            max_q=max_quantile
            
        white_imgs, white_rois = load_rescale_images(image_folder, samples_list, white, min_v, max_q)
        roi_master.append(white_rois)
        white='w_'+white+"_"
        
    # Calculate number of ROIs found    
    num_rois = np.array([len(x) for x in [red_rois, green_rois, blue_rois,magenta_rois,cyan_rois,yellow_rois,white_rois]]).max()
    print(f'Found {num_rois} regions of interest')
    
    
    # Set non-used colours to empty
    if red is None:
        red_imgs = [0 for x in range(num_rois)]
        red=''
        
    if green is None:
        green_imgs = [0 for x in range(num_rois)]
        green=''
        
    if blue is None:
        blue_imgs = [0 for x in range(num_rois)]
        blue=''
        
    if magenta is None:
        magenta_imgs = [0 for x in range(num_rois)]
        magenta=''
        
    if cyan is None:
        cyan_imgs = [0 for x in range(num_rois)]
        cyan=''
        
    if yellow is None:
        yellow_imgs = [0 for x in range(num_rois)]
        yellow=''           
                
    if white is None:
        white_imgs = [0 for x in range(num_rois)]
        white=''      
    
    # Add up the various colours to make an RGB compatible colour space   
    red_summary = [np.clip((white_imgs[x] + red_imgs[x] + magenta_imgs[x] + yellow_imgs[x]),0,1) for x in range(num_rois)]
    blue_summary = [np.clip((white_imgs[x] + blue_imgs[x] + magenta_imgs[x] + cyan_imgs[x]),0,1) for x in range(num_rois)]
    green_summary = [np.clip((white_imgs[x] + green_imgs[x] + cyan_imgs[x] + yellow_imgs[x]),0,1) for x in range(num_rois)]
          
    
    # If using images which have no R, G or B at all, then use empty values
    for sample, r, g, b in zip(roi_master[0], red_summary, green_summary, blue_summary):
    
        print('Saving: '+sample)
        
        if np.shape(b)==():
            if np.shape(g)==():
                b=np.zeros(r.shape)
                g=np.zeros(r.shape)
            elif np.shape(r)==():
                b=np.zeros(g.shape)                
                r=np.zeros(g.shape)
            else:
                b=np.zeros(r.shape)
        elif np.shape(r)==():
            if np.shape(g)==():
                r=np.zeros(b.shape)
                g=np.zeros(b.shape)
            else:
                r=np.zeros(b.shape)
        elif np.shape(g)==():
            g=np.zeros(r.shape)
                  
        
        stack = np.dstack((r,g,b))
                
        # If using sample_file_names, then each image just gets saved as its ROI name
        if not simple_file_names:
            filename=f'{name_prefix}{sample}_{red}{green}{blue}{yellow}{cyan}{magenta}{white}'
            filename = filename.rstrip('_')
        else:
            filename=sample
        
        if not roi_folder_save:
            save_path=os.path.join(output_folder,f'{filename}.png')
        else:
            # Create output directory if doesn't exist
            
            roi_folder=os.path.join(output_folder,sample)           

            try:
                os.makedirs(roi_folder)
            except:
                pass
            
            save_path=os.path.join(roi_folder,f'{filename}.png')
        
        if save_subfolder!='':
             
            sub_path=os.path.join(output_folder,save_subfolder)           

            try:
                os.makedirs(sub_path)
            except:
                pass
            
            save_path=os.path.join(sub_path,f'{filename}.png')       
                    
        io.imsave(save_path,img_as_ubyte(stack))
               

def load_rescale_images(image_folder, samples_list,marker, minimum, max_val):
    
    import numpy as np
    import os
    from pathlib import Path
    from skimage import exposure
    from itertools import compress
    
    ''' Helper function to rescale images for above function'''
    
    mode = 'value'
    if str(max_val)[0]=='q':
        max_quantile=float(str(max_val)[1:])
        mode='quantile'

    
    # Load the imaes
    image_list, _, folder_list = load_imgs_from_directory(image_folder,marker,quiet=True)

    # Get the list of ROIs
    roi_list = [os.path.basename(Path(x)) for x in folder_list]
        
    # Filter out any samples not in the samples list
    sample_filter = [x in samples_list for x in roi_list]
    image_list = list(compress(image_list, sample_filter))
    roi_list = list(compress(roi_list, sample_filter))
    
    # Calculate the value at which to cap off the staining by taking the average of the max quantile value    
    if mode=='quantile':
        max_value = [np.quantile(i, max_quantile) for i in image_list]    
        max_value = np.array(max_value).mean()
        print(f'Marker: {marker}, Min value: {minimum}, Quantile: {max_quantile}, Calculated max value: {max_value}')
    else:
        max_value = max_val
        print(f'Marker: {marker}, Min value: {minimum},  Max value: {max_value}')
    
    # Clip
    image_list = [i.clip(minimum, max_value) for i in image_list]
    
    # Rescale intensity
    image_list = [exposure.rescale_intensity(i) for i in image_list]

    return image_list, roi_list

def cells_in_environment(so, samples, image_folder, save_folder, marker, low_val=0.2, upper_val='q0.99', show=True, node_size=10):
    
    from pathlib import Path
    import os
    
    # If only one sample, make into a list anyway
    if not isinstance(samples, list):
        samples = [samples] 
    
    # Create output folder if not already made
    save_folder = Path(save_folder)
    try:
        os.makedirs(save_folder)
    except:
        pass
    
    image_list, roi_list = load_rescale_images(image_folder, samples, marker, low_val, upper_val)


    for image, roi in zip(image_list,roi_list):

        #raw_image  = skio.imread('processed/1A1/32_20_Dy163_GLUT1.tiff')

        #norm_value = np.quantile(raw_image, q=0.99)

        #norm_image = np.clip((raw_image / norm_value),0,1)

        fig, ax = plt.subplots(figsize=(20,20),dpi=300)
        ax.imshow(image, vmin=0, vmax = 1, cmap='gist_gray')

        #sh.pl.spatial(so, '1A1', attr='population_broad_id', mode='mask', ax=ax, background_color='white')
        sh.pl.spatial(so_myeloid, roi, attr='population_broad_id', ax=ax, node_size=node_size)

        filename=f'{roi}_{marker}.png'
        fig.savefig(os.path.join(save_folder,filename), bbox_inches='tight')

        if not show:
            plt.close()


def subset_typeofnode(G, pop, pop_attr='population_broad'):
    '''return those nodes in graph G that match type = typestr.'''
    return [name for name, d in G.nodes(data=True) 
            if pop_attr in d and (d[pop_attr] == pop)]

#All computations happen in this function
def find_nearest(G, typeofnode, fromnode):

    import networkx as nx
    
    #Calculate the length of paths from fromnode to all other nodes
    lengths=nx.single_source_dijkstra_path_length(G, fromnode, weight='distance')
    #paths = nx.single_source_dijkstra_path(G, fromnode)

    #We are only interested in a particular type of node
    subnodes = subset_typeofnode(G, typeofnode)
    subdict = {k: v for k, v in lengths.items() if k in subnodes}
    
    #if fromnode in subdict:
    #    del subdict[fromnode]

    #return the smallest of all lengths to get to typeofnode
    if subdict: #dict of shortest paths to all entrances/toilets
        nearest =  min(subdict, key=subdict.get) #shortest value among all the keys
        #return(nearest, subdict[nearest], paths[nearest])
        return subdict[nearest]

    else: #not found, no path from source to typeofnode
        return None

    
    
    
def average_nearest(roi_name, G, to_pop, from_pop, pop_attr='population_broad', distance=True):

    roi_list=[]
        
    distance_list=[]
    distance_std_list=[]
    
    nearest_list=[]
    nearest_std_list=[]
    #edgecore_list=[]
    number_from_list=[]
    number_to_list=[]
    from_list=[]
    to_list=[]

    from copy import copy

    from_nodes = subset_typeofnode(G, from_pop, pop_attr)
    to_nodes =  subset_typeofnode(G, to_pop, pop_attr)
  
    if not distance:
        Nearest = [find_nearest(G, to_pop, x) for x in from_nodes]
        Distance = [0 for x in from_nodes]
    else:        
        #if (len(from_nodes)!=0)&(len(from_nodes)!=0):
        Raw = [find_nearest_distance(G, to_pop, x) for x in from_nodes]
        Unzip = list(zip(*Raw))
        
        try:
            Nearest = Unzip[0]
            Distance = Unzip[1]
        except:
            Nearest=[]
            Distance=[]
        #else:
        #    Nearest = np.nan
        #    Distance = np.nan
        
            
    roi_list.append(copy(roi_name))        
    from_list.append(copy(from_pop))
    to_list.append(copy(to_pop))                                                          
    
    try:
        nearest_list.append(np.array(Nearest).mean())
        nearest_std_list.append(np.std(Nearest))
    except:
        nearest_list.append(np.nan)
        nearest_std_list.append(np.nan)
    
    try:
        distance_list.append(np.array(Distance).mean())
        distance_std_list.append(np.std(Distance))
    except:
        distance_list.append(np.nan)
        distance_std_list.append(np.nan)        
        
    
    #edgecore_list.append(copy(so.spl.loc[s,'HEClass_2class']))
    number_from_list.append(len(from_nodes))
    number_to_list.append(len(to_nodes))
    
    nearest_df = pd.DataFrame(zip(roi_list, from_list, to_list, nearest_list,nearest_std_list,distance_list,distance_std_list,number_from_list, number_to_list), columns=['ROI','From_pop','To_pop','Avg_Nearest','STD_Nearest','Avg_Distance','STD_Distance','Number_from_pop','Number_to_pop'])
                              
    return nearest_df


def find_nearest_distance(G, typeofnode, fromnode, X_loc='X_loc', Y_loc='Y_loc'):
 
    global subdict
    global subdict_distance
    #We are only interested in a particular type of node
    subnodes = subset_typeofnode(G, typeofnode)
    
    #Calculate the WEIGHT length of paths from fromnode to all other nodes, ie each cell jump is 1
    lengths=nx.single_source_dijkstra_path_length(G, fromnode, weight='weight')
    subdict = {k: v for k, v in lengths.items() if k in subnodes}

    #Calculate the DISTANCE of paths from fromnode to all other nodes
    lengths_distance=nx.single_source_dijkstra_path_length(G, fromnode, weight='distance')
    subdict_distance = {k: v for k, v in lengths_distance.items() if k in subnodes}
    
    #If the 'from' and 'to' populations are the same, looks for next nearest instead by removing 'fromnode' from list
    if fromnode in subdict:
        del subdict[fromnode]
        del subdict_distance[fromnode]
            
    #return the smallest of all lengths to get to typeofnode
    if subdict: #dict of shortest paths to all entrances/toilets
        nearest =  min(subdict, key=subdict.get) #shortest value among all the keys   
        nearest_distance =  min(subdict_distance, key=subdict_distance.get) #shortest distance path among all the keys   
              
        #Calculate euclidian distance between nodes directly, rather than using a path between nodes
        distance = math.dist([G.nodes[fromnode][X_loc], G.nodes[fromnode][Y_loc]],[G.nodes[nearest_distance][X_loc], G.nodes[nearest_distance][Y_loc]])
        
        return subdict[nearest], distance

    else: #not found, no path from source to typeofnode
        return None, None
    
def graph_add_distances(Graph, X_loc='X_loc', Y_loc='Y_loc',edge_attr='distance'):

    #import math

    G = Graph.copy()

    for u,v,a in G.edges(data=True):
        G.edges[u, v][edge_attr]=math.dist([G.nodes[u][X_loc], G.nodes[u][Y_loc]],[G.nodes[v][X_loc], G.nodes[v][Y_loc]]) 

    return G

def randomise_graph(g, attr):

    import random

    g_perm = g.copy()

    attr_list = [g_perm.nodes[x][attr] for x in g_perm.nodes()]
    random.shuffle(attr_list)

    for a, n in zip(attr_list, g_perm.nodes()):
        g_perm.nodes[n].update({attr:a})

    return g_perm

def predict_distances(from_number, to_number, itterations=10, roi_size=1500, to_cells=None):

    import numpy as np
    import scipy.stats
    import matplotlib.pyplot as plt
    import math

    if (from_number==0) or (to_number==0):
        return np.nan, np.nan, np.nan
    
    #Setup size of ROI

    #Simulation window parameters
    xMin=0;xMax=roi_size;
    yMin=0;yMax=roi_size;
    xDelta=xMax-xMin;yDelta=yMax-yMin; #rectangle dimensions
    areaTotal=xDelta*yDelta;

    results=[]

    for x in range(itterations):
        #Simulate Poisson point process
        from_xx = xDelta*scipy.stats.uniform.rvs(0,1,((from_number,1)))+xMin#x coordinates of Poisson points
        from_yy = yDelta*scipy.stats.uniform.rvs(0,1,((from_number,1)))+yMin#y coordinates of Poisson points
        from_cells = list(zip(from_xx,from_yy))

        if not to_cells:
            to_xx = xDelta*scipy.stats.uniform.rvs(0,1,((to_number,1)))+xMin#x coordinates of Poisson points
            to_yy = yDelta*scipy.stats.uniform.rvs(0,1,((to_number,1)))+yMin#y coordinates of Poisson points
            to_cells = list(zip(to_xx,to_yy))        

        #For each cells in from_cells, calculate the minimum distance to the nearest to_cell, then take the average overall
        results.append(np.mean([min([math.dist(f, t) for t in to_cells]) for f in from_cells]))

    return np.mean(results), np.std(results), (np.std(results, ddof=1) / np.sqrt(np.size(results)))

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import AxesGrid

def shiftedColorMap(cmap, start=0, midpoint=0.5, stop=1.0, name='shiftedcmap'):
    '''
    Taken from: https://stackoverflow.com/questions/7404116/defining-the-midpoint-of-a-colormap-in-matplotlib
    
    Function to offset the "center" of a colormap. Useful for
    data with a negative min and positive max and you want the
    middle of the colormap's dynamic range to be at zero.

    Input
    -----
      cmap : The matplotlib colormap to be altered
      start : Offset from lowest point in the colormap's range.
          Defaults to 0.0 (no lower offset). Should be between
          0.0 and `midpoint`.
      midpoint : The new center of the colormap. Defaults to 
          0.5 (no shift). Should be between 0.0 and 1.0. In
          general, this should be  1 - vmax / (vmax + abs(vmin))
          For example if your data range from -15.0 to +5.0 and
          you want the center of the colormap at 0.0, `midpoint`
          should be set to  1 - 5/(5 + 15)) or 0.75
      stop : Offset from highest point in the colormap's range.
          Defaults to 1.0 (no upper offset). Should be between
          `midpoint` and 1.0.
    '''
    cdict = {
        'red': [],
        'green': [],
        'blue': [],
        'alpha': []
    }

    # regular index to compute the colors
    reg_index = np.linspace(start, stop, 257)

    # shifted index to match the data
    shift_index = np.hstack([
        np.linspace(0.0, midpoint, 128, endpoint=False), 
        np.linspace(midpoint, 1.0, 129, endpoint=True)
    ])

    for ri, si in zip(reg_index, shift_index):
        r, g, b, a = cmap(ri)

        cdict['red'].append((si, r, r))
        cdict['green'].append((si, g, g))
        cdict['blue'].append((si, b, b))
        cdict['alpha'].append((si, a, a))

    newcmap = matplotlib.colors.LinearSegmentedColormap(name, cdict)
    plt.register_cmap(cmap=newcmap)

    return newcmap

@ignore_warnings(category=ConvergenceWarning)
def mlm_two_way(data_full, 
                value_cols, 
                groups_col, #Assumes only two groups - check
                populations_col,
                control_population=None, 
                populations_list=None,
                Case_col='Case',
                use_cell_level=True,
                ROI_col='ROI',
                mult_comp='holm-sidak',
                factors='both'): #'both','populations', 'groups'
    
    from copy import copy
    import statsmodels as sm
    import pandas as pd
    import warnings
    
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)

    # Check that the factors is one of the options
    assert factors=='both' or factors=='populations' or factors=='groups', 'Factors must be one of: both, populations, groups'
    
    # If only a single marker given, make into a list anyway
    if not isinstance(value_cols, list):
        value_cols=[value_cols]
    
    # If not populations given, just use all of them from the data
    if not populations_list: 
        populations_list = data_full[populations_col].unique().tolist()
        
    # Assign a control pop if not given
    if not control_population:        
        control_population = data_full[populations_col].value_counts().index[0]
        print(f'No control populaton provided, using most abundant population: {control_population}')
    
    # Check number of groups is two
    groups_list = data_full[groups_col].unique().tolist()
    assert len(groups_list)==2, f'More than two groups found in the groups column: {groups_list}'
    
    # Setup MLM parameters if using cell-level data, if not will assume values are at ROI level
    if use_cell_level:
        vc = {f'{ROI_col}': f'0 + C({ROI_col})'}
        re_formula='1'
    else:
        vc=None
        re_formula=None
    
    results_list=[]
    
    for marker in value_cols:
        
        marker_list=[]
        pop_list=[]
        p_value=[]
        formula_list=[]             
        
        
        if factors=='groups' or factors=='both':
        
            #################### Compare groups WITHIN each population               
            for pop in populations_list:

                # Filter down to population of interest

                data=data_full.loc[data_full[populations_col]==pop, :]
                mlm_two_way.data = data

                formula = f"{marker} ~ {groups_col}"

                md = sm.regression.mixed_linear_model.MixedLM.from_formula(formula=formula,
                                                                           vc_formula=vc, 
                                                                           data=data, 
                                                                           groups=Case_col, 
                                                                           re_formula=re_formula)
                mdf = md.fit()

                marker_list.append(copy(marker))
                formula_list.append(copy(formula))
                pop_list.append(copy(pop))                    
                p_value.append(mdf.pvalues[1])
            
   
               
        if factors=='populations' or factors=='both':

            #################### Compare populations, regardless of groups 
            
            populations_limited = populations_list.copy()
            populations_limited.remove(control_population)

            for pop in populations_limited:

                # Filter down to population of interest AND control pop only
                data=data_full.loc[data_full[populations_col].isin([pop, control_population]), :]
                data[populations_col].cat.remove_unused_categories(inplace=True)

                mlm_two_way.data = data

                formula = f"{marker} ~ {populations_col}"

                md = sm.api.regression.mixed_linear_model.MixedLM.from_formula(formula=formula,
                                                                           vc_formula=vc, 
                                                                           data=data, 
                                                                           groups=Case_col, 
                                                                           re_formula=re_formula)
                mdf = md.fit()

                marker_list.append(copy(marker))
                formula_list.append(copy(formula))
                pop_list.append(f'{control_population} -- {pop}')                    
                p_value.append(mdf.pvalues[1])
            
        marker_dataframe = pd.DataFrame(zip(marker_list, formula_list, pop_list, p_value), columns=['Marker','Formula','Population','Pvalue'])
        
        if mult_comp:
            marker_dataframe['Pval_corr'] = sm.stats.multitest.multipletests(marker_dataframe['Pvalue'],alpha=0.05,method=mult_comp)[1]
            marker_dataframe['Significant'] = sm.stats.multitest.multipletests(marker_dataframe['Pvalue'],alpha=0.05,method=mult_comp)[0]
                        
        results_list.append(marker_dataframe)
    
    results_dataframe=pd.concat(results_list)
    
    return results_dataframe


def get_top_columns(row, number):
    top = tuple(row.nlargest(number).index)
    return '__'.join(top)


def backgating(adata,
               cell_index,
               radius,
               image_folder,
                red=None,
                red_range=None,
                green=None,
                green_range=None,
                blue=None, 
                blue_range=None,
                magenta=None,
                magenta_range=None,
                cyan=None,
                cyan_range=None,
                yellow=None,
                yellow_range=None,
                white=None,
                white_range=None,
                output_folder='Backgating',
              roi_obs='ROI',
              x_loc_obs='X_loc',
              y_loc_obs='Y_loc',
              cell_index_obs='Master_Index',
              use_masks=False,
              cells_per_row=5,
              overview_images=True,
              save_subfolder='',
              minimum=0.2,
              max_quantile='q0.97',
              training=False):

    """This function perform a backgating assessment on a given selection of cells as specified by a cell index
    Args:
        adata: adata object
        cell_index: List of cells to visualise
        radius=15: Size of square over cell to visualise

        image_folder: Folder with raw or denoised images. Each ROI has it's own folder.
        {colour}: Channel to use for that colours
        {colour_range}: Format is tuple, with lower and upper range. If not given, will calcuate from quartiles.
        
        roi_obs: adata.obs identifying ROIs,
        x_loc_obs: adata.obs identifying X location,
        y_loc_obs: adata.obs identifying Y location,
        cell_index_obs: Identifier for individual cells,
        use_masks: If specified, is a .csv file that has two ,
        cells_per_row: Cells per row to plot,
        overview_images: Whether or not to save ROI images with cell areas overlayed,
        output_folder: Output folder for saving
        save_subfolder: Specify a subfolder within output folder to save,
        minimum=0.2,
        max_quantile='q0.97',
        training=False
        
    Output:
        Will save everthing to a folder called 'Backgating', with each population having it's own subfolder
    """    
        
    
    
    from skimage import io, segmentation
    from skimage.draw import polygon, rectangle_perimeter
    import os
    import copy

    # If only one cell index is supplied, make it into a list anyway
    if not isinstance(cell_index, list):
        cell_index=[cell_index]
        
    #Extract only the data for cells of index
    adata_obs_cells = adata.obs.loc[adata.obs[cell_index_obs].isin(cell_index),:].copy()
    
    # Get a list of the ROIs to generate images for:
    roi_list = adata_obs_cells[roi_obs].unique().tolist()
                    
    # Create images for all ROIs we're taking cells from
    make_images(image_folder=image_folder, 
                samples_list=roi_list,
                output_folder=output_folder,
                red=red,
                red_range=red_range,
                green=green,
                green_range=green_range,
                blue=blue, 
                blue_range=blue_range,
                magenta=magenta,
                magenta_range=magenta_range,
                cyan=cyan,
                cyan_range=cyan_range,
                yellow=yellow,
                yellow_range=yellow_range,
                white=white,
                white_range=white_range,
                simple_file_names=True,
                minimum=minimum,
                max_quantile=max_quantile,
                save_subfolder=save_subfolder
                )
    
    # Load all images, and the sizes of the images
    images = [io.imread(os.path.join(output_folder,save_subfolder, (x+'.png'))) for x in roi_list]
    y_lengths = [i.shape[0] for i in images]
    x_lengths = [i.shape[1] for i in images]
    
    img_df = pd.DataFrame(zip(roi_list,images,x_lengths,y_lengths),columns=[roi_obs,'image','x_length','y_length']).set_index(roi_obs)
    
    # Map back into the sampled dataframe
    adata_obs_cells['x_max']=adata_obs_cells[roi_obs].map(dict(zip(roi_list,x_lengths)))
    adata_obs_cells['y_max']=adata_obs_cells[roi_obs].map(dict(zip(roi_list,y_lengths)))
    
    # Filter cells which are out of bounds -ie, a box of given radius wont go over the edge of the image        
    adata_obs_cells['in_range']=(adata_obs_cells[x_loc_obs]-radius>=0) & (adata_obs_cells[y_loc_obs]-radius>=0) & ((adata_obs_cells[y_loc_obs]+radius)<=adata_obs_cells['y_max']) & ((adata_obs_cells[x_loc_obs]+radius)<=adata_obs_cells['x_max'])  
    adata_obs_cells_filtered = adata_obs_cells[adata_obs_cells['in_range']==True]
    
    # Count number of exluded cells
    excluded_cell_nums = str(len(adata_obs_cells) - len(adata_obs_cells_filtered))
    cell_nums = len(adata_obs_cells_filtered)
    
    print(f'{excluded_cell_nums} cells out of bounds for plotting, proceeding with plotting ')
    
    if use_masks:
        # Load the mask dictionary
        masks = pd.read_csv(use_masks).set_index(roi_obs)
        
        # Map the mask_paths
        img_df['mask_path']=img_df.index.map(masks['mask_path'].to_dict())
        
        # Load the images into the images_df
        img_df['mask']=[io.imread(x) for x in img_df['mask_path']]  
        
    if overview_images:    
        img_overviews = img_df.copy()

                                   
    # Create the figs and axs 
    rows = rounded_up = -(-cell_nums // cells_per_row)                        
    fig, axs = plt.subplots(rows, cells_per_row, figsize=(10, rows*2), dpi=100)                                                 
                               
    # Create an itterable for axes
    axs_iter = iter(axs.flatten())
                              
    # Create empty list to store cell data frames
    cell_dfs=[]
    
    # Loop through each ROI in turn 
    for roi, image in img_df.iterrows():
                                              
        # Filter down to only the cells from this specific ROI                      
        cells = adata_obs_cells_filtered.loc[adata_obs_cells_filtered[roi_obs]==roi,:].copy()
                                      
        # loop through each of the cells in this ROI
        for i, cell_row in cells.iterrows():
            
            y_cell = int(round(cell_row[y_loc_obs]))    
            x_cell = int(round(cell_row[x_loc_obs]))
            
            thumb=image['image'][(y_cell-radius):(y_cell+radius), (x_cell-radius):(x_cell+radius), :]
            ax = next(axs_iter)            
            ax.imshow(thumb)
            
            if use_masks:
                # Trim mask to right areas
                mask=image['mask'][(y_cell-radius):(y_cell+radius), (x_cell-radius):(x_cell+radius)]
                orig_mask = mask
                
                # Here, we should be able to remove any masks that aren't in the centre of the image
                # Find the pixel value at the centre, corresponding to centre cell
                centre_cell = mask[int(mask.shape[0]/2),int(mask.shape[1]/2)]
                
                # Remove non-centre cells from the mask
                mask = np.where(mask != centre_cell, 0, mask)
                
                # Calculate boundaries
                boundaries = segmentation.find_boundaries(mask, connectivity=1, mode='inner')
                
                # Convert boundaries into a true/false mask
                boundaries = np.ma.masked_where(boundaries == 0, boundaries)
                
                # Display mask
                ax.imshow(boundaries,  interpolation='none', cmap="gray", alpha=1, vmin=0, vmax=1)


            ax.grid(None)
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            ax.set_title(label=f'{roi}--{i}', fontdict={'fontsize':10})
            
            if training:
                fig_temp, ax_temp = plt.subplots(figsize=(5,5))
                ax_temp.imshow(thumb)
                ax_temp.imshow(boundaries,  interpolation='none', cmap="gray", alpha=1, vmin=0, vmax=1)
                ax_temp.grid(None)
                ax_temp.get_xaxis().set_visible(False)
                ax_temp.get_yaxis().set_visible(False)
                ax_temp.set_title(label=f'{roi}--{i}', fontdict={'fontsize':10})
                
                show_figure(fig_temp)
                
                cells.loc[i,'training']=copy(answer)
                
        # Append a list of the viewed cells
        cell_dfs.append(cells.copy())                    
            
            
    # Create the subfolders, if using
    
    save_path=os.path.join(output_folder, save_subfolder)
    
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    if overview_images:

        for roi, image in img_overviews.iterrows():
            
            # Filter down to only the cells from this specific ROI                      
            cells = adata_obs_cells_filtered.loc[adata_obs_cells_filtered[roi_obs]==roi,:].copy()
            
            # loop through each of the cells in this ROI
            for i, cell_row in cells.iterrows():
                
                y_cell = int(round(cell_row[y_loc_obs]))    
                x_cell = int(round(cell_row[x_loc_obs]))
            
                # Get coordinates of a square over cell
                rr, cc = rectangle_perimeter((y_cell - radius, x_cell - radius), extent=(radius*2, radius*2), shape=image['image'].shape)

                # Set the coordinates to white
                img_overviews.loc[roi,'image'][rr, cc, :]=255
                
            io.imsave(os.path.join(save_path, (roi+'_overview.png')), image['image']) 
                
    fig.tight_layout()
    fig.suptitle(f'Population: {save_subfolder}. Red: {red}. Green: {green}. Blue: {blue}')
    fig.show()
    
    fig.savefig(os.path.join(save_path, ('Cells.png')), bbox_inches='tight',dpi=200)
    
    cell_dfs =pd.concat(cell_dfs)
    cell_dfs.to_csv(os.path.join(save_path, 'cells_list.csv'))
    


def backgating_assessment(adata,                          
                          image_folder,
                          pop_obs,
                          cells_per_group=50,
                          radius=15,
                          roi_obs='ROI',
                          x_loc_obs='X_loc',
                          y_loc_obs='Y_loc',
                          cell_index_obs='Master_Index',
                          use_masks='mask_dict.csv',
                          output_folder='Backgating',
                          minimum=0.4,
                          max_quantile='q0.98',
                          markers_exclude=[],
                          only_use_markers=[],
                          number_top_markers=3,
                          mode='full',
                         specify_red=None,
                         specify_green=None,
                         specify_blue=None):
    
    """This function perform a backgating assessment on a supplied adata.obs grouping, usually a populations
    Args:
        adata: adata object
        image_folder: Folder with raw or denoised images. Each ROI has it's own folder.
        pop_obs: adata.obs that identifies populations
        cells_per_group: Cells per group to plot
        radius=15: Size of square over cell to visualise
        roi_obs: adata.obs identifying ROIs,
        x_loc_obs: adata.obs identifying X location,
        y_loc_obs: adata.obs identifying Y location,
        cell_index_obs: Identifier for individual cells,
        use_masks: If specified, is a .csv file that maps ROIs onto paths to masks,
        output_folder: Output folder for saving,
        minimum: Minimum pixel value for images,
        max_quantile: Quantile of signal for max intensity of images,
        markers_exclude: List of markers to exclude when calculating highest expression of markers in group,
        only_use_markers: Supply a list of markers to use when calculating highest expression of markers in group,
        number_top_markers: Number of highest expression of markers, between 1 and 3
        mode:
            full: Will calculate top markers, then immediately backgate
            save_markers: Will only calculate markers then save in a file, which can be reviewed and modified
            load_markers: Loads a saved marker file, then runs backgating           
        specify_red:Specify a channel which all pops will use for red,
        specify_green:Specify a channel which all pops will use for green,
        specify_blue:Specify a channel which all pops will use for blue
        
    Output:
        Will save everthing to output, with each population having it's own subfolder
    """    
    
    

    from IPython.display import display
    import os

    
    if mode == 'full' or mode == 'save_markers':
    
        # Get a table of the mean expression of the different populations
        res = pd.DataFrame(columns=adata.var_names, index=adata.obs[pop_obs].cat.categories)                                                                                                 

        for clust in adata.obs[pop_obs].cat.categories: 
            res.loc[clust] = adata[adata.obs[pop_obs].isin([clust]),:].X.mean(0)

        # Convert to numbers
        res = res.astype('float64')

        # Drop any markers we don't want in the 'top' assessment
        res = res.drop(columns=markers_exclude)

        # If given, only use markers from a specific list
        if only_use_markers != []:
            res=res.loc[:,only_use_markers]

        # Extract top markers in descending order 
        res['top'] = res.apply(lambda x : get_top_columns(x, number_top_markers), axis=1)
                
        # Extract colours from top columns
        if number_top_markers==1:
            res['Red'] = res['top']
            res[['Green','Blue']]=None
                
        elif number_top_markers==2:
            res[['Red','Green']] = res['top'].str.split("__", expand = True)
            res['Blue']=None
                
        elif number_top_markers==3:
            res[['Red','Green','Blue']] = res['top'].str.split("__", expand = True)
                
        # If channels have been specified, select them here
        if specify_red:
            res['Red']=specify_red
        if specify_green:
            res['Green']=specify_green
        if specify_blue:
            res['Blue']=specify_blue
                                    
        # Save to file    
        res.to_csv(os.path.join(output_folder,'markers_used.csv'))
        
    elif mode == 'load_markers':
        
        
        res = pd.read_csv(os.path.join(output_folder,'markers_used.csv'),index_col=0)
        
        # 'None' values are saved as strings, so converting them back into actual None
        for c in ['Red', 'Blue', 'Green']:
            res.loc[res[c]=='None', c]=None
        
                
    if mode == 'full' or mode == 'load_markers':
        
        print('Calculated or loaded markers:')
        display(res[['Red','Green','Blue']])         
        
        for pop, row in res.iterrows():

            print(f'Backgating population: {pop}')

            backgating(adata=adata,
                        cell_index=list(adata[adata.obs[pop_obs]==pop].obs[cell_index_obs].sample(cells_per_group)),
                        radius=radius,
                        image_folder=image_folder,
                        red=row['Red'],
                        green=row['Green'],
                        blue=row['Blue'],
                        use_masks=use_masks,
                        output_folder=output_folder,
                        overview_images=True,
                        minimum=minimum,
                        max_quantile=max_quantile,
                        save_subfolder=clean_text(pop)
                        )

            plt.close()
    else:
        print('Proposed markers:')
        display(res[['Red','Green','Blue']])
            
        return res
        
        
            
def abundance_clustermap(adata,
                         row_obs,
                         col_obs,
                         row_colourmap=None,
                         col_colourmap=None,
                         row_cluster=True,
                         col_cluster=True,
                         ROI_obs='ROI',
                         log10=False,
                         figsize=(10,10),
                         zscore=None,
                         cmap=None, 
                         vmin=None, 
                         vmax=None,
                         save=None,
                         palette=sc.plotting.palettes.zeileis_28
                        ):

    import pandas as pd
    import numpy as np
    import seaborn as sb
    import scanpy as sc

    # Crosstab to get abundances
    plotting = pd.crosstab(adata.obs[row_obs],[adata.obs[col_obs],adata.obs[ROI_obs]],normalize=False)

    if not row_colourmap:
        row_colourmap = dict(zip(adata.obs[row_obs].unique(), 
                                 [palette[x] for x in range(len(adata.obs[row_obs].unique()))]))
        
    if not col_colourmap:
        col_colourmap = dict(zip(adata.obs[col_obs].unique(), 
                                 [palette[x] for x in range(len(adata.obs[col_obs].unique()))]))
                
        
    row_colours = list(plotting.reset_index()[row_obs].map(row_colourmap))
    col_colours = list(plotting.T.reset_index()[col_obs].map(col_colourmap))
    

    if log10:
        plotting= np.log10(plotting)
        plotting.replace([np.inf, -np.inf], 0, inplace=True)

    clustermap = sb.clustermap(plotting,
                                figsize=figsize, 
                                linewidths=0.5,
                                method='single',
                                z_score=zscore,
                                row_cluster=row_cluster,
                                col_cluster=col_cluster,
                                col_colors=col_colours,
                                row_colors=row_colours,
                                cmap=cmap, 
                                vmin=vmin, 
                                vmax=vmax
                 )

    if save:
        clustermap.savefig(save, dpi=200, bbox_inches='tight')                         





        

        