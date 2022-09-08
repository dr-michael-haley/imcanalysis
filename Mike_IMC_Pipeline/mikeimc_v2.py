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
        row = row.lower()
        
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
        adata = sc.AnnData(markers_normalised)
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
                
    
def pop_stats(adata_plotting,groups,Case_id,ROI_id,x_axis,display_tables=True,fig_size=(5,5), confidence_interval=68,save=False, log_scale=True):

    import seaborn as sb
    import pandas as pd
    import statsmodels as sm
    import scipy as sp
    import matplotlib.pyplot as plt 
    
    cells = pd.crosstab([adata_plotting.obs[groups], adata_plotting.obs[Case_id], adata_plotting.obs[ROI_id]],adata_plotting.obs[x_axis])
    cells.columns=cells.columns.astype('str')

    cells_long = cells.reset_index().melt(id_vars=[groups,Case_id,ROI_id])
    cells_long.columns=cells_long.columns.astype('str')

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
        sh.neigh.interactions(so_copy, s, o, mode=m, prediction_type='diff', graph_key=g)
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

def analyse_cell(raw_image, size0, size1, radius, cell_axis0, cell_axis1):
    """ Used to multithread extraction of regions around cells""" 
    try:
    
        # Create a circle mask
        cell_mask = np.zeros((size0, size1))
        rr, cc = disk((cell_axis0,cell_axis1),radius)
        cell_mask[rr, cc] = 1

        # Convert to a label
        cell_label = label(cell_mask)
        cell_properties = regionprops(cell_label,raw_image, cache=False)

        # Return the area intensity
        return cell_properties[0].intensity_mean
    
    except:
        
       #  Return NaN if can't perform, usually if the circle goes over edge of image
       return np.nan
        
        
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
                        cmap='coolwarm'):

    import seaborn as sb
    import matplotlib.pyplot as plt 
    from scipy.stats import ttest_1samp
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
            stats_row = results_df.reset_index()[results_df.index==i]['diff']
            pvalue = ttest_1samp(stats_row,0).pvalue
            pvalues.append(pvalue)

        summary['pvalue']=pvalues

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
        sb.heatmap(data=df1, cmap=cmap, robust=True,ax=ax,linewidths=.5,annot=sig_array, fmt="",
                   annot_kws=annot_kws,
                  cbar_kws=cbar_kws)      
    else:
        sb.heatmap(data=df1, cmap=cmap, robust=True,ax=ax,linewidths=.5, cbar_kws=cbar_kws)
    
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
