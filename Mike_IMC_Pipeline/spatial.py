import pandas as pd
import numpy as np
import seaborn as sns
import anndata as ad
import os
import shutil

def run_spoox(adata, 
             population_obs, 
             groupby=None, 
             samples=None,
             specify_functions=None,
             spoox_output_dir='spooxout', 
             spoox_output_summary_dir='spooxout_summary', 
             output_file='stats.txt',
             index_obs='Master_Index',
             roi_obs='ROI',
             xloc_obs='X_loc',
             yloc_obs='Y_loc',
             masks_source_directory='masks',
             masks_destination_directory='spoox_mask',
             run_analysis=True,
             analyse_samples_together=True,
             summary=True):
    
    '''
    Run the SpOOx analyis pipeline from an AnnData object.
    More information can be found here: https://github.com/Taylor-CCB-Group/SpOOx/tree/main/src/spatialstats
    
    Any errors in running functions will be saved in errors.csv

    Parameters
    ----------
    adata
        AnnData object, or path (string) to a saved h5ad file.
    population_obs
        The .obs that defines the population for each cell
    groupby
        If specifed, should be a .obs that identifies different groups in the data.
        In the summary step, it will then compare 
    samples
        Specify a list of samples, if None will process all samples
    specify_functions
        By default will run the follow functions from the Spoox pipeline: paircorrelationfunction morueta-holme networkstatistics 
        This is a complete list that will be run if 'all' is used: 
            paircorrelationfunction
            localclusteringheatmaps
            celllocationmap
            contourplots
            quadratcounts
            quadratcelldistributions
            morueta-holme
            networkstatistics
    run_analysis
        Whether or not to run the analyis, or just create the spatialstats file
    analyse_samples_together
        Whether to analyse all samples together
    summary
        Whether to run summary script
        
    Returns
    ----------
    Creates two folders with outputs of SpOOx pipeline
        
    '''
    
    # Load from file if given a string
    if type(adata) == str: 
        adata = ad.read_h5ad(adata)
        
    if not samples:
        samples = adata.obs[roi_obs].unique().tolist()
        print(f'Following samples found in {roi_obs}:')
        print(samples)
    else:
        print(f'Only analysing these samples from {roi_obs}:')
        print(samples)
        
    # Specify functions to run, by default will run all
    if specify_functions=='all':
        functions=''
    elif specify_functions:
        functions=f' -f {specify_functions}'    
    else:
        functions=' -f paircorrelationfunction morueta-holme networkstatistics'

    # Copy over masks into correct format
    if os.path.isdir(masks_destination_directory):
        print('Spoox_masks directory already exists')
    else:
    
        # Create a copy of the original directory
        shutil.copytree(masks_source_directory, masks_destination_directory)

        # Get the list of files in the copied directory
        files = os.listdir(masks_destination_directory)

        for filename in files:
            if os.path.isfile(os.path.join(masks_destination_directory, filename)):
                # Create a subdirectory with the same name as the file
                subdir = os.path.join(masks_destination_directory, os.path.splitext(filename)[0])
                os.makedirs(subdir, exist_ok=True)

                # Move the file to the subdirectory and rename it
                new_filepath = os.path.join(subdir, "deepcell.tif")
                shutil.move(os.path.join(masks_destination_directory, filename), new_filepath)
                #print(f"Moved file: {filename} -> {new_filepath}")
        
        print(f'Created spoox compatible masks in directory: {masks_destination_directory}')
    
    # Create output folders
    os.makedirs(spoox_output_dir, exist_ok = True)
    os.makedirs(spoox_output_summary_dir, exist_ok = True)

    spatial_stats = adata.obs.copy()
    
    # Create spatial stats dataframe from adata.obs
    cols = [index_obs, roi_obs,xloc_obs, yloc_obs, population_obs]
    cols_rename = {index_obs:'cellID',roi_obs:'sample_id', xloc_obs:'x', yloc_obs:'y', population_obs:'cluster'}
    #print(cols)
    
    if groupby:
        cols.append(str(groupby))
        cols_rename.update({groupby: 'Conditions'})
    
   # print(cols)
    spatial_stats = spatial_stats[cols]
    spatial_stats = spatial_stats.rename(columns=cols_rename)
        

    # This may not be neded, just 'label'
    #spatial_stats.cellID = spatial_stats.cellID.astype('int') + 1

    spatial_stats.cellID = 'ID_' + spatial_stats.cellID

    # Add a label column that will match cell numbers to their labels in the mask files (hopefully!)
    for i in spatial_stats.sample_id.unique().tolist():

        df_location = spatial_stats.loc[spatial_stats.sample_id==i, 'cellID']
        spatial_stats.loc[spatial_stats.sample_id==i, 'label'] = [int(x+1) for x in range(0,df_location.shape[0])]

    spatial_stats['label'] = spatial_stats['label'].astype('int')

    spatial_stats.to_csv(output_file, sep='\t')
    print(f'Saved to file: {output_file}')

    ### Create conditions file
    ################

    if groupby:
    
        result = {}

        for roi, niches in zip(adata.obs[~adata.obs.duplicated(roi_obs)][roi_obs], adata.obs[~adata.obs.duplicated(roi_obs)][groupby]):
            if niches in result:
                result[niches].append(roi)
            else:
                result[niches] = [roi]

        formatted_result = {"conditions": result}
        print(formatted_result)


        import json

        with open('conditions.json', 'w') as json_file:
            json.dump(formatted_result, json_file, indent=1)              
    
    
    
    ### Run analysis
    ################
    
    if run_analysis:
    
        if not analyse_samples_together:
        
            # Run each sample individually
            for s in samples:

                spatial_stats_sample = spatial_stats[spatial_stats.sample_id==s]
                spatial_stats_sample.to_csv('sample.txt', sep='\t')
                print(f'Running sample {s}...')

                #command = "python spatialstats\\spatialstats.py -i stats.txt -o spooxout -d nf2_masks -cl cluster -f quadratcounts quadratcelldistributions paircorrelationfunction morueta-holme networkstatistics"
                command = f"python spatialstats\\spatialstats.py -i sample.txt -o {spoox_output_dir} -d {masks_destination_directory} -cl cluster{functions}"

                os.system(command)
        else:
            
            #Run all samples together
            spatial_stats_sample = spatial_stats[spatial_stats.sample_id.isin(samples)]
            spatial_stats_sample.to_csv('sample.txt', sep='\t')
            print(f'Running {str(len(samples))} samples...')

            #command = "python spatialstats\\spatialstats.py -i stats.txt -o spooxout -d nf2_masks -cl cluster -f quadratcounts quadratcelldistributions paircorrelationfunction morueta-holme networkstatistics"
            command = f"python spatialstats\\spatialstats.py -i sample.txt -o {spoox_output_dir} -d {masks_destination_directory} -cl cluster{functions}"

            os.system(command)            
                      
        
        if groupby:

            print('Calculating group averages...')

            command = f"python spatialstats\\average_by_condition.py -i stats.txt -p {spoox_output_dir} -o {spoox_output_summary_dir} -cl cluster -j conditions.json"

            os.system(command)
            
            
        if summary:
        
            print('Summarising data...')

            command = f'python spatialstats\\summary.py -p {spoox_output_dir}'

            os.system(command)