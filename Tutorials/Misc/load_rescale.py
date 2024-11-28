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
        mode='mean_quantile'
    elif str(max_val)[0]=='i':
        max_quantile=float(str(max_val)[1:])
        mode='individual_quantile'
    elif str(max_val)[0]=='m':
        max_quantile=float(str(max_val)[1:])
        mode='minimum_quantile'
    elif str(max_val)[0]=='x':
        max_quantile=float(str(max_val)[1:])
        mode='max_quantile'
        
    # Load the imaes
    image_list, _, folder_list = load_imgs_from_directory(image_folder,marker,quiet=True)

    # Get the list of ROIs
    roi_list = [os.path.basename(Path(x)) for x in folder_list]
        
    # Filter out any samples not in the samples list
    sample_filter = [x in samples_list for x in roi_list]
    image_list = list(compress(image_list, sample_filter))
    roi_list = list(compress(roi_list, sample_filter))
    
    # Calculate the value at which to cap off the staining by taking the average of the max quantile value    
    if mode=='mean_quantile':
        max_value = [np.quantile(i, max_quantile) for i in image_list]    
        max_value = np.array(max_value).mean()
        print(f'Marker: {marker}, Mode: {mode}. Min value: {minimum}, Quantile: {max_quantile}, Calculated max value: {max_value}')
    elif mode=='individual_quantile':
        max_value = [np.quantile(i, max_quantile) for i in image_list]    
        print(f'Marker: {marker}, Mode: {mode}. Min value: {minimum}, Quantile: {max_quantile}, Max value specific of for each image.')
    elif mode=='minimum_quantile':
        max_value = [np.quantile(i, max_quantile) for i in image_list]    
        max_value = np.array(max_value).min()
        print(f'Marker: {marker}, Mode: {mode}. Min value: {minimum}, Quantile: {max_quantile}, Calculated max value: {max_value}')
    elif mode=='max_quantile':
        max_value = [np.quantile(i, max_quantile) for i in image_list]    
        max_value = np.array(max_value).max()
        print(f'Marker: {marker}, Mode: {mode}. Min value: {minimum}, Quantile: {max_quantile}, Calculated max value: {max_value}')
    else:
        max_value = max_val
        print(f'Marker: {marker}, Min value: {minimum},  Max value: {max_value}')
    
    # Clip
    if mode != 'individual_quantile':
        image_list = [i.clip(np.float64(minimum), np.float64(max_value)) for i in image_list]
    else:
        image_list = [i.clip(np.float64(minimum), np.float64(x)) for (i,x) in zip(image_list, max_value)] 
    
    # Rescale intensity
    image_list = [exposure.rescale_intensity(i) for i in image_list]

    return image_list, roi_list