import datetime 
from types import ModuleType
import pandas as pd
from os import getlogin


def compare_lists(L1, L2, L1_name, L2_name, return_error=True):
    '''
    Function for thoroughly comparing lists
    '''
                  
    L1_not_L2 = [x for x in L1 if x not in L2]
    L2_not_L1 = [x for x in L2 if x not in L1]    
    
    try:
        assert L1_not_L2 == L2_not_L1, "Lists did not match:"
    except:                
        print(f'{L1_name} items NOT in {L2_name}:')
        print(L1_not_L2)
        print(f'{L2_name} items NOT in {L1_name}:')
        print(L2_not_L1)
        
        if return_error:
            raise TypeError(f"{L1_name} and {L2_name} should have exactly the same items")
            
        
def print_full(x):
    '''
    Prints a full pandas table
    '''
    pd.set_option('display.max_rows', len(x))
    display(x)
    pd.reset_option('display.max_rows')

    

def adlog(adata, 
              entry=None, 
              module=None,
              module_name=None,
              module_version=None,
              save=False,
              temp_file='adata_temp.h5ad'):
    '''
    This function saves a log in the AnnData
    '''
    
    
    # Get user login
    try:
        login_user = getlogin()
    except:
        login_user = 'Unknown'

    # Get time and date
    now = datetime.datetime.now()
    time_now = now.strftime('%Y-%m-%d')
    date_now = now.strftime('%H:%M:%S')

    # Check if log exists, create if not
    try: 
        log = adata.uns['logging']
    except:
        print('No log in AnnData, creating new log')
        log = pd.DataFrame(columns=['Date','Time','Entry','Module','Version', 'User'])
        adata.uns.update({'logging':log})
        
    # If a module is supplied, get its name and version number
    if isinstance(module, ModuleType):
        module_name = module.__name__
        
        try:
            module_version = module.__version__
        except:
            module_version = None
    
    # Otherwise just use raw inputs
    elif isinstance(module, str):
        module_name = module

         
    # Add new entry, if given
    if entry:
        log.loc[len(log.index)] = [str(x) for x in [date_now, time_now, entry, module_name, module_version, login_user]]
        
    if save:
        
        # Get time and date
        now = datetime.datetime.now()
        time_now = now.strftime('%Y-%m-%d')
        date_now = now.strftime('%H:%M:%S')
        
        print(f'Saving temporary file: {temp_file}')
        adata.write(temp_file)
        log.loc[len(log.index)] = [str(x) for x in [date_now, time_now, f'Saved AnnData backup: {temp_file}', None, None, login_user]]

        