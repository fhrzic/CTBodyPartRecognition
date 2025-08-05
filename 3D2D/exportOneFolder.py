import zipfile
import os
from Utils.Containers import *
import matplotlib.pyplot as plt
from Utils.Conversion import *
from multiprocessing import Process
from more_itertools import chunked 
import shutil
import sys

def process_app(input_path:str, results_dir:str):
    """
    Simple function for multi processing.

    Args:
        * input_path, str, path to the input folder
        * results_dir, str, path to results export dir
    """
    print(f"Process started!")

    # Extract zip to a temp folder
    print(f"Process started extraction of given path!")
    
    # Obtain series
    print(f"Process started processing series!")
    _container = SeriesContainer(verbose = True)
    _container.obtain_data_from_series(path_dir = input_path, mode='o')
    print(f"Process finished processing series of")
    
    # Obtain Case number
    _case_number = [_s for _s in input_path.split(os.sep)][-1]
    

    # Extract data
    print(f"Process started exporting data!")
    _container.export_data_to_directory(export_dir_path = os.path.join(results_dir, _case_number), 
                                        export_npy = False )
    print(f"Process finished exporting data!")
    
    # Extract data
    print(f"Process started exporting images!")
    _container.export_images_to_directory(export_dir_path = os.path.join(results_dir, _case_number),
                                            resample_spacing = [0.5, 0.5, 0.5])
    print(f"Process finished exporting images !")
    
    print(f"Process finished!")

def main(**kwargs):
    # Obtain necesarry data
    assert 'input_dir' in kwargs.keys() and \
        'results_dir' in kwargs.keys() and \
        f"Missing arguments input_dir, results_dir!!!"
    _input_dir = kwargs['input_dir']
    _results_dir = kwargs['results_dir']
    # Read zips in zip file

    # Create results dir
    shutil.rmtree(_results_dir, ignore_errors=True)
    os.makedirs(_results_dir)

    
    # Run through all processes
    process_app(_input_dir, _results_dir)

# Main function            
if __name__ == '__main__':
    print(sys.argv)
    main(**dict(arg.split('=') for arg in sys.argv[1:]))
