import zipfile
import os
from Utils.Containers import *
import matplotlib.pyplot as plt
from Utils.Conversion import *
from multiprocessing import Process
from more_itertools import chunked 
import shutil
import sys

def multi_process_app(index:int, zip_paths:list, results_dir:str):
    """
    Simple function for multi processing.

    Args:
        * index, int, necessary to avoid colision between proccesses when
        extracting zip files,
        * zip_paths, list, list containing paths to the zips
        * results_dir, str, path to results export dir
    """
    print(f"Process {index}, started!")
    for _p, _zip_filename in enumerate(zip_paths):
        # Extract zip to a temp folder
        print(f"Process {index+1} started extraction of {_p+1}/{len(zip_paths)}")
        _zip_export_dir = extract_zip(input_zip_path = _zip_filename, 
                                        target_dir_path = ".",
                                        preserve_dir_structure = True,
                                        verbose = True)
    
        print(f"Process {index+1} finished extracting at {_p+1}/{len(zip_paths)}!")

        # Obtain series
        print(f"Process {index+1} started processing series of {_p+1}/{len(zip_paths)}")
        _container = SeriesContainer(verbose = False)
        _container.obtain_data_from_series(path_dir = _zip_export_dir, mode='o')
        print(f"Process {index+1} finished processing series of {_p+1}/{len(zip_paths)}")
        
        # Obtain Case number
        _case_number = [_s for _s in _zip_filename.split(os.sep) if "case" in _s][0]
        _case_number = os.path.splitext(_case_number)[0]

        # Extract data
        print(f"Process {index+1} started exporting data of {_p+1}/{len(zip_paths)}")
        _container.export_data_to_directory(export_dir_path = os.path.join(results_dir, _case_number), 
                                            export_npy = False )
        print(f"Process {index+1} finished exporting data of {_p+1}/{len(zip_paths)}")
        
        # Extract data
        print(f"Process {index+1} started exporting images of {_p+1}/{len(zip_paths)}")
        _container.export_images_to_directory(export_dir_path = os.path.join(results_dir, _case_number),
                                              resample_spacing = [0.5, 0.5, 0.5])
        print(f"Process {index+1} finished exporting images of {_p+1}/{len(zip_paths)}")
        
        # Remove dir for no confusion
        shutil.rmtree(_zip_export_dir)
        while os.path.exists(_zip_export_dir): # check if it exists
            pass
        #Info
        print(f"Process {index} finished {_p+1}/{len(zip_paths)}!")

    print(f"Process {index}, finished!")

def divide_list(input_list:list, number_of_sublists: int)->list:
    """
    Divide list in equal sublists

    Args:
        * input_list, list, list which must be divede into chunks
        * number_of_sublists, int, number represetning number of chunks

    Returns:
        * sublists, list, list of sublist 
    """
    # Calculate the number of elements per sublist
    _elements_per_sublist = len(input_list) // number_of_sublists
    _remainder = len(input_list) % number_of_sublists

    # Initialize starting index and result list
    _start = 0
    _sublists = []

    # Iterate over each sublist
    for _i in range(number_of_sublists):
        # Calculate the sublist size considering the remainder
        _sublist_size = _elements_per_sublist + (1 if _i < _remainder else 0)
        
        # Append sublist to the result list
        _sublists.append(input_list[_start:_start+_sublist_size])
        
        # Update the starting index for the next sublist
        _start += _sublist_size

    return _sublists


def main(**kwargs):
    # Obtain necesarry data
    assert 'zips_dir' in kwargs.keys() and \
        'results_dir' in kwargs.keys() and \
            'number_of_proc' in kwargs.keys() and \
                'limit_number' in kwargs.keys(),\
        f"Missing arguments zips_dir, results_dir, number_of_proc, limit_number !!!"
    _zips_dir = kwargs['zips_dir']
    _results_dir = kwargs['results_dir']
    _number_of_processes = int(kwargs['number_of_proc'])
    try:
        _limit = int(kwargs['limit_number'])
    except:
        _limit = None
    # Read zips in zip file
    _zip_files = [os.path.join(_zips_dir,_filename) for _filename in os.listdir(_zips_dir) if _filename.endswith('.zip')]
    if _limit != None:
        _zip_files = _zip_files[0:_limit]

    # Precussion massure
    if len(_zip_files) < _number_of_processes:
        _number_of_processes = len(_zip_files)
    
    # Split zip files into sublists    
    _zip_sublists = divide_list(_zip_files, _number_of_processes)

    # Create results dir
    shutil.rmtree(_results_dir, ignore_errors=True)
    os.makedirs(_results_dir)

    
    # Run through all processes
    for _i, _list in enumerate(_zip_sublists):
        _p = Process(target=multi_process_app, args=(_i,_list, _results_dir))
        _p.start()
    _p.join()

# Main function            
if __name__ == '__main__':
    print(sys.argv)
    main(**dict(arg.split('=') for arg in sys.argv[1:]))
