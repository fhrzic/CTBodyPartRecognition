import zipfile
import os
from Utils.Containers import *
import matplotlib.pyplot as plt
from Utils.Conversion import *
from multiprocessing import Process
from more_itertools import chunked 
import shutil
import sys

def multi_process_app(index:int, input_path_list:list, results_dir:str):
    """
    Simple function for multi processing.

    Args:
        * index, int, necessary to avoid colision between proccesses when
        extracting files,
        * input_path, list, path to the input folder
        * results_dir, str, path to results export dir
    """
    print(f"Process  {index} started!")

    # Extract zip to a temp folder
    print(f"Process {index} started extraction of given path!")
    
    # Obtain series
    print(f"Process {index} started processing files!")
    for _index, _input_path in enumerate(input_path_list):
        print(f"Process {index} working on {_index}/{len(input_path_list)}!")    
        _container = SeriesContainer(verbose = True)
        _container.obtain_data_from_series(path_dir = _input_path, mode='o')
        print(f"Process {index} finished processing series of")
        
        # Obtain Case number
        _case_number = [_s for _s in _input_path.split(os.sep)][-1]
        

        # Extract data
        print(f"Process {index} started exporting data ({_index}/{len(input_path_list)})!")
        _container.export_data_to_directory(export_dir_path = os.path.join(results_dir, _case_number), 
                                            export_npy = False )
        print(f"Process {index} finished exporting data ({_index}/{len(input_path_list)})!")
        
        # Extract data
        print(f"Process {index} started exporting images ({_index}/{len(input_path_list)})!")
        _container.export_images_to_directory(export_dir_path = os.path.join(results_dir, _case_number),
                                                resample_spacing = [0.5, 0.5, 0.5])
        print(f"Process {index} finished exporting images ({_index}/{len(input_path_list)})!")
        
    print(f"Process {index} finished!")


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
    assert 'input_dir' in kwargs.keys() and \
        'results_dir' in kwargs.keys() and \
        'number_of_proc' in kwargs.keys() and \
        'limit_number' in kwargs.keys(),\
        f"Missing arguments zips_dir, results_dir, number_of_proc, limit_number !!!"
    _number_of_processes = int(kwargs['number_of_proc'])

    try:
        _limit = int(kwargs['limit_number'])
    except:
        _limit = None
    
    _input_dir = kwargs['input_dir']
    _results_dir = kwargs['results_dir']

    # Obtain files
    _files_from_input_dir = [os.path.join(_input_dir,_filename) for _filename in os.listdir(_input_dir)]

    # Parse only few files (limited number of them)
    if _limit != None:
        _input_dir = _input_dir[0:_limit]


    # Split input dirs into sublists
    _paths_sublists = divide_list(_files_from_input_dir, _number_of_processes)

    # Create results dir
    shutil.rmtree(_results_dir, ignore_errors=True)
    os.makedirs(_results_dir)

    # Run through all processes
    for _i, _list in enumerate(_paths_sublists):
        _p = Process(target=multi_process_app, args=(_i,_list, _results_dir))
        _p.start()
    _p.join()

# Main function            
if __name__ == '__main__':
    print(sys.argv)
    main(**dict(arg.split('=') for arg in sys.argv[1:]))
