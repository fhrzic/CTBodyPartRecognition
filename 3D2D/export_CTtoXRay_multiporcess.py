import zipfile
import os
from Utils.Conversion import *
from multiprocessing import Process
from more_itertools import chunked 
import shutil
import sys

def multi_process_app(index:int, paths: list, _final_results_output_dir:str):
    """
    Simple function for multi processing.

    Args:
        * index, int, necessary to avoid colision between proccesses when
        extracting zip files,
        * paths, list, list containing paths to the nii.gz
        * results_dir, str, path to results export dir
    """
    
    # Info
    print(f"Process {index}, started!")
    
    # Export
    export_nifti_to_dirs_from_list(input_dir_path = paths,
                    export_dir_path = _final_results_output_dir,
                    verbose = True)
    # Info
    print(f"Process {index+1}, finished!")



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
        'axial_output_dir' in kwargs.keys() and \
        'final_results_output_dir' in kwargs.keys() and \
        'number_of_proc' in kwargs.keys() and \
        'limit_number' in kwargs.keys(),\
        f"Missing arguments zips_dir, results_dir, number_of_proc, limit_number !!!"
    
    # Parse it
    _input_dir = "AXIAL_CTs" # kwargs['input_dir']
    _axial_output_dir = "Axial_CTs" #kwargs['axial_output_dir']
    _final_results_output_dir = "Exported_To_X-ray" # kwargs['final_results_output_dir']
        _number_of_processes = int(kwargs['number_of_proc'])
    
    try:
        _limit = int(kwargs['limit_number'])
    except:
        _limit = None
    
    # Obtain nifti files
    _nifti_file_list = []
    for _root, _dirs, _files in os.walk(_input_dir):
        for _file in _files:
            if _file.endswith('.nii.gz'):
                _nifti_file_list.append(os.path.join(_root, _file))

    # In this case, the CT's are already exported to axial so no need to do it again.
    print(f"Found {len(_nifti_file_list)} in {_input_dir}")

    if _axial_output_dir != "None":
        # Export axial
        for _i, _nifti_file in enumerate(_nifti_file_list):
            print(f"Exporting file {_i+1}/{len(_nifti_file_list)}")
            _name = os.path.basename(_nifti_file)
            export_nifti_to_axial(_nifti_file, os.path.join(_axial_output_dir, _name))

        # Update list
        _nifti_file_list = []
        for _root, _dirs, _files in os.walk(_axial_output_dir):
            for _file in _files:
                if _file.endswith('.nii.gz'):
                    _nifti_file_list.append(os.path.join(_root, _file))

    # Precussion massure
    if len(_nifti_file_list) < _number_of_processes:
        _number_of_processes = len(_nifti_file_list)
    

    # Split zip files into sublists    
    _sublists = divide_list(_nifti_file_list, _number_of_processes)

    
    # Run through all processes
    for _i, _list in enumerate(_sublists):
        _p = Process(target=multi_process_app, args=(_i,_list, _final_results_output_dir))
        _p.start()
    _p.join()

# Main function            
if __name__ == '__main__':
    print(sys.argv)
    main(**dict(arg.split('=') for arg in sys.argv[1:]))
