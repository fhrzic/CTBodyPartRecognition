import os
from multiprocessing import Process
from more_itertools import chunked 
import shutil
import sys
from Utils.Conversion import *
from Utils.PatchExport import *


xlsx_dir = "/mnt/HDD/SAROS/SAROS_working_data"
        
def multi_process_app(index:int, paths:list):
    """
    Simple function for multi processing.

    Args:
        * index, int, necessary to avoid colision between proccesses when
        extracting zip files,
        * paths, list, list containing paths to the zips
    """
    print(f"Process {index}, started!")
    for _p, _filename in enumerate(paths):
        # Extract zip to a temp folder
        print(f"Process {index+1} started extraction of {_p+1}/{len(paths)}")
        print(f"Process {index+1} working on {_filename}!")
        _input_dir_path = _filename
        _my_container = patch_container(_input_dir_path)
        _my_container.obtain_raw_blobs(_input_dir_path)
        print(f"Process {index} obtained blobs {_p+1}/{len(paths)}!")
        _my_container.merge_blobs(export_path = _input_dir_path,
                        merging_criteria = "reduced_cluster_remaped")

        _study = os.path.basename(_filename)
        _xlsx_path = os.path.join(xlsx_dir,_study,"results.xlsx")        
        _my_container.filter_blobs(xlsx_path = _xlsx_path,
                          export_path = _input_dir_path)
        _my_container.generate_nifti(export_path = _input_dir_path, file_name = "filtered.nii")
        _my_container.find_biggest_blobs(export_path=_input_dir_path)
        _my_container.generate_nifti(export_path = _input_dir_path, file_name = "biggest.nii")

        _name = os.path.basename(_input_dir_path)

        _input_file = os.path.join(_input_dir_path, f"{_name}_biggest_blobs.json")
        print(f"Process {index} Exporting images {_p+1}/{len(paths)}!")
        _segmentation_volume = os.path.join(_input_dir_path, f"{_name}_filtered.nii")
        _my_container.create_central_patches(biggest_blobs_path = _input_file,
                                            segmentation_volume_path=_segmentation_volume,
                                            width_height_dict = {"min": 128, "max": 400, "n":10})

        print(f"Process {index} finished {_p+1}/{len(paths)}!")

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
    assert 'dir' in kwargs.keys() and \
           'number_of_proc' in kwargs.keys() and \
           'limit_number' in kwargs.keys(),\
        f"Missing arguments dir, number_of_proc, limit_number !!!"
    _dir = kwargs['dir']
    _number_of_processes = int(kwargs['number_of_proc'])
    try:
        _limit = int(kwargs['limit_number'])
    except:
        _limit = None
    # Read zips in zip file
    _files = [os.path.join(_dir, _subdir) for _subdir in os.listdir(_dir) if os.path.isdir(os.path.join(_dir, _subdir))]
    if _limit != None:
        _files = _files[0:_limit]

    # Precussion massure
    if len(_files) < _number_of_processes:
        _number_of_processes = len(_files)
    
    # Split zip files into sublists    
    _sublists = divide_list(_files, _number_of_processes)
    
    # Run through all processes
    for _i, _list in enumerate(_sublists):
        _p = Process(target=multi_process_app, args=(_i,_list))
        _p.start()
    _p.join()

# Main function            
if __name__ == '__main__':
    print(sys.argv)
    main(**dict(arg.split('=') for arg in sys.argv[1:]))
