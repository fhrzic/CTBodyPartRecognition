import pandas as pd
import numpy as np
import os
import SimpleITK as sitk
from Utils.GenerateNPY import *
import cv2
from multiprocessing import Process
from more_itertools import chunked 
import scipy.ndimage

def main(index: int = 0, all_cases_list: list = None, nii_gz_dir: str = None):
    """
    Main function for feature extraction for full cases.

    Args:
        * index, int , for multi porcessing index (reporting)
        * all_cases_list, list, list conataining all cases which the dedicated process needs to evalaute
        * nii_gz_dir, str, path to the dir where the ni.gz files are stored
    """
    
    # Go through cases
    for _i, _case in enumerate(all_cases_list):
        print(f"Proces {index} wokrking on {_i+1}/{len(all_cases_list)}!!!")
        _id_study_name = os.path.basename(_case)
        _id = _id_study_name.split("_exp")[0]
        print(_id)
        _nifti_file = os.path.join(nii_gz_dir, _id, f"{_id_study_name}.nii.gz")
        print(_nifti_file)
        print("--------------------------------------------------------------------------------------------------")
                
        # Load it
        _main_volume = sitk.ReadImage(_nifti_file)

        # Obtain original volume
        _main_coronal_volume, _main_reduced_volume = process_volume(volume = _main_volume,
                                                                    series_info_dict_path = _nifti_file.replace(".nii.gz", ".json"))

        # Export main volume to _NPY
        crop_and_export_to_NPY(_main_coronal_volume, 
                               _main_reduced_volume, 
                               _case)

# Main function
if __name__ == '__main__':
    # Define input dir
    _nii_gz_dir = "SAROS_working_data"
    _export_dir = "Full_SAROS" 
    _number_of_processes = 30

    # Obtain all cases
    _all_cases_list = [os.path.join(_export_dir, _f) for _f in os.listdir(_export_dir)] 
    # Split zip files into sublists    
    _sublists = divide_list(_all_cases_list, _number_of_processes)

    
    # Run through all processes
    for _i, _list in enumerate(_sublists):
        _p = Process(target=main, args=(_i,_list, _nii_gz_dir))
        _p.start()
    _p.join()
