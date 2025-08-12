import pandas as pd
import numpy as np
import os
import SimpleITK as sitk
from Utils.GenerateNPY import *
import cv2
from multiprocessing import Process
from more_itertools import chunked 
import scipy.ndimage

def main(index: int = 0, all_cases_list: list = None, nii_gz_dirs: str = None):
    """
    Main function for feature extraction for full cases.

    Args:
        * index, int , for multi porcessing index (reporting)
        * all_cases_list, list, list conataining all cases which the dedicated process needs to evalaute
        * nii_gz_dir, str, path to the dir where the ni.gz files are stored
    """
    # Obtain list of nifties
    _nifti_list = []
    for _dir in nii_gz_dirs:
        for _root, _, _files in os.walk(_dir):
            for _file in _files:
                if _file.endswith(".nii.gz"):
                    _nifti_list.append(os.path.join(_root, _file))
    
    # Go through cases
    for _i, _case in enumerate(all_cases_list):
        print(f"Proces {index} wokrking on {_i+1}/{len(all_cases_list)}!!!")
        print(os.path.basename(_case))
        _target_name = os.path.basename(_case)
        # Find nifti
        for _nii_file in _nifti_list:
            _study_name = os.path.basename(_nii_file)
            _id = os.path.basename(os.path.dirname(_nii_file))
            _name = "_".join([_id, _study_name]).split(".nii")[0]
            if _name == _target_name:
                _nifti_file = _nii_file
                break
        
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

# Main function   _study_name = os.path.basename(_case)
        _id = os.path.basename(os.path.dirname(_case))
        _name = "_".join([_id, _study_name])
        print(_name)
        
if __name__ == '__main__':
    # Define input dir
    _nii_gz_dirs = ["CT_2D",
                   "Part1_CT_2D",
                   "Part2_CT_2D",
                   "Part3_CT_2D",
                   "Part4_CT_2D"]
    
    _export_dir = "VERIFIED_BCH" 
    _number_of_processes = 30

    # Obtain all cases
    _all_cases_list = [os.path.join(_export_dir, _f) for _f in os.listdir(_export_dir)] 
    # Split zip files into sublists    
    _sublists = divide_list(_all_cases_list, _number_of_processes)

    
    # Run through all processes
    for _i, _list in enumerate(_sublists):
        _p = Process(target=main, args=(_i,_list, _nii_gz_dirs))
        _p.start()
    _p.join()
