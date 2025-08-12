import pandas as pd
import numpy as np
import os
import SimpleITK as sitk
from Utils.GenerateNPY import *
import cv2
from multiprocessing import Process
from more_itertools import chunked 
import scipy.ndimage


def main(index: int = 0, all_cases_list: list = None):
    """
    Script which generates numpy array from ct and stores it properly in the folder

    Args:
        * index, int , for multi porcessing index (reporting)
        * all_cases_list, list, list conataining all cases which the dedicated process needs to evalaute
    Returns:
    """
    
    # Export Case to numpy array
    print(f"Proces {index} started!!!")
    for _i, _case_path in enumerate(all_cases_list):
        print(f"Proces {index} wokrking on {_i+1}/{len(all_cases_list)}!!!")
        # Obtain .nii.gz file
        _nifti_file = [os.path.join(_case_path, _f) for _f in os.listdir(_case_path) if _f.endswith('.nii.gz')][0]

        # Load it
        _main_volume = sitk.ReadImage(_nifti_file)

        # Obtain original volume
        _main_coronal_volume, _main_reduced_volume = process_volume(volume = _main_volume,
                                                                    series_info_dict_path = _nifti_file.replace(".nii.gz", ".json"))

        # Export main volume to _NPY
        crop_and_export_to_NPY(_main_coronal_volume, 
                               _main_reduced_volume, 
                               _case_path)

        # Obtain main image
        _image_file = [os.path.join(_case_path, _f) for _f in os.listdir(_case_path) if _f.endswith('reducted_image.png')][0]
        _main_image = cv2.imread(os.path.join(_case_path, _image_file), cv2.IMREAD_GRAYSCALE)
        print(_image_file)
        
        # Go through images if egsists
        print(f"Proces {index} working on patches {_i+1}/{len(all_cases_list)}!!!")
        if os.path.exists(os.path.join(_case_path, "images")):
            # Obtain files
            _patches_images_paths = []
            for _dirpath, _dirnames, _filenames in os.walk(os.path.join(_case_path, "images")):
                for _filename in _filenames:
                    if _filename == "main.png":
                        _full_path = os.path.join(_dirpath, _filename)
                        _patches_images_paths.append(_full_path)

            # Obtain positions
            for _patch_path in _patches_images_paths:
                # Load images
                _main_image = cv2.imread(os.path.join(_case_path, _image_file), cv2.IMREAD_GRAYSCALE)
                _patch = cv2.imread(_patch_path, cv2.IMREAD_GRAYSCALE)

                # Obtain crop cordinates
                _x, _y = obtain_crop_cordinates_2D(main_image = _main_image,
                                                   patch = _patch)
                
                crop_and_export_to_NPY(volume_coronal = _main_coronal_volume,
                                       volume_reducted = _main_reduced_volume,
                                       export_dir = os.path.dirname(_patch_path),
                                         crop_cordinates = [_y, _x])

        print(f"Proces {index} Finished {_i+1}/{len(all_cases_list)}!!!")
    print(f"Proces {index} finished!!!")


# Main Function
if __name__ == '__main__':
    # Define input dir
    _export_dir = "FIXExport/FAIL"
    _number_of_processes = 30

    # Obtain all cases
    _all_cases_list = [os.path.join(_export_dir, _f) for _f in os.listdir(_export_dir)] 
    
    # Split zip files into sublists    
    _sublists = divide_list(_all_cases_list, _number_of_processes)

    
    # Run through all processes
    for _i, _list in enumerate(_sublists):
        _p = Process(target=main, args=(_i,_list))
        _p.start()
    _p.join()

