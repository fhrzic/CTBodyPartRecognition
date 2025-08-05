import pandas as pd
import numpy as np
import os
import json
    
def main(export_dir: str = None):
    """
    Main script for for extracting data - generating results.xlsx in each
    folder.

    Args:
        * export_dir, str, path to the export dir which contains folder where
        every folder represent one case which requires json file
    """
    # Define json

    projection = {"series_tags": {"0020|0037": ["1\\0\\0\\0\\1\\0", "ImageOrientationPatient"]}}
    
    # Obtain folders
    _folder_list = [os.path.join(export_dir, _f) for _f in os.listdir(export_dir)]

    # Save
    for _folder in _folder_list:
        # Obtain case name
        _case_name = os.path.basename(_folder)
        
        # Dump data
        with open(os.path.join(_folder, f"{_case_name}.json"), "w") as _outfile:
            json.dump(projection, _outfile)
        

if __name__ == '__main__':
    main(export_dir = "/mnt/HDD/SAROS/SAROS_working_data")
