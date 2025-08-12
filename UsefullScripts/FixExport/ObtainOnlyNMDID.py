import os
import pandas as pd
from tqdm import tqdm
import shutil

def main(path_to_xlsx: str = None,
         dir: str = None,
         export_dir: str = None):
    # Create dir
    # Remove the directory if it exists
    if os.path.exists(export_dir):
        shutil.rmtree(export_dir)  # Removes the export_dir and all its contents

    # Create the export_dir
    os.makedirs(export_dir)  
    
    # Read df
    _df = pd.read_excel(path_to_xlsx)
    _files = _df["Original_paths"]

    # Obtain all files
    _all_files = []
    for root, dirs, files in os.walk(dir):
        for file in files:
            _all_files.append(os.path.join(root, file))


    # Copy
    _all_files_of_interest = []
    for _file in _files:
        _name = _file.split("/")[5]
        _all_files_of_interest.append(_name)
    _all_files_of_interest = set(_all_files_of_interest)
    #print(len(_all_files_of_interest))
    
    # Copy
    for _file in tqdm(_all_files, total = len(_all_files)):
        #print(_file)
        for _case in _all_files_of_interest:
            if _case in _file:
                _dir = os.path.join(export_dir,_case)
                os.makedirs(_dir, exist_ok=True)
                shutil.copy(_file, os.path.join(_dir, os.path.basename(_file)))

if __name__ == "__main__":
    _xlsx = "SourceDataset.xlsx"
    _dir = "Exported_To_X-ray"
    _export_dir = "NMDID_PATCHES"
    main(path_to_xlsx=_xlsx,
         dir = _dir,
         export_dir= _export_dir)