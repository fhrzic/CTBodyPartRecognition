import os
import shutil
from tqdm import tqdm
def list_directories(directory):
    """
    Lists only the directories directly under the specified directory.

    Args:
        directory (str): Path to the directory to process.

    Returns:
        list: A list of directories directly under the given directory.
    """
    if not os.path.isdir(directory):
        print(f"The path {directory} is not a valid directory.")
        return []

    try:
        directories = [os.path.join(directory, d) for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d))]
        return directories
    except Exception as e:
        print(f"Error accessing directory {directory}: {e}")
        return []

def main(input_dir_all: str = None,
         input_dir_rest: str = None,
         export_dir:str = None):

        # Find names to be exported
        _solved_cases = list_directories(export_dir)

        # Names
        _solved_cases_names = []
        for _item in _solved_cases:
            _solved_cases_names.append(os.path.basename(_item))
        
        # Obtain ones which needs to be exported
        _unsolved_cases = list_directories(input_dir_rest)

        # Names
        _unsolved_cases_names = []
        for _item in _unsolved_cases:
            _unsolved_cases_names.append(os.path.basename(_item))

        _unsolved_cases_names = set(_unsolved_cases_names) - set(_solved_cases_names)


        # Obtain all files
        # Obtain all files
        _all_files = []
        for root, dirs, files in os.walk(input_dir_all):
            for file in files:
                _all_files.append(os.path.join(root, file))

        # Copy
        for _file in tqdm(_all_files, total = len(_all_files)):
            #print(_file)
            for _case in _unsolved_cases_names:
                if _case in _file:
                    _dir = os.path.join(export_dir,_case)
                    os.makedirs(_dir, exist_ok=True)
                    shutil.copy(_file, os.path.join(_dir, os.path.basename(_file)))

if __name__ == "__main__":
    _input_dir_all = "/mnt/HDD/NMDD/Exported_To_X-ray"
    _input_dir_rest = "/mnt/HDD/NMDD/Full_NMDD"
    _export_dir = "/mnt/SSD/Franko/Desktop/FIXExport/NMDID_PATCHES"

    main(input_dir_all= _input_dir_all, 
         input_dir_rest = _input_dir_rest,
         export_dir = _export_dir)