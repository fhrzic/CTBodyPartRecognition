import os
import pandas as pd


def list_all_directories(directory):
    """
    Lists all directories in the specified directory.

    Args:
        directory (str): Path to the directory to process.

    Returns:
        list: A list of all directories in the given directory.
    """
    if not os.path.isdir(directory):
        print(f"The path {directory} is not a valid directory.")
        return []

    directories = []
    for root, subdirs, _ in os.walk(directory):
        directories.extend([os.path.join(root, d) for d in subdirs])
    return directories

def clean_directories_based_on_images(image_paths, directories_to_check):
    """
    Deletes directories that are not in the list of valid directories and contain specified images.

    Args:
        image_paths (list): List of image file paths.
        valid_directories (list): List of valid directories to keep.
    """
    valid_directories = [os.path.dirname(image).split("SAROS_working_data")[-1] for image in image_paths]

    _cnt = 0
    _cnt_2 = 0
    for dir_path in directories_to_check:
        _new_dir_path = dir_path.split("SAROS_working_data")[-1]
        if os.path.basename(_new_dir_path) == "images" or "case" in os.path.basename(_new_dir_path):
            continue  

        if _new_dir_path not in valid_directories:
            _cnt_2 += 1
            try:
                print(f"Deleting directory and its contents: {dir_path}")
                for root, subdirs, files in os.walk(dir_path, topdown=False):
                    for file in files:
                        file_path = os.path.join(root, file)
                        print(f"Deleting file: {file_path}")
                        os.remove(file_path)
                    for subdir in subdirs:
                        subdir_path = os.path.join(root, subdir)
                        print(f"Deleting subdirectory: {subdir_path}")
                        os.rmdir(subdir_path)
                os.rmdir(dir_path)
            except Exception as e:
                print(f"Error deleting directory {dir_path}: {e}")

        else:
            _cnt += 1
    print(_cnt, _cnt_2)

def main(path_to_xlsxs: str = None,
         extension: str = None):
    """
    Main script for obtaining data for patches

    Args:
        * path_to_xlsx, str, path to the place where the xlsx files are located
        * extension, str, name of the file extension to be copy (.npy)
    """

    # Obtain labels from patches NMDD 
    # Read xlsx
    try:
        _mapping_df = pd.read_excel(os.path.join(path_to_xlsxs, "SourceDataset.xlsx"))
    except:
        print(f"Error while reading {_mapping_df}! Please check out data file path!")
        return -1
    
    # Read batches
    try:
        # Storage
        _good_images_list = []
        _file_list = []
        # Obtain files
        for _root, _, _files in os.walk(path_to_xlsxs):
            for _file in _files:
                # Check if _file name contains 'batch' and has a .xlsx extension
                if 'batch' in _file and _file.endswith('.xlsx'):
                    _file_list.append(os.path.join(_root, _file))

        # Load content 
        for _file in _file_list:
            _df = pd.read_excel(_file)
            _good_images_list += _df["good"].to_list()
    except:
        print(f"Couldn not read batch xlsx in folder {path_to_xlsxs}!. Please check out data file path!")
        return -1
    
    # Remap paths
    _images_paths = []
    for _path in _good_images_list:
        _column = _mapping_df[_mapping_df["Remaped_paths"] == _path]
        _org_path = _column["Original_paths"].to_list()
        _images_paths += _org_path
    
    _new_images_paths = []
    # Go trought data and build data list
    for _index, _image_path in enumerate(_images_paths):
        # Obtan label path
        _dir = os.path.dirname(_image_path)
        _label_path = os.path.join(_dir, "labels.xlsx")                
        
        # Check label name
        if not os.path.isfile(_label_path):
            if self.dataset_config.verbose:
                print(f"Could not find image: {_label_path}: Skipping!!!")
            continue

        # Check image path
        _image_path = os.path.join(_dir, f"main{extension}")
        _new_images_paths.append(_image_path)
        # Check if file exist - skip otherwise
        if not os.path.isfile(_image_path):
            if self.dataset_config.verbose:
                print(f"Could not find image: {_image_path}: Skipping!!!")
            continue
        
    # Return
    return _new_images_paths

if __name__ == "__main__":
    _path_to_xlsx = "/mnt/SSD/Franko/Desktop/FIXExport/Patches_SAROS"
    _extension = ".png"

    _good_images = main(path_to_xlsxs=_path_to_xlsx,
                        extension=_extension)

    _folders = "/mnt/HDD/SAROS/SAROS_working_data".strip()
    _directories = list_all_directories(_folders)

    clean_directories_based_on_images(_good_images, _directories)


