import os
import pandas as pd
import shutil
# Copies images from input xlsx column to target dir
_output_dir = "/run/user/1001/gvfs/smb-share:server=mig_data1,share=mig_projects/Object_Detection/BCH_CT_TO_LABEL"
_input_dir = "/mnt/SSD/BCH_CT/CT_2D/"

# Files and dirs
_list_of_files = []
for _root, dirs, _files in os.walk(_input_dir):
    for _file in _files:
        if _file.endswith('_reducted_image.png'):
            _list_of_files.append(os.path.join(_root, _file))


# Copy files to new file names
for _file in _list_of_files:
    # Obtain case name
    _case = os.path.basename(os.path.dirname(_file))
   
    # Name
    _name = os.path.basename(_file)

    # Full name
    _full_name = f"{_case}_{_name}"

    # Copy
    shutil.copyfile(_file, os.path.join(_output_dir, _full_name))
    
