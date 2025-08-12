#########################
"""
Total segmentor exports everything into same folder. This scripts organize
total its export and orginial nifty into the folders. This is not the case
if the nii.gz files for before organized into separate folders
"""
#########################
# libs
import os
import shutil

# Define input and export dirs
_input_dir = "AXIAL_CTs"
_input_dir_images = "Exported_To_X-ray" 
_export_dir = "SAROS_working_data"

# Create dir
if os.path.exists(_export_dir):
        shutil.rmtree(_export_dir)
os.makedirs(_export_dir)

# List nii.gz files
_nii_gz_file_list = [_f for _f in os.listdir(_input_dir) if _f.endswith('.nii.gz')]

# Go through files
for _file in _nii_gz_file_list:
    # Obtain name
    _name = os.path.basename(_file).split(".")[0]

    # Create dir
    _export_dir_case = os.path.join(_export_dir, _name)
    os.makedirs(_export_dir_case)

    # Copy all files with that name in the case
    shutil.copy(os.path.join(_input_dir, f"{_name}.nii.gz"), os.path.join(_export_dir_case, f"{_name}.nii.gz"))
    shutil.copy(os.path.join(_input_dir, f"{_name}_TotalSegmentatior_total.nii"), os.path.join(_export_dir_case, f"{_name}_TotalSegmentatior_total.nii"))
    shutil.copy(os.path.join(_input_dir, f"{_name}_TotalSegmentatior_appendicular_bones.nii"), os.path.join(_export_dir_case, f"{_name}_TotalSegmentatior_appendicular_bones.nii"))
    shutil.copy(os.path.join(_input_dir_images, f"{_name}_reducted_image.png"), os.path.join(_export_dir_case, f"{_name}_reducted_image.png"))
    shutil.copy(os.path.join(_input_dir_images, f"{_name}_original_image.png"), os.path.join(_export_dir_case, f"{_name}_original_image.png"))
    shutil.copy(os.path.join(_input_dir_images, f"{_name}_coronal_image.png"), os.path.join(_export_dir_case, f"{_name}_coronal_image.png"))


