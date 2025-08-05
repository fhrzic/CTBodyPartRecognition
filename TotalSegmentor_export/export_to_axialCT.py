import pandas as pd
import os
import shutil
from Utils.Conversion import export_nifti_to_axial
#########################################################################
# input csv
_csv_file_path = "/mnt/HDD/SAROS2/Segmentations/Segmentation Info.csv"
_export_path = "/mnt/HDD/SAROS2/AXIAL_CTs"
_input_dir = "/mnt/HDD/SAROS2/data"
#########################################################################

# load to dataframe
_df = pd.read_csv(_csv_file_path, index_col = 0)

# select only indexes of interest
_white_list_cases = _df[_df["anatomic_region"] == "wholebody"].index.to_list()

# Create output dir
# Remove the directory and its contents
if os.path.exists(_export_path):
    shutil.rmtree(_export_path)
os.makedirs(_export_path)

# Obtain nifti files
_nifti_file_list = []
for _root, _dirs, _files in os.walk(_input_dir):
    for _file in _files:
        if _file.endswith('image_original.nii.gz'):
            _nifti_file_list.append(os.path.join(_root, _file))

# Filter and export
for _nifti in _nifti_file_list:
    _base_dir = os.path.basename(os.path.dirname(_nifti))
    if _base_dir in _white_list_cases:
        export_nifti_to_axial(_nifti, os.path.join(_export_path, f"{_base_dir}.nii.gz"))
    