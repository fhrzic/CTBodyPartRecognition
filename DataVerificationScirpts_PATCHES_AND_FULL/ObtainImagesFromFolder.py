import os
import shutil
import pandas as pd

# Dict where source files are located (top level dict)
_src_dir = "SAROS_working_data"
_dst_dir = "Images2Label"

# Create fres output dir (empty one)
if os.path.exists(_dst_dir):
    shutil.rmtree(_dst_dir)
os.makedirs(_dst_dir)

# Obtain files in dir (all.png)
_all_png_files = []

# Walk through the directory structure
for _dirpath, _dirnames, _filenames in os.walk(_src_dir):
    for _filename in _filenames:
        if _filename == 'all.png':
            _file_path = os.path.join(_dirpath, _filename)
            _all_png_files.append(_file_path)

_df = pd.DataFrame()
_df["Original_paths"] = _all_png_files

# Copy images and build remap
_remap_list = []
for _index, _src_img_path in enumerate(_all_png_files):
    # Generate name
    _map_name = f"{_index}.png"
    # Add it to list
    _remap_list.append(_map_name)
    # Create dst img path
    _dst_img_path = os.path.join(_dst_dir, _map_name)
    shutil.copy(_src_img_path, _dst_img_path)

# Add it to df
_df["Remaped_paths"] = _remap_list
_df.to_excel("SourceDataset.xlsx", index = False, sheet_name = "mapping")
