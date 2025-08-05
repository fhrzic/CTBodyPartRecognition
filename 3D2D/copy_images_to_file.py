import os
import pandas as pd
import shutil
# Copies images from input xlsx column to target dir
_xlsx_path = r"/run/user/1001/gvfs/smb-share:server=mig_data1,share=mig_projects/Object_Detection_QC/unique_CT_from_MRCT_TotalSegmentator/unique_CT_from_MRCT_TotalSegmentator/Output-Labels.xlsx"
_output_dir = r"/run/user/1001/gvfs/smb-share:server=mig_data1,share=mig_projects/Object_Detection_QC/unique_CT_from_MRCT_TotalSegmentator/unique_CT_from_MRCT_TotalSegmentator/Images"
_input_dir = r"/mnt/HDD/NMDD/Exported_To_X-ray/"

# Files and dirs
_list_of_files = [os.listdir(_input_dir)]

# Read pandas
_df = pd.read_excel(_xlsx_path, sheet_name='Volume')

for _id in _df['ID']:
    _name = f"{_id}_reducted_image.png"
    _case = _id.split('_exp')[0]
    _path_src = os.path.join(_input_dir, _case, _name)
    _path_trg = os.path.join(_output_dir, _name)

    if (os.path.isfile(_path_src)):
        shutil.copyfile(_path_src, _path_trg)
    else:
        print(f"File not found {_case}/{_name}!")

