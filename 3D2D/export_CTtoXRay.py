from Utils.Conversion import *
import os
_input_dir = r"Z:\Object_Detection_QC\unique_CT_from_MRCT_TotalSegmentator\unique_CT_from_MRCT_TotalSegmentator\CT_unique"
_axial_output_dir = r"Z:\Object_Detection_QC\unique_CT_from_MRCT_TotalSegmentator\unique_CT_from_MRCT_TotalSegmentator\Axial_CT"
_final_results_output_dir = r"Z:\Object_Detection_QC\unique_CT_from_MRCT_TotalSegmentator\unique_CT_from_MRCT_TotalSegmentator\Results_CTtoXRay"

_nifti_file_list = []
for _root, _dirs, _files in os.walk(_input_dir):
    for _file in _files:
        if _file.endswith('.nii.gz'):
            _nifti_file_list.append(os.path.join(_root, _file))

print(f"Found {len(_nifti_file_list)} in {_input_dir} and stored them in axial to the {_axial_output_dir}")

for _i, _nifti_file in enumerate(_nifti_file_list):
    print(f"Exporting file {_i+1}/{len(_nifti_file_list)}")
    _name = os.path.basename(_nifti_file)
    export_nifti_to_axial(_nifti_file, os.path.join(_axial_output_dir, _name))


export_nifti_to_dirs(input_dir_path = _axial_output_dir,
                     export_dir_path = _final_results_output_dir,
                     verbose = True)