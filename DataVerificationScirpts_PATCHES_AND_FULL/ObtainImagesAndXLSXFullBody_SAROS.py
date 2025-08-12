# Libs
import os
import shutil


# input dirs
_input_dir = "SAROS_working_data"
_output_dir = "SAROS_LABEL"

# Create results dir
shutil.rmtree(_output_dir, ignore_errors=True)
os.makedirs(_output_dir)

# Obtain all directories/cases
_all_cases = os.listdir(_input_dir)

# iterate over cases
for _case in _all_cases:
    # Obtain_names
    _case_name = os.path.basename(_case)
    _image_name = f"{_case_name}_reducted_image.png"
    _xlsx_name = f"{_case_name}_results.xlsx"

    # Copy to output dir
    shutil.copy(os.path.join(_input_dir, _case_name, _image_name), os.path.join(_output_dir, _image_name))
    shutil.copy(os.path.join(_input_dir, _case_name, "results.xlsx"), os.path.join(_output_dir, _xlsx_name))
