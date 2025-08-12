# Script for extracting targeted cases from BCH data
# Libs
import os
import pandas as pd
import shutil
import numpy as np


# Config variables
#####################################################
_labels_xlsx_file = "Labels.xlsx"
filter = ["foot", "forearm", "hand", "patella", "tarsal"]
output_dir = "BCH_LABEL_Part1"
input_dir = "CT_2D/"
#####################################################

# Functions
def load_labels(xlsx_file_path:str = None)->pd.DataFrame:
    """
    Function which loads xlsx file and loads it to the dataframe. Also it filters and fixes data frame
    to have proper names and removes unnecessary columns

    Args:
        * xlsx_file_path, str, path to the label extracted by Ata

    Return:
        * pd.DataFrame, data frame with all loaded columns that matters with fixed naming
    """
    # Load file
    assert os.path.exists(xlsx_file_path), f"File does not exists {xlsx_file_path}"
    _df = pd.read_excel(xlsx_file_path, index_col=0)
    
    # Drop columns
    _df = _df.drop("Exam Code", axis = 1)
    _df = _df.drop("Exam Description", axis = 1)
    
    # Rename columns
    # Label names
    _labels = ["skull", "shoulder", "humerus", "vertebrae_C", "thorax", "vertebrae_L", 
                "forearm", "pelvis", "femur", "hand", "patella", "shin", "tarsal", "foot"]
    _df.columns = _labels

    # Return
    return _df

def filter_labels(df: pd.DataFrame = None, filter_list: list = None, amount: int = 1000)->list:
    """
    Function to filter the data based on given filter list. Return list of ids 

    Args:
        * df, pd.DataFrame, input dataframe obtained by load_labels function
        * filter_list, list, list contatainge all column names that wants to be included into the final otput list
        * amount, int, forces the maximum number of extracted items. It is not hard margine (it will try to stick 
        around given number)
    
    Return:
        * list, list of ids which are selected by filtering
    """
    # generate output id list
    _id_output_list = []

    # Iterate through filter
    for _filter in filter_list:BCH_CT_Accession_2010_2018_
        # Obtain list with indexes of interes
        _extracted_list = df[df[_filter] > 0].index
        _extracted_list = _extracted_list.to_list()
        
        # Cap it
        if len(_extracted_list) > amount:
            _extracted_list = _extracted_list[0:amount]
        
        # Add it to outpu list
        _id_output_list += _extracted_list

    # Remove duplcates
    _id_output_list = list(set(_id_output_list))
    # Return
    return _id_output_list

def export_files(input_dir:str = None, output_dir:str = None, filter_list: list = None):
    """
    Method which obtain only images and saves them to the output dir. Also applies
    filter if provided

    Args:
        * input_dir, str, path to the input dir where cases are located
        * output_dir, str, path to the output dir where the reducted images should be exported
        * filter_list, list, list of indexes to be exported

    """
    # Files and dirs
    _list_of_files = []
    for _root, dirs, _files in os.walk(input_dir):
        for _file in _files:
            if _file.endswith('_reducted_image.png'):
                _list_of_files.append(os.path.join(_root, _file))

    # Copy files to new file names
    for _file in _list_of_files:
        # Obtain case name
        _case = os.path.basename(os.path.dirname(_file))
        
        # Apply filter
        if filter_list != None:
            if int(_case) not in filter_list:
                continue
        
        # Name
        _name = os.path.basename(_file)

        # Full name
        _full_name = f"{_case}_{_name}"
        
        # Copy
        shutil.copyfile(_file, os.path.join(output_dir, _full_name))
    

def main():
    """
    Main script
    """
    _df = load_labels(_labels_xlsx_file)
    _idx_list = filter_labels(_df, filter)
    export_files(input_dir, output_dir, _idx_list)

if __name__ == '__main__':
    main()