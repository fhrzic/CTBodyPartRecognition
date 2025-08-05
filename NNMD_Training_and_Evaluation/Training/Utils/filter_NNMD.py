import pandas as pd
import matplotlib.pyplot as plt
from Utils.config_NNMD import datasetConfig
import os
import numpy as np

#*******************************************************#
# Filtering dataset based on different criteria
#*******************************************************# 

class dataFrameFilterContainter():
    def __init__(self, 
                 path_to_xlsx:str = None,
                 verbose: bool = False):
        """
        Init of the class which requires path to xlsx. It load xlsx and verify that 
        xlsx is obtained from the NNMD script: "run_generate_xlsx_from_ni". 
        The xlsx file must contain all necessary data which are volumes from the 
        total segmentator.

        Args:
            path_to_xlsx, str, path to the xlsx - his location ond the os.
            verbose, bool, write messages of the filtering process
        """
        # Check if path exists
        assert path_to_xlsx != None, f"Path must not be none: {path_to_xlsx}!"
        assert os.path.isfile(path_to_xlsx), f"File does not exists: {path_to_xlsx}!"

        # Load xslx
        self._main_dataframe = pd.read_excel(path_to_xlsx, index_col = 0, header = 0)
        self.original_backup = pd.read_excel(path_to_xlsx, index_col = 0, header = 0)
        
        # Remap dict - dictionary conatining labels and their mappings. This are filter out labels
        # remap_dict: dictionary for rempaing the labels. This was done based on image provided at (from skull to foot for bones)
        # https://github.com/wasserth/TotalSegmentator/blob/master/resources/imgs/overview_classes_v2.png
        self.remap_dict =  {
            91: [1, 1],
            73: [2.1, 2],
            74: [2.2, 3],
            71: [3.1, 4],
            72: [3.2, 5],
            69: [4.1, 6],
            70: [4.2, 7],
            50: [5.1, 8],
            49: [5.2, 9],
            48: [5.3, 10],
            47: [5.4, 11],
            46: [5.5, 12],
            45: [5.6, 13],
            44: [5.7, 14],
            116:[6, 15],
            92: [7.1, 16],
            104: [7.2, 17],
            93: [8.1, 18],
            105: [8.2, 19],
            94: [9.1, 20],
            106: [9.2, 21],
            95: [10.1, 22],
            107: [10.2, 23],
            96: [11.1, 24],
            108: [11.2, 25],
            97: [12.1, 26],
            109: [12.2, 27],
            98: [13.1, 28],
            110: [13.2, 29],
            99: [14.1, 30],
            111: [14.2, 31],
            100: [15.1, 32],
            112: [15.2, 33],
            101: [16.1, 34],
            113: [16.2, 35],
            102: [17.1, 36],
            114: [17.2, 37],
            103: [18.1, 38],
            115: [18.2, 39],
            43: [19.1, 40],
            42: [19.2, 41],
            41: [19.3, 42],
            40: [19.4, 43],
            39: [19.5, 44],
            38: [19.6, 45],
            37: [19.7, 46],
            36: [19.8, 47],
            35: [19.9, 48],
            34: [19.11, 49],
            33: [19.12, 50],
            32: [19.13, 51],
            31: [20.1, 52],
            30: [20.2, 53],
            29: [20.3, 54],
            28: [20.4, 55],
            27: [20.5, 56],
            26: [21, 57],
            124: [22, 58],
            125: [23, 59],
            77: [24.1, 60],
            78: [24.2, 61],
            25: [25, 62],
            75: [26.1, 63],
            76: [26.2, 64],
            126: [27, 65],
            127: [28, 66],
            128: [29, 67],
            118: [30, 68],
            119: [31, 69],
            120: [32, 70],
            121: [33, 71],
            122: [34, 72],
            123: [35, 73],
        }

        # Set up verbose
        self.verbose = verbose

        # Report
        if self.verbose:
            print(f"Successfully load the dataframe!")

    def drop_columns(self):
        """
        Method which drop the columns based on the remap dict
        """
        # Create temporary dataset frame
        _temp_df = pd.DataFrame()

        # Filter out data
        for _key_mdf in self._main_dataframe:
            if _key_mdf in self.remap_dict:
                _temp_df[_key_mdf] = self._main_dataframe[_key_mdf]

        # Update main dataframe
        self._main_dataframe = _temp_df

    
    def swap_names(self, 
                   xlsx_swap_path:str = None):
        """
        Method which swaps names in the dataset so they are human readable.
        It requires xlsx file which has columns Label and Name where Label is a number to
        which name is associated. For instance "0 Skull", "1 LowerLeg". One such pair can
        be obtained in xslx sheet "Task" obtainable by NNMD script: "run_generate_xlsx_from_ni"

        Args:
            * xlsx_swap_path, str, path to xlsx which contains Label and Name columns
        """
        # Check if path exists
        assert xlsx_swap_path != None, f"Path must not be none: {xlsx_swap_path}!"
        assert os.path.isfile(xlsx_swap_path), f"File does not exists: {xlsx_swap_path}!"
    
        # Read xlsx
        _swap_df = pd.read_excel(xlsx_swap_path)
        
        # Obtain swap dict if possible
        try:
            # Swap names
            _name_dict = _swap_df.set_index('Label')['Name'].to_dict()
            self._main_dataframe.columns = [{**_name_dict, **{_v:_k for _k,_v in _name_dict.items()}}.get(_x, _x) for _x in self._main_dataframe.columns]

            # Retreive ID column
            _id_column = self.original_backup['ID'] 
            self._main_dataframe = pd.concat([_id_column, self._main_dataframe], axis = 1,  ignore_index=False)
        except:
            print(f"Error: given xlsx for name swap is missing columns named: Name or/and Label")

    def discard_outlayers_mean_std(self, 
                                    skip_columns:list = []):
        """
        Method which is dicarding potential outlayers based on the lower STD bound.

        Args:
            * skip_columns, list, list of columns which will be skipped
        """
        # Iterate over columns
        _drop_index_list = []
        for _column in self._main_dataframe.columns:
            # Skip first column or any which must be skipped
            if _column in skip_columns:
                continue
            # Calculate threshold
            _threshold = self._main_dataframe[_column].mean() - self._main_dataframe[_column].std()

            # Apply threshold
            _rows_to_be_deleted = self._main_dataframe[_column].loc[self._main_dataframe[_column] <= _threshold]
            
            # Add tehm to list
            _drop_index_list += _rows_to_be_deleted.index.tolist()
            
            # Report
            if self.verbose:
                print(f"Based on column {_column}, {len(_rows_to_be_deleted)} rows has been removed")

        # Remove duplicates
        _drop_index_list = list(set(_drop_index_list))

        # Rremoving indexes from the data
        self._main_dataframe = self._main_dataframe.drop(_drop_index_list)

    def preview_discard_outlayers(self, 
                                  column_name: str = None, 
                                  criterium: str = "mean_std"):
        """
        Function which previews discarded data based on the selected criteria.

        Args:
            * colum_name, str, 
        """
        # Check data
        #assert criterium in ["mean_std"], f"Given criterium: {criterium} is not implemented!"
        #assert column_name in self._main_dataframe, f"Given criterium: {criterium} is not implemented!"

        # Obtain column
        _column = self._main_dataframe[column_name]

        # Apply criterium
        if criterium == "mean_std":
            print("STATS", _column.mean(), _column.std())
            _threshold = _column.mean() - _column.std()
            _filter = _column.loc[_column <= _threshold]
            _cnt = len(_filter)

        # Plot it
        plt.plot(_column.index, _column, 'b.')
        plt.plot(_filter.index, _filter, 'gx')
        plt.plot((0, len(_column.index)), (_threshold, _threshold), "r--", label = criterium)
        plt.title(f"{column_name}:{_cnt}")
        plt.legend()
        plt.show()
        
    def export_maindataframe(self, name:str = "maindataframe.xlsx"):
        """
        Method for expotring the main dataframe to xlsx.

        Args:
            * name, str, save name to which dataframe will be exported to
            xlsx
        """
        self._main_dataframe.to_excel(name)



    @property
    def main_dataframe(self):
        """
        Getter for main data frame
        """
        return self._main_dataframe
    
    @main_dataframe.setter
    def main_dataframe(self, data_frame: pd.DataFrame):
        """
        Setter for main dataframe.

        Args:
            * data_frame, pd.DataFrame, setter for the pandas dataframe 
        """
        assert isinstance(data_frame, pd.DataFrame), f"ERROR: {data_frame} is not pd.DataFrame!"
        self._main_dataframe = data_frame
        self.original_backup = data_frame