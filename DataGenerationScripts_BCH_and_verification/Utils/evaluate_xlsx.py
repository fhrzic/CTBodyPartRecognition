import os
import pandas as pd

class statisticContainer():
    """
    class for generating statistic and inspecting results on the each fold
    """
    def __init__(self, path_to_cheatsheet_names: str = None):
        """
        Init of the class, inits all necessary storage.

        Args:
            * path_to_cheatsheet_names, str, Path to cheatsheet where the names of all variables are stored

        """
        # Set cheatsheet names
        self.names_df = pd.read_excel(path_to_cheatsheet_names, sheet_name="reduced_cluster_remaped")

        # Main storage for predictions
        self._main_data_storage = {}

        # main storage for results
        self._main_results_storage = {}

        # working storage
        self.working_storage = {}

    def load_dir(self, root_path: str = None):
        """
        Load xlsx file to pd data frame and saves in into dictionary where
        key is paht, and value is data inside that path loaded as pd dataframe
        
        Args:
            * path, str, path to dir which contains xlsx files
        """
        # Reset working storage
        self.working_storage = {}
        # Find _xlsx_files
        _xlsx_files = []
        for _root, _, _files in os.walk(root_path):
            for _file in _files:
                if _file.endswith('.xlsx'):
                    _xlsx_files.append(os.path.join(_root, _file)) 
        
        # Load it as pandas
        for _xlsx_file in _xlsx_files:
            _df = pd.read_excel(_xlsx_file, index_col=0)
            self.working_storage[_xlsx_file] = _df
    
    def generate_statistics(self, true_key: str = "True"):
        """
        Method to generate statistic for the working storage.

        Args:
            * true_key, str, string which defines where the true labels are stored. Values: True or True_Original
        """
        # Check if storage is not empty
        assert len(self.working_storage) > 0, f"Working storage is empty, use load_dir to add data to it!"

        # Create statistics container
        _statistic_container = {}
        self.statistic_container = None
        for _item in self.names_df["Name"]:
            _statistic_container[_item] = {"TP": 0,
                                           "FP": 0,
                                           "FN": 0,
                                           "TN":0}
        # Calculate statistics -> basic
        for _xlsx in self.working_storage:
            _df = self.working_storage[_xlsx]
            
            # Populate statistics
            for _key in _statistic_container:
                try:
                    _true = _df.loc[true_key, _key]
                    _predicted = _df.loc['Predicted', _key]
                    if _true and _predicted:
                        _statistic_container[_key]["TP"] += 1
                    if _true and _predicted == 0:
                        _statistic_container[_key]["FN"] += 1
                    if _true == 0 and _predicted == 1:
                        _statistic_container[_key]["FP"] += 1
                    if _true == 0 and _predicted == 0:
                        _statistic_container[_key]["TN"] += 1                  
                except:
                    continue
            
        # Calculate metrices
        for _key in _statistic_container:
            _statistic_container[_key]["support"] = _statistic_container[_key]["TP"] + _statistic_container[_key]["FN"]
            try:
                _statistic_container[_key]["precision"] = _statistic_container[_key]["TP"] / (_statistic_container[_key]["TP"] + _statistic_container[_key]["FP"])
            except:
                _statistic_container[_key]["precision"] = "N/A"
            try:
                _statistic_container[_key]["recall"] = _statistic_container[_key]["TP"] / (_statistic_container[_key]["TP"] + _statistic_container[_key]["FN"])
            except:
                _statistic_container[_key]["recall"] = "N/A"
            try:    
                _statistic_container[_key]["accuracy"] = (_statistic_container[_key]["TP"] + _statistic_container[_key]["TN"]) / (_statistic_container[_key]["TP"]+_statistic_container[_key]["FP"]+_statistic_container[_key]["FN"]+_statistic_container[_key]["TN"])
            except:
                _statistic_container[_key]["accuracy"] = "N/A"
            try:    
                _statistic_container[_key]["f1"] = 2 * _statistic_container[_key]["precision"] * _statistic_container[_key]["recall"] / (_statistic_container[_key]["precision"] + _statistic_container[_key]["recall"])
            except:
                _statistic_container[_key]["f1"] = "N/A"

        # Generate df
        _df = pd.DataFrame.from_dict(_statistic_container, orient='index')
        _df.index.name = 'body_part'
        _df = _df.reset_index()
        self.statistic_container = _df

    def dump_data(self, name: str = None, true_key: str = "True"):
        """
        Method which dumps currently obtained data to main storage.

        Args:
            * name, str, name of the sheet/dict_key where the data will be dumped
             * true_key, str, string which defines where the true labels are stored. Values: True or True_Original
        """
        # Obtain sheet name
        if name == None:
            _name = str(len(self._main_storage))
        else:
            _name = name

        # Create main storage
        _main_export_dict = {}
        for _xlsx in self.working_storage:
            _case_id = os.path.basename(os.path.dirname(_xlsx))
            # Dict to save data export
            _tmp_dict = {}
            for _key in self.names_df["Name"]:
                _tmp_dict[f"{_key}_true"] = self.working_storage[_xlsx].loc[true_key, _key]
                _tmp_dict[f"{_key}_pred"] = self.working_storage[_xlsx].loc['Predicted', _key]
            _main_export_dict[_case_id] = _tmp_dict

        # Generate df
        _df = pd.DataFrame.from_dict(_main_export_dict, orient='index')
        _df.index.name = "Case_ID"
        _df = _df.reset_index()

        # Store
        self._main_data_storage[_name] = _df
        self._main_results_storage[_name] = self.statistic_container

        # Reset
        self.statistic_container = None
        self.working_storage = None

    def export_to_xlsx(self, export_path:str = None):
        """
        Export all data to a single xlsx file.

        Args:
            * export_path, str, path to the export xlsx file
        """
        # Create writer
        _writer = pd.ExcelWriter(export_path, engine='xlsxwriter')
        _workbook = _writer.book
        _center_alignment = _workbook.add_format({'align': 'center', 'valign': 'vcenter'})
	
        # Go trough data
        for _key in self._main_data_storage:
            # Obtain data
            _data_df = self._main_data_storage[_key]
            _results_df = self._main_results_storage[_key]

            # Export statistics
            _data_df.to_excel(_writer, sheet_name = f"{_key}_data", index = False)
            _results_df.to_excel(_writer, sheet_name = f"{_key}_statistics", index = False)
            
            # Fix col width
            _worksheet = _writer.sheets[f"{_key}_data"]

            # Set column width
            for _i, _col in enumerate(_data_df.columns):
                # Get the maximum length of the column content and the column header
                _max_len = max(_data_df[_col].astype(str).map(len).max(), len(_col)) + 2  

                # Set the column width
                _worksheet.set_column(_i, _i, _max_len, _center_alignment)

            _worksheet = _writer.sheets[f"{_key}_statistics"]
            # Set column width
            for _i, _col in enumerate(_results_df.columns):
                # Get the maximum length of the column content and the column header
                _max_len = max(_results_df[_col].astype(str).map(len).max(), len(_col)) + 2  

                # Set the column width
                _worksheet.set_column(_i, _i, _max_len, _center_alignment)
        
        # Create merged table
        _tmp_list = []
        for _key in self._main_data_storage:
            _tmp_list.append(self._main_data_storage[_key])
        _merged_df = pd.concat(_tmp_list, ignore_index = True)
	
	# Fix merged table columns size
	# Export statistics
        _merged_df.to_excel(_writer, sheet_name = f"merged_data", index = False)
	
	# Fix col width
        _worksheet = _writer.sheets[f"merged_data"]

	# Set column width
        for _i, _col in enumerate(_merged_df.columns):
            # Get the maximum length of the column content and the column header
            _max_len = max(_merged_df[_col].astype(str).map(len).max(), len(_col)) + 2  

            # Set the column width
            _worksheet.set_column(_i, _i, _max_len, _center_alignment)        
	
	# Obtain merged statistics
        _statistic_container = {}
        for _item in self.names_df["Name"]:
            _statistic_container[_item] = {"TP": 0, "FP": 0, "FN": 0, "TN":0}
            for _key in self._main_results_storage:
                _df = self._main_results_storage[_key]
                _statistic_container[_item]["TP"] += int(_df[_df["body_part"] == _item]["TP"].values[0])
                _statistic_container[_item]["FP"] += int(_df[_df["body_part"] == _item]["FP"].values[0])
                _statistic_container[_item]["TN"] += int(_df[_df["body_part"] == _item]["TN"].values[0])
                _statistic_container[_item]["FN"] += int(_df[_df["body_part"] == _item]["FN"].values[0])
        
        # Calculate metrices
        for _key in _statistic_container:
            _statistic_container[_key]["support"] = _statistic_container[_key]["TP"] + _statistic_container[_key]["FN"]
            try:
                _statistic_container[_key]["precision"] = _statistic_container[_key]["TP"] / (_statistic_container[_key]["TP"] + _statistic_container[_key]["FP"])
            except:
                _statistic_container[_key]["precision"] = "N/A"
            try:
                _statistic_container[_key]["recall"] = _statistic_container[_key]["TP"] / (_statistic_container[_key]["TP"] + _statistic_container[_key]["FN"])
            except:
                _statistic_container[_key]["recall"] = "N/A"
            try:    
                _statistic_container[_key]["accuracy"] = (_statistic_container[_key]["TP"] + _statistic_container[_key]["TN"]) / (_statistic_container[_key]["TP"]+_statistic_container[_key]["FP"]+_statistic_container[_key]["FN"]+_statistic_container[_key]["TN"])
            except:
                _statistic_container[_key]["accuracy"] = "N/A"
            try:    
                _statistic_container[_key]["f1"] = 2 * _statistic_container[_key]["precision"] * _statistic_container[_key]["recall"] / (_statistic_container[_key]["precision"] + _statistic_container[_key]["recall"])
            except:
                _statistic_container[_key]["f1"] = "N/A"
        
        
        _df = pd.DataFrame.from_dict(_statistic_container, orient='index')
        _df.index.name = 'body_part'
        _df = _df.reset_index()
        _df.to_excel(_writer, sheet_name = f"merged_statistics", index = False)
        
        # Fix col width
        _worksheet = _writer.sheets[f"merged_statistics"]

	# Set column width
        for _i, _col in enumerate(_df.columns):
            # Get the maximum length of the column content and the column header
            _max_len = max(_df[_col].astype(str).map(len).max(), len(_col)) + 2  

            # Set the column width
            _worksheet.set_column(_i, _i, _max_len, _center_alignment)        
	
        
        # Export
        _writer.close()

    # Getters and setters for main properties and containeers
    @property
    def main_data_storage(self): 
         return self._main_data_storage

    @main_data_storage.setter 
    def main_data_storage(self): 
         self._main_data_storage = {}


    @property
    def main_results_storage(self): 
         return self._results_storage

    @main_results_storage.setter 
    def main_results_storage(self): 
         self._results_storage = {}
