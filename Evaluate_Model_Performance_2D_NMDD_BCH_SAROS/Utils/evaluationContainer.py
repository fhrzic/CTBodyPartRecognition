import pandas as pd
import numpy as np
import os
import json
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import confusion_matrix
import sklearn.metrics as skm
import matplotlib.pyplot as plt
import seaborn as sns
import shutil

######################################################################################################################################################################################
def generate_combined_xlsx(input_dir: str = None, output_xlsx: str = None):
    """
    Function which goes through directories and builds up the xslx directory with paths and predictions necessary for evaluation_container

    Args:
        * input_dir, str, path to the directoriy containing data (it is recursive se top dir will do the job)
        * output_dir, str, path to the output dir xlsx
    
    Exports:
        * Exports to dict
    """
    # Check validity of given path
    assert os.path.exists(input_dir), f"Directory: {input_dir} does not exists!"
    assert output_xlsx != None, f"Directory : {output_dir} should not be None!"

    # Obtain files
    _xlsx_files = []

    # Walk through the directory and subdirectories
    for _root, _, _files in os.walk(input_dir):
        for _file in _files:
            if _file.endswith('.xlsx'):
                _xlsx_files.append(os.path.join(_root, _file))

    # Create dfs
    _list_true = []
    _list_binary = []
    _list_real = []

    # Go through xlsx
    for _xlsx_file in _xlsx_files:
        # Open xlsx as df
        _df_tmp = pd.read_excel(_xlsx_file, index_col = 0)

        # Extract rows
        _true_row = _df_tmp.loc["True"]
        _binary_row = _df_tmp.loc["Model_rounded"]
        _real_row = _df_tmp.loc["Model_real"]

        # Add paths at the begining
        _extracted_true_df = _true_row.to_frame().T
        _extracted_binary_df = _binary_row.to_frame().T
        _extracted_real_df = _real_row.to_frame().T

        # Omit invalids (true labels all 0)
        if (_true_row == 0).all():
            continue

        # Add labels and img path
        _extracted_true_df.insert(0, "Labels", _xlsx_file)
        _extracted_true_df.insert(0, "Image", _xlsx_file.replace("labels.xlsx", "reducted_image.png"))
        _extracted_binary_df.insert(0, "Labels", _xlsx_file)
        _extracted_binary_df.insert(0, "Image", _xlsx_file.replace("labels.xlsx", "reducted_image.png"))
        _extracted_real_df.insert(0, "Labels", _xlsx_file)
        _extracted_real_df.insert(0, "Image", _xlsx_file.replace("labels.xlsx", "reducted_image.png"))
        
        # Store
        _list_true.append(_extracted_true_df)
        _list_binary.append(_extracted_binary_df)
        _list_real.append(_extracted_real_df)
    
    # Creat dfs
    _df_true = pd.concat(_list_true, ignore_index=True)
    _df_binary = pd.concat(_list_binary, ignore_index=True)
    _df_real = pd.concat(_list_real, ignore_index=True)

    # Export
    with pd.ExcelWriter(output_xlsx, engine='xlsxwriter') as _writer:
        _df_true.to_excel(_writer, sheet_name = "true", index = False)
        _df_binary.to_excel(_writer, sheet_name = "binary", index = False)
        _df_real.to_excel(_writer, sheet_name = "real", index = False)

 
######################################################################################################################################################################################
class evaluation_container():
    """
    Class for evaluation of the f1 socres
    """
    def __init__(self, predictions_xlsx: str = None):
        """
        Init method of the class 

        Args:
            * prediction_xlsx, str or df for hip evaluation, path to the xlsx file containing all predictions. It can be obtained by "obtain_predictions_dl_NMDD_BCH_SAROS.py" or by
            the "generate_xlsx" function located in this script. 
        """

        # Check if file exists
        assert os.path.exists(predictions_xlsx), f"Missing xlsx, check the path {predictions_xlsx}"

        # Load data
        self.true = pd.read_excel(predictions_xlsx, sheet_name='true')
        self.predicted = pd.read_excel(predictions_xlsx, sheet_name='real')

        # Define body parts
        self.body_parts = ["skull", "shoulder", "humerus", "vertebrae_C", 
          "thorax", "vertebrae_L", "forearm", "pelvis", 
          "femur", "hand", "patella", "shin", "tarsal", "foot"]
        
        # Main threshold
        self.threshold_dict = {}
        for _key in self.body_parts:
                self.threshold_dict[_key] = 0.5
    

    def __obtain_statistics_data(self, true_arr:np.array = None, pred_arr:np.array = None)->dict:
        """
        Function which accepts true values and predictions for given body parts and returns
        F1 scores based on different thresholds. In case of a tie, the highest possible threshold
        is taken.

        Args:
            * true_arr, np.array, numpy array with true values of the model
            * pred_arr, np.array, numpy array representing predicted values of a model
        Returns:
            * dictionary containing keys named "interval" (list), "f1-score"(list), "recall" (list), "precision" (list), and "best-thresh" for best threshold and
            "best-f1" for the best f1-score value.
        """
        # Storage
        _f1_scores = []
        _thresholds = []
        _precisions = []
        _recalls = []
        
        # Obtain predictions
        for _threshold in np.linspace(0,1, 51):
            _thresholded_arr = [1 if float(_element) > _threshold else 0 for _element in pred_arr]
            
            # Metrices
            _f1 = f1_score(true_arr, _thresholded_arr)
            _recall = recall_score(true_arr, _thresholded_arr)
            _precision = precision_score(true_arr, _thresholded_arr)

            if _f1 == 0 or _recall == 0 or _precision == 0:
                continue
            
            # Storage
            _f1_scores.append(_f1)
            _thresholds.append(_threshold)
            _precisions.append(_precision)
            _recalls.append(_recall)
            
        # Return
        return {"interval": _thresholds,
            "precision": _precisions,
            "recall": _recalls,
            "f1-score": _f1_scores}

    def obtain_thresholds_with_tolerance(self, tolerance: float = 0.0):
        """
        Method to obtain thresholds based on tolerance. Methods seeks for the best possbile threshold bases on
        f1- score. Then it seeks for the existance of the better score inside the tolerance interval regarding the
        found best f1 - score

        Args:
            tolerance, float, tolerance inteval - recommended 1-2% (0.01, 0.02)
        """
        
        _tmp_scores_dict = {}

        # Iterate through body parts and calculate thresholds
        for _body_part in self.body_parts:
            _stats = self.__obtain_statistics_data(true_arr = self.true[_body_part], 
                                                   pred_arr = self.predicted[_body_part])            
            
             # Calculate tolerance
            _recall_threshold = np.max(_stats["recall"]) - tolerance * np.max(_stats["recall"])
            _precision_threshold = np.max(_stats["precision"]) - tolerance * np.max(_stats["precision"])
            _f1_threshold = np.max(_stats["f1-score"]) - tolerance * np.max(_stats["f1-score"])

            # Find the indices of the last values above the thresholds
            _index_recall_last_above = np.where(_stats["recall"] >= _recall_threshold)[0][-1]
            _index_precision_last_above = np.where(_stats["precision"] >= _precision_threshold)[0][-1]
            _index_f1_last_above = np.where(_stats["f1-score"] >= _f1_threshold)[0][-1]

            _tmp_scores_dict[_body_part] = _stats["interval"][_index_f1_last_above]
        
        self.threshold_dict = _tmp_scores_dict

    def obtain_threshold_from_xlsx(self, xlsx_path:str = None)->dict:
        """
        Method which obtains threshold dict from the given xlsx file in "Threshold" sheet.
        """
        # Check if ifle exists
        assert os.path.exists(xlsx_path), f"File path {xlsx_path} is invalid"

        # Try to obtain dict
        try:
            _df_tmp = pd.read_excel(xlsx_path, sheet_name = "Thresholds", index_col = 0)
            _data_dict = _df_tmp[0].to_dict()
            self.threshold_dict = _data_dict
        except:
            print(f"Something went wrong with loading thresholds from dict {xlsx_path}! Please check path and xlsx file")
            return -1
        return _data_dict

    def obtain_scores(self, 
                      threshold = 0.5,
                      output_xlsx: str = None)->dict:
        """
        Method which obtaines predictions. It accepts threshold as a parameter.

        Args:
            * threshold, <float, dict, str, None>, it can be float - same threshold is applied to all body parts,
            dict (key = bodypart, value = threshold value for that body part), str path to the json holding the dict
            with threshold. One can be obtained with the method obtain_threhsolds. In case of None, local caluclated storage
            is loaded.

            * output_xlsx, str, path to the xlsx file where the final scoring will be calculated
        
        Returns:
            * dict, dict containing necessary obtained scores for each body part and global (meaning how many times it predicted 
            corectly the whole case
        """
        # Create storage
        _thresholds_dict = {}
        # Populate storage
        # None case
        if threshold == None:
            threshold = self.threshold_dict
       
        # Float case
        if isinstance(threshold, float):
            for _key in self.body_parts:
                _thresholds_dict[_key] = threshold
        
        # String case
        if isinstance(threshold, str):
            with open(threshold, 'r') as _file:
                _thresholds_dict = json.load(_file)

        # Dict case
        if isinstance(threshold, dict):
            _thresholds_dict = threshold

        # Update main
        self.threshold_dict = _thresholds_dict

        # Obtain scores
        _score_dict = {}
        for _body_part in self.body_parts:
            # Obtain predictions
            _true_array = self.true[_body_part]
            _predict_array = self.predicted[_body_part]

            # Apply threshold
            _thresholded_arr = [1 if float(_element) > _thresholds_dict[_body_part] else 0 for _element in _predict_array]

            # Calcualte metrices
            _f1 = f1_score(_true_array, _thresholded_arr)
            _recall = recall_score(_true_array, _thresholded_arr)
            _precision = precision_score(_true_array, _thresholded_arr)
            #print(_true_array, _thresholded_arr)
            # Confusion matrix can colaps if there is an issue with TP (for instance body part is not presented)
            try:
                _tn, _fp, _fn, _tp = confusion_matrix(_true_array, _thresholded_arr).ravel()
            except:
                _tn, _fp, _fn, _tp = (0,0,0,0)
            
            # Storage
            _body_dict = {"support": _fn + _tp,
                          "tp": _tp,
                          "fp": _fp,
                          "fn": _fn,
                          "tn": _tn,
                          "recall": _recall,
                          "precision": _precision,
                          "f1-score": _f1}
            _score_dict[_body_part] = _body_dict
        
        # Export to xlsx
        if output_xlsx != None:
            with pd.ExcelWriter(output_xlsx, engine='xlsxwriter') as _writer:
                _df = pd.DataFrame.from_dict(_score_dict, orient='index')
                _df.loc['Average'] = _df.mean()
                _df.to_excel(_writer, sheet_name = "Results", index = True)

                _df = pd.DataFrame.from_dict(self.threshold_dict, orient='index')
                _df.to_excel(_writer, sheet_name = "Thresholds", index = True)

        # Return 
        return _score_dict
    

    def obtain_in_depth_scores(self, threshold = 0.5, output_dir:str = None):
        """
        This is in depth method for obtaining scores and exporting actual images to the files. Script is based on
        evaluate_dl_NMDD_BCH_SAROS.py

        Args:
            * threshold, <float, dict, str, None>, it can be float - same threshold is applied to all body parts,
            dict (key = bodypart, value = threshold value for that body part), str path to the json holding the dict
            with threshold. One can be obtained with the method obtain_threhsolds. In case of None, local caluclated storage
            is loaded.
            * output_dir, str, place where the outputs will be stored
        """
        # To be connected with output_dir
        _root_path = output_dir
        # Remove the directory if it exists
        if os.path.exists(_root_path):
            shutil.rmtree(_root_path)

        # Create a new directory
        os.makedirs(_root_path)

        # Create storage
        _thresholds_dict = {}
        # Populate storage
        # None case
        if threshold == None:
            threshold = self.threshold_dict
       
        # Float case
        if isinstance(threshold, float):
            for _key in self.body_parts:
                _thresholds_dict[_key] = threshold
        
        # String case
        if isinstance(threshold, str):
            with open(threshold, 'r') as _file:
                _thresholds_dict = json.load(_file)

        # Dict case
        if isinstance(threshold, dict):
            _thresholds_dict = threshold

        # Update main
        self.threshold_dict = _thresholds_dict
        _thresholds_list = list(self.threshold_dict.values())
        # Obtain scores
        _paths = self.true["Image"].to_list()
        _binary_true = self.true.iloc[:, 2:].values.tolist()
        _binary_true = [[int(_element) for _element in _row] for _row in _binary_true]
        _binary_predictions = self.predicted.iloc[:, 2:].values.tolist()
        _binary_predictions = [[1 if _element > _thresholds_list[_i] else 0 for _i, _element in enumerate(_row)] for _row in _binary_predictions]

        # Storage dict
        _storage_dict = {}
        for _i, _key in enumerate(self.body_parts):
            _TP = []
            _TN = []
            _FP = []
            _FN = []
            for _true, _pred, _path in zip(_binary_true, _binary_predictions, _paths):
                _t = _true[_i]
                _p = _pred[_i]
                # TP
                if _t == 1 and _p == 1:
                    _TP.append(_path)
                
                # TN
                if _t == 0 and _p == 0:
                    _TN.append(_path)
                
                # FN
                if _t == 1 and _p == 0:
                    _FN.append(_path)
                
                # FP
                if _t == 0 and _p == 1:
                    _FP.append(_path)
            
            # Update sotrage
            _storage_dict[_key] = {"TP": _TP, "TN": _TN, "FN": _FN, "FP": _FP}

        _cm = skm.multilabel_confusion_matrix(np.array(_binary_true), np.array(_binary_predictions))
        _report = skm.classification_report(np.array(_binary_true), np.array(_binary_predictions), target_names = self.body_parts, output_dict=True)
        _df = pd.DataFrame(_report).transpose()
        _df.to_excel(os.path.join(_root_path, "general_report.xlsx"))

        # Plot info
        for _i, _key in enumerate(self.body_parts):
            
            # Create dir
            if os.path.exists(os.path.join(_root_path, _key)):
                    shutil.rmtree(os.path.join(_root_path, _key))
            os.makedirs(os.path.join(_root_path, _key))
            
            # Export path
            os.makedirs(os.path.join(_root_path, _key, "TP"))
            for _index, _path in enumerate(_storage_dict[_key]["TP"]):
                _path_components = _path.split(os.sep)
                _new_name = f"{_path_components[5]}_{_path_components[7]}_{_index}_main.png"        
                shutil.copy(_path, os.path.join(_root_path,_key, "TP", _new_name))
            
            os.makedirs(os.path.join(_root_path, _key, "FP"))
            for _index, _path in enumerate(_storage_dict[_key]["FP"]):
                _path_components = _path.split(os.sep)
                _new_name = f"{_path_components[5]}_{_path_components[7]}_{_index}_main.png"
                shutil.copy(_path, os.path.join(_root_path,_key, "FP", _new_name))

            os.makedirs(os.path.join(_root_path, _key, "TN"))
            for _index, _path in enumerate(_storage_dict[_key]["TN"]):
                _path_components = _path.split(os.sep)
                _new_name = f"{_path_components[5]}_{_path_components[7]}_{_index}_main.png"
                shutil.copy(_path, os.path.join(_root_path,_key, "TN", _new_name))

            os.makedirs(os.path.join(_root_path, _key, "FN"))
            for _index, _path in enumerate(_storage_dict[_key]["FN"]):
                _path_components = _path.split(os.sep)
                _new_name = f"{_path_components[5]}_{_path_components[7]}_{_index}_main.png"

                shutil.copy(_path, os.path.join(_root_path,_key, "FN", _new_name))



            _report = skm.classification_report(np.array(_binary_true), 
                                                np.array(_binary_predictions), 
                                                labels = [_i], 
                                                target_names = [_key],
                                                output_dict = True)
                
            plt.figure(figsize = (10,10))
            _group_names = ["True Neg","False Pos","False Neg","True Pos"]
            _group_counts = ["{0:0.0f}".format(_value) for _value in _cm[_i].flatten()]
            _group_percentages = ["{0:.2%}".format(_value) for _value in _cm[_i].flatten()/np.sum(_cm[_i])]
            _labels = [f"{_v1}\n{_v2}\n{_v3}" for _v1, _v2, _v3 in
                    zip(_group_names, _group_counts, _group_percentages)]
            _labels = np.asarray(_labels).reshape(2,2)
            _cm_plot = sns.heatmap(_cm[_i], annot= _labels, fmt="", cmap='Blues')
            _cm_plot.set_title(f"{_key}", size = 20)
            _cm_plot.set_xlabel(f"precision: {_report[_key]['precision']:.2}\n recall: {_report[_key]['recall']:.2}\n f1-score: {_report[_key]['f1-score']:.2}\n support: {_report[_key]['support']:4.2}\n", size = 20)
            _cm_plot.figure.savefig(os.path.join(_root_path, _key, f"{_key}.png"),  bbox_inches='tight')
