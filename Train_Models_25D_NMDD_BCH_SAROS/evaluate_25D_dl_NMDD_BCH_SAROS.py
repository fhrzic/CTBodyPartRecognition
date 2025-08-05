# Import libs
from Utils.config_25D_NMDD_BCH_SAROS import *
from Utils.dataloader_25D_NMDD_BCH_SAROS import *
from Utils.models_25D_NMDD_BCH_SAROS import *
from Utils.train_model_25D_NMDD_BCH_SAROS import *
import sklearn.metrics as skm
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Create dataset
# Load configs 
_data_config = datasetConfig()
_loader_config = loaderConfig()
_model_config = modelConfig()

#Create data
loaderConfig.use_gpu = 'cuda:2'
loaderConfig.number_of_workers = 4
loaderConfig.batch_size  = 8
print(_loader_config)

# Training dataset
datasetConfig.mapping_xlsx_path_NMDD = "/mnt/SSD/Franko/Desktop/Train_Models_2.5D_BCH_NMDD_SAROS/FinalTrainingData/Patches_NMDD/SourceDataset.xlsx"
datasetConfig.batch_xlsx_path_NMDD = "/mnt/SSD/Franko/Desktop/Train_Models_2.5D_BCH_NMDD_SAROS/FinalTrainingData/Patches_NMDD"
datasetConfig.mapping_xlsx_path_SAROS = "/mnt/SSD/Franko/Desktop/Train_Models_2.5D_BCH_NMDD_SAROS/FinalTrainingData/Patches_SAROS/SourceDataset.xlsx"
datasetConfig.batch_xlsx_path_SAROS = "/mnt/SSD/Franko/Desktop/Train_Models_2.5D_BCH_NMDD_SAROS/FinalTrainingData/Patches_SAROS"
datasetConfig.dir_path_BCH = "/mnt/SSD/Franko/Desktop/Train_Models_2.5D_BCH_NMDD_SAROS/FinalTrainingData/VERIFIED_BCH"
datasetConfig.dir_path_NMDD = "/mnt/SSD/Franko/Desktop/Train_Models_2.5D_BCH_NMDD_SAROS/FinalTrainingData/Full_NMDD"
datasetConfig.dir_path_SAROS = "/mnt/SSD/Franko/Desktop/Train_Models_2.5D_BCH_NMDD_SAROS/FinalTrainingData/Full_SAROS"
      
      
datasetConfig.split_ratio = 0.75
datasetConfig.image_dimension = 224
datasetConfig.folds = None
datasetConfig.type = "test"
datasetConfig.verbose = False
print(_data_config)

# Test data loade
_dl = init_dataloader(_loader_config, _data_config)

# Model config
modelConfig.gpu = 'cuda:2'
modelConfig.number_of_output_classes = 14
modelConfig.model_name = 'dense161'
modelConfig.loss = 'bce'
modelConfig.valid_epochs = "1_1"
modelConfig.early_stopping = 15
modelConfig.learning_rate = 1e-3#"Auto" #"Auto" #1e-3
modelConfig.opt_name = 'ADAMW'
modelConfig.epochs = 500
modelConfig.wandb = False
modelConfig.info = f"densenet161_25D_NMDD_BCH_SAROS"

# Set augumentation method
modelConfig.augmentation_model = 'GRAY_Augmentation'

# Pretrained
modelConfig.pretrained = False
print(_model_config)

# Set path
_root_path = "25D_densen161_evaluation/"

# Load model
_training = model_training_app(None, None, _model_config, _root_path)
_training.load_model("/mnt/SSD/Franko/Desktop/Train_Models_2.5D_BCH_NMDD_SAROS/denseNet161_25D_NMDD_BCH_SAROS/dense161_ADAMW_0.001_bcevalid_best_model.pth")

# Obtain data
_prediction_dict = _training.model_predict_from_dl_plain_prediction(_dl)

# Convert data
_binary_predictions = [[1 if _element > 0.5 else 0 for _element in _row] for _row in _prediction_dict["Pred"]]
_binary_true =  [[int(_element) for _element in _row] for _row in _prediction_dict["True"]]
_paths = _prediction_dict["Paths"]
_dataset = _prediction_dict["Datasets"]

#from collections import Counter
#counter = Counter(_dataset)
#print("Number of data:", counter)

# Label _keys
_keys = list(datasetConfig.remap_dict.keys())

# Storage dict
_storage_dict = {}
for _i, _key in enumerate(_keys):
    _TP = []
    _TN = []
    _FP = []
    _FN = []
    for _true, _pred, _path, _data in zip(_binary_true, _binary_predictions, _paths, _dataset):
        # Fix path to point to the image based on dataset
        if _data == "patches_NMDD":
            _path = _path.replace("reduced.npy", "main.png")
            
        if _data == "patches_SAROS":
            _path = _path.replace("reduced.npy", "main.png")
            
        if _data == "full_NMDD":
            _dir_name = os.path.basename(os.path.dirname(_path))
            _path = _path.replace("reduced.npy", f"{_dir_name}_reducted_image.png")

        if _data == "full_SAROS":
            _path = _path.replace("reduced.npy", "reducted_image.png")

        if _data == "BCH":
            _path = _path.replace("reduced.npy", "reducted_image.png")

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
_report = skm.classification_report(np.array(_binary_true), np.array(_binary_predictions), target_names =_keys, output_dict=True)
_df = pd.DataFrame(_report).transpose()
_df.to_excel(os.path.join(_root_path, "general_report.xlsx"))

# Plot info
for _i, _key in enumerate(_keys):
       
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

