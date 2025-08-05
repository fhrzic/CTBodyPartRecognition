# Import libs
from Utils.config_3D_NMDD_BCH_SAROS import *
from Utils.dataloader_3D_NMDD_BCH_SAROS import *
from Utils.train_model_3D_NMDD_BCH_SAROS import *
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
loaderConfig.use_gpu = 'cuda:0'
loaderConfig.number_of_workers = 4
loaderConfig.batch_size  = 4
print(_loader_config)

# Training dataset
datasetConfig.mapping_xlsx_path_NMDD = "/mnt/SSD/Franko/Desktop/Train_Models_3D_BCH_NMDD_SAROS/FinalTrainingData/Patches_NMDD/SourceDataset.xlsx"
datasetConfig.batch_xlsx_path_NMDD = "/mnt/SSD/Franko/Desktop/Train_Models_3D_BCH_NMDD_SAROS/FinalTrainingData/Patches_NMDD"
datasetConfig.mapping_xlsx_path_SAROS = "/mnt/SSD/Franko/Desktop/Train_Models_3D_BCH_NMDD_SAROS/FinalTrainingData/Patches_SAROS/SourceDataset.xlsx"
datasetConfig.batch_xlsx_path_SAROS = "/mnt/SSD/Franko/Desktop/Train_Models_3D_BCH_NMDD_SAROS/FinalTrainingData/Patches_SAROS"
datasetConfig.dir_path_BCH = "/mnt/SSD/Franko/Desktop/Train_Models_3D_BCH_NMDD_SAROS/FinalTrainingData/VERIFIED_BCH"
datasetConfig.dir_path_NMDD = "/mnt/SSD/Franko/Desktop/Train_Models_3D_BCH_NMDD_SAROS/FinalTrainingData/Full_NMDD"
datasetConfig.dir_path_SAROS = "/mnt/SSD/Franko/Desktop/Train_Models_3D_BCH_NMDD_SAROS/FinalTrainingData/Full_SAROS"
      
datasetConfig.split_ratio = 0.75
datasetConfig.image_dimension = 340
datasetConfig.folds = None
datasetConfig.type = "valid"
datasetConfig.verbose = False
print(_data_config)

# Test data loade
_dl = init_dataloader(_loader_config, _data_config)


# Model config
modelConfig.gpu = 'cuda:0'
modelConfig.number_of_output_classes = 14
modelConfig.input_channels = 1
modelConfig.input_shape = {"input_shape": (1, datasetConfig.image_dimension, datasetConfig.image_dimension, datasetConfig.image_dimension)}
modelConfig.model_name = 'VoxCNN'
modelConfig.loss = 'bce'
modelConfig.valid_epochs = "1_1"
modelConfig.early_stopping = 15
modelConfig.learning_rate = 1e-3#"Auto" #"Auto" #1e-3
modelConfig.opt_name = 'ADAMW'
modelConfig.epochs = 500
modelConfig.wandb = False
modelConfig.info = f"VoxCNN_3D_NMDD_BCH_SAROS_vol2"
# Set augumentation method
modelConfig.augmentation_model = 'GRAY_Augmentation'

# Pretrained
modelConfig.pretrained = False
print(_model_config)

# Set path
_root_path = "VoxCNN_3D_NMDD_BCH_SAROS_eval_preidctions_dummy_for_obtaining/"

# Load model
_training = model_training_app(None, None, _model_config, _root_path)
_training.load_model("/mnt/SSD/Franko/Desktop/Train_Models_3D_BCH_NMDD_SAROS/VoxCNN_3D_NMDD_BCH_SAROS_vol2/VoxCNN_ADAMW_0.001_bcevalid_best_model.pth")

# Obtain data
_prediction_dict = _training.model_predict_from_dl_plain_prediction(_dl)

# Convert data
_binary_predictions = [[1 if _element > 0.5 else 0 for _element in _row] for _row in _prediction_dict["Pred"]]
_real_predictions = [[float(_element) for _element in _row] for _row in _prediction_dict["Pred"]]
_binary_true =  [[int(_element) for _element in _row] for _row in _prediction_dict["True"]]
_paths = _prediction_dict["Paths"]
_dataset = _prediction_dict["Datasets"]

# Label _keys
_keys = list(datasetConfig.remap_dict.keys())

# Create info dataframe
_info_df = pd.DataFrame({"Paths": _paths, "Dataset": _dataset})

# Create dataframe
_binary_predictions_df = pd.DataFrame(_binary_predictions, columns=_keys)
_real_predictions_df = pd.DataFrame(_real_predictions, columns=_keys)
_true_predictions_df = pd.DataFrame(_binary_true, columns=_keys)

# Merge
_binary_predictions_df = pd.concat([_info_df, _binary_predictions_df], axis = 1)
_real_predictions_df = pd.concat([_info_df, _real_predictions_df], axis = 1)
_true_predictions_df = pd.concat([_info_df, _true_predictions_df], axis = 1)

# Create a Pandas Excel writer using XlsxWriter as the engine
with pd.ExcelWriter(f"{datasetConfig.type}_VoxCNN_3D_Size340_output.xlsx", engine='xlsxwriter') as writer:
    # Write each DataFrame to a different sheet
    _binary_predictions_df.to_excel(writer, sheet_name='binary', index=False)
    _real_predictions_df.to_excel(writer, sheet_name='real', index=False)
    _true_predictions_df.to_excel(writer, sheet_name='true', index=False)
