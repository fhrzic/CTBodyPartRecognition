import os
from Utils.config_NNMD import *
from Utils.dataloader_NNMD import *
from Utils.models_NNMD import *
from Utils.train_model_NNMD import *

# OPTIONAL: Load the "autoreload" extension so that code can change
#%load_ext autoreload

# OPTIONAL: always reload modules so that as you change code in src, it gets loaded
#%autoreload 2

# Load configs 
_data_config = datasetConfig()
_loader_config = loaderConfig()
_model_config = modelConfig()

# Set dataset
datasetConfig.labels_xlsx_path = '/home/franko/Desktop/BodyPartTraining/Dataset/Output-Labels.xlsx'
datasetConfig.cheetsheet_xlsx_path = '/home/franko/Desktop/BodyPartTraining/Dataset/cheetsheet.xlsx'
datasetConfig.imgs_png_home_path = '/home/franko/Desktop/BodyPartTraining/Dataset/Images'
datasetConfig.image_sufix = "_reducted_image"
datasetConfig.label_type = "cluster_remaped"
datasetConfig.label_dimension = 35
datasetConfig.split_ratio = 0.75
datasetConfig.image_dimension = 224

# Data loaders
datasetConfig.type = 'train'
_train_data_loader = init_dataloader(_loader_config, _data_config)
datasetConfig.type = 'valid'
_valid_data_loader = init_dataloader(_loader_config, _data_config)
datasetConfig.type = 'test'
_test_data_loader = init_dataloader(_loader_config, _data_config)



# Config training
# Model config
modelConfig.gpu = 'cuda:1'
modelConfig.number_of_output_classes = 35
modelConfig.model_name = 'eff0'
modelConfig.loss = 'bce'
modelConfig.valid_epochs = "1_1"
modelConfig.early_stopping = 50
modelConfig.learning_rate = 1e-3#"Auto" #"Auto" #1e-3
modelConfig.opt_name = 'ADAMW'
modelConfig.epochs = 500
modelConfig.wandb = False
modelConfig.custom_info = 'TestTrainingFirstTime'
print(_model_config)

# Set augumentation method
modelConfig.augmentation_model = 'GRAY_Augmentation'
modelConfig.pretrained = False


_training = model_training_app(_train_data_loader, _valid_data_loader, _model_config, "TEST1/")
_training.load_model("/home/franko/Desktop/BodyPartTraining/Scripts/NNMD_VGG_TestTraining/eff0_ADAMW_0.001_bcevalid_best_model.pth")
_training.model_predict_from_dl(_train_data_loader,"train")
_training.model_predict_from_dl(_valid_data_loader,"valid")
_training.model_predict_from_dl(_test_data_loader,"test")



