# Libs
import os
from Utils.config_NNMD import *
from Utils.dataloader_NNMD import *
from Utils.models_NNMD import *
from Utils.train_model_NNMD import *

# Test Data loader
# Load configs 
_model_config = modelConfig()
    
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
modelConfig.custom_info = 'Predicion'

# Set augumentation method
modelConfig.augmentation_model = 'GRAY_Augmentation'
modelConfig.pretrained = False
print(_model_config)

# Load model
_evaluation = model_training_app(None, None, _model_config, "TEST1/")
_evaluation.load_model("/home/franko/Desktop/BodyPartTraining/Scripts/NNMD_EFF0_Training4k/eff0_ADAMW_0.001_bcevalid_best_model.pth")
print("**************************************")

# Predict one image
_image = "/home/franko/Desktop/BodyPartTraining/Evaluation/TEST.png"
_predictions = _evaluation.model_predict_from_image_full_body(input_image_path = _image, input_xlsx_path =  None, input_cheatsheet_path = None)

# Create export
# Names
_path = "/home/franko/Desktop/BodyPartTraining/Dataset/cheatsheet_names.xlsx"
_cheatsheet_names_df = pd.read_excel(_path, sheet_name = "cluster_remaped")
_columns = _cheatsheet_names_df.set_index('Label')['Name'].to_dict()
_columns = list(_columns.values())

# Obtain export data and recalculate areas
_predicted = _predictions["prediction"]
    
#Generate data frame
_data = {"Predicted": _predicted}

# Create export
_dir = os.path.basename(_image).split(datasetConfig.image_sufix)[0]
_dir = os.path.join("Output", _dir)

# Create saving dirs
if os.path.exists(_dir):
   shutil.rmtree(_dir)
os.makedirs(_dir)
    
# Save data
_export_df = pd.DataFrame(_data, index=_columns).T
_export_df.to_excel(os.path.join(_dir, 'results.xlsx'), engine='xlsxwriter')

# Save image
import matplotlib.image
matplotlib.image.imsave(os.path.join(_dir, os.path.basename(_image)), _predictions["image"])
shutil.copy(_predictions["image_path"], os.path.join(_dir, "original.png"))
