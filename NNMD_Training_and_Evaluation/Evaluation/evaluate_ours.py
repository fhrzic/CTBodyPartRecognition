# Libs
import os
from Utils.config_NNMD import *
from Utils.dataloader_NNMD import *
from Utils.models_NNMD import *
from Utils.train_model_NNMD import *

# Test Data loader
# Load configs 
_data_config = datasetConfig()
_loader_config = loaderConfig()
_model_config = modelConfig()

# Set dataset
datasetConfig.labels_xlsx_path = "/home/franko/Desktop/BodyPartTraining/Dataset_Ours/Output-Labels.xlsx"
datasetConfig.cheetsheet_xlsx_path = "/home/franko/Desktop/BodyPartTraining/Dataset_Ours/cheetsheet.xlsx"
datasetConfig.imgs_png_home_path = "/home/franko/Desktop/BodyPartTraining/Dataset_Ours/Images"
datasetConfig.image_sufix = "_reducted_image"
datasetConfig.label_type = "cluster_remaped"
datasetConfig.label_dimension = 35
datasetConfig.split_ratio = 1
datasetConfig.image_dimension = 224
datasetConfig.type = 'train'
print(_data_config)

_test_data_loader = init_dataloader(_loader_config, _data_config)

# Create storage in case that it is from DL
_paths_storage = []
for _item in _test_data_loader:
    # Obtain data
    _, _, _paths = _item
    # Store paths
    for _path in _paths:
    	_paths_storage += [f"/home/franko/Desktop/BodyPartTraining/Dataset_Ours/Images/{_path}{datasetConfig.image_sufix}.png"]
    
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
#_image = _paths_storage[0]
#_predictions = _evaluation.model_predict_from_image_full_body(input_image_path = _image, input_xlsx_path = datasetConfig.labels_xlsx_path, input_cheatsheet_path = datasetConfig.cheetsheet_xlsx_path)

# Predict on all images
for _i, _image in enumerate(_paths_storage):
    print(f"Working on: {_i}/{len(_paths_storage)}...")
    _predictions = _evaluation.model_predict_from_image_full_body(input_image_path = _image, input_xlsx_path = datasetConfig.labels_xlsx_path, input_cheatsheet_path = datasetConfig.cheetsheet_xlsx_path)

    # Create export
    # Names
    _path = "/home/franko/Desktop/BodyPartTraining/Dataset_Ours/cheatsheet_names.xlsx"
    _cheatsheet_names_df = pd.read_excel(_path, sheet_name = "cluster_remaped")
    _columns = _cheatsheet_names_df.set_index('Label')['Name'].to_dict()
    _columns = list(_columns.values())

    # Obtain export data and recalculate areas
    _predicted = _predictions["prediction"]
    _true = _predictions["true"]
    _area_short = _predictions["area"]
    _area_full = []
    _index = 0

    for _value in _true:
        if _value == 1:
           _area_full.append(_area_short[_index])
           _index += 1
        else:
           _area_full.append(0)
    
    # Generate data frame
    _data = {"Area": _area_full, "True": _true, "Predicted": _predicted}

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
    
