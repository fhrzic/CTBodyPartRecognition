# Libs
import os
from Utils.config_NNMD import *
from Utils.dataloader_NNMD import *
from Utils.models_NNMD import *
from Utils.train_model_NNMD import *
import shutil
import time
# Test Data loader
# Load configs 
_data_config = datasetConfig()
_loader_config = loaderConfig()
_model_config = modelConfig()

# Set dataset
_k = 4
for _k in [0,1,2,3,4]:
    datasetConfig.labels_xlsx_path = '/home/franko/Desktop/BodyPartTraining/Dataset/Output-Labels.xlsx'
    datasetConfig.cheetsheet_xlsx_path = '/home/franko/Desktop/BodyPartTraining/Dataset/cheetsheet.xlsx'
    datasetConfig.imgs_png_home_path = '/home/franko/Desktop/BodyPartTraining/Dataset/Images'
    datasetConfig.image_sufix = "_reducted_image"
    datasetConfig.label_type = "reduced_cluster_remaped"
    datasetConfig.label_dimension = 14
    datasetConfig.split_ratio = 0.75
    datasetConfig.image_dimension = 224
    datasetConfig.folds = [_k, 5]
    datasetConfig.type = 'test'
    print(_data_config)

    _test_data_loader = init_dataloader(_loader_config, _data_config)

    # Model config
    modelConfig.gpu = 'cuda:1'
    modelConfig.number_of_output_classes = 14
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
    _name = f"/home/franko/Desktop/BodyPartTraining/Scripts/NNMD_EFF0_Training4k_fold_{_k}/eff0_ADAMW_0.001_bcevalid_best_model.pth"
    _evaluation.load_model(_name)
    print("**************************************")


    # Predict on all images
    # Crate storage
    _output_dir_name = f"Output_Folds/Fold_{_k}"
    os.makedirs(_output_dir_name)

    _predictions = _evaluation.model_predict_from_dl_full_body(input_data_loader = _test_data_loader, input_image_path = datasetConfig.imgs_png_home_path, input_xlsx_path = datasetConfig.labels_xlsx_path, input_cheatsheet_path = datasetConfig.cheetsheet_xlsx_path, label_type = "reduced_cluster_remaped")

    for _item in _predictions:

        # Create export
        # Names
        _path = "cheatsheet_names.xlsx"
        _cheatsheet_names_df = pd.read_excel(_path, sheet_name = "reduced_cluster_remaped")
        _columns = _cheatsheet_names_df.set_index('Label')['Name'].to_dict()
        _columns = list(_columns.values())

        # Obtain export data and recalculate areas
        _predicted = _item["prediction"]
        _true = _item["true"]
        _area = _item["area"]

        # Generate data frame
        _data = {"Area": _area, "True": _true, "Predicted": _predicted}

        # Create export
        _dir = os.path.join(_output_dir_name, _item["id"])

        # Create saving dirs
        if os.path.exists(_dir):
            shutil.rmtree(_dir)
        os.makedirs(_dir)

        # Save data
        _export_df = pd.DataFrame(_data, index=_columns).T
        _export_df.to_excel(os.path.join(_dir, 'results.xlsx'), engine='xlsxwriter')

        # Save image
        import matplotlib.image
        matplotlib.image.imsave(os.path.join(_dir, os.path.basename(_item["image_path"])), _item["image"])
        shutil.copy(_item["image_path"], os.path.join(_dir, "original.png"))
        

