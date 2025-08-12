import os
from Utils.config_NNMD import *
from Utils.dataloader_NNMD import *
from Utils.models_NNMD import *
from Utils.train_model_NNMD import *


def main():
    # Load configs 
    _data_config = datasetConfig()
    _loader_config = loaderConfig()
    _model_config = modelConfig()

    #Create data
    loaderConfig.use_gpu = 'cuda:1'
    loaderConfig.number_of_workers = 4
    loaderConfig.batch_size  = 32
    print(_loader_config)

    # Training dataset
    datasetConfig.labels_xlsx_path = 'Output-Labels.xlsx'
    datasetConfig.cheetsheet_xlsx_path = 'cheetsheet.xlsx'
    datasetConfig.imgs_png_home_path = 'Images'
    datasetConfig.image_sufix = "_reducted_image"
    datasetConfig.label_type = "cluster_remaped"
    datasetConfig.label_dimension = 35
    datasetConfig.split_ratio = 0.75
    datasetConfig.image_dimension = 224
    datasetConfig.type = "train"
    datasetConfig.verbose = False
    print(_data_config)
    # Train data loader
    _train_dl = init_dataloader(_loader_config, _data_config)

    # Validation dataset
    datasetConfig.type = "valid"
    print(_data_config)
    # Valid data loader
    _valid_dl = init_dataloader(_loader_config, _data_config)
    
    # Test dataset
    # Validation dataset
    datasetConfig.type = "test"
    print(_data_config)
    # Valid data loader
    _valid_dl = init_dataloader(_loader_config, _data_config)

    # Model config
    modelConfig.gpu = 'cuda:1'
    modelConfig.number_of_output_classes = 35
    modelConfig.model_name = 'eff0'
    modelConfig.loss = 'bce'
    modelConfig.valid_epochs = "1_1"
    modelConfig.early_stopping = 100
    modelConfig.learning_rate = 1e-3#"Auto" #"Auto" #1e-3
    modelConfig.opt_name = 'ADAMW'
    modelConfig.epochs = 500
    modelConfig.wandb = True
    modelConfig.info = 'Training4kdataset'

    # Set augumentation method
    modelConfig.augmentation_model = 'GRAY_Augmentation'

    # Pretrained
    modelConfig.pretrained = True
    print(_model_config)


    _training = model_training_app(_train_dl, _valid_dl, _model_config, "NNMD_EFF0_Training4k/")
    _training.freeze_unfreeze_model(freeze = False)
    _training.start_training()

# Workarround for multiprocessing
if __name__ == '__main__':
    main()
