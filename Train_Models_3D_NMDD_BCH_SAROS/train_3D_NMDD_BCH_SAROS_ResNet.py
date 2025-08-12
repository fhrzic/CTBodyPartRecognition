import os
from Utils.config_3D_NMDD_BCH_SAROS import *
from Utils.dataloader_3D_NMDD_BCH_SAROS import *
from Utils.models_3D_NMDD_BCH_SAROS import *
from Utils.train_model_3D_NMDD_BCH_SAROS import *
from multiprocessing import Process
import shutil
import time

def main():
    # Load configs 
    _data_config = datasetConfig()
    _loader_config = loaderConfig()
    _model_config = modelConfig()

    #Create data
    loaderConfig.use_gpu = 'cuda:1'
    loaderConfig.number_of_workers = 4
    loaderConfig.batch_size  = 4
    print(_loader_config)

    # Training dataset
    datasetConfig.mapping_xlsx_path_NMDD = "Patches_NMDD/SourceDataset.xlsx"
    datasetConfig.batch_xlsx_path_NMDD = "Patches_NMDD"
    datasetConfig.mapping_xlsx_path_SAROS = "Patches_SAROS/SourceDataset.xlsx"
    datasetConfig.batch_xlsx_path_SAROS = "Patches_SAROS"
    datasetConfig.dir_path_BCH = "VERIFIED_BCH"
    datasetConfig.dir_path_NMDD = "Full_NMDD"
    datasetConfig.dir_path_SAROS = "Full_SAROS"
      
    
    datasetConfig.split_ratio = 0.75
    datasetConfig.image_dimension = 224
    datasetConfig.folds = None
    datasetConfig.type = "train"
    datasetConfig.verbose = False
    print(_data_config)

    # Train data loader
    _train_dl = init_dataloader(_loader_config, _data_config)

    # Test dataset
    # Validation dataset
    datasetConfig.type = "valid"

    # Valid data loader
    _valid_dl = init_dataloader(_loader_config, _data_config)
    
    # Model config
    modelConfig.gpu = 'cuda:1'
    modelConfig.number_of_output_classes = 14
    modelConfig.number_of_output_channels = 1
    modelConfig.input_shape = {"input_shape": (1, datasetConfig.image_dimension, datasetConfig.image_dimension, datasetConfig.image_dimension)}
    modelConfig.model_name = 'ResNet'
    modelConfig.loss = 'bce'
    modelConfig.valid_epochs = "1_1"
    modelConfig.early_stopping = 30
    modelConfig.learning_rate = 1e-3#"Auto" #"Auto" #1e-3
    modelConfig.opt_name = 'ADAMW'
    modelConfig.epochs = 500
    modelConfig.wandb = False
    modelConfig.info = f"ResNet_3D_NMDD_BCH_SAROS"

    # Set augumentation method
    modelConfig.augmentation_model = 'GRAY_Augmentation'#'GRAY_Simple'#'GRAY_Augmentation'

    # Pretrained
    modelConfig.pretrained = False
    print(_model_config)


    _training = model_training_app(_train_dl, _valid_dl, _model_config, f"ResNet_3D_NMDD_BCH_SAROS/")
    _training.freeze_unfreeze_model(freeze = False)
    _training.start_training()
        

# Workarround for multiprocessing
if __name__ == '__main__':
    main()
