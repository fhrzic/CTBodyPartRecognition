import os
from Utils.config_25D_NMDD_BCH_SAROS import *
from Utils.dataloader_25D_NMDD_BCH_SAROS import *
from Utils.models_25D_NMDD_BCH_SAROS import *
from Utils.train_model_25D_NMDD_BCH_SAROS import *
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
    loaderConfig.batch_size  = 32
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
    datasetConfig.type = "train"
    datasetConfig.verbose = False
    print(_data_config)

    # Train data loader
    _train_dl = init_dataloader(_loader_config, _data_config)

    # Test dataset
    # Validation dataset
    datasetConfig.type = "valid"
    print(_data_config)
    # Valid data loader
    _valid_dl = init_dataloader(_loader_config, _data_config)

    # Model config
    modelConfig.gpu = 'cuda:1'
    modelConfig.number_of_output_classes = 14
    modelConfig.model_name = 'dense161'
    modelConfig.loss = 'bce'
    modelConfig.valid_epochs = "1_1"
    modelConfig.early_stopping = 30
    modelConfig.learning_rate = 1e-3#"Auto" #"Auto" #1e-3
    modelConfig.opt_name = 'ADAMW'
    modelConfig.epochs = 500
    modelConfig.wandb = False
    modelConfig.info = f"denseNet161_25D_NMDD_BCH_SAROS"

    # Set augumentation method
    modelConfig.augmentation_model = 'GRAY_Augmentation'

    # Pretrained
    modelConfig.pretrained = False
    print(_model_config)


    _training = model_training_app(_train_dl, _valid_dl, _model_config, f"denseNet161_25D_NMDD_BCH_SAROS/")
    _training.freeze_unfreeze_model(freeze = False)
    _training.start_training()
        

# Workarround for multiprocessing
if __name__ == '__main__':
    main()
