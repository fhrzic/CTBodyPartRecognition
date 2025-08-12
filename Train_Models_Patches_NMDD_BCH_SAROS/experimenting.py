from Utils.config_Patches_NMDD_BCH_SAROS import *
from Utils.dataloader_Patches_NMDD_BCH_SAROS import *
from Utils.models_Patches_NMDD_BCH_SAROS import *

import torch

def main():
    # Load configs 
    _data_config = datasetConfig()
    _loader_config = loaderConfig()
    _model_config = modelConfig()

    #Create data
    loaderConfig.use_gpu = 'cuda:2'
    loaderConfig.number_of_workers = 4
    loaderConfig.batch_size  = 32
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
    datasetConfig.image_dimension = 380
    datasetConfig.folds = None
    datasetConfig.type = "train"
    datasetConfig.verbose = False
    print(_data_config)

    # Train data loader
    #_train_dl = init_dataloader(_loader_config, _data_config)

    # Test datasetpwd
    datasetConfig.type = "valid"

    # Valid data loader
    _valid_dl = init_dataloader(_loader_config, _data_config)

    _model = VoxCNN_Patches(input_shape = (1,128,128,128), output_classes = 14)
    _model = _model.to("cuda:2")
    _aug_model = TransformToGray_AugmentedPatches()
    for _item in _valid_dl:
        _img, _label, _, _  = _item
        print("LABELA", _label.shape)
        print(_img.shape)
        
        _split_tensors = torch.split(_img, 1, dim=0)
        for _img in _split_tensors:
            _img_2 = _aug_model(_img)
            _img_2 = _img_2.to("cuda:2")
            print(_img_2.shape)
            _out = _model(_img_2)
            print(_img_2.shape)
            print(_out.shape)
        print("DONE")

    """
    # Model config
    modelConfig.gpu = 'cuda:2'
    modelConfig.number_of_output_classes = 14
    modelConfig.input_channels = 3
    modelConfig.input_shape = {"pretrained": False}
    modelConfig.model_name = 'R3D_18'
    modelConfig.loss = 'bce'
    modelConfig.valid_epochs = "1_1"
    modelConfig.early_stopping = 15
    modelConfig.learning_rate = 1e-3#"Auto" #"Auto" #1e-3
    modelConfig.opt_name = 'ADAMW'
    modelConfig.epochs = 500
    modelConfig.wandb = True
    modelConfig.info = f"R3D_18_3D_NMDD_BCH_SAROS"

    # Set augumentation method
    modelConfig.augmentation_model = 'GRAY_Augmentation'#'GRAY_Simple'#'GRAY_Augmentation'

    # Pretrained
    modelConfig.pretrained = False
    print(_model_config)


    _training = model_training_app(_train_dl, _valid_dl, _model_config, f"R3D_18_3D_NMDD_BCH_SAROS/")
    _training.freeze_unfreeze_model(freeze = False)
    _training.start_training()
    """

# Workarround for multiprocessing
if __name__ == '__main__':
    main()
