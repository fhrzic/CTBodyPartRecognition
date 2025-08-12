import pandas as pd
import os 
import matplotlib.pyplot as plt
from Utils.dataloader_NMDD_BCH_SAROS import *
from Utils.config_NMDD_BCH_SAROS import *

def obtain_data_in_each_subset(data_type = "train"):
    # Create dataset
    # Load configs 
    _data_config = datasetConfig()
    _loader_config = loaderConfig()
    _model_config = modelConfig()

    #Create data
    loaderConfig.use_gpu = 'cuda:0'
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
    datasetConfig.image_dimension = 300
    datasetConfig.folds = None
    datasetConfig.type = data_type
    datasetConfig.verbose = False
    print(_data_config)

    # Loader
    # Test data loade
    _dl = init_dataloader(_loader_config, _data_config)
   
    _true_list = []
    for _item in _dl:
        _, _true, _, _ = _item
        _true_list += _true.tolist()

    return _true_list

# Obtain list in dataloader
for _subset in ['train', 'valid', 'test']:
    _label_list = obtain_data_in_each_subset(_subset)

    # Obtain statistics
    _label_dict = {"skull": 0, 
            "shoulder": 0,
            "humerus": 0,
            "vertebrae_C": 0,
            "thorax": 0,
            "vertebrae_L": 0,
            "forearm": 0,
            "pelvis": 0,
            "femur": 0,
            "hand": 0,
            "patella": 0,
            "shin": 0,
            "tarsal": 0,
            "foot": 0}

    for _labels in _label_list:
        for _i, _key in enumerate(list(_label_dict.keys())):
            _label_dict[_key] += _labels[_i]

    # Plot it
    # Sort them
    _label_dict = {key: int(_label_dict[key]) for key in sorted(_label_dict.keys())}
    # Extract keys and values
    _keys = list(_label_dict.keys())
    _values = list(_label_dict.values())

    # Create a bar plot
    plt.bar(_keys, _values, color='skyblue')

    # Add titles and labels
    plt.title('Data-plot')
    plt.xlabel('Count')
    plt.ylabel('Amount')

    # Add text labels on the bars
    for _i, _value in enumerate(_values):
        plt.text(_i, _value + 0.5, str(_value), ha='center', va='bottom')

    # Rotate x-axis labels if needed
    plt.xticks(rotation=90)

    # Add grid lines for better readability
    plt.title(_subset)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig(f"{_subset}.png",  bbox_inches='tight')
    plt.close()