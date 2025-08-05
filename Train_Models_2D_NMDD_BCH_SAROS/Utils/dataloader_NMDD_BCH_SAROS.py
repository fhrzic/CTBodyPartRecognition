from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import Utils.config_NMDD_BCH_SAROS as cfg
import pandas as pd
import numpy as np
import random
import os
import time
import diskcache
import cv2
import torch

#**********************************************************************#
# Data loading helpful functions
#**********************************************************************#
def get_cache(scope_str):
    """
    Cashing Descriptor function
    """
    return diskcache.FanoutCache('data_cache/' + scope_str,
                       shards=32,
                       timeout=1,
                       size_limit=3e11,
                       )

my_cache = get_cache('NMDD2')

@my_cache.memoize(typed=True)
def get_data_sample(sample, image_dimension):
    """
    Middleman function for cashing Fast is smooth, smooth is fast
    """
    _data = ProcessData(sample, image_dimension)
    _output = _data.get_sample()
    return _output



#**********************************************************************#
# Class for processing data --> loading image etc.
#**********************************************************************#
class ProcessData: 
    """
    Class for loading data from json
    """
    def __init__(self, sample:dict = None, 
                 image_dimension:int = 224,
                 ):
        """
        Init function.

        Args: 
            * sample, dict, dictionary contating following keys:
                --> "case_name", "image_path", "labels", "label_names"
            * image_dimensons, integer, scaling factor for images.
        """
        # Dataset
        self.dataset = sample["dataset"]

        # Case name
        self.image_path = sample["image_path"]

        # Obtain extract data
        _image_path = sample["image_path"]

        # Labels
        _label_path = sample["labels_path"]
       
        # Images   
        self.image = cv2.imread(_image_path, cv2.IMREAD_GRAYSCALE)
        # Set dimension
        self.image_dimension = image_dimension
        
        # Pad image and resize it
        self.__pad_image(self.image_dimension)

        # Obtain label
        self.label = self.__obtain_labels(label_path = _label_path)

    def __pad_image(self, image_desired_size:int):
        """
        Method for resizing the image to the desired dimensions
        First the image is resized then it is zero-padded to the desired 
        size given in the argument

        Args:
            * image_desired_dimension, int, new size of the image

        Output:
            * None, self.image is updated
        """
        # Grab the old_size
        _old_size = self.image.shape[:2] # old_size is in (height, width) format
        
        # Calculate new size
        _ratio = float(image_desired_size)/max(_old_size)
        _new_size = tuple([int(_x*_ratio) for _x in _old_size])

        # new_size should be in (width, height) format
        self.image = cv2.resize(self.image, (_new_size[1], _new_size[0]))
        
        # Calculate padding
        _delta_w = image_desired_size - _new_size[1]
        _delta_h = image_desired_size - _new_size[0]
        _top, _bottom = _delta_h//2, _delta_h-(_delta_h//2)
        _left, _right = _delta_w//2, _delta_w-(_delta_w//2)
        
        # Pad
        color = [0]
        
        self.image = cv2.copyMakeBorder(self.image, _top, _bottom, _left, _right, cv2.BORDER_CONSTANT, value=color)
        # Change to grayscale
        
        self.image = self.image

    def __obtain_labels(self, label_path):
        """
        Method for obtaining labels from xlsx file given in label path.

        Args:
            * label_path, str, path to the label xlsx folder

        Returns:
            * labels, np.array,  np.array, array contianing all labels (zero code encoded)
        """
        # Read dataframe
        _df = pd.read_excel(label_path, index_col=0)
        
        # Obtain label based on the dataset
        if self.dataset == "patches_NMDD":     
            _row = np.array(_df.loc["Value"].to_list())
            _row =  np.where(_row > 10, float(1), float(0))

        if self.dataset == "full_NMDD":       
            _row = np.array(_df.loc["True"].to_list())
            _row =  np.where(_row != 0, 1, 0)

        if self.dataset == "BCH":       
            _row = np.array(_df.loc[True].to_list())
            _row =  np.where(_row != 0, 1, 0)

        # Obtain label based on the dataset
        if self.dataset == "patches_SAROS":     
            _row = np.array(_df.loc["Value"].to_list())
            _row =  np.where(_row > 10, float(1), float(0))

        if self.dataset == "full_SAROS":       
            _row = np.array(_df.loc[True].to_list())
            _row =  np.where(_row != 0, 1, 0)


        # Return label
        return _row

    def get_sample(self):
        """
        Return sample --> loaded image and its annotations
        """
        return (self.image, self.label, self.image_path, self.dataset)
        

#**********************************************************************#
# Main class for handeling dataset
#**********************************************************************#
class NMDD_BCH_SAROS_dataset:
    """
    Class for handling the dataset. It is the plain train,valid,split

    Random seed is set to 5070

    Link for data:

    """

    def __init__(self, dataset_config: cfg.datasetConfig):
        """
        Init function which handles the dataset loading
        Args:
            * dataset_config, see config.py in Utils --> datasetConfig
            All neccessary data for config
                
        """

        # Set self config
        self.dataset_config = dataset_config
        # Set random seed
        random.seed(5070)
        
        # Obtain data lists --> all data as a list with structured paths     
        _s_time = time.time()
        self.data_list = self.__obtain_structured_data()
        _e_time = time.time()
        if self.dataset_config.verbose:
            print(f"Elapsed time for obtaining sturcutred data: {(_e_time-_s_time):.2f} sec")

        # Check if folds are activated
        if self.dataset_config.folds != None:
            try:
                _k, _n = self.dataset_config.folds
            except:
                print(f"dataset_config.folds has a wrong format! Expected [k ,n] but got {self.dataset_config.folds}")
            _data_list_splited = self.__split_to_folds(data_list=self.data_list,
                                                  fold = _k,
                                                  number_of_folds= _n)
            
            # Select dataset and store it in self.data 
            _selector = lambda _type, _data_sublists: {
                'train': _data_sublists[0],
                'test': _data_sublists[1],
            }[_type]   

            # Obtain sublists
            self.data_list = _selector(self.dataset_config.type, _data_list_splited)
        else:   
            _data_list_splited = self.__split_to_subsets(data_list = self.data_list, 
                                                        split_ratio = self.dataset_config.split_ratio)

            # Select dataset and store it in self.data 
            _selector = lambda _type, _data_sublists: {
                'train': _data_sublists[0],
                'valid': _data_sublists[1],
                'test': _data_sublists[2],
            }[_type]   

            # Obtain sublists
            self.data_list = _selector(self.dataset_config.type, _data_list_splited)

        # Take partition
        #if dataset_config.partition != None:
        #    self.data_list = self.data_list[0:int(dataset_config.partition * len(self.data_list))]

        # Get count it
        self.samples_cnt = len(self.data_list)
    
    def __len__(self):
        """
        Returns number of samples in dataset
        """
        return self.samples_cnt
    
    def shuffle_samples(self):
        """
        Simply shuffles the dataset -- necessary for batches
        """
        # Shuffeling dataset
        random.seed(5070)
        random.shuffle(self.data_list)

    def __getitem__(self, indx):
        """
        Gets data from the dataset

        Args:
            * indx, int, index of the data sample
        
        Output: data sample
        """
        # Get sample
        _sample = self.data_list[indx]
        
        # Obtain image (input) and annotation(output)
        _preprocesed_data = get_data_sample(_sample, 
                                            self.dataset_config.image_dimension, 
                                        )     
     
         # Image and image normalization
        _image = torch.from_numpy(_preprocesed_data[0])
        _image = _image.to(torch.float32)
        _image /= 255.0
       
         # label
        _label = torch.tensor(_preprocesed_data[1], dtype = torch.float)

        # Case
        _image_path = _preprocesed_data[2]

        # Dataset
        _dataset = _preprocesed_data[3]
        
        # Return
        return (_image, _label, _image_path, _dataset)
    
    def __obtain_structured_data(self)->list:
        """
        Function which return a list of data where each data sample is actually a dict with all usefull information.
        Each dict contains: 
            --> path_to_image, str
            --> labels, dict, key is label, value is a list having true label name and area of the label presented
            --> image_name, string, name of the image (usefull to extract ni and nii.gz files)
        
        Returns:
            * list containing 3 sublists - each for type of subset one list
        
        """
        # Main output lists
        _main_output_list_1 = []
        _main_output_list_2 = []
        _main_output_list_3 = []
        _main_output_list_4 = []
        _main_output_list_5 = []

        # Obtain labels from patches NMDD 
        # Read xlsx
        try:
            _mapping_df = pd.read_excel(self.dataset_config.mapping_xlsx_path_NMDD)
        except:
            print(f"Error while reading {_mapping_df}! Please check out data file path!")
            return -1
        
        # Read batches
        try:
            # Storage
            _good_images_list = []
            _file_list = []
            # Obtain files
            for _root, _, _files in os.walk(self.dataset_config.batch_xlsx_path_NMDD):
                for _file in _files:
                    # Check if _file name contains 'batch' and has a .xlsx extension
                    if 'batch' in _file and _file.endswith('.xlsx'):
                        _file_list.append(os.path.join(_root, _file))

            # Load content 
            for _file in _file_list:
                _df = pd.read_excel(_file)
                _good_images_list += _df["good"].to_list()
        except:
            print(f"Couldn not read batch xlsx in folder {self.dataset_config.batch_xlsx_path_NMDD}!. Please check out data file path!")
            return -1
        
        # Remap paths
        _images_paths = []
        for _path in _good_images_list:
            _column = _mapping_df[_mapping_df["Remaped_paths"] == _path]
            _org_path = _column["Original_paths"].to_list()
            _images_paths += _org_path

        # Go trought data and build data list
        for _index, _image_path in enumerate(_images_paths):
            # Obtan label path
            _dir = os.path.dirname(_image_path)
            _label_path = os.path.join(_dir, "labels.xlsx")                
            
            # Check label name
            if not os.path.isfile(_label_path):
                if self.dataset_config.verbose:
                    print(f"Could not find image: {_label_path}: Skipping!!!")
                continue

            # Check image path
            _image_path = os.path.join(_dir, "main.png")
                    
            # Check if file exist - skip otherwise
            if not os.path.isfile(_image_path):
                if self.dataset_config.verbose:
                    print(f"Could not find image: {_image_path}: Skipping!!!")
                continue
            
            _main_output_list_1.append({
                "good_name": _good_images_list[_index],
                "image_path": _image_path,
                "labels_path": _label_path,
                "dataset": "patches_NMDD"
            })          


        # Obtain labels from BCH
        assert os.path.exists(self.dataset_config.dir_path_BCH), f"Path to BCH data is not valid! {self.dataset_config.dir_path_BCH}!"
        _dir_paths = [_x[0] for _x in os.walk(self.dataset_config.dir_path_BCH)][1:]

        # Obtain data samples
        for _path in _dir_paths:
            # Define paths
            _label_path = os.path.join(_path, "labels.xlsx")
            _image_path = os.path.join(_path, "reducted_image.png")

            # Check label name
            if not os.path.isfile(_label_path):
                if self.dataset_config.verbose:
                    print(f"Could not find image: {_label_path}: Skipping!!!")
                continue

            # Check if file exist - skip otherwise
            if not os.path.isfile(_image_path):
                if self.dataset_config.verbose:
                    print(f"Could not find image: {_image_path}: Skipping!!!")
                continue
            
            # Add it
            _main_output_list_2.append({
                "good_name": "N/A",
                "image_path": _image_path,
                "labels_path": _label_path,
                "dataset": "BCH"
            })   
        
        # Obtain labels from NMDD full
        assert os.path.exists(self.dataset_config.dir_path_NMDD), f"Path to NMDD data is not valid! {self.dataset_config.dir_path_NMDD}!"
        _dir_paths = [_x[0] for _x in os.walk(self.dataset_config.dir_path_NMDD)][1:]

        # Obtain data samples
        for _path in _dir_paths:
            # Define paths
            _label_path = os.path.join(_path, "results.xlsx")
            _image_path = os.path.join(_path, "original.png")

            # Check label name
            if not os.path.isfile(_label_path):
                if self.dataset_config.verbose:
                    print(f"Could not find image: {_label_path}: Skipping!!!")
                continue

            # Check if file exist - skip otherwise
            if not os.path.isfile(_image_path):
                if self.dataset_config.verbose:
                    print(f"Could not find image: {_image_path}: Skipping!!!")
                continue
            
            # Add it
            _main_output_list_3.append({
                "good_name": "N/A",
                "image_path": _image_path,
                "labels_path": _label_path,
                "dataset": "full_NMDD"
            })   


        # Obtain labels from patches SAROS 
        # Read xlsx
        try:
            _mapping_df = pd.read_excel(self.dataset_config.mapping_xlsx_path_SAROS)
        except:
            print(f"Error while reading {_mapping_df}! Please check out data file path!")
            return -1
        
        # Read batches
        try:
            # Storage
            _good_images_list = []
            _file_list = []
            # Obtain files
            for _root, _, _files in os.walk(self.dataset_config.batch_xlsx_path_SAROS):
                for _file in _files:
                    # Check if _file name contains 'batch' and has a .xlsx extension
                    if 'batch' in _file and _file.endswith('.xlsx'):
                        _file_list.append(os.path.join(_root, _file))

            # Load content 
            for _file in _file_list:
                _df = pd.read_excel(_file)
                _good_images_list += _df["good"].to_list()
        except:
            print(f"Couldn not read batch xlsx in folder {self.dataset_config.batch_xlsx_path_SAROS}!. Please check out data file path!")
            return -1
        
        # Remap paths
        _images_paths = []
        for _path in _good_images_list:
            _column = _mapping_df[_mapping_df["Remaped_paths"] == _path]
            _org_path = _column["Original_paths"].to_list()
            _images_paths += _org_path

        # Go trought data and build data list
        for _index, _image_path in enumerate(_images_paths):
            # Obtan label path
            _dir = os.path.dirname(_image_path)
            _label_path = os.path.join(_dir, "labels.xlsx")                
            
            # Check label name
            if not os.path.isfile(_label_path):
                if self.dataset_config.verbose:
                    print(f"Could not find image: {_label_path}: Skipping!!!")
                continue

            # Check image path
            _image_path = os.path.join(_dir, "main.png")
                    
            # Check if file exist - skip otherwise
            if not os.path.isfile(_image_path):
                if self.dataset_config.verbose:
                    print(f"Could not find image: {_image_path}: Skipping!!!")
                continue
            
            _main_output_list_4.append({
                "good_name": _good_images_list[_index],
                "image_path": _image_path,
                "labels_path": _label_path,
                "dataset": "patches_SAROS"
            })          
        

        # Obtain labels from SAROS full
        assert os.path.exists(self.dataset_config.dir_path_SAROS), f"Path to SAROS data is not valid! {self.dataset_config.dir_path_SAROS}!"
        _dir_paths = [_x[0] for _x in os.walk(self.dataset_config.dir_path_SAROS)][1:]

        # Obtain data samples
        for _path in _dir_paths:
            # Define paths
            _label_path = os.path.join(_path, "labels.xlsx")
            _image_path = os.path.join(_path, "reducted_image.png")

            # Check label name
            if not os.path.isfile(_label_path):
                if self.dataset_config.verbose:
                    print(f"Could not find image: {_label_path}: Skipping!!!")
                continue

            # Check if file exist - skip otherwise
            if not os.path.isfile(_image_path):
                if self.dataset_config.verbose:
                    print(f"Could not find image: {_image_path}: Skipping!!!")
                continue
            
            # Add it
            _main_output_list_5.append({
                "good_name": "N/A",
                "image_path": _image_path,
                "labels_path": _label_path,
                "dataset": "full_SAROS"
            })   


        # Return data
        return [_main_output_list_1, _main_output_list_2, _main_output_list_3, _main_output_list_4, _main_output_list_5]
    
    def __split_to_folds(self, 
                         data_list:list = None, 
                         fold: int = 0, 
                         number_of_folds: int =5)->list:
        """
        Method which builds a trening. valid and test sets based on folds. In this case 
        there exists only training and test sets.

        Args:
            * fold, int, which fold should be retreived. Indexes from 0 to number_of_folds-1
            * number_of_folds, int, total number of folds in which te dataset will be splitted
        Output:
            * two lists of samples, train and test 
        """
        # Set random seed and shuffle data
        random.seed(5070)
        random.shuffle(data_list)

        # Find number of splits
        _chunks, _remainder = divmod(len(data_list), number_of_folds)

        # Generate folds
        _folds = [data_list[_i * _chunks + min(_i, _remainder):(_i + 1) * _chunks + min(_i + 1, _remainder)] for _i in range(number_of_folds)]

        # Split train and test
        _test_fold = _folds[fold]

        # Obtain training set
        _training_folds = []
        for _i, _fold in enumerate(_folds):
            if _i != fold:
                _training_folds += _fold
        # Retrieve
        return [_training_folds, _test_fold]
    
    def __split_to_subsets(self, data_list:list = None, split_ratio: float = 0.75)->list:
        """
        Method which builds train, valid, test subset of the given list by respecting the given split ratio
        
        Args:
            * data_list, list, list which containes samples
            * split_ratio, float, split ratio for dataset. valid=test=(1-_train_split_ratio)/2
        
        Output:
            * three lists of samples: train, valid, test.
        """
        # Create storages
        _train_data_split = []
        _validation_data_split = []
        _test_data_split = []

        # Obtain data lists
        for _list in data_list:
            # Set random seed and shuffle data
            random.seed(5070)
            random.shuffle(_list)

            # Set split ratio # Valid/test is what is left from
            _size = len(_list)
            _train_split_ratio = split_ratio
            _validation_split_ratio = (1-split_ratio) / 2         
        
            # Define storage
            _train_data_split += _list[: int(_train_split_ratio*(_size))] 
            _validation_data_split += _list[int(_train_split_ratio*(_size)):int((_train_split_ratio+_validation_split_ratio)*(_size))]
            _test_data_split += _list[int((_train_split_ratio+_validation_split_ratio)*(_size)):]
            
        # Return values
        random.shuffle(_train_data_split)
        random.shuffle(_validation_data_split)
        random.shuffle(_test_data_split)
        return([_train_data_split, _validation_data_split, _test_data_split])
    
#**********************************************************************#
# Generate dataloader
#**********************************************************************#
def init_dataloader(data_loader_params: cfg.loaderConfig, dataset_params: cfg.datasetConfig)->DataLoader:
    """
        Init of the  data loader. NOT TESTED FOR MULTIPLE GPU
        Creating wrapper arround data class. 
_train
        ARGS:
            * batch_size, int, size of the batch
            * num_wokers, int, number of workers for data loading 
            * use_gpu, boolean, if gpu used
            * dataset_info --> data_params object

        Output:
            * Torch DataLoader
    """
    _ds = NMDD_BCH_SAROS_dataset(dataset_params)

    _dl = DataLoader(
        _ds,
        batch_size = data_loader_params.batch_size,
        num_workers = data_loader_params.number_of_workers,
        pin_memory = data_loader_params.use_gpu,
    )  
    return _dl
