from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import Utils.config_NNMD as cfg
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

my_cache = get_cache('NNMD2')

@my_cache.memoize(typed=True)
def get_data_sample(sample, image_dimension, label_dimension, label_type):
    """
    Middleman function for cashing Fast is smooth, smooth is fast
    """
    _data = ProcessData(sample, image_dimension, label_dimension, label_type)
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
                 label_dimension: int = 35,
                 label_type: str = "cluster_remaped"):
        """
        Init function.

        Args: 
            * sample, dict, dictionary contating following keys:
                --> "case_name", "image_path", "labels"
                --> "labels": "original_label", "label_name", "cluster_remaped", "remaped"
            * image_dimensopns, integer, scaling factor for images.
            * label_type, str, type of label based on which the dataset will be build (output, variable)
            IF LABEL TYPE IS SET TO None, LABEL IS IGNORED and so is self.case_name

        """
        # Obtain extract data
        _image_path =sample["image_path"]

        # Labels
        if sample["labels"] == None:
            self.label = None
            self.case_name = None
        else:
            self.__obtain_label(labels = sample["labels"],
                            label_dimension=label_dimension,
                            label_type=label_type) 
       
        # Images   
        self.image = cv2.imread(_image_path, cv2.IMREAD_GRAYSCALE)
        
        # Set dimension
        self.image_dimension = image_dimension
        
        # Pad image and resize it
        self.__pad_image(self.image_dimension)

        # Case name
        self.case_name = sample["case_name"]

    def __obtain_label(self, 
                       labels:list = None, 
                       label_dimension: int = 35,
                       label_type: str = "cluster_remaped"):
        """
        Method for obtianing a vector of 1 and 0 of length label_dimension.
        Type of label is given by label type

        Args:
            * labels, list, list of labels dicts
            * label_dimension, int, length of label output vector
            * label_type, str, type of label to be retreived
        """

        # Create storage
        _label_list = []
        # Get indices
        for _label in labels:
            _sub_label = _label[label_type]
            _label_list.append(int(_sub_label))
        # Remove doubles
        _label_list = list(set(_label_list))

        # Reduce by 1 to obtain indexes
        _np_label_list = np.array(_label_list) - 1

        # Output np_array
        _output_np_array = np.zeros(label_dimension)
        _output_np_array[_np_label_list] = 1

        # Set label return value
        self.label = _output_np_array 

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

        
    def get_sample(self):
        """
        Return sample --> loaded image and its annotations
        """
        return (self.image, self.label, self.case_name)
        

#**********************************************************************#
# Main class for handeling dataset
#**********************************************************************#
class NNMD_dataset:
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
        
        # Check if data path is ok
        assert os.path.exists(self.dataset_config.imgs_png_home_path), f"Path {self.dataset_config.imgs_png_home_path} does not exist"
        assert os.path.exists(self.dataset_config.cheetsheet_xlsx_path), f"Path {self.dataset_config.cheetsheet_xlsx_path} does not exist"
        assert os.path.exists(self.dataset_config.labels_xlsx_path), f"Path {self.dataset_config.labels_xlsx_path} does not exist"
        
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
            _data_list_splited = self.__split_to_subsets(data_list = 
                                                        self.data_list, 
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
                                            self.dataset_config.label_dimension,
                                            self.dataset_config.label_type)     
     
         # Image and image normalization
        _image = torch.from_numpy(_preprocesed_data[0])
        _image = _image.to(torch.float32)
        _image /= 255.0
       
         # label
        _label = torch.tensor(_preprocesed_data[1], dtype = torch.float)

        # Case
        _case = _preprocesed_data[2]
        
        # Return
        return (_image, _label, _case)
    
    def __blacklist_labels(self, cheatsheet:pd.DataFrame = None)->list:
        """
        Based on the provided cheatsheet some labels are blacklisted.
        This function apply different criteria to remove labels.

        Criteria implemented:
            --> keep, go trough keep column and remove all that are marked as 0

        Input:
            * cheatsheet, pd.DataFrame, input dataframe which is obtained by cheatsheet_xlsx_path 
            given in confing file

        Output:
            * blacklist, list, list containing indexes of labels that should be removed from dataset
        """
        # Create storage
        _blacklist = []
        # Go through keep criteria
        try:
            _wrong_labels = np.array(cheatsheet["Label"].tolist()) * np.array(cheatsheet["Keep"].tolist())
            _wrong_labels = np.where(_wrong_labels == 0)[0]
            _wrong_labels += 1
            _blacklist = list(_wrong_labels)
        except:
            print(f"Error during blacklisting: possible missing collumns 'Label' and 'Keep' in cheetsheat xlsx!!!")

        # Return blacklist
        return _blacklist

    def __obtain_structured_data(self)->list:
        """
        Function which return a list of data where each data sample is actually a dict with all usefull information.
        Each dict contains: 
            --> path_to_image, str
            --> labels, dict, key is label, value is a list having true label name and area of the label presented
            --> image_name, string, name of the image (usefull to extract ni and nii.gz files)
        
        Returns:
            * list where every member is a dict with relevant information: path to image, gender and years.
        
        """

        # Save dir
        _main_output_list = []
        
        # Obtain xlsx data
        _cheatsheet_df = pd.read_excel(self.dataset_config.cheetsheet_xlsx_path)
        _data_df = pd.read_excel(self.dataset_config.labels_xlsx_path)

        # Obtain blacklist labels
        if self.dataset_config.blacklist:
            _blacklist_labels = self.__blacklist_labels(cheatsheet = _cheatsheet_df) 
            if self.dataset_config.verbose:
                print(f"Successfully generated blacklist!!!")
        
        # Remaping dictionaries 
        _remap_dict = self.dataset_config.remap_dict 
        _name_dict = _cheatsheet_df.set_index('Label')['Name'].to_dict()

        # Go trought data and build data list
        for _i, _row in _data_df.iterrows():
            # Obtain image name
            _image_name = _row["ID"]
            
            # Obtain image path
            _image_name_joined = "".join((_image_name, self.dataset_config.image_sufix, ".png"))
            _image_path = os.path.join(self.dataset_config.imgs_png_home_path, _image_name_joined)

            # Check if file exist - skip otherwise
            if not os.path.isfile(_image_path):
                if self.dataset_config.verbose:
                    print(f"Could not find image for id {_image_name}: Skipping!!!")
                continue
            
            # Obtain labels: original label, label name, prefix clustered name, remaped name.
            _labels = []
            for _id in range(1,len(_name_dict.keys())+1):
                # APPLY BLACKLIST
                if self.dataset_config.blacklist and _id in _blacklist_labels:
                    continue
                
                _area = _row[_id]
                # Check if label exists _area not nan 
                if pd.isna(_area):
                    continue
                
                # label name
                _original_label_name = _id

                # string label name
                _string_name = _name_dict[_id]

                # Derived names
                _prefix_gathered_name, _remaped_name, _reduced_name = _remap_dict[_id]

                # Add label to list
                _labels.append({
                    "original_label": _original_label_name,
                    "label_name": _string_name,
                    "cluster_remaped": _prefix_gathered_name,
                    "remaped": _remaped_name,
                    "reduced_cluster_remaped": _reduced_name
                })
            
            _main_output_list.append({
                "case_name": _image_name,
                "image_path": _image_path,
                "labels": _labels
            })          
        # Return data
        return _main_output_list
    
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
        # Set random seed and shuffle data
        random.seed(5070)
        random.shuffle(data_list)

        # Set split ratio # Valid/test is what is left from
        _size = len(data_list)
        _train_split_ratio = split_ratio
        _validation_split_ratio = (1-split_ratio) / 2         
       
        # Define storage
        _train_data_split = data_list[: int(_train_split_ratio*(_size))] 
        _validation_data_split = data_list[int(_train_split_ratio*(_size)):int((_train_split_ratio+_validation_split_ratio)*(_size))]
        _test_data_split = data_list[int((_train_split_ratio+_validation_split_ratio)*(_size)): ]
        
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
    _ds = NNMD_dataset(dataset_params)

    _dl = DataLoader(
        _ds,
        batch_size = data_loader_params.batch_size,
        num_workers = data_loader_params.number_of_workers,
        pin_memory = data_loader_params.use_gpu,
    )  
    return _dl
